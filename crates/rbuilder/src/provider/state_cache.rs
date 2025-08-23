//! A simple SQLite-based key-value cache for storing blockchain state using sqlx.
//!
//! This module provides a `StateCache` struct that encapsulates interactions
//! with a SQLite database via an `sqlx` connection pool. It is used by the
//! `HttpStateProvider` to persist state data fetched from a remote RPC node.

use alloy_primitives::{B256, U256};
use reth_primitives::{Account, Bytecode};
use serde::{de::DeserializeOwned, Serialize};
use sqlx::{
    sqlite::{SqlitePool, SqlitePoolOptions},
    Executor, Row,
};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::Path;

/// A thread-safe, shareable database cache using an `sqlx` connection pool.
/// The pool is designed for concurrent access and is cloneable.
#[derive(Clone, Debug)]
pub struct StateCache {
    pool: SqlitePool,
}

impl StateCache {
    /// Creates a new `StateCache` or opens an existing one at the given path.
    /// This function is async to align with `sqlx`'s async-first design.
    pub async fn new(db_path: &Path) -> Result<Self, sqlx::Error> {
        // Ensure the parent directory exists.
        if let Some(parent) = db_path.parent() {
            // Use tokio's async fs operation.
            tokio::fs::create_dir_all(parent).await.map_err(sqlx::Error::Io)?;
        }

        let db_url = format!("sqlite:{}?mode=rwc", db_path.to_str().unwrap());
        let pool = SqlitePoolOptions::new().max_connections(5).connect(&db_url).await?;

        let pragmas = [
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA cache_size = -10000;",
        ];
        for pragma in pragmas.iter() {
            pool.execute(*pragma).await?;
        }

        // Create the key-value table if it doesn't exist.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS kv_store (
                key   TEXT PRIMARY KEY,
                value BLOB NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Retrieves and deserializes a value from the cache.
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, sqlx::Error> {
        let row_opt = sqlx::query("SELECT value FROM kv_store WHERE key = ?1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;

        match row_opt {
            Some(row) => {
                let blob: Vec<u8> = row.try_get("value")?;
                // Convert bincode error into an sqlx::Error::Decode
                let value: T =
                    bincode::deserialize(&blob).map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Serializes and sets a value in the cache using an "upsert".
    pub async fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<(), sqlx::Error> {
        // FIX: Map the bincode error to a valid sqlx::Error variant, like Protocol.
        let blob = bincode::serialize(value).map_err(|e| sqlx::Error::Protocol(e.to_string()))?;

        sqlx::query(
            "INSERT INTO kv_store (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        )
        .bind(key)
        .bind(blob)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

/// A helper struct to create type-safe cache accessors.
#[derive(Clone)]
pub struct CacheAccessor<T> {
    cache: StateCache,
    _phantom: PhantomData<T>,
}

impl<T> CacheAccessor<T>
where
    // Add `Send` bound because it's used across await points.
    T: Serialize + DeserializeOwned + Send,
{
    pub fn new(cache: StateCache) -> Self {
        Self { cache, _phantom: PhantomData }
    }

    /// Gets a value from the cache. This is now an async operation.
    pub async fn get(&self, key: &str) -> Option<T> {
        self.cache.get(key).await.unwrap_or(None)
    }

    /// Sets a value in the cache. This is now an async operation.
    pub async fn set(&self, key: &str, value: &T) {
        if let Err(e) = self.cache.set(key, value).await {
            eprintln!("Failed to write to cache: {}", e);
        }
    }
}

// Type aliases for different cache accessors remain the same for readability.
pub type AccountCache = CacheAccessor<Account>;
pub type StorageCache = CacheAccessor<U256>;
pub type BytecodeCache = CacheAccessor<Bytecode>;
pub type BlockHashCache = CacheAccessor<B256>;

/// A convenient container for all cache accessors.
#[derive(Clone)]
pub struct CacheDB {
    pub accounts: AccountCache,
    pub storage: StorageCache,
    pub bytecode: BytecodeCache,
    pub block_hashes: BlockHashCache,
}

impl CacheDB {
    /// Creates a new `CacheDB` instance from a `StateCache`.
    pub fn new(cache: StateCache) -> Self {
        Self {
            accounts: AccountCache::new(cache.clone()),
            storage: StorageCache::new(cache.clone()),
            bytecode: BytecodeCache::new(cache.clone()),
            block_hashes: BlockHashCache::new(cache),
        }
    }
}
