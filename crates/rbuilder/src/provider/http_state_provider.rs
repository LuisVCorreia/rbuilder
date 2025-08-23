use super::{state_cache::*, StateProviderFactory};
use alloy_network::Ethereum;
use alloy_provider::{Provider, ProviderBuilder};
use alloy_rpc_types::{BlockId, BlockNumberOrTag};
use alloy_primitives::{Address, B256, Bytes, StorageKey, StorageValue, BlockNumber, BlockHash};
use reth_errors::ProviderResult;
use reth_primitives::{Account, Bytecode, Header};
use reth_provider::{
    errors::any::AnyError, AccountReader, BlockHashReader, ProviderError, StateProofProvider, StateProvider, StateProviderBox, StateRootProvider, HashedPostStateProvider, StorageRootProvider
};
use reth_trie::{updates::TrieUpdates, AccountProof, HashedPostState, TrieInput, MultiProof, MultiProofTargets, StorageMultiProof, StorageProof, HashedStorage};
use alloy_eips::BlockNumHash;
use revm::database::BundleState;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::runtime::{Builder, Handle};

#[derive(Clone)]
pub struct HttpStateProviderFactory {
    provider: Arc<dyn Provider<Ethereum> + Send + Sync>,
    cache_db: CacheDB,
}

// runtime-agnostic block_on
pub fn block_on_compat<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    if let Ok(handle) = Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio rt")
            .block_on(fut)
    }
}

impl HttpStateProviderFactory {
    pub fn new_with_url_and_cache(url: &str, cache_path: PathBuf) -> Self {
        let provider = ProviderBuilder::new()
            .network::<Ethereum>()
            .on_http(url.parse().expect("Failed to parse provider URL"));

        let state_cache = block_on_compat(StateCache::new(&cache_path))
            .expect("Failed to open or create state cache");

        let cache_db = CacheDB::new(state_cache);

        Self {
            provider: Arc::new(provider),
            cache_db,
        }
    }
}

impl StateProviderFactory for HttpStateProviderFactory {
    fn history_by_block_number(&self, block_number: BlockNumber) -> ProviderResult<StateProviderBox> {
        let provider = self.provider.clone();
        let res = block_on_compat(async {
            provider
                .get_block_by_number(BlockNumberOrTag::Number(block_number))
                .await
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?
        .ok_or_else(|| ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Block not found",
        ))))?;

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
            self.cache_db.clone(),
        ))
    }

    fn latest(&self) -> ProviderResult<StateProviderBox> {
        let provider = self.provider.clone();
        let res = block_on_compat(async { provider.get_block(BlockId::latest()).await })
            .map_err(|e| ProviderError::Other(AnyError::new(e)))?
            .ok_or_else(|| ProviderError::Other(AnyError::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Block not found",
            ))))?;

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
            self.cache_db.clone(),
        ))
    }

    fn history_by_block_hash(&self, block_hash: B256) -> ProviderResult<StateProviderBox> {
        let provider = self.provider.clone();
        let res = block_on_compat(async { provider.get_block_by_hash(block_hash).await })
            .map_err(|e| ProviderError::Other(AnyError::new(e)))?
            .ok_or_else(|| ProviderError::Other(AnyError::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Block not found",
            ))))?;

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
            self.cache_db.clone(),
        ))
    }
    
    fn header(&self, block_hash: &BlockHash) -> ProviderResult<Option<Header>> {
        let block_hash = *block_hash;
        let provider = self.provider.clone();
        let res = block_on_compat(async { provider.get_block_by_hash(block_hash).await })
            .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
        Ok(res.map(|block| block.header.inner))
    }

    fn last_block_number(&self) -> ProviderResult<BlockNumber> {
        let provider = self.provider.clone();
        let block_number = block_on_compat(async { provider.get_block_number().await })
            .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
        Ok(block_number)
    }

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        let provider = self.provider.clone();
        let block = block_on_compat(async {
            provider
                .get_block_by_number(BlockNumberOrTag::Number(number))
                .await
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
        Ok(block.map(|b| b.header.hash))
    }

    fn best_block_number(&self) -> ProviderResult<BlockNumber> {
        self.last_block_number()
    }

    fn header_by_number(&self, num: u64) -> ProviderResult<Option<Header>> {
        let provider = self.provider.clone();
        let block = block_on_compat(async {
            provider
                .get_block_by_number(BlockNumberOrTag::Number(num))
                .await
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
        Ok(block.map(|b| b.header.inner))
    }

    fn root_hasher(&self, _parent_num_hash: BlockNumHash) -> ProviderResult<Box<dyn super::RootHasher>> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not implemented - not needed for backtesting",
        ))))
    }
}

pub struct HttpStateProvider {
    provider: Arc<dyn Provider<Ethereum> + Send + Sync>,
    hash: B256,
    cache_db: CacheDB,
}

impl HttpStateProvider {
    pub fn new(
        provider: Arc<dyn Provider<Ethereum> + Send + Sync>,
        hash: B256,
        cache_db: CacheDB,
    ) -> Box<Self> {
        Box::new(Self { provider, hash, cache_db })
    }
}

impl StateProvider for HttpStateProvider {
    fn storage(&self, address: Address, storage_key: StorageKey) -> ProviderResult<Option<StorageValue>> {
        let cache_key = format!("storage:{:?}:{:?}:{:?}", self.hash, address, storage_key);
        block_on_compat(async {
            if let Some(cached_value) = self.cache_db.storage.get(&cache_key).await {
                return Ok(Some(cached_value.into()));
            }
            let block_id = BlockId::hash(self.hash);
            let res = self.provider
                .get_storage_at(address, storage_key.into())
                .block_id(block_id)
                .await
                .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            self.cache_db.storage.set(&cache_key, &res).await;
            Ok(Some(res.into()))
        })
    }

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        let cache_key = format!("bytecode:{:?}:{:?}", self.hash, code_hash);
        block_on_compat(async {
            if let Some(cached_bytecode) = self.cache_db.bytecode.get(&cache_key).await {
                return Ok(Some(cached_bytecode));
            }
            let block_id = BlockId::hash(self.hash);
            let code_hash_val = *code_hash;
            let code: Option<Bytes> = self.provider
                .client()
                .request("debug_codeByHash", (code_hash_val, block_id))
                .await
                .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            match code {
                Some(bytes) if !bytes.is_empty() => {
                    let bytecode = Bytecode::new_raw(bytes);
                    self.cache_db.bytecode.set(&cache_key, &bytecode).await;
                    Ok(Some(bytecode))
                }
                _ => Ok(None),
            }
        })
    }
    
    fn account_nonce(&self, address: &Address) -> ProviderResult<Option<u64>> {
        match self.basic_account(address)? {
            Some(account) => Ok(Some(account.nonce)),
            None => Ok(None),
        }
    }
}

impl BlockHashReader for HttpStateProvider {
    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        let cache_key = format!("block_hash:{}", number);
        block_on_compat(async {
            if let Some(cached_hash) = self.cache_db.block_hashes.get(&cache_key).await {
                return Ok(Some(cached_hash));
            }
            let block = self.provider
                .get_block_by_number(BlockNumberOrTag::Number(number))
                .await
                .map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            if let Some(ref b) = block {
                self.cache_db.block_hashes.set(&cache_key, &b.header.hash).await;
            }
            Ok(block.map(|b| b.header.hash))
        })
    }

    fn canonical_hashes_range(&self, start: BlockNumber, end: BlockNumber) -> ProviderResult<Vec<B256>> {
        let mut res = vec![];
        for i in start..end {
            if let Some(hash) = self.block_hash(i)? {
                res.push(hash);
            }
        }
        Ok(res)
    }
}

impl AccountReader for HttpStateProvider {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        let cache_key = format!("account:{:?}:{:?}", self.hash, address);
        block_on_compat(async {
            if let Some(cached_account) = self.cache_db.accounts.get(&cache_key).await {
                return Ok(Some(cached_account));
            }
            let block_id = BlockId::hash(self.hash);
            let (balance_res, nonce_res, code_res) = tokio::join!(
                self.provider.get_balance(*address).block_id(block_id),
                self.provider.get_transaction_count(*address).block_id(block_id),
                self.provider.get_code_at(*address).block_id(block_id)
            );
            let balance = balance_res.map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            let nonce = nonce_res.map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            let code = code_res.map_err(|e| ProviderError::Other(AnyError::new(e)))?;
            let bytecode_hash = if code.is_empty() { None } else { Some(alloy_primitives::keccak256(&code)) };
            let result = Account { balance, nonce, bytecode_hash };
            self.cache_db.accounts.set(&cache_key, &result).await;
            Ok(Some(result))
        })
    }
}

impl StateRootProvider for HttpStateProvider {
    fn state_root(&self, _hashed_state: HashedPostState) -> ProviderResult<B256> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }
    fn state_root_from_nodes(&self, _input: TrieInput) -> ProviderResult<B256> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }
    fn state_root_with_updates(&self, _hashed_state: HashedPostState) -> ProviderResult<(B256, TrieUpdates)> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }
    fn state_root_from_nodes_with_updates(&self, _input: TrieInput) -> ProviderResult<(B256, TrieUpdates)> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }
}
impl StorageRootProvider for HttpStateProvider {
    fn storage_root(&self, _address: Address, _hashed_storage: HashedStorage) -> ProviderResult<B256> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage root calculation not supported for HTTP provider",
        ))))
    }
    fn storage_proof(&self, _address: Address, _slot: B256, _hashed_storage: HashedStorage) -> ProviderResult<StorageProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage proof not supported for HTTP provider",
        ))))
    }
    fn storage_multiproof(&self, _address: Address, _slots: &[B256], _hashed_storage: HashedStorage) -> ProviderResult<StorageMultiProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage multiproof not supported for HTTP provider",
        ))))
    }
}
impl StateProofProvider for HttpStateProvider {
    fn proof(&self, _input: TrieInput, _address: Address, _slots: &[B256]) -> ProviderResult<AccountProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Proof generation not supported for HTTP provider",
        ))))
    }
    fn multiproof(&self, _input: TrieInput, _targets: MultiProofTargets) -> ProviderResult<MultiProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Multiproof generation not supported for HTTP provider",
        ))))
    }
    fn witness(&self, _input: TrieInput, _target: HashedPostState) -> ProviderResult<Vec<Bytes>> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Witness generation not supported for HTTP provider",
        ))))
    }
}
impl HashedPostStateProvider for HttpStateProvider {
    fn hashed_post_state(&self, _bundle_state: &BundleState) -> HashedPostState {
        HashedPostState::default()
    }
}
