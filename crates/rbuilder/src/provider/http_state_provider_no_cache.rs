use super::StateProviderFactory;

use alloy_network::Ethereum;

use alloy_provider::{Provider, ProviderBuilder};

use alloy_rpc_types::{BlockId, BlockNumberOrTag};

use alloy_primitives::{Address, BlockHash, BlockNumber, Bytes, StorageKey, StorageValue, B256};

use reth_errors::ProviderResult;

use reth_primitives::{Account, Bytecode, Header};

use reth_provider::{
    errors::any::AnyError, AccountReader, BlockHashReader, HashedPostStateProvider, ProviderError,
    StateProofProvider, StateProvider, StateProviderBox, StateRootProvider, StorageRootProvider,
};

use reth_trie::{
    updates::TrieUpdates, AccountProof, HashedPostState, HashedStorage, MultiProof,
    MultiProofTargets, StorageMultiProof, StorageProof, TrieInput,
};

use alloy_eips::BlockNumHash;

use revm::database::BundleState;

use std::sync::Arc;

/// The factory now holds a shared, thread-safe provider instance as a trait object.

#[derive(Clone)]

pub struct HttpStateProviderFactory {
    /// The provider is stored as a trait object `Arc<dyn Provider<...>>`.

    /// This abstracts away the concrete provider type returned by the builder (e.g., `FillProvider`),

    /// making the code more flexible and robust.

    /// The correct generic order is Network, then Transport.
    provider: Arc<dyn Provider<Ethereum> + Send + Sync>,
}

impl HttpStateProviderFactory {
    /// Creates a new factory. The HTTP provider is initialized here once and reused.

    pub fn new_with_url(url: &str) -> Self {
        // The builder returns a concrete `FillProvider` type that implements the `Provider` trait.

        let provider = ProviderBuilder::new()
            .network::<Ethereum>()
            .on_http(url.parse().expect("Failed to parse provider URL"));

        // We wrap the concrete provider in an `Arc` and coerce it into a trait object.

        Self {
            provider: Arc::new(provider),
        }
    }
}

impl StateProviderFactory for HttpStateProviderFactory {
    // Standard ETH JSON RPC implementations (like Alchemy) support these methods

    // We use eth_getBlockByNumber, eth_getBlockByHash, eth_blockNumber, etc.

    fn history_by_block_number(
        &self,

        block_number: BlockNumber,
    ) -> ProviderResult<StateProviderBox> {
        // Clone the Arc, which is a cheap operation, to get a new reference to the shared provider.

        let provider = self.provider.clone();

        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                provider
                    .get_block_by_number(BlockNumberOrTag::Number(block_number))
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?
        .ok_or_else(|| {
            ProviderError::Other(AnyError::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Block not found",
            )))
        })?;

        // Pass the shared provider trait object to the state provider instance.

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
        ))
    }

    fn latest(&self) -> ProviderResult<StateProviderBox> {
        let provider = self.provider.clone();

        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { provider.get_block(BlockId::latest()).await })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?
        .ok_or_else(|| {
            ProviderError::Other(AnyError::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Block not found",
            )))
        })?;

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
        ))
    }

    fn history_by_block_hash(&self, block_hash: B256) -> ProviderResult<StateProviderBox> {
        let provider = self.provider.clone();

        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { provider.get_block_by_hash(block_hash).await })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?
        .ok_or_else(|| {
            ProviderError::Other(AnyError::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Block not found",
            )))
        })?;

        Ok(HttpStateProvider::new(
            self.provider.clone(),
            res.header.hash,
        ))
    }

    /// Get header by block hash

    fn header(&self, block_hash: &BlockHash) -> ProviderResult<Option<Header>> {
        let block_hash = *block_hash;

        let provider = self.provider.clone();

        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { provider.get_block_by_hash(block_hash).await })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(res.map(|block| block.header.inner))
    }

    fn last_block_number(&self) -> ProviderResult<BlockNumber> {
        let provider = self.provider.clone();

        let block_number = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { provider.get_block_number().await })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(block_number)
    }

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        let provider = self.provider.clone();

        let block = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                provider
                    .get_block_by_number(BlockNumberOrTag::Number(number))
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(block.map(|b| b.header.hash))
    }

    fn best_block_number(&self) -> ProviderResult<BlockNumber> {
        self.last_block_number()
    }

    fn header_by_number(&self, num: u64) -> ProviderResult<Option<Header>> {
        let provider = self.provider.clone();

        let block = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                provider
                    .get_block_by_number(BlockNumberOrTag::Number(num))
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(block.map(|b| b.header.inner))
    }

    fn root_hasher(
        &self,
        _parent_num_hash: BlockNumHash,
    ) -> ProviderResult<Box<dyn super::RootHasher>> {
        // State root calculation is not needed for backtesting purposes

        // This would require complex state root calculations

        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not implemented - not needed for backtesting",
        ))))
    }
}

/// The state provider now holds a reference to the shared provider trait object.

pub struct HttpStateProvider {
    provider: Arc<dyn Provider<Ethereum> + Send + Sync>,

    hash: B256,
}

impl HttpStateProvider {
    /// Updated `new` to accept the shared provider trait object.

    pub fn new(provider: Arc<dyn Provider<Ethereum> + Send + Sync>, hash: B256) -> Box<Self> {
        Box::new(Self { provider, hash })
    }
}

impl StateProvider for HttpStateProvider {
    // Using standard ETH JSON RPC calls: eth_getStorageAt, eth_getBalance,

    // eth_getTransactionCount, eth_getCode

    // Also using debug_codeByHash for bytecode_by_hash (available in reth)

    /// Get storage of given account.

    fn storage(
        &self,

        address: Address,

        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        let block_id = BlockId::hash(self.hash);

        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Use the shared provider directly.

                self.provider
                    .get_storage_at(address, storage_key.into())
                    .block_id(block_id)
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(Some(res.into()))
    }

    /// Get account code by its hash

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        let block_id = BlockId::hash(self.hash);

        let code_hash = *code_hash;

        let code: Option<Bytes> = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.provider
                    .client()
                    .request("debug_codeByHash", (code_hash, block_id))
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        match code {
            Some(bytes) => {
                if bytes.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(Bytecode::new_raw(bytes)))
                }
            }

            None => Ok(None),
        }
    }

    /// Get account nonce

    fn account_nonce(&self, address: &Address) -> ProviderResult<Option<u64>> {
        match self.basic_account(address)? {
            Some(account) => Ok(Some(account.nonce)),

            None => Ok(None),
        }
    }
}

impl BlockHashReader for HttpStateProvider {
    /// Get the hash of the block with the given number. Returns `None` if no block with this number

    /// exists.

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.provider
                    .get_block_by_number(BlockNumberOrTag::Number(number))
                    .await
            })
        })
        .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

        Ok(res.map(|block| block.header.hash))
    }

    fn canonical_hashes_range(
        &self,

        start: BlockNumber,

        end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        let mut res = vec![];

        for i in start..end {
            let block = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    self.provider
                        .get_block_by_number(BlockNumberOrTag::Number(i))
                        .await
                })
            })
            .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

            if let Some(block) = block {
                res.push(block.header.hash);
            }
        }

        Ok(res)
    }
}

impl AccountReader for HttpStateProvider {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        let block_id = BlockId::hash(self.hash);

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Get balance, nonce, and code using the shared provider.

                let balance = self
                    .provider
                    .get_balance(*address)
                    .block_id(block_id)
                    .await
                    .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

                let nonce = self
                    .provider
                    .get_transaction_count(*address)
                    .block_id(block_id)
                    .await
                    .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

                let code = self
                    .provider
                    .get_code_at(*address)
                    .block_id(block_id)
                    .await
                    .map_err(|e| ProviderError::Other(AnyError::new(e)))?;

                // Calculate code hash

                let bytecode_hash = if code.is_empty() {
                    None
                } else {
                    Some(alloy_primitives::keccak256(&code))
                };

                Ok::<Account, ProviderError>(Account {
                    balance,

                    nonce,

                    bytecode_hash,
                })
            })
        })?;

        Ok(Some(result))
    }
}

// No changes needed for the following trait implementations as they don't make network calls.

// They correctly return an error indicating they are not supported.

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

    fn state_root_with_updates(
        &self,

        _hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }

    fn state_root_from_nodes_with_updates(
        &self,

        _input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "State root calculation not supported for HTTP provider",
        ))))
    }
}

impl StorageRootProvider for HttpStateProvider {
    fn storage_root(
        &self,

        _address: Address,

        _hashed_storage: HashedStorage,
    ) -> ProviderResult<B256> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage root calculation not supported for HTTP provider",
        ))))
    }

    fn storage_proof(
        &self,

        _address: Address,

        _slot: B256,

        _hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage proof not supported for HTTP provider",
        ))))
    }

    fn storage_multiproof(
        &self,

        _address: Address,

        _slots: &[B256],

        _hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Storage multiproof not supported for HTTP provider",
        ))))
    }
}

impl StateProofProvider for HttpStateProvider {
    fn proof(
        &self,

        _input: TrieInput,

        _address: Address,

        _slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        Err(ProviderError::Other(AnyError::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Proof generation not supported for HTTP provider",
        ))))
    }

    fn multiproof(
        &self,

        _input: TrieInput,

        _targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
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
        // For HTTP provider, we can't generate hashed post state

        // This would require complex state calculations

        HashedPostState::default()
    }
}
