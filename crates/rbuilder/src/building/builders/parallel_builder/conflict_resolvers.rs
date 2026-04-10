use ahash::{HashMap, HashSet};
use alloy_primitives::{Address, U256};
use rbuilder_primitives::evm_inspector::{SlotKey, UsedStateTrace};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use rand::{seq::SliceRandom, SeedableRng};
use reth::providers::StateProvider;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;

use super::{
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
};

use crate::building::{
    BlockBuildingContext, BlockState, ExecutionError, ExecutionResult, OrderErr, PartialBlock,
    ThreadBlockBuildingContext,
};
use rbuilder_primitives::{OrderId, SimulatedOrder};

/// Context for resolving conflicts in merging tasks.

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ResolverContext {
    #[derivative(Debug = "ignore")]
    pub state: Arc<dyn StateProvider>,
    pub ctx: BlockBuildingContext,
    pub cancellation_token: CancellationToken,
    pub simulation_cache: Arc<SharedSimulationCache>,
}

impl ResolverContext {
    /// Creates a new `ResolverContext`.
    ///
    /// # Arguments
    ///
    /// * `provider_factory` - Factory for creating state providers.
    /// * `ctx` - Context for block building.
    /// * `cancellation_token` - Token for cancelling operations.
    /// * `cache` - Optional cached reads for optimization.
    /// * `simulation_cache` - Shared cache for simulation results.
    pub fn new(
        state: Arc<dyn StateProvider>,
        ctx: BlockBuildingContext,
        cancellation_token: CancellationToken,
        simulation_cache: Arc<SharedSimulationCache>,
    ) -> Self {
        ResolverContext {
            state,
            ctx,
            cancellation_token,
            simulation_cache,
        }
    }

    /// Runs a merging task and returns the best [ResolutionResult] found.
    ///
    /// # Arguments
    ///
    /// * `task` - The [ConflictTask] to run.
    ///
    /// # Returns
    ///
    /// The best [ResolutionResult] and corresponding sequence of order indices found.
    pub fn run_conflict_task(&mut self, task: ConflictTask) -> Result<ResolutionResult> {
        trace!(
            "run_conflict_task: {:?} with algorithm {:?}",
            task.group.id,
            task.algorithm
        );

        let sequence_to_try = generate_sequences_of_orders_to_try(&task);

        let mut best_resolution_result = ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        };

        for sequence_of_orders in sequence_to_try {
            let (resolution_result, _state) =
                self.process_sequence_of_orders(sequence_of_orders, &task, self.state.clone())?;
            self.update_best_result(resolution_result, &mut best_resolution_result);
        }

        trace!(
            "Resolved conflict task {:?} with profit: {:?} and algorithm: {:?}",
            task.group.id,
            best_resolution_result.total_profit,
            task.algorithm
        );

        if task.group.orders.len() > 1700 && task.algorithm == Algorithm::DexDirectionBalanced {
            let mut greedy_best = ResolutionResult {
                total_profit: U256::ZERO,
                sequence_of_orders: vec![],
            };
            for seq in generate_greedy_sequence(&task, false) {
                let (result, _) =
                    self.process_sequence_of_orders(seq, &task, self.state.clone())?;
                self.update_best_result(result, &mut greedy_best);
            }
            let dex_profit = best_resolution_result.total_profit;
            let greedy_profit = greedy_best.total_profit;
            let (diff_sign, diff) = if dex_profit >= greedy_profit {
                ("+", dex_profit - greedy_profit)
            } else {
                ("-", greedy_profit - dex_profit)
            };
            let dex_len = best_resolution_result.sequence_of_orders.len();
            let greedy_len = greedy_best.sequence_of_orders.len();
            eprintln!(
                "[DBG grp=326] COMPARE: DexBalanced profit={dex_profit} txs={dex_len} vs Greedy profit={greedy_profit} txs={greedy_len} (dex{diff_sign}{diff})",
            );
        }

        Ok(best_resolution_result)
    }

    /// Updates the best result if a better one is found.
    ///
    /// # Arguments
    ///
    /// * `new_result` - The newly processed result.
    /// * `best_result` - The current best result to update.
    fn update_best_result(
        &mut self,
        new_result: ResolutionResult,
        best_result: &mut ResolutionResult,
    ) {
        if best_result.total_profit < new_result.total_profit {
            best_result.total_profit = new_result.total_profit;
            best_result.sequence_of_orders = new_result.sequence_of_orders;
        }
    }

    /// Processes a single sequence of orders, utilizing the simulation cache.
    ///
    /// # Arguments
    ///
    /// * `sequence_of_orders` - The order of transaction indices to process.
    /// * `task` - The current conflict task.
    /// * `state_provider` - The state provider for the current block.
    ///
    /// # Returns
    ///
    /// A tuple containing the resolution result and the final block state.
    fn process_sequence_of_orders(
        &mut self,
        sequence_of_orders: Vec<usize>,
        task: &ConflictTask,
        state_provider: Arc<dyn StateProvider>,
    ) -> Result<(ResolutionResult, BlockState)> {
        // @todo actually reuse it for the duration of the block
        let mut local_ctx = ThreadBlockBuildingContext::default();

        let order_id_to_index = self.initialize_order_id_to_index_map(task);
        let full_sequence_of_orders = self.initialize_full_order_ids_vec(&sequence_of_orders, task);

        // Check for cached simulation state
        let (cached_state_option, cached_up_to_index) = self
            .simulation_cache
            .get_cached_state(&full_sequence_of_orders);

        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(state_provider);
        partial_block.pre_block_call(&self.ctx, &mut local_ctx, &mut state)?;

        // Initialize sequenced_order_result
        let mut sequenced_order_result =
            self.initialize_result_order_sequence(&cached_state_option, &order_id_to_index);

        let mut total_profit = cached_state_option
            .as_ref()
            .map_or(U256::ZERO, |cached| cached.total_profit);

        let mut per_order_profits = cached_state_option
            .as_ref()
            .map_or(Vec::new(), |cached| cached.per_order_profits.clone());

        // Prepare the sequence of orders to try, skipping already cached orders
        let mut remaining_orders = sequence_of_orders[cached_up_to_index..].to_vec();
        remaining_orders.reverse(); // Use as a stack: pop from the end

        let mut pending_orders: HashMap<(Address, u64), usize> = HashMap::default();

        // Processing loop
        while let Some(order_idx) = remaining_orders.pop() {
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            let sim_order = &task.group.orders[order_idx];

            // For DexDirectionBalanced: orders that the static analysis classified as
            // price-neutral must also prove to be price-neutral in the actual execution
            // context (the block state may differ from top-of-block simulation).
            // If the actual trace shows a price change, roll back the order and skip it.
            let is_predicted_neutral = task.algorithm == Algorithm::DexDirectionBalanced
                && sim_order
                    .used_state_trace
                    .as_ref()
                    .map(|t| t.is_price_neutral())
                    .unwrap_or(false);

            let result = if is_predicted_neutral {
                partial_block.commit_order(
                    sim_order,
                    &self.ctx,
                    &mut local_ctx,
                    &mut state,
                    &|_, actual_trace: Option<&UsedStateTrace>| {
                        if actual_trace.map(|t| t.is_price_neutral()).unwrap_or(true) {
                            Ok(())
                        } else {
                            Err(ExecutionError::OrderError(OrderErr::NotPriceNeutral))
                        }
                    },
                )?
            } else {
                partial_block.commit_order(
                    sim_order,
                    &self.ctx,
                    &mut local_ctx,
                    &mut state,
                    &|_, _| Ok(()),
                )?
            };

            match result {
                Ok(res) => {
                    if is_predicted_neutral {
                        let expected =
                            sim_order.sim_value.full_profit_info().coinbase_profit();
                        let actual = res.coinbase_profit;
                        eprintln!(
                            "[DBG neutral-profit] order={:?} expected={expected} actual={actual} match={}",
                            sim_order.order.id(),
                            expected == actual
                        );
                    }
                    self.handle_successful_commit(
                        res,
                        sim_order,
                        order_idx,
                        &mut pending_orders,
                        &mut remaining_orders,
                        &mut sequenced_order_result,
                        &mut total_profit,
                        &mut per_order_profits,
                    )
                }
                Err(err) => {
                    if matches!(err, ExecutionError::OrderError(OrderErr::NotPriceNeutral)) {
                        eprintln!(
                            "[DBG neutral-discarded] order {:?} discarded: changed pool prices in actual execution",
                            sim_order.order.id()
                        );
                    }
                    self.handle_err(&err, sim_order, &mut pending_orders, order_idx)
                }
            }
        }

        self.store_simulation_state(
            &full_sequence_of_orders,
            &state,
            total_profit,
            &per_order_profits,
        );

        let resolution_result = ResolutionResult {
            total_profit,
            sequence_of_orders: sequenced_order_result,
        };
        Ok((resolution_result, state))
    }

    /// Helper function to handle a successful commit of an order.
    #[allow(clippy::too_many_arguments)]
    fn handle_successful_commit(
        &mut self,
        res: ExecutionResult,
        sim_order: &SimulatedOrder,
        order_idx: usize,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        remaining_orders: &mut Vec<usize>,
        sequenced_order_result: &mut Vec<(usize, U256)>,
        total_profit: &mut U256,
        per_order_profits: &mut Vec<(OrderId, U256)>,
    ) {
        for (address, nonce) in res.nonces_updated {
            if let Some(pending_order) = pending_orders.remove(&(address, nonce)) {
                remaining_orders.push(pending_order);
            }
        }
        let order_id = sim_order.order.id();
        *total_profit += res.coinbase_profit;
        per_order_profits.push((order_id, res.coinbase_profit));
        sequenced_order_result.push((order_idx, res.coinbase_profit));
    }

    /// Helper function to handle an error in committing an order.
    fn handle_err(
        &mut self,
        err: &ExecutionError,
        sim_order: &SimulatedOrder,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        order_idx: usize,
    ) {
        if let Some((address, nonce)) = err.try_get_tx_too_high_error(&sim_order.order) {
            pending_orders.insert((address, nonce), order_idx);
        };
    }

    /// Initializes a HashMap of order id to index.
    fn initialize_order_id_to_index_map(&self, task: &ConflictTask) -> HashMap<OrderId, usize> {
        task.group
            .orders
            .iter()
            .enumerate()
            .map(|(idx, sim_order)| (sim_order.order.id(), idx))
            .collect()
    }

    /// Initializes a vector of full order ids corresponding to the sequence of orders.
    fn initialize_full_order_ids_vec(
        &self,
        sequence_of_orders: &[usize],
        task: &ConflictTask,
    ) -> Vec<OrderId> {
        sequence_of_orders
            .iter()
            .map(|&idx| task.group.orders[idx].order.id())
            .collect()
    }

    /// Initializes the tuple of (order_idx, profit) for the resolution result using the cached state if available.
    fn initialize_result_order_sequence(
        &self,
        cached_state_option: &Option<Arc<CachedSimulationState>>,
        order_id_to_index: &HashMap<OrderId, usize>,
    ) -> Vec<(usize, U256)> {
        if let Some(cached_state) = &cached_state_option {
            cached_state
                .per_order_profits
                .iter()
                .filter_map(|(order_id, profit)| {
                    order_id_to_index.get(order_id).map(|&idx| (idx, *profit))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    /// Initializes the block state, using a cached state if available.
    fn initialize_block_state(&mut self, state_provider: Arc<dyn StateProvider>) -> BlockState {
        BlockState::new_arc(state_provider)
    }

    /// Stores the simulation state in the cache.
    fn store_simulation_state(
        &self,
        full_order_ids: &[OrderId],
        state: &BlockState,
        total_profit: U256,
        per_order_profits: &[(OrderId, U256)],
    ) {
        let (bundle_state, _) = state.clone().into_parts();
        let cached_simulation_state = CachedSimulationState {
            bundle_state,
            total_profit,
            per_order_profits: per_order_profits.to_owned(),
        };
        self.simulation_cache
            .store_cached_state(full_order_ids, cached_simulation_state);
    }
}

/// Generates different sequences of orders to try based on the conflict task command.
///
/// # Arguments
///
/// * `task` - The conflict task containing the algorithm for generating sequences.
///
/// # Returns
///
/// A vector of different sequences of order indices to try.
fn generate_sequences_of_orders_to_try(task: &ConflictTask) -> Vec<Vec<usize>> {
    match task.algorithm {
        Algorithm::Greedy => generate_greedy_sequence(task, false),
        Algorithm::ReverseGreedy => generate_greedy_sequence(task, true),
        Algorithm::Length => generate_length_based_sequence(task),
        Algorithm::AllPermutations => generate_all_permutations(task),
        Algorithm::Random { seed, count } => generate_random_permutations(task, seed, count),
        Algorithm::DexDirectionBalanced => generate_dex_marginal_centered_sequence(task),
    }
}

/// Generates random permutations of sequences of order indices.
///
/// # Arguments
///
/// * `task` - The current conflict task.
/// * `seed` - Seed for the random number generator.
/// * `count` - Number of random permutations to generate.
///
/// # Returns
///
/// A vector of randomly generated sequences of order indices.
fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    let mut sequences_of_orders = vec![];

    let order_group = &task.group;
    let mut indexes = (0..order_group.orders.len()).collect::<Vec<_>>();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        sequences_of_orders.push(indexes.clone());
    }

    sequences_of_orders
}

/// Generates all possible permutations of sequences of order indices.
///
/// # Arguments
///
/// * `task` - The current conflict task.
///
/// # Returns
///
/// A vector of all possible sequences of order indices.
fn generate_all_permutations(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let sequences_of_orders = (0..order_group.orders.len()).collect::<Vec<_>>();
    sequences_of_orders
        .into_iter()
        .permutations(order_group.orders.len())
        .collect()
}

/// Generates static sequences of order indices based on gas price and coinbase profit.
///
/// # Arguments
///
/// * `task` - The current conflict task.
/// * `reverse` - Whether to reverse the sorting order (e.g. sorting by min coinbase profit and mev_gas_price)
///
/// # Returns
///
/// A vector of static sequences of order indices, sorted by coinbase profit and mev_gas_price.
fn generate_greedy_sequence(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
    let order_group = &task.group;

    let create_sequence = |value_extractor: fn(&SimulatedOrder) -> U256| {
        let mut ids_and_value: Vec<_> = order_group
            .orders
            .iter()
            .enumerate()
            .map(|(idx, order)| (idx, value_extractor(order)))
            .collect();

        ids_and_value.sort_by(|a, b| {
            if reverse {
                a.1.cmp(&b.1)
            } else {
                b.1.cmp(&a.1)
            }
        });
        ids_and_value.into_iter().map(|(idx, _)| idx).collect()
    };

    vec![
        create_sequence(|sim_order| sim_order.sim_value.full_profit_info().coinbase_profit()),
        // create_sequence(|sim_order| sim_order.sim_value.full_profit_info().mev_gas_price()),
    ]
}

/// Generates length based sequences of order indices based on the length of the orders.
/// e.g. prioritizes longer bundles first
///
/// # Arguments
///
/// * `task` - The current conflict task.
///
/// # Returns
///
/// A vector of length based sequences of order indices.
fn generate_length_based_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
    let mut sequences_of_orders = vec![];
    let order_group = &task.group;

    let mut order_data: Vec<(usize, usize, U256)> = order_group
        .orders
        .iter()
        .enumerate()
        .map(|(idx, order)| {
            (
                idx,
                order.order.list_txs().len(),
                order.sim_value.full_profit_info().coinbase_profit(),
            )
        })
        .collect();

    // Sort by length (descending) and then by profit (descending) as a tie-breaker
    order_data.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));

    // Extract the sorted indices
    let length_based_sequence: Vec<usize> = order_data.into_iter().map(|(idx, _, _)| idx).collect();

    sequences_of_orders.push(length_based_sequence);
    sequences_of_orders
}

#[inline]
fn coinbase_profit(task: &ConflictTask, idx: usize) -> U256 {
    task.group.orders[idx]
        .sim_value
        .full_profit_info()
        .coinbase_profit()
}

/// Place price-neutral bundles first (sorted by profit), then impacting orders (sorted by profit).
///
/// An order is "price-neutral" if for every pool it touches, the pool price after all its
/// transactions exactly equals the price before any of them (e.g. user tx + MEV backrun that
/// fully restores the price). These bundles can be included at the top of the block without
/// perturbing pool prices seen by later orders.
///
/// Before placing a candidate-neutral order in the neutral bucket we also verify that it does
/// not write to any storage slot that an impacting order reads.  This catches the common case
/// where a neutral bundle consumes a nonce (or other state) that a high-profit impacting bundle
/// depends on, which would cause the impacting bundle to revert if the neutral runs first.
///
/// Orders without a state trace are treated as impacting (placed after neutrals).
fn generate_dex_marginal_centered_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
    let n = task.group.orders.len();

    // Initial split by price-neutrality.
    let (candidate_neutral, mut impacting): (Vec<usize>, Vec<usize>) =
        (0..n).partition(|&i| {
            task.group.orders[i]
                .used_state_trace
                .as_ref()
                .map(|t| t.is_price_neutral())
                .unwrap_or(false)
        });

    // Collect every storage slot READ by any impacting order (those that have a trace).
    let impacting_reads: HashSet<&SlotKey> = impacting
        .iter()
        .filter_map(|&i| task.group.orders[i].used_state_trace.as_ref())
        .flat_map(|t| t.read_slot_values.keys())
        .collect();

    // Keep a candidate neutral only if none of its WRITES overlap with the impacting reads.
    // Neutrals that do write to such slots (e.g. consuming a nonce that an impacting bundle
    // also needs) are demoted to impacting so the profit-based sort handles them correctly.
    let mut neutral: Vec<usize> = Vec::new();
    for i in candidate_neutral {
        let demote = task.group.orders[i]
            .used_state_trace
            .as_ref()
            .map(|t| {
                t.written_slot_values
                    .keys()
                    .any(|k| impacting_reads.contains(k))
            })
            .unwrap_or(false);
        if demote {
            eprintln!(
                "[DBG neutral-demote] order {:?} demoted to impacting: writes slots read by impacting orders (likely shared nonce/state)",
                task.group.orders[i].order.id()
            );
            impacting.push(i);
        } else {
            neutral.push(i);
        }
    }

    neutral.sort_by(|&a, &b| coinbase_profit(task, b).cmp(&coinbase_profit(task, a)));
    impacting.sort_by(|&a, &b| coinbase_profit(task, b).cmp(&coinbase_profit(task, a)));

    neutral.extend(impacting);
    vec![neutral]
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, TxHash, B256, U256};
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use super::*;
    use crate::building::builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority};
    use rbuilder_primitives::{
        Bundle, Metadata, Order, SimValue, SimulatedOrder, TransactionSignedEcRecoveredWithBlobs,
        LAST_BUNDLE_VERSION,
    };

    struct DataGenerator {
        last_used_id: u64,
    }
    impl DataGenerator {
        pub fn new() -> DataGenerator {
            DataGenerator { last_used_id: 0 }
        }

        pub fn create_u64(&mut self) -> u64 {
            self.last_used_id += 1;
            self.last_used_id
        }

        pub fn create_u256(&mut self) -> U256 {
            U256::from(self.create_u64())
        }

        pub fn create_hash(&mut self) -> TxHash {
            TxHash::from(self.create_u256())
        }

        pub fn create_tx(&mut self) -> Recovered<TransactionSigned> {
            let tx_legacy = TxLegacy {
                nonce: self.create_u64(),
                ..Default::default()
            };
            Recovered::new_unchecked(
                TransactionSigned::new_unchecked(
                    Transaction::Legacy(tx_legacy),
                    alloy_primitives::Signature::test_signature(),
                    self.create_hash(),
                ),
                Address::default(),
            )
        }

        pub fn create_order_with_length(
            &mut self,
            coinbase_profit: U256,
            mev_gas_price: U256,
            num_of_orders: usize,
        ) -> Arc<SimulatedOrder> {
            let mut txs = Vec::new();
            for _ in 0..num_of_orders {
                txs.push(
                    TransactionSignedEcRecoveredWithBlobs::new_no_blobs(self.create_tx()).unwrap(),
                );
            }

            let sim_value = SimValue::new_test_no_gas(coinbase_profit, mev_gas_price);

            let bundle = Bundle {
                block: Some(0),
                min_timestamp: None,
                max_timestamp: None,
                txs,
                reverting_tx_hashes: Vec::new(),
                hash: B256::ZERO,
                uuid: Uuid::new_v4(),
                replacement_data: None,
                signer: None,
                metadata: Metadata::default(),
                dropping_tx_hashes: Vec::new(),
                refund: None,
                refund_identity: None,
                version: LAST_BUNDLE_VERSION,
                external_hash: None,
            };

            Arc::new(SimulatedOrder {
                order: Order::Bundle(bundle),
                used_state_trace: None,
                sim_value,
            })
        }
    }

    // Helper function to create an order group
    fn create_mock_order_group(
        id: GroupId,
        orders: Vec<Arc<SimulatedOrder>>,
        conflicting_ids: HashSet<GroupId>,
    ) -> ConflictGroup {
        ConflictGroup {
            id,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(conflicting_ids.into_iter().collect()),
        }
    }

    fn create_mock_task(
        group_idx: usize,
        group: ConflictGroup,
        algorithm: Algorithm,
        priority: TaskPriority,
        created_at: Instant,
    ) -> ConflictTask {
        ConflictTask {
            group_idx,
            group,
            algorithm,
            priority,
            created_at,
        }
    }

    #[test]
    fn test_all_permutations() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(100), 1), // index: 0, Length 1, profit 100, mev_gas_price 100
                data_generator.create_order_with_length(U256::from(200), U256::from(200), 3), // index: 1, Length 3, profit 200, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(150), U256::from(150), 2), // index: 2, Length 2, profit 150, mev_gas_price 150
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::AllPermutations,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 6);
        assert_eq!(sequences[0], vec![0, 1, 2]);
        assert_eq!(sequences[1], vec![0, 2, 1]);
        assert_eq!(sequences[2], vec![1, 0, 2]);
        assert_eq!(sequences[3], vec![1, 2, 0]);
        assert_eq!(sequences[4], vec![2, 0, 1]);
        assert_eq!(sequences[5], vec![2, 1, 0]);
    }

    #[test]
    fn test_generate_length_based_sequence() {
        let mut data_generator = DataGenerator::new();

        let orders = vec![
            data_generator.create_order_with_length(U256::from(100), U256::from(100), 1), // Length 1, profit 100
            data_generator.create_order_with_length(U256::from(200), U256::from(200), 3), // Length 3, profit 200
            data_generator.create_order_with_length(U256::from(150), U256::from(150), 2), // Length 2, profit 150
            data_generator.create_order_with_length(U256::from(300), U256::from(300), 1), // Length 1, profit 300
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());

        let task = create_mock_task(
            0,
            group,
            Algorithm::Length,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0], vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_max_profit_and_mev_gas_price_sequences() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(300), 1), // index: 0, Length 1, profit 100, mev_gas_price 300
                data_generator.create_order_with_length(U256::from(200), U256::from(150), 3), // index: 1, Length 3, profit 200, mev_gas_price 150
                data_generator.create_order_with_length(U256::from(150), U256::from(200), 2), // index: 2, Length 2, profit 150, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(300), U256::from(100), 1), // index: 3, Length 1, profit 300, mev_gas_price 100
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::Greedy,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 1);

        // Coinbase profit descending
        assert_eq!(sequences[0], vec![3, 1, 2, 0]);
    }

    #[test]
    fn test_reverse_max_profit_and_mev_gas_price_sequences() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(300), 1), // index: 0, Length 1, profit 100, mev_gas_price 300
                data_generator.create_order_with_length(U256::from(200), U256::from(150), 3), // index: 1, Length 3, profit 200, mev_gas_price 150
                data_generator.create_order_with_length(U256::from(150), U256::from(200), 2), // index: 2, Length 2, profit 150, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(300), U256::from(100), 1), // index: 3, Length 1, profit 300, mev_gas_price 100
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::ReverseGreedy,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 1);

        // Coinbase profit ascending
        assert_eq!(sequences[0], vec![0, 2, 1, 3]);
    }

    // ── DexDirectionBalanced (zero-net-first) tests ──────────────────────────

    use rbuilder_primitives::evm_inspector::{PoolKind, SlotKey, UsedStateTrace};

    fn b256_from_u256(v: U256) -> B256 {
        B256::from(v.to_be_bytes::<32>())
    }

    fn slot0_key(pool: Address) -> SlotKey {
        SlotKey { address: pool, key: B256::ZERO }
    }

    fn slot8_key(pool: Address) -> SlotKey {
        SlotKey { address: pool, key: b256_from_u256(U256::from(8u64)) }
    }

    fn pack_v2_reserves(r0: U256, r1: U256) -> B256 {
        b256_from_u256(r0 | (r1 << 112))
    }

    /// Build a V3 UsedStateTrace where price was NOT restored (impacting).
    fn v3_trace_impacting(pool: Address, sqrt_before: U256, sqrt_after: U256) -> UsedStateTrace {
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256_from_u256(sqrt_before));
        trace.written_slot_values.insert(slot0_key(pool), b256_from_u256(sqrt_after));
        trace
    }

    /// Build a V3 UsedStateTrace where price was restored (neutral): no write entry.
    /// Simulates the SSTORE hook removing the write because final value == initial value.
    fn v3_trace_neutral(pool: Address, sqrt_price: U256) -> UsedStateTrace {
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256_from_u256(sqrt_price));
        trace
    }

    /// Build a V2 UsedStateTrace where reserves were NOT restored (impacting).
    fn v2_trace_impacting(
        pool: Address,
        r0_before: U256,
        r1_before: U256,
        r0_after: U256,
        r1_after: U256,
    ) -> UsedStateTrace {
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV2);
        trace.read_slot_values.insert(slot8_key(pool), pack_v2_reserves(r0_before, r1_before));
        trace.written_slot_values.insert(slot8_key(pool), pack_v2_reserves(r0_after, r1_after));
        trace
    }

    /// Build a V2 UsedStateTrace where reserves were restored (neutral): no write entry.
    fn v2_trace_neutral(pool: Address, r0: U256, r1: U256) -> UsedStateTrace {
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV2);
        trace.read_slot_values.insert(slot8_key(pool), pack_v2_reserves(r0, r1));
        trace
    }

    /// Create a SimulatedOrder with an explicit UsedStateTrace and profit.
    fn create_order_with_trace(profit: U256, trace: UsedStateTrace) -> Arc<SimulatedOrder> {
        Arc::new(SimulatedOrder {
            order: Order::Bundle(Bundle {
                block: Some(0),
                min_timestamp: None,
                max_timestamp: None,
                txs: vec![],
                reverting_tx_hashes: vec![],
                hash: B256::ZERO,
                uuid: Uuid::new_v4(),
                replacement_data: None,
                signer: None,
                metadata: Metadata::default(),
                dropping_tx_hashes: vec![],
                refund: None,
                refund_identity: None,
                version: LAST_BUNDLE_VERSION,
                external_hash: None,
            }),
            used_state_trace: Some(trace),
            sim_value: SimValue::new_test_no_gas(profit, U256::ZERO),
        })
    }

    #[test]
    fn test_dex_zero_net_neutral_goes_first() {
        // idx 0: price-neutral V3 bundle (backrun restores price), profit=10
        // idx 1: impacting V3 single swap, profit=50
        // idx 2: trace with no touched pools (vacuously neutral), profit=200
        //
        // Expected: neutral sorted by profit first → [2 (200), 0 (10)],
        // then impacting → [1 (50)]. Final: [2, 0, 1]
        let pool_a = Address::repeat_byte(0xAA);
        let p0 = U256::from(1_000_000u64);
        let p1 = U256::from(1_001_000u64);

        let orders = vec![
            create_order_with_trace(U256::from(10), v3_trace_neutral(pool_a, p0)),
            create_order_with_trace(U256::from(50), v3_trace_impacting(pool_a, p0, p1)),
            create_order_with_trace(U256::from(200), UsedStateTrace::default()),
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());
        let task = create_mock_task(0, group, Algorithm::DexDirectionBalanced, TaskPriority::Low, Instant::now());
        let sequences = generate_sequences_of_orders_to_try(&task);

        assert_eq!(sequences.len(), 1);
        assert_eq!(&sequences[0], &vec![2, 0, 1],
            "neutral orders (idx 2 profit=200, idx 0 profit=10) come before impacting (idx 1)");
    }

    #[test]
    fn test_dex_impacting_sorted_by_profit_when_no_neutrals() {
        // All orders have no trace → all impacting → sorted by profit descending
        let mut dg = DataGenerator::new();
        let orders = vec![
            dg.create_order_with_length(U256::from(100), U256::ZERO, 1), // idx 0, profit 100
            dg.create_order_with_length(U256::from(300), U256::ZERO, 1), // idx 1, profit 300
            dg.create_order_with_length(U256::from(200), U256::ZERO, 1), // idx 2, profit 200
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());
        let task = create_mock_task(0, group, Algorithm::DexDirectionBalanced, TaskPriority::Low, Instant::now());
        let sequences = generate_sequences_of_orders_to_try(&task);

        assert_eq!(sequences[0], vec![1, 2, 0]);
    }

    #[test]
    fn test_dex_neutral_profit_ordering() {
        // Three neutral bundles with different profits — sorted descending.
        let pool_a = Address::repeat_byte(0xBB);
        let p0 = U256::from(500_000u64);

        let orders = vec![
            create_order_with_trace(U256::from(100), v3_trace_neutral(pool_a, p0)),
            create_order_with_trace(U256::from(300), v3_trace_neutral(pool_a, p0)),
            create_order_with_trace(U256::from(200), v3_trace_neutral(pool_a, p0)),
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());
        let task = create_mock_task(0, group, Algorithm::DexDirectionBalanced, TaskPriority::Low, Instant::now());
        let sequences = generate_sequences_of_orders_to_try(&task);

        assert_eq!(sequences[0], vec![1, 2, 0]);
    }

    #[test]
    fn test_dex_completeness_multi_pool() {
        // Both orders touch different pools; both impacting. All orders must appear in result.
        let pool_a = Address::repeat_byte(0xCC);
        let pool_b = Address::repeat_byte(0xDD);
        let p0 = U256::from(1_000u64);
        let p1 = U256::from(1_100u64);

        // Order 0: impacting on pool_a and pool_b
        let mut trace0 = v3_trace_impacting(pool_a, p0, p1);
        trace0.touched_pools.insert(pool_b, PoolKind::UniV3);
        trace0.read_slot_values.insert(slot0_key(pool_b), b256_from_u256(p0));
        trace0.written_slot_values.insert(slot0_key(pool_b), b256_from_u256(p1));

        let orders = vec![
            create_order_with_trace(U256::from(50), trace0),
            create_order_with_trace(U256::from(50), v3_trace_impacting(pool_a, p1, p0)),
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());
        let task = create_mock_task(0, group, Algorithm::DexDirectionBalanced, TaskPriority::Low, Instant::now());
        let sequences = generate_sequences_of_orders_to_try(&task);

        assert_eq!(sequences[0].len(), 2);
        assert!(sequences[0].contains(&0) && sequences[0].contains(&1));
    }

    #[test]
    fn test_dex_zero_net_v2_exact_cancel() {
        // V2 pool: user tx moves reserves, backrun restores them exactly.
        // After the full bundle the SSTORE hook sees final value == read value and removes
        // the write entry, so is_price_neutral() returns true (no write → neutral).
        let pool_v2 = Address::repeat_byte(0xEE);
        let (r0_init, r1_init) = (U256::from(100u64), U256::from(200u64));
        let (r0_mid, r1_mid) = (U256::from(110u64), U256::from(181u64));

        let orders = vec![
            // idx 0: V2 neutral bundle (price restored, no write entry)
            create_order_with_trace(U256::from(10), v2_trace_neutral(pool_v2, r0_init, r1_init)),
            // idx 1: impacting V2 single swap
            create_order_with_trace(U256::from(50), v2_trace_impacting(pool_v2, r0_init, r1_init, r0_mid, r1_mid)),
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());
        let task = create_mock_task(0, group, Algorithm::DexDirectionBalanced, TaskPriority::Low, Instant::now());
        let sequences = generate_sequences_of_orders_to_try(&task);

        let seq = &sequences[0];
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0], 0, "V2 neutral bundle should be placed first despite lower profit");
        assert_eq!(seq[1], 1);
    }
}
