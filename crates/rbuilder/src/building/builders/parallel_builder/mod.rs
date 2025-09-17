pub mod block_building_result_assembler;
pub mod conflict_resolvers;
pub mod conflict_resolving_pool;
pub mod conflict_task_generator;
pub mod groups;
pub mod order_intake_store;
pub mod results_aggregator;
pub mod simulation_cache;
pub mod task;
use alloy_primitives::U256;
pub use groups::*;
pub mod nonce_handling;
pub mod genetic_algo;
pub use conflict_task_generator::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::sync::mpsc::RecvTimeoutError;
use rayon::join;

use ahash::HashMap;
use conflict_resolving_pool::{ConflictResolvingPool, TaskQueue};
use crossbeam::queue::SegQueue;
use eyre::Result;
use results_aggregator::BestResults;
use reth_provider::StateProvider;
use serde::Deserialize;
use simulation_cache::SharedSimulationCache;
use std::{
    sync::{mpsc as std_mpsc, Arc},
    thread,
    time::Instant,
    cmp::Ordering as CmpOrdering,
};
use task::*;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use tracing::{error, trace};

use crate::{
    building::builders::{
        BacktestSimulateBlockInput, Block, BlockBuildingAlgorithm, BlockBuildingAlgorithmInput,
        LiveBuilderInput,
    },
    provider::StateProviderFactory,
};

use self::{
    block_building_result_assembler::BlockBuildingResultAssembler,
    order_intake_store::OrderIntakeStore, results_aggregator::ResultsAggregator,
};

pub type GroupId = usize;
pub type ConflictResolutionResultPerGroup = (GroupId, (ResolutionResult, ConflictGroup));

/// ParallelBuilderConfig configures parallel builder.
/// * `num_threads` - number of threads to use for merging.
/// * `merge_wait_time_ms` - time to wait for merging to finish before consuming new orders.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParallelBuilderConfig {
    pub discard_txs: bool,
    pub num_threads: usize,
    #[serde(default)]
    pub coinbase_payment: bool,
}

fn get_communication_channels() -> (
    std_mpsc::Sender<ConflictResolutionResultPerGroup>,
    std_mpsc::Receiver<ConflictResolutionResultPerGroup>,
) {
    std_mpsc::channel()
}

fn get_shared_data_structures() -> (Arc<BestResults>, TaskQueue) {
    let best_results = Arc::new(BestResults::new());
    let task_queue = Arc::new(SegQueue::new());
    (best_results, task_queue)
}

fn cmp_res(a: &ResolutionResult, b: &ResolutionResult) -> CmpOrdering {
    // Primary: total_profit desc
    let c = b.total_profit.cmp(&a.total_profit);
    if c != CmpOrdering::Equal { return c; }

    // Secondary: gas_used asc (prefer cheaper for same profit)
    let c = a.gas_used.cmp(&b.gas_used);
    if c != CmpOrdering::Equal { return c; }

    // Final tie-breaker: lexicographic by order indices (intra-group determinism)
    let mut ia = a.sequence_of_orders.iter().map(|(i, _, _)| *i);
    let mut ib = b.sequence_of_orders.iter().map(|(i, _, _)| *i);
    loop {
        match (ia.next(), ib.next()) {
            (Some(x), Some(y)) => if x != y { return x.cmp(&y); },
            (None, Some(_)) => return CmpOrdering::Less,
            (Some(_), None) => return CmpOrdering::Greater,
            (None, None) => return CmpOrdering::Equal, // truly identical
        }
    }
}


struct ParallelBuilder<P> {
    order_intake_consumer: OrderIntakeStore,
    conflict_finder: ConflictFinder,
    conflict_task_generator: ConflictTaskGenerator,
    conflict_resolving_pool: ConflictResolvingPool<P>,
    results_aggregator: ResultsAggregator,
    block_building_result_assembler: BlockBuildingResultAssembler,
}

#[derive(Copy, Clone, Debug)]
pub enum BuildMode {
    GreedyProfit,   // groups contiguous, sorted by total_profit
    GreedyMgp,      // groups contiguous, sorted by group-level mev_gas_price
    HeapMgp,        // interleave by per-tx MEV gas price
    HeapProfit,     // interleave by per-tx profit only
}

impl<P> ParallelBuilder<P>
where
    P: StateProviderFactory + Clone + 'static,
{
    /// Creates a ParallelBuilder.
    /// Sets up the various components and communication channels.
    pub fn try_new(
        input: LiveBuilderInput<P>,
        config: &ParallelBuilderConfig,
    ) -> eyre::Result<Self> {
        let (group_result_sender, group_result_receiver) = get_communication_channels();
        let group_result_sender_for_task_generator = group_result_sender.clone();

        let (best_results, task_queue) = get_shared_data_structures();

        let simulation_cache = Arc::new(SharedSimulationCache::new());

        let conflict_finder = ConflictFinder::new();

        let conflict_task_generator = ConflictTaskGenerator::new(
            Arc::clone(&task_queue),
            group_result_sender_for_task_generator,
        );

        let conflict_resolving_pool = ConflictResolvingPool::new(
            config.num_threads,
            Arc::clone(&task_queue),
            group_result_sender,
            input.cancel.clone(),
            input.ctx.clone(),
            input.provider.clone(),
            Arc::clone(&simulation_cache),
        );

        let results_aggregator =
            ResultsAggregator::new(group_result_receiver, Arc::clone(&best_results));

        let block_state = input
            .provider
            .history_by_block_hash(input.ctx.attributes.parent)?
            .into();

        let block_building_result_assembler = BlockBuildingResultAssembler::new(
            config,
            Arc::clone(&best_results),
            block_state,
            input.ctx.clone(),
            input.cancel.clone(),
            input.builder_name.clone(),
            input.sink.can_use_suggested_fee_recipient_as_coinbase(),
            Some(input.sink.clone()),
        );

        let order_intake_consumer = OrderIntakeStore::new(input.input);

        Ok(Self {
            order_intake_consumer,
            conflict_finder,
            conflict_task_generator,
            conflict_resolving_pool,
            results_aggregator,
            block_building_result_assembler,
        })
    }

    /// Initializes the orders in the cached groups.
    fn initialize_orders(&mut self) {
        let initial_orders = self.order_intake_consumer.get_orders();
        trace!("Initializing with {} orders", initial_orders.len());
        self.conflict_finder.add_orders(initial_orders);
    }
}

/// Runs the parallel builder algorithm to construct blocks from incoming orders.
///
/// This function implements a continuous block building process that:
/// 1. Consumes orders from an intake store.
/// 2. Identifies conflict groups among the orders.
/// 3. Manages conflicts and attempts to resolve them.
/// 4. Builds blocks from the best results of conflict resolution.
///
/// The process involves several key components:
/// - [OrderIntakeStore]: Provides a continuous stream of incoming orders.
/// - [ConflictFinder]: Identifies and manages conflict groups among orders.
/// - [ConflictTaskGenerator]: Decides which conflicts to attempt to resolve and what priority.
/// - [ConflictResolvingPool]: A pool of workers that resolve conflicts between orders, producing "results" which are resolved conflicts.
/// - [ResultsAggregator]: Collects results from workers and initiates block building.
/// - [BlockBuildingResultAssembler]: Builds blocks from the best results collected.
///
/// The function runs in a loop, continuously processing new orders and building blocks
/// until cancellation is requested. It uses separate processes for
/// 1. Identifying conflicts and processing which conflicts to attempt to resolve in what priority
/// 2. Resolving conflicts
/// 3. Block building given conflict resolution results
///
/// By separating these processes we can continuously take in new flow, triage that flow intelligently, and build blocks continuously with the best results.
///
/// # Arguments
/// * `input`: LiveBuilderInput containing necessary context and resources for block building.
/// * `config`: Configuration parameters for the parallel builder.
///
/// # Type Parameters
/// * `DB`: The database type, which must implement Database, Clone, and have a static lifetime.
pub fn run_parallel_builder<P>(input: LiveBuilderInput<P>, config: &ParallelBuilderConfig)
where
    P: StateProviderFactory + Clone + 'static,
{
    let cancel_for_results_aggregator = input.cancel.clone();
    let cancel_for_block_building_result_assembler = input.cancel.clone();
    let cancel_for_process_orders_loop = input.cancel.clone();

    let mut builder = match ParallelBuilder::try_new(input, config) {
        Ok(builder) => builder,
        Err(err) => {
            error!(?err, "Failed to create parallel builder, cancelling");
            return;
        }
    };
    builder.initialize_orders();

    // Start task processing
    match builder.conflict_resolving_pool.start() {
        Ok(()) => {}
        Err(err) => {
            error!(
                ?err,
                "Failed to start parallel builder conflict_resolving_pool, cancelling"
            );
            return;
        }
    }

    // Process that collects conflict resolution results from workers and triggers block building
    tokio::spawn(async move {
        builder
            .results_aggregator
            .run(cancel_for_results_aggregator)
            .await;
    });

    // Process that builds blocks from the best conflict resolution results
    thread::spawn(move || {
        builder
            .block_building_result_assembler
            .run(cancel_for_block_building_result_assembler);
    });

    // Process that consumes orders from the intake store, updates the cached groups, and triggers new conflict resolution tasks for the worker pool
    run_order_intake(
        &cancel_for_process_orders_loop,
        &mut builder.order_intake_consumer,
        &mut builder.conflict_finder,
        &mut builder.conflict_task_generator,
    );
}

fn run_order_intake(
    cancel_token: &CancellationToken,
    order_intake_consumer: &mut OrderIntakeStore,
    conflict_finder: &mut ConflictFinder,
    conflict_task_generator: &mut ConflictTaskGenerator,
) {
    'building: loop {
        if cancel_token.is_cancelled() {
            break 'building;
        }

        match order_intake_consumer.consume_next_batch() {
            Ok(ok) => {
                if !ok {
                    break 'building;
                }
            }
            Err(err) => {
                error!(?err, "Error consuming next order batch");
                continue;
            }
        }

        let new_orders = order_intake_consumer.try_drain_new_orders_if_no_cancellations();

        // We can update conflict_finder if we have ONLY adds
        if let Some(new_orders) = new_orders {
            if !new_orders.is_empty() {
                let time_start = Instant::now();
                let len = new_orders.len();
                conflict_finder.add_orders(new_orders);
                trace!(
                    new_orders_count = len,
                    groups_count = conflict_finder.get_order_groups().len(),
                    time_taken_ms = %time_start.elapsed().as_millis(),
                    "Order intake: added new orders and processing groups"
                );
                conflict_task_generator.process_groups(conflict_finder.get_order_groups());
            }
        }
    }
}


pub fn parallel_build_backtest<P>(
    input: BacktestSimulateBlockInput<'_, P>,
    config: ParallelBuilderConfig,
) -> Result<Block>
where
    P: StateProviderFactory + Clone + 'static,
{
    let start_time = Instant::now();

    // Initialization stage
    let init_start = Instant::now();
    let (best_results, task_queue) = get_shared_data_structures();

    let (group_result_sender, group_result_receiver) = get_communication_channels();
    let group_result_sender_for_task_generator = group_result_sender.clone();

    let mut conflict_finder = ConflictFinder::new();

    let sorted_orders = {
        let mut orders = input.sim_orders.clone();
        orders.sort_by_key(|o| o.order.id());
        orders
    };

    let simulation_cache = Arc::new(SharedSimulationCache::new());
    let init_duration = init_start.elapsed();

    // Worker pool and conflict manager creation
    let setup_start = Instant::now();

    let cancel_token = CancellationToken::new();
    let outstanding = Arc::new(AtomicUsize::new(0));
    let conflict_resolving_pool = ConflictResolvingPool::new(
        config.num_threads,
        Arc::clone(&task_queue),
        group_result_sender,
        cancel_token.clone(),
        input.ctx.clone(),
        input.provider.clone(),
        Arc::clone(&simulation_cache),
    ).with_outstanding_counter(outstanding.clone());

    let setup_duration = setup_start.elapsed();

    let block_state: Arc<dyn StateProvider> = input
        .provider
        .history_by_block_hash(input.ctx.attributes.parent)?
        .into();

    // Group processing
    conflict_finder.add_orders(sorted_orders);
    let groups = conflict_finder.get_order_groups();

    // Generate tasks using the same logic as live builder
    let mut task_generator = ConflictTaskGenerator::new(Arc::clone(&task_queue), group_result_sender_for_task_generator);
    let processing_start = Instant::now();
    task_generator.process_groups(groups.clone());

    // Initialise outstanding from the actual number of enqueued tasks
    let planned = task_queue.len();
    outstanding.store(planned, Ordering::Release);

    // Start worker threads (after tasks are enqueued)
    if let Err(err) = conflict_resolving_pool.start() {
        return Err(err);
    }
    
    let mut results: Vec<(GroupId, (ResolutionResult, ConflictGroup))> = Vec::new();
    let mut last_progress = Instant::now();
    loop {
        match group_result_receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(res) => {
                results.push(res);
                last_progress = Instant::now();
            }
            Err(RecvTimeoutError::Timeout) => {
                if outstanding.load(Ordering::Acquire) == 0 {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    // Stop workers
    cancel_token.cancel();

    let processing_end = last_progress; // updated on every Ok(res)
    let processing_duration = processing_end.duration_since(processing_start);

    // Block building result assembler creation
    let assembler_start = Instant::now();
        let mut asm_greedy = BlockBuildingResultAssembler::new(
        &config,
        Arc::clone(&best_results),
        block_state.clone(),
        input.ctx.clone(),
        CancellationToken::new(),
        "backtest_builder_greedy".into(),
        true,
        None,
    );

    let mut asm_mgp = BlockBuildingResultAssembler::new(
        &config,
        Arc::clone(&best_results),
        block_state.clone(),
        input.ctx.clone(),
        CancellationToken::new(),
        "backtest_builder_mgp".into(),
        true,
        None,
    );
    let assembler_duration = assembler_start.elapsed();

    // Best results collection
    let collection_start = Instant::now();

    let mut best_results: HashMap<GroupId, (ResolutionResult, ConflictGroup)> = HashMap::default();

    for (gid, (res, grp)) in results.into_iter() {
        match best_results.get_mut(&gid) {
            None => { best_results.insert(gid, (res, grp)); }
            Some((cur_res, cur_grp)) => {
                if cmp_res(&res, cur_res).is_lt() {
                    *cur_res = res;
                    *cur_grp = grp;
                }
            }
        }
    }

    let collection_duration = collection_start.elapsed();

    // Block building
    let building_start = Instant::now();
    let orders_closed_at = OffsetDateTime::now_utc();
    let (res_greedy, res_mgp) = join(
        || asm_greedy.build_backtest_block(best_results.clone(), orders_closed_at, BuildMode::HeapProfit),
        || asm_mgp.build_backtest_block(best_results.clone(), orders_closed_at, BuildMode::HeapMgp),
    );

    let helper_greedy = res_greedy?;
    let helper_mgp    = res_mgp?;

    let val_greedy = if config.coinbase_payment { U256::ZERO } else { helper_greedy.true_block_value()? };
    let val_mgp    = if config.coinbase_payment { U256::ZERO } else { helper_mgp.true_block_value()? };

    let (chosen_helper, mut chosen_asm) = if val_mgp > val_greedy {
        (helper_mgp, asm_mgp)
    } else {
        (helper_greedy, asm_greedy)
    };

    let payout_tx_value = if config.coinbase_payment { None } else { Some(chosen_helper.true_block_value()?) };
    let finalize_block_result = chosen_helper.finalize_block(
        &mut chosen_asm.local_ctx,
        payout_tx_value,
        None,
    )?;

    let building_duration = building_start.elapsed();
    let total_duration = start_time.elapsed();

    trace!("Initialization time: {:?}", init_duration);
    trace!("Setup time: {:?}", setup_duration);
    trace!("Group processing time: {:?}", processing_duration);
    trace!("Assembler creation time: {:?}", assembler_duration);
    trace!("Best results collection time: {:?}", collection_duration);
    trace!("Block building time: {:?}", building_duration);
    trace!("Total time taken: {:?}", total_duration);

    Ok(finalize_block_result.block)
}


#[derive(Debug)]
pub struct ParallelBuildingAlgorithm {
    config: ParallelBuilderConfig,
    name: String,
}

impl ParallelBuildingAlgorithm {
    pub fn new(config: ParallelBuilderConfig, name: String) -> Self {
        Self { config, name }
    }
}

impl<P> BlockBuildingAlgorithm<P> for ParallelBuildingAlgorithm
where
    P: StateProviderFactory + Clone + 'static,
{
    fn name(&self) -> String {
        self.name.clone()
    }

    fn build_blocks(&self, input: BlockBuildingAlgorithmInput<P>) {
        let live_input = LiveBuilderInput {
            provider: input.provider,
            ctx: input.ctx.clone(),
            input: input.input,
            sink: input.sink,
            builder_name: self.name.clone(),
            cancel: input.cancel,
        };
        run_parallel_builder(live_input, &self.config);
    }
}
