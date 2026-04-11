pub mod block_building_result_assembler;
pub mod conflict_resolvers;
pub mod conflict_resolving_pool;
pub mod conflict_task_generator;
pub mod groups;
pub mod order_intake_store;
pub mod results_aggregator;
pub mod simulation_cache;
pub mod task;
use alloy_primitives::{Address, I256};
pub use groups::*;
pub mod nonce_handling;
pub mod genetic_algo;
pub use conflict_task_generator::*;
use rbuilder_primitives::{Order, SimulatedOrder};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use rayon::join;

use ahash::HashMap;
use conflict_resolving_pool::ConflictResolvingPool;
use crossbeam::channel::{Sender, Receiver};
use eyre::Result;
use results_aggregator::BestResults;
use reth_provider::StateProvider;
use serde::Deserialize;
use simulation_cache::SharedSimulationCache;
use std::{
    sync::{mpsc as std_mpsc, Arc},
    thread,
    time::{Duration, Instant},
    cmp::Ordering as CmpOrdering,
};
use task::*;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use tracing::{error, trace};

use crate::{
    building::builders::{
        BacktestSimulateBlockInput, Block, BlockBuildingAlgorithm, BlockBuildingAlgorithmInput,
        BuiltBlockIdSource, LiveBuilderInput,
    },
    live_builder::block_output::bidding_service_interface::CompetitionBidContext,
    provider::StateProviderFactory,
    utils::elapsed_ms,
};
use conflict_resolvers::{AlgoRecord, GAGenRecord};

use self::{
    block_building_result_assembler::BlockBuildingResultAssembler,
    order_intake_store::OrderIntakeStore, results_aggregator::ResultsAggregator,
};

pub type GroupId = usize;
pub type ConflictResolutionResultPerGroup = (GroupId, (ResolutionResult, ConflictGroup));
pub type TaskQueueSender = Sender<ConflictTask>;
pub type TaskQueueReceiver = Receiver<ConflictTask>;

/// ParallelBuilderConfig configures parallel builder.
/// * `num_threads` - number of threads to use for merging.
/// * `merge_wait_time_ms` - time to wait for merging to finish before consuming new orders.
/// * `safe_sorting_only` - Will only use sort modes that don't risk breaking much the "best refund for user"
///   since random sorting might put the worst kickback first and let a blind backrun win.
///   This flag is just to test the algo until we solve every issue.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParallelBuilderConfig {
    pub discard_txs: bool,
    pub num_threads: usize,
    pub safe_sorting_only: bool,
}

fn get_communication_channels() -> (
    std_mpsc::Sender<ConflictResolutionResultPerGroup>,
    std_mpsc::Receiver<ConflictResolutionResultPerGroup>,
) {
    std_mpsc::channel()
}

fn get_shared_data_structures() -> (Arc<BestResults>, TaskQueueSender, TaskQueueReceiver) {
    let best_results = Arc::new(BestResults::new());
    let (sender, receiver) = crossbeam::channel::unbounded();
    (best_results, sender, receiver)
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

        let (best_results, task_queue_sender, task_queue_receiver) = get_shared_data_structures();

        let simulation_cache = Arc::new(SharedSimulationCache::new());

        let conflict_finder = ConflictFinder::new();

        let conflict_task_generator = ConflictTaskGenerator::new(
            config.safe_sorting_only,
            task_queue_sender,
            group_result_sender_for_task_generator,
            false, // live builder: single default DexDirectionBalanced task
        );

        let conflict_resolving_pool = ConflictResolvingPool::new(
            config.num_threads,
            task_queue_receiver.clone(),
            config.safe_sorting_only,
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
            Some(input.sink.clone()),
            input.built_block_id_source.clone(),
            input.max_order_execution_duration_warning,
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
                    time_taken_ms = %elapsed_ms(time_start),
                    "Order intake: added new orders and processing groups"
                );
                conflict_task_generator.process_groups(conflict_finder.get_order_groups());
            }
        }
    }
}

fn dedup_by_bundle_signer_and_slots(orders: Vec<Arc<SimulatedOrder>>) -> Vec<Arc<SimulatedOrder>> {
    use std::collections::HashMap;

    let mut keyed: HashMap<(Address, Vec<(Address, u64)>), Arc<SimulatedOrder>> = HashMap::new();
    let mut unkeyed: Vec<Arc<SimulatedOrder>> = Vec::new();

    for order in orders {
        // Only applies to bundles with a signer.
        let signer = match order.order.as_ref() {
            Order::Bundle(b) => match b.signer {
                Some(s) => s,
                None => { unkeyed.push(order); continue; }
            },
            _ => { unkeyed.push(order); continue; }
        };

        // Key on the sorted set of (sender, nonce) slots the bundle touches.
        let txs = order.order.list_txs();
        let mut slots: Vec<(Address, u64)> = txs
            .iter()
            .map(|(tx, _)| (tx.signer(), tx.nonce()))
            .collect();
        slots.sort_unstable();
        slots.dedup();

        let key = (signer, slots);
        let entry = keyed.entry(key).or_insert_with(|| order.clone());
        // Keep the more profitable one.
        if order.sim_value.full_profit_info().coinbase_profit()
            > entry.sim_value.full_profit_info().coinbase_profit()
        {
            *entry = order;
        }
    }

    let mut result = unkeyed;
    result.extend(keyed.into_values());
    result
}

// ── DEX hyperparameter sweep analytics ───────────────────────────────────────

/// One (alpha, lambda) combination result for a single conflict group.
#[derive(Debug, serde::Serialize)]
struct DexCombo {
    alpha: f64,
    lambda: f64,
    profit_wei: String,
}

/// Per-group summary of the DEX hyperparameter sweep.
#[derive(Debug, serde::Serialize)]
struct DexGroupHyperparamResult {
    group_id: usize,
    order_count: usize,
    greedy_profit_wei: String,
    best_dex_profit_wei: String,
    /// `(best_dex - greedy) / greedy * 100`; `null` when greedy profit is zero.
    improvement_over_greedy_pct: Option<f64>,
    /// All (alpha, lambda) combinations that tied for the best DEX profit.
    winning_combos: Vec<DexCombo>,
    /// All 15 combinations, sorted alpha asc then lambda asc.
    all_combos: Vec<DexCombo>,
}

/// Top-level file written per block.
#[derive(Debug, serde::Serialize)]
struct DexHyperparamFile {
    block_number: u64,
    block_profit_wei: String,
    /// Number of groups that had >120 valid orderings and ran the sweep.
    eligible_groups: usize,
    groups: Vec<DexGroupHyperparamResult>,
}

/// Build per-group DEX hyperparameter results from the flat list of `AlgoRecord`s.
fn build_dex_hyperparam_results(algo_records: &[conflict_resolvers::AlgoRecord]) -> Vec<DexGroupHyperparamResult> {
    use alloy_primitives::U256;
    use std::str::FromStr;

    // Bucket records by group_id.
    let mut by_group: HashMap<usize, Vec<&conflict_resolvers::AlgoRecord>> = HashMap::default();
    for rec in algo_records {
        by_group.entry(rec.group_id).or_default().push(rec);
    }

    let mut results: Vec<DexGroupHyperparamResult> = Vec::new();

    for (group_id, records) in &by_group {
        // Split into Greedy and DEX records.
        let greedy_records: Vec<_> = records.iter()
            .filter(|r| r.algo == "Greedy" && r.dex_alpha.is_none())
            .collect();
        let dex_records: Vec<_> = records.iter()
            .filter(|r| r.dex_alpha.is_some())
            .collect();

        // Skip groups that didn't run any DEX tasks (e.g. they fell through to the legacy path).
        if dex_records.is_empty() {
            continue;
        }

        let order_count = records.first().map(|r| r.order_count).unwrap_or(0);

        // Best greedy profit (take the max if somehow multiple records exist).
        let greedy_profit: U256 = greedy_records.iter()
            .filter_map(|r| U256::from_str(&r.profit_wei).ok())
            .max()
            .unwrap_or(U256::ZERO);

        // Find the maximum DEX profit across all (alpha, lambda) combinations.
        let best_dex_profit: U256 = dex_records.iter()
            .filter_map(|r| U256::from_str(&r.profit_wei).ok())
            .max()
            .unwrap_or(U256::ZERO);

        // Collect all combos that tied for best.
        let mut winning_combos: Vec<DexCombo> = dex_records.iter()
            .filter(|r| U256::from_str(&r.profit_wei).ok() == Some(best_dex_profit))
            .map(|r| DexCombo {
                alpha: r.dex_alpha.unwrap_or(0.0),
                lambda: r.dex_lambda.unwrap_or(0.0),
                profit_wei: r.profit_wei.clone(),
            })
            .collect();
        winning_combos.sort_by(|a, b| a.alpha.partial_cmp(&b.alpha)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.lambda.partial_cmp(&b.lambda).unwrap_or(std::cmp::Ordering::Equal)));

        // All combos sorted alpha asc, lambda asc.
        let mut all_combos: Vec<DexCombo> = dex_records.iter()
            .map(|r| DexCombo {
                alpha: r.dex_alpha.unwrap_or(0.0),
                lambda: r.dex_lambda.unwrap_or(0.0),
                profit_wei: r.profit_wei.clone(),
            })
            .collect();
        all_combos.sort_by(|a, b| a.alpha.partial_cmp(&b.alpha)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.lambda.partial_cmp(&b.lambda).unwrap_or(std::cmp::Ordering::Equal)));

        let improvement_over_greedy_pct = if greedy_profit.is_zero() {
            None
        } else {
            // Compute as f64: (best_dex - greedy) / greedy * 100
            // Use saturating sub to avoid panic on underflow.
            let diff = best_dex_profit.saturating_sub(greedy_profit);
            let diff_f64: f64 = diff.to_string().parse().unwrap_or(0.0);
            let greedy_f64: f64 = greedy_profit.to_string().parse().unwrap_or(1.0);
            Some(diff_f64 / greedy_f64 * 100.0)
        };

        results.push(DexGroupHyperparamResult {
            group_id: *group_id,
            order_count,
            greedy_profit_wei: greedy_profit.to_string(),
            best_dex_profit_wei: best_dex_profit.to_string(),
            improvement_over_greedy_pct,
            winning_combos,
            all_combos,
        });
    }

    // Sort by group_id for deterministic output.
    results.sort_by_key(|r| r.group_id);
    results
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
    let (best_results, task_queue_sender, task_queue_receiver) = get_shared_data_structures();

    let (group_result_sender, group_result_receiver) = get_communication_channels();
    let group_result_sender_for_task_generator = group_result_sender.clone();

    let mut conflict_finder = ConflictFinder::new();

    let sorted_orders = {
        let mut orders = input.sim_orders.clone();
        orders.sort_by_key(|o| o.order.id());
        orders
    };

    // println!("Num orders before dedup: {}", input.sim_orders.len());
    // let sorted_orders = {
    //     let mut orders = input.sim_orders.clone();
    //     orders.sort_by_key(|o| o.order.id());
    //     dedup_by_bundle_signer_and_slots(orders)
    // };
    // println!("After dedup: {}", sorted_orders.len());

    let simulation_cache = Arc::new(SharedSimulationCache::new());
    let init_duration = init_start.elapsed();

    // Worker pool and conflict manager creation
    let setup_start = Instant::now();

    let cancel_token = CancellationToken::new();
    let outstanding = Arc::new(AtomicUsize::new(0));
    let mut conflict_resolving_pool = ConflictResolvingPool::new(
        config.num_threads,
        task_queue_receiver.clone(),
        config.safe_sorting_only,
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

    // Keep orders only from the largest group
    // if let Some(largest_group) = groups.clone().into_iter().max_by_key(|group| group.orders.len()) {
    //     groups.retain(|group| group.orders.len() == largest_group.orders.len());
    // }
    // println!("After keeping only largest group(s), {} groups remain with {} orders", groups.len(), groups.iter().map(|g| g.orders.len()).sum::<usize>());

    // Generate tasks using the same logic as live builder
    let mut task_generator = ConflictTaskGenerator::new(
        config.safe_sorting_only,
        task_queue_sender.clone(),
        group_result_sender_for_task_generator,
        true, // backtest: sweep all (alpha,lambda) hyperparameter combinations
    );
    // let processing_start = Instant::now();
    task_generator.process_groups(groups.clone());

    // // Initialise outstanding from the actual number of enqueued tasks
    // let planned = task_queue_receiver.len();
    // outstanding.store(planned, Ordering::Release);

    // // Start worker threads (after tasks are enqueued)
    // if let Err(err) = conflict_resolving_pool.start() {
    //     return Err(err);
    // }
    
    // let mut results: Vec<(GroupId, (ResolutionResult, ConflictGroup))> = Vec::new();
    // let mut last_progress = Instant::now();
    // loop {
    //     match group_result_receiver.recv_timeout(Duration::from_millis(250)) {
    //         Ok(res) => {
    //             results.push(res);
    //             last_progress = Instant::now();
    //         }
    //         Err(RecvTimeoutError::Timeout) => {
    //             if outstanding.load(Ordering::Acquire) == 0 {
    //                 break;
    //             }
    //         }
    //         Err(RecvTimeoutError::Disconnected) => break,
    //     }
    // }

    // // Stop workers
    // cancel_token.cancel();

    // let processing_end = last_progress; // updated on every Ok(res)
    // let processing_duration = processing_end.duration_since(processing_start);

    let (results, algo_records, ga_records) = conflict_resolving_pool.process_groups_backtest(
        groups,
        &input.ctx,
        block_state.clone(),
        Arc::clone(&simulation_cache),
        true, // backtest: sweep all (alpha,lambda) hyperparameter combinations
    );


    // Block building result assembler creation
    let assembler_start = Instant::now();
    let built_block_id_source = Arc::new(BuiltBlockIdSource::new());
    let mut asm_greedy = BlockBuildingResultAssembler::new(
        &config,
        Arc::clone(&best_results),
        block_state.clone(),
        input.ctx.clone(),
        CancellationToken::new(),
        "backtest_builder_greedy".into(),
        None,
        Arc::clone(&built_block_id_source),
        None,
    );

    let mut asm_mgp = BlockBuildingResultAssembler::new(
        &config,
        Arc::clone(&best_results),
        block_state.clone(),
        input.ctx.clone(),
        CancellationToken::new(),
        "backtest_builder_mgp".into(),
        None,
        Arc::clone(&built_block_id_source),
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

    // NOTE: coinbase_payment functionality has been removed, always use payout tx
    let val_greedy = helper_greedy.true_block_value()?;
    let val_mgp    = helper_mgp.true_block_value()?;

    let (mut chosen_helper, mut chosen_asm) = if val_mgp > val_greedy {
        (helper_mgp, asm_mgp)
    } else {
        (helper_greedy, asm_greedy)
    };

    let payout_tx_value = chosen_helper.true_block_value()?;
    let finalize_block_result = chosen_helper.finalize_block(
        &mut chosen_asm.local_ctx,
        payout_tx_value,
        I256::ZERO,
        CompetitionBidContext::no_competition_bid(),
    )?;

    let building_duration = building_start.elapsed();
    let total_duration = start_time.elapsed();

    trace!("Initialization time: {:?}", init_duration);
    trace!("Setup time: {:?}", setup_duration);
    trace!("Assembler creation time: {:?}", assembler_duration);
    trace!("Best results collection time: {:?}", collection_duration);
    trace!("Block building time: {:?}", building_duration);
    trace!("Total time taken: {:?}", total_duration);

    // Write parallel builder analytics
    {
        let analytics_dir = {
            let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let workspace_root = manifest_dir
                .ancestors()
                .find(|p| p.join("Cargo.lock").exists())
                .unwrap_or(manifest_dir.as_path())
                .to_path_buf();
            workspace_root
                .parent()
                .unwrap_or(workspace_root.as_path())
                .join("parallel_builder_analytics")
        };
        if let Err(e) = std::fs::create_dir_all(&analytics_dir) {
            error!(%e, "Failed to create parallel_builder_analytics directory");
        } else {
            let block = input.ctx.block();

            // File 1: algo results + block profit
            #[derive(serde::Serialize)]
            struct AlgoResultsFile<'a> {
                block_number: u64,
                block_profit_wei: String,
                algo_results: &'a [AlgoRecord],
            }
            let algo_file = analytics_dir.join(format!("algo_results_{}.json", block));
            let payload = AlgoResultsFile {
                block_number: block,
                block_profit_wei: payout_tx_value.to_string(),
                algo_results: &algo_records,
            };
            if let Ok(json) = serde_json::to_string_pretty(&payload) {
                if let Err(e) = std::fs::write(&algo_file, json) {
                    error!(%e, "Failed to write algo_results analytics");
                }
            }

            // File 2: GA per-generation records
            if !ga_records.is_empty() {
                let ga_file = analytics_dir.join(format!("ga_analytics_{}.json", block));
                if let Ok(json) = serde_json::to_string_pretty(&ga_records) {
                    if let Err(e) = std::fs::write(&ga_file, json) {
                        error!(%e, "Failed to write ga_analytics");
                    }
                }
            }

            // File 3: DEX hyperparameter sweep results
            let dex_groups = build_dex_hyperparam_results(&algo_records);
            if !dex_groups.is_empty() {
                let dex_payload = DexHyperparamFile {
                    block_number: block,
                    block_profit_wei: payout_tx_value.to_string(),
                    eligible_groups: dex_groups.len(),
                    groups: dex_groups,
                };
                let dex_file = analytics_dir.join(format!("dex_hyperparam_{}.json", block));
                if let Ok(json) = serde_json::to_string_pretty(&dex_payload) {
                    if let Err(e) = std::fs::write(&dex_file, json) {
                        error!(%e, "Failed to write dex_hyperparam analytics");
                    }
                }
            }
        }
    }

    Ok(finalize_block_result.block)
}


#[derive(Debug)]
pub struct ParallelBuildingAlgorithm {
    config: ParallelBuilderConfig,
    max_order_execution_duration_warning: Option<Duration>,
    name: String,
}

impl ParallelBuildingAlgorithm {
    pub fn new(
        config: ParallelBuilderConfig,
        max_order_execution_duration_warning: Option<Duration>,
        name: String,
    ) -> Self {
        Self {
            config,
            max_order_execution_duration_warning,
            name,
        }
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
            built_block_cache: input.built_block_cache,
            built_block_id_source: input.built_block_id_source,
            max_order_execution_duration_warning: self.max_order_execution_duration_warning,
        };
        run_parallel_builder(live_input, &self.config);
    }
}
