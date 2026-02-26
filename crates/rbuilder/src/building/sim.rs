use super::{
    create_sim_value,
    tracers::{AccumulatorSimulationTracer, SimulationTracer},
    OrderErr, PartialBlockFork, ThreadBlockBuildingContext,
};
use crate::{
    building::{
        order_is_worth_executing, BlockBuildingContext, BlockBuildingSpaceState, BlockState,
        CriticalCommitOrderError, NullPartialBlockForkExecutionTracer,
    },
    live_builder::order_input::mempool_txs_detector::MempoolTxsDetector,
    provider::StateProviderFactory,
    telemetry::{add_order_simulation_time, mark_order_pending_nonce},
    utils::NonceCache,
};
use ahash::{HashMap, HashSet};
use alloy_primitives::Address;
use rand::seq::SliceRandom;
use rbuilder_primitives::{Order, OrderId, SimulatedOrder};
use reth_errors::ProviderError;
use reth_provider::StateProvider;
use std::{
    cmp::{max, min, Ordering},
    collections::hash_map::Entry,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, trace, info};

// Simulation analytics types

/// Record emitted for every successful simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimulationRecord {
    /// Unique id of the simulation task
    pub sim_id: SimulationId,
    /// Human-readable order id
    pub order_id: String,
    /// "bundle" or "mempool_tx" (or "unknown" if we can't tell)
    pub order_kind: String,
    /// Whether any transaction in the order carries blob (EIP-4844) sidecar data
    pub has_blob_txs: bool,
    /// Wall-clock time the simulation took (including parent simulations), in microseconds
    pub simulation_time_us: u64,
    /// Gas consumed by the simulation (includes reverted txs)
    pub gas_used: u64,
    /// Number of transactions in the order
    pub num_txs: usize,
    /// Number of parent orders that had to be simulated first
    pub num_parents: usize,
    /// Coinbase profit of the simulated order (wei, stored as string to avoid f64 precision loss)
    pub coinbase_profit_wei: String,
}

/// Accumulates [`SimulationRecord`]s and can flush them to a JSON file.
#[derive(Debug, Default)]
pub struct SimulationAnalytics {
    records: Vec<SimulationRecord>,
}

impl SimulationAnalytics {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub fn push(&mut self, record: SimulationRecord) {
        self.records.push(record);
    }

    pub fn records(&self) -> &[SimulationRecord] {
        &self.records
    }

    /// Write all collected records to `path` as a JSON array.
    /// Appends to existing data if the file already exists.
    pub fn flush_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;

        let mut all_records: Vec<SimulationRecord> = Vec::new();

        // If the file already exists, load previous records so we append
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(path) {
                if let Ok(mut existing) =
                    serde_json::from_str::<Vec<SimulationRecord>>(&contents)
                {
                    all_records.append(&mut existing);
                }
            }
        }

        all_records.extend(self.records.iter().cloned());

        let json = serde_json::to_string_pretty(&all_records)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;

        info!(
            path = %path.display(),
            new_records = self.records.len(),
            total_records = all_records.len(),
            "Flushed simulation analytics"
        );

        Ok(())
    }
}

// Helpers to classify orders

/// Returns `"bundle"` or `"mempool_tx"` (or `"unknown"`).
fn classify_order_kind(order: &Order) -> &'static str {
    match order {
        Order::Bundle(_) => "bundle",
        Order::Tx(_) => "mempool_tx",
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum OrderSimResult {
    Success(Arc<SimulatedOrder>, Vec<(Address, u64)>),
    Failed(OrderErr),
}

#[derive(Debug)]
pub struct OrderSimResultWithGas {
    pub result: OrderSimResult,
    /// gas_used includes ANY gas consumed (eg: reverted txs)
    pub gas_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NonceKey {
    pub address: Address,
    pub nonce: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingOrder {
    order: Arc<Order>,
    unsatisfied_nonces: usize,
}

pub type SimulationId = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimulationRequest {
    pub id: SimulationId,
    pub order: Arc<Order>,
    pub parents: Vec<Arc<Order>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedResult {
    pub id: SimulationId,
    pub simulated_order: Arc<SimulatedOrder>,
    pub previous_orders: Vec<Arc<Order>>,
    pub nonces_after: Vec<NonceKey>,
    pub simulation_time: Duration,
}

// @Feat replaceable orders
#[derive(Debug)]
pub struct SimTree {
    // fields for nonce management
    nonces: NonceCache,

    sims: HashMap<SimulationId, SimulatedResult>,
    sims_that_update_one_nonce: HashMap<NonceKey, SimulationId>,

    pending_orders: HashMap<OrderId, PendingOrder>,
    pending_nonces: HashMap<NonceKey, Vec<OrderId>>,

    ready_orders: Vec<SimulationRequest>,
}

#[derive(Debug)]
enum OrderNonceState {
    Invalid,
    PendingNonces(Vec<NonceKey>),
    Ready(Vec<Arc<Order>>),
}

impl SimTree {
    pub fn new(nonce_cache_ref: NonceCache) -> Self {
        Self {
            nonces: nonce_cache_ref,
            sims: HashMap::default(),
            sims_that_update_one_nonce: HashMap::default(),
            pending_orders: HashMap::default(),
            pending_nonces: HashMap::default(),
            ready_orders: Vec::default(),
        }
    }

    fn push_order(&mut self, order: Arc<Order>) -> Result<(), ProviderError> {
        if self.pending_orders.contains_key(&order.id()) {
            return Ok(());
        }

        let order_nonce_state = self.get_order_nonce_state(&order)?;

        let order_id = order.id();

        match order_nonce_state {
            OrderNonceState::Invalid => {
                return Ok(());
            }
            OrderNonceState::PendingNonces(pending_nonces) => {
                mark_order_pending_nonce(order_id);
                let unsatisfied_nonces = pending_nonces.len();
                for nonce in pending_nonces {
                    self.pending_nonces
                        .entry(nonce)
                        .or_default()
                        .push(order.id());
                }
                self.pending_orders.insert(
                    order.id(),
                    PendingOrder {
                        order,
                        unsatisfied_nonces,
                    },
                );
            }
            OrderNonceState::Ready(parents) => {
                self.ready_orders.push(SimulationRequest {
                    id: rand::random(),
                    order,
                    parents,
                });
            }
        }
        Ok(())
    }

    fn get_order_nonce_state(&mut self, order: &Order) -> Result<OrderNonceState, ProviderError> {
        let mut onchain_nonces_incremented = HashSet::default();
        let mut pending_nonces = Vec::new();
        let mut parent_orders = Vec::new();

        for nonce in order.nonces() {
            let onchain_nonce = self.nonces.nonce(nonce.address)?;

            match onchain_nonce.cmp(&nonce.nonce) {
                Ordering::Equal => {
                    // nonce, valid
                    onchain_nonces_incremented.insert(nonce.address);
                    continue;
                }
                Ordering::Greater => {
                    // nonce invalid, maybe its optional
                    if !nonce.optional {
                        // this order will never be valid
                        trace!(
                            order = ?order.id(),
                            ?nonce,
                            "Dropping order because of nonce"
                        );
                        return Ok(OrderNonceState::Invalid);
                    } else {
                        // we can ignore this tx
                        continue;
                    }
                }
                Ordering::Less => {
                    if onchain_nonces_incremented.contains(&nonce.address) {
                        // we already considered this account nonce
                        continue;
                    }
                    // mark this nonce as considered
                    onchain_nonces_incremented.insert(nonce.address);

                    let nonce_key = NonceKey {
                        address: nonce.address,
                        nonce: nonce.nonce,
                    };

                    if let Some(sim_id) = self.sims_that_update_one_nonce.get(&nonce_key) {
                        // we have something that fills this nonce
                        let sim = self.sims.get(sim_id).expect("we never delete sims");
                        parent_orders.extend_from_slice(&sim.previous_orders);
                        parent_orders.push(sim.simulated_order.order.clone());
                        continue;
                    }

                    pending_nonces.push(nonce_key);
                }
            }
        }

        if pending_nonces.is_empty() {
            Ok(OrderNonceState::Ready(parent_orders))
        } else {
            Ok(OrderNonceState::PendingNonces(pending_nonces))
        }
    }

    pub fn push_orders(&mut self, orders: Vec<Arc<Order>>) -> Result<(), ProviderError> {
        for order in orders {
            self.push_order(order)?;
        }
        Ok(())
    }

    pub fn pop_simulation_tasks(&mut self, limit: usize) -> Vec<SimulationRequest> {
        let limit = min(limit, self.ready_orders.len());
        self.ready_orders.drain(..limit).collect()
    }

    // we don't really need state here because nonces are cached but its smaller if we reuse pending state fn
    fn process_simulation_task_result(
        &mut self,
        result: SimulatedResult,
    ) -> Result<(), ProviderError> {
        self.sims.insert(result.id, result.clone());
        let mut orders_ready = Vec::new();
        if result.nonces_after.len() == 1 {
            let updated_nonce = result.nonces_after.first().unwrap().clone();

            match self.sims_that_update_one_nonce.entry(updated_nonce.clone()) {
                Entry::Occupied(mut entry) => {
                    let current_sim_profit = {
                        let sim_id = entry.get_mut();
                        self.sims
                            .get(sim_id)
                            .expect("we never delete sims")
                            .simulated_order
                            .sim_value
                            .full_profit_info()
                            .coinbase_profit()
                    };
                    if result
                        .simulated_order
                        .sim_value
                        .full_profit_info()
                        .coinbase_profit()
                        > current_sim_profit
                    {
                        entry.insert(result.id);
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(result.id);

                    if let Some(pending_orders) = self.pending_nonces.remove(&updated_nonce) {
                        for order in pending_orders {
                            match self.pending_orders.entry(order) {
                                Entry::Occupied(mut entry) => {
                                    let pending_order = entry.get_mut();
                                    pending_order.unsatisfied_nonces -= 1;
                                    if pending_order.unsatisfied_nonces == 0 {
                                        orders_ready.push(entry.remove().order);
                                    }
                                }
                                Entry::Vacant(_) => {
                                    error!("SimTree bug order not found");
                                    // @Metric bug counter
                                }
                            }
                        }
                    }
                }
            }
        }

        for ready_order in orders_ready {
            let pending_state = self.get_order_nonce_state(&ready_order)?;
            match pending_state {
                OrderNonceState::Ready(parents) => {
                    self.ready_orders.push(SimulationRequest {
                        id: rand::random(),
                        order: ready_order,
                        parents,
                    });
                }
                OrderNonceState::Invalid => {
                    // @Metric bug counter
                    error!("SimTree bug order became invalid");
                }
                OrderNonceState::PendingNonces(_) => {
                    // @Metric bug counter
                    error!("SimTree bug order became pending again");
                }
            }
        }
        Ok(())
    }

    pub fn submit_simulation_tasks_results(
        &mut self,
        results: Vec<SimulatedResult>,
    ) -> Result<(), ProviderError> {
        for result in results {
            self.process_simulation_task_result(result)?;
        }
        Ok(())
    }
}

/// Non-interactive usage of sim tree that will simply simulate all orders.
/// `randomize_insertion` is used to debug if sim tree works correctly when orders are inserted in a different order
/// outputs should be independent of this arg.
pub fn simulate_all_orders_with_sim_tree<P>(
    provider: P,
    ctx: &BlockBuildingContext,
    orders: &[Arc<Order>],
    randomize_insertion: bool,
) -> Result<(Vec<Arc<SimulatedOrder>>, Vec<OrderErr>), CriticalCommitOrderError>
where
    P: StateProviderFactory + Clone,
{
    let nonces = {
        let state = provider.history_by_block_hash(ctx.attributes.parent)?;
        NonceCache::new(state.into())
    };
    let mut sim_tree = SimTree::new(nonces);

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
            .join("simulation_analytics")
    };
    std::fs::create_dir_all(&analytics_dir).unwrap_or_else(|e| {
        error!(%e, "Failed to create analytics directory");
    });
    let analytics_path = analytics_dir.join(format!("sim_analytics_{}.json", ctx.block()));

    let mut orders = orders.to_vec();
    let random_insert_size = max(orders.len() / 20, 1);
    if randomize_insertion {
        let mut rng = rand::thread_rng();
        // shuffle orders
        orders.shuffle(&mut rng);
    } else {
        sim_tree.push_orders(orders.clone())?;
    }

    let mut sim_errors = Vec::new();
    let mut analytics = SimulationAnalytics::new();
    let mut state_for_sim =
        Arc::<dyn StateProvider>::from(provider.history_by_block_hash(ctx.attributes.parent)?);
    let mut local_ctx = ThreadBlockBuildingContext::default();
    loop {
        // mix new orders into the sim_tree
        if randomize_insertion && !orders.is_empty() {
            let insert_size = min(random_insert_size, orders.len());
            let orders = orders.drain(..insert_size).collect::<Vec<_>>();
            sim_tree.push_orders(orders)?;
        }

        let sim_tasks = sim_tree.pop_simulation_tasks(1000);
        if sim_tasks.is_empty() {
            if randomize_insertion && !orders.is_empty() {
                continue;
            } else {
                break;
            }
        }

        let mut sim_results = Vec::new();
        for sim_task in sim_tasks {
            let start_time = Instant::now();
            let mut block_state = BlockState::new_arc(state_for_sim);
            let sim_result = simulate_order(
                sim_task.parents.clone(),
                sim_task.order.clone(),
                ctx,
                &mut local_ctx,
                &mut block_state,
            )?;
            let (_, provider) = block_state.into_parts();
            state_for_sim = provider;
            match sim_result.result {
                OrderSimResult::Failed(err) => {
                    trace!(
                        order = sim_task.order.id().to_string(),
                        ?err,
                        "Order simulation failed"
                    );
                    sim_errors.push(err);
                    continue;
                }
                OrderSimResult::Success(ref sim_order, ref nonces) => {
                    let elapsed = start_time.elapsed();

                    // Record analytics for successful simulation
                    let record = SimulationRecord {
                        sim_id: sim_task.id,
                        order_id: sim_task.order.id().to_string(),
                        order_kind: classify_order_kind(&sim_task.order).to_string(),
                        has_blob_txs: sim_task.order.has_blobs(),
                        simulation_time_us: elapsed.as_micros() as u64,
                        gas_used: sim_result.gas_used,
                        num_txs: sim_task.order.list_txs_len(),
                        num_parents: sim_task.parents.len(),
                        coinbase_profit_wei: sim_order
                            .sim_value
                            .full_profit_info()
                            .coinbase_profit()
                            .to_string(),
                    };
                    analytics.push(record);

                    let result = SimulatedResult {
                        id: sim_task.id,
                        simulated_order: sim_order.clone(),
                        previous_orders: sim_task.parents,
                        nonces_after: nonces
                            .iter()
                            .map(|(address, nonce)| NonceKey {
                                address: *address,
                                nonce: *nonce,
                            })
                            .collect(),
                        simulation_time: elapsed,
                    };
                    sim_results.push(result);
                }
            }
        }
        sim_tree.submit_simulation_tasks_results(sim_results)?;
    }

    if let Err(e) = analytics.flush_to_file(analytics_path.as_path()) {
        error!(%e, "Failed to write simulation analytics");
    }

    Ok((
        sim_tree
            .sims
            .into_values()
            .map(|sim| sim.simulated_order)
            .collect(),
        sim_errors,
    ))
}

/// Prepares context (fork + tracer) and calls simulate_order_using_fork
pub fn simulate_order(
    parent_orders: Vec<Arc<Order>>,
    order: Arc<Order>,
    ctx: &BlockBuildingContext,
    local_ctx: &mut ThreadBlockBuildingContext,
    state: &mut BlockState,
) -> Result<OrderSimResultWithGas, CriticalCommitOrderError> {
    let mut tracer = AccumulatorSimulationTracer::new();
    let mut fork = PartialBlockFork::new(state, ctx, local_ctx).with_tracer(&mut tracer);
    let rollback_point = fork.rollback_point();
    let sim_res =
        simulate_order_using_fork(parent_orders, order, &mut fork, &ctx.mempool_tx_detector);
    fork.rollback(rollback_point);
    let sim_res = sim_res?;
    Ok(OrderSimResultWithGas {
        result: sim_res,
        gas_used: tracer.used_gas,
    })
}

/// Simulates order (including parent (those needed to reach proper nonces) orders) using a precreated fork
pub fn simulate_order_using_fork<Tracer: SimulationTracer>(
    parent_orders: Vec<Arc<Order>>,
    order: Arc<Order>,
    fork: &mut PartialBlockFork<'_, '_, '_, '_, Tracer, NullPartialBlockForkExecutionTracer>,
    mempool_tx_detector: &MempoolTxsDetector,
) -> Result<OrderSimResult, CriticalCommitOrderError> {
    let start = Instant::now();
    // simulate parents
    let mut space_state = BlockBuildingSpaceState::ZERO;
    // We use empty combined refunds because the value of the bundle will
    // not change from batching.
    let combined_refunds = std::collections::HashMap::default();
    for parent in &parent_orders {
        let result = fork.commit_order(parent, space_state, true, &combined_refunds)?;
        match result {
            Ok(res) => {
                space_state.use_space(res.space_used);
            }
            Err(err) => {
                tracing::trace!(parent_order = ?parent.id(), ?err, "failed to simulate parent order");
                return Ok(OrderSimResult::Failed(err));
            }
        }
    }

    // simulate
    let result = fork.commit_order(&order, space_state, true, &combined_refunds)?;
    let sim_time = start.elapsed();
    add_order_simulation_time(sim_time, "sim", result.is_ok()); // we count parent sim time + order sim time time here

    match result {
        Ok(res) => {
            let sim_value = create_sim_value(&order, &res, mempool_tx_detector);
            if let Err(err) = order_is_worth_executing(&sim_value) {
                return Ok(OrderSimResult::Failed(err));
            }
            let new_nonces = res.nonces_updated.into_iter().collect::<Vec<_>>();
            Ok(OrderSimResult::Success(
                Arc::new(SimulatedOrder::new(order, sim_value, res.used_state_trace)),
                new_nonces,
            ))
        }
        Err(err) => Ok(OrderSimResult::Failed(err)),
    }
}
