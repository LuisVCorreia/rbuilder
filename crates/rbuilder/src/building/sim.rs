use super::{
    tracers::{AccumulatorSimulationTracer, SimulationTracer},
    OrderErr, PartialBlockFork, ThreadBlockBuildingContext,
};
use crate::{
    building::{BlockBuildingContext, BlockState, CriticalCommitOrderError},
    primitives::{Order, OrderId, SimValue, SimulatedOrder},
    provider::StateProviderFactory,
    telemetry::{add_order_simulation_time, mark_order_pending_nonce},
    utils::NonceCache,
};
use ahash::{HashMap, HashSet};
use alloy_primitives::Address;
use rand::seq::SliceRandom;
use reth_errors::ProviderError;
use reth_provider::StateProvider;
use std::{
    cmp::{max, min, Ordering},
    collections::hash_map::Entry,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, trace};

use std::fs;
use serde_json::json;
use tikv_jemalloc_ctl::{epoch, stats};
#[cfg(target_os = "linux")]
use libc::{getrusage, rusage, RUSAGE_SELF};

#[derive(Clone, Copy, Default, Debug)]
struct AllocStats {
    allocated: u64,
    active: u64,
    resident: u64,
}

fn read_alloc_stats() -> AllocStats {
    // Refresh jemalloc epoch so stats reflect recent frees/mallocs.
    let _ = epoch::advance();
    AllocStats {
        allocated: stats::allocated::read().unwrap_or(0) as u64,
        active: stats::active::read().unwrap_or(0) as u64,
        resident: stats::resident::read().unwrap_or(0) as u64,
    }
}

fn read_rss_and_hwm_bytes() -> (u64, u64) {
    // Linux: parse /proc/self/status for VmRSS and VmHWM (kB -> bytes).
    let mut rss = 0;
    let mut hwm = 0;
    if let Ok(s) = fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.split_whitespace().find_map(|t| t.parse::<u64>().ok()) {
                    rss = kb.saturating_mul(1024);
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb) = rest.split_whitespace().find_map(|t| t.parse::<u64>().ok()) {
                    hwm = kb.saturating_mul(1024);
                }
            }
        }
    }
    (rss, hwm)
}

#[cfg(target_os = "linux")]
fn read_ru_maxrss_bytes() -> u64 {
    unsafe {
        let mut r: rusage = std::mem::zeroed();
        if getrusage(RUSAGE_SELF, &mut r) == 0 {
            (r.ru_maxrss as u64).saturating_mul(1024)
        } else {
            0
        }
    }
}
#[cfg(not(target_os = "linux"))]
fn read_ru_maxrss_bytes() -> u64 { 0 }


#[derive(Debug)]
pub enum OrderSimResult {
    Success(SimulatedOrder, Vec<(Address, u64)>),
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
    order: Order,
    unsatisfied_nonces: usize,
}

pub type SimulationId = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimulationRequest {
    pub id: SimulationId,
    pub order: Order,
    pub parents: Vec<Order>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedResult {
    pub id: SimulationId,
    pub simulated_order: SimulatedOrder,
    pub previous_orders: Vec<Order>,
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
    Ready(Vec<Order>),
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

    fn push_order(&mut self, order: Order) -> Result<(), ProviderError> {
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

    pub fn push_orders(&mut self, orders: Vec<Order>) -> Result<(), ProviderError> {
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
                            .coinbase_profit
                    };
                    if result.simulated_order.sim_value.coinbase_profit > current_sim_profit {
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
    orders: &[Order],
    randomize_insertion: bool,
) -> Result<(Vec<SimulatedOrder>, Vec<OrderErr>), CriticalCommitOrderError>
where
    P: StateProviderFactory + Clone,
{
    let nonces = {
        let state = provider.history_by_block_hash(ctx.attributes.parent)?;
        NonceCache::new(state.into())
    };
    let mut sim_tree = SimTree::new(nonces);

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
    let mut state_for_sim =
        Arc::<dyn StateProvider>::from(provider.history_by_block_hash(ctx.attributes.parent)?);
    let mut local_ctx = ThreadBlockBuildingContext::default();

    // Collect per-tx perf rows
    let mut perf_records: Vec<serde_json::Value> = Vec::new();

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

            let alloc_before = read_alloc_stats();
            let (rss_before, hwm_before) = read_rss_and_hwm_bytes();
            let ru_before = read_ru_maxrss_bytes();

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

            let sim_dur = start_time.elapsed();
            let alloc_after = read_alloc_stats();
            let (rss_after, hwm_after) = read_rss_and_hwm_bytes();
            let ru_after = read_ru_maxrss_bytes();

            // let is_success = matches!(sim_result.result, OrderSimResult::Success(_, _));
            // println!("Simulation result: {}, gas used: {}", is_success, sim_result.gas_used);
            let alloc_delta = alloc_after.allocated.saturating_sub(alloc_before.allocated);
            let active_delta = alloc_after.active.saturating_sub(alloc_before.active);
            let resident_delta = alloc_after.resident.saturating_sub(alloc_before.resident);
            let rss_delta = rss_after.saturating_sub(rss_before);
            let hwm_delta = hwm_after.saturating_sub(hwm_before);
            let ru_delta = ru_after.saturating_sub(ru_before);

            match sim_result.result {
                OrderSimResult::Failed(err) => {
                    println!(
                        "Order simulation failed: order = {}, err = {}",
                        sim_task.order.id(),
                        err
                    );
                    sim_errors.push(err);
                    continue;
                }
                OrderSimResult::Success(sim_order, nonces) => {
                    perf_records.push(json!({
                        "order_id": format!("{}", sim_task.order.id()),
                        "success": true,
                        "time_ms": sim_dur.as_secs_f64() * 1_000.0,
                        "gas_used": sim_order.sim_value.gas_used,
                        "blob_gas_used": sim_order.sim_value.blob_gas_used,
                        "coinbase_profit_wei": format!("{}", sim_order.sim_value.coinbase_profit),

                        // allocator / OS metrics
                        "jemalloc_allocated_delta": alloc_delta,
                        "jemalloc_active_delta": active_delta,
                        "jemalloc_resident_delta": resident_delta,
                        "rss_before_bytes": rss_before,
                        "rss_after_bytes": rss_after,
                        "rss_delta_bytes": rss_delta,
                        "hwm_before_bytes": hwm_before,
                        "hwm_after_bytes": hwm_after,
                        "hwm_delta_bytes": hwm_delta,
                        "ru_maxrss_before_bytes": ru_before,
                        "ru_maxrss_after_bytes": ru_after,
                        "ru_maxrss_delta_bytes": ru_delta
                    }));

                    let result = SimulatedResult {
                        id: sim_task.id,
                        simulated_order: sim_order,
                        previous_orders: sim_task.parents,
                        nonces_after: nonces
                            .into_iter()
                            .map(|(address, nonce)| NonceKey { address, nonce })
                            .collect(),

                        simulation_time: start_time.elapsed(),
                    };
                    sim_results.push(result);
                }
            }
        }
        sim_tree.submit_simulation_tasks_results(sim_results)?;
    }

    // let out_dir = PathBuf::from("performance_testing/ave_speed_and_mem");
    // if let Err(e) = fs::create_dir_all(&out_dir) {
    //     tracing::warn!(?e, "failed to create ave_speed_and_mem dir");
    // } else {
    //     let out_file = out_dir.join(format!("{}.json", ctx.evm_env.block_env.number));
    //     match serde_json::to_string_pretty(&perf_records) {
    //         Ok(s) => {
    //             if let Err(e) = fs::write(&out_file, s) {
    //                 tracing::warn!(?e, ?out_file, "failed to write perf metrics json");
    //             }
    //         }
    //         Err(e) => tracing::warn!(?e, "failed to serialize perf metrics json"),
    //     }
    // }

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
    parent_orders: Vec<Order>,
    order: Order,
    ctx: &BlockBuildingContext,
    local_ctx: &mut ThreadBlockBuildingContext,
    state: &mut BlockState,
) -> Result<OrderSimResultWithGas, CriticalCommitOrderError> {
    let mut tracer = AccumulatorSimulationTracer::new();
    let mut fork = PartialBlockFork::new(state, ctx, local_ctx).with_tracer(&mut tracer);
    let rollback_point = fork.rollback_point();
    let sim_res = simulate_order_using_fork(parent_orders, order, &mut fork);
    fork.rollback(rollback_point);
    let sim_res = sim_res?;
    Ok(OrderSimResultWithGas {
        result: sim_res,
        gas_used: tracer.used_gas,
    })
}

/// Simulates order (including parent (those needed to reach proper nonces) orders) using a precreated fork
pub fn simulate_order_using_fork<Tracer: SimulationTracer>(
    parent_orders: Vec<Order>,
    order: Order,
    fork: &mut PartialBlockFork<'_, '_, '_, '_, Tracer>,
) -> Result<OrderSimResult, CriticalCommitOrderError> {
    let start = Instant::now();
    // simulate parents
    let mut gas_used = 0;
    let mut blob_gas_used = 0;
    for parent in parent_orders {
        let result = fork.commit_order(&parent, gas_used, 0, blob_gas_used, true)?;
        match result {
            Ok(res) => {
                gas_used += res.gas_used;
                blob_gas_used += res.blob_gas_used;
            }
            Err(err) => {
                tracing::trace!(parent_order = ?parent.id(), ?err, "failed to simulate parent order");
                return Ok(OrderSimResult::Failed(err));
            }
        }
    }

    // simulate
    let result = fork.commit_order(&order, gas_used, 0, blob_gas_used, true)?;
    let sim_time = start.elapsed();
    add_order_simulation_time(sim_time, "sim", result.is_ok()); // we count parent sim time + order sim time time here

    match result {
        Ok(res) => {
            let sim_value = SimValue::new(
                res.coinbase_profit,
                res.gas_used,
                res.blob_gas_used,
                res.paid_kickbacks,
            );
            let new_nonces = res.nonces_updated.into_iter().collect::<Vec<_>>();
            Ok(OrderSimResult::Success(
                SimulatedOrder {
                    order,
                    sim_value,
                    used_state_trace: res.used_state_trace,
                },
                new_nonces,
            ))
        }
        Err(err) => Ok(OrderSimResult::Failed(err)),
    }
}
