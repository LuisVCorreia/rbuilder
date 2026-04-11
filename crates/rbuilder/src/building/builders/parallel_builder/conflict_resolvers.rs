use ahash::{HashMap, HashSet};
use alloy_primitives::{Address, U256};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use rand::seq::SliceRandom;
use reth::providers::StateProvider;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use rayon::prelude::*;

use super::{
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
    nonce_handling::{GroupDeps, GreedyKey, allowed_indices_after_nonce_dedup, enumerate_all_with_choices, ALL_PERMS_CAP},
    genetic_algo::*,
};

use crate::building::builders::parallel_builder::nonce_handling::ordering_with_biased_choices_and_weighted_topo;
use crate::building::{
    BlockBuildingContext, BlockBuildingSpaceState, BlockState, ExecutionError, ExecutionResult, PartialBlock,
    ThreadBlockBuildingContext, order_is_worth_executing,
};
use rbuilder_primitives::{BlockSpace, SimValue};
use rbuilder_primitives::{OrderId, SimulatedOrder};

// Analytics types

/// One record per algorithm run on a conflict group.
#[derive(Debug, serde::Serialize)]
pub struct AlgoRecord {
    pub group_id: usize,
    pub algo: String,
    pub order_count: usize,
    pub elapsed_ms: f64,
    pub profit_wei: String,
    pub sequence_tx_count: usize,
    pub gas_used: u64,
    /// Set only for `DexDirectionBalanced` runs; `None` for all other algorithms.
    pub dex_alpha: Option<f64>,
    pub dex_lambda: Option<f64>,
}

/// One record per GA generation (plus an "initial" record before gen 0).
#[derive(Debug, serde::Serialize)]
pub struct GAGenRecord {
    pub group_id: usize,
    pub is_initial: bool,
    pub generation: usize,
    pub best_profit_wei: String,
    pub improved: bool,
    // children diagnostics (0 for is_initial)
    pub num_children: usize,
    pub children_evaluated: usize,
    pub children_failed: usize,
    pub children_zero_profit: usize,
    pub children_better: usize,
    pub children_equal: usize,
    pub children_worse: usize,
    pub child_profit_min_wei: String,
    pub child_profit_median_wei: String,
    pub child_profit_max_wei: String,
    pub dc_replacements: usize,
    pub dc_kept: usize,
    // timing in microseconds (0 for is_initial)
    pub gen_time_us: u64,
    pub eval_time_us: u64,
    pub dc_time_us: u64,
    pub migration_time_us: u64,
    pub total_elapsed_us: u64,
    // population stats
    pub pop_size: usize,
    pub unique_genomes: usize,
    pub unique_profits: usize,
    pub zero_profit_in_pop: usize,
    pub pop_profit_min_wei: String,
    pub pop_profit_median_wei: String,
    pub pop_profit_max_wei: String,
}

fn build_group_deps(task: &ConflictTask) -> Option<GroupDeps> {
    GroupDeps::from_group(&task.group)
}

#[derive(Derivative)]
#[derivative(Debug, Clone)]
pub struct ResolverContext {
    #[derivative(Debug = "ignore")]
    pub state: Arc<dyn StateProvider + Send + Sync>,
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
    pub fn run_conflict_task(&mut self, task: ConflictTask, local_ctx: &mut ThreadBlockBuildingContext) -> Result<(ResolutionResult, AlgoRecord, Vec<GAGenRecord>)> {
        trace!(
            "run_conflict_task: {:?} with algorithm {:?}",
            task.group.id,
            task.algorithm
        );

        let start = Instant::now();
        let algo = format!("{:?}", task.algorithm);
        let group_id = task.group.id;
        let order_count = task.group.orders.len();

        let inner: Result<(ResolutionResult, Vec<GAGenRecord>)> = match task.algorithm {
            Algorithm::Genetic {
                population,
                crossover_rate,
                mutation_rate,
                max_generations,
                time_ms,
                seed,
                num_islands,
                migration_interval,
                w_choice,
                early_stopping_generations,
                temp_tight_low,
                temp_tight_high,
                temp_broad_low,
                temp_broad_high,
                tight_fraction,
            } => {
                let params = GAParams {
                    population,
                    crossover_rate,
                    mutation_rate,
                    max_generations,
                    time_ms,
                    seed,
                    num_islands,
                    migration_interval,
                    w_choice,
                    early_stopping_generations,
                    temp_tight_low,
                    temp_tight_high,
                    temp_broad_low,
                    temp_broad_high,
                    tight_fraction,
                };
                let (res, ga_records) = self.run_genetic(&task, params, local_ctx)?;
                trace!(
                    "Resolved GA task {:?} with profit: {:?}",
                    task.group.id,
                    res.total_profit
                );
                Ok((res, ga_records))
            }
            Algorithm::GreedyHeap => {
                let (res_profit, res_mgp) = rayon::join(
                    || self.process_orders_greedy(&task, self.state.clone(), &mut local_ctx.clone(), GreedyKey::Profit),
                    || self.process_orders_greedy(&task, self.state.clone(), &mut local_ctx.clone(), GreedyKey::MevGasPrice),
                );

                let mut resolution_result = res_profit?.0;
                if let Ok((mgp_res, _)) = res_mgp {
                    self.update_best_result(mgp_res, &mut resolution_result);
                }

                trace!(
                    "Resolved greedy task {:?} with profit: {:?}",
                    task.group.id,
                    resolution_result.total_profit
                );
                Ok((resolution_result, vec![]))
            }
            _ => {
                let sequences = generate_sequences_of_orders_to_try(&task);

                let results: Vec<ResolutionResult> = sequences
                    .into_par_iter()
                    .map_init(
                        || local_ctx.clone(),
                        |thread_ctx, sequence_of_orders| {
                            self.process_sequence_of_orders(sequence_of_orders, &task, self.state.clone(), thread_ctx)
                                .map(|(resolution_result, _)| resolution_result)
                        },
                    )
                    .filter_map(|r| r.ok())
                    .collect();

                let mut best_resolution_result = ResolutionResult::new(U256::ZERO, 0, vec![]);
                for resolution_result in results {
                    self.update_best_result(resolution_result, &mut best_resolution_result);
                }
                trace!(
                    "Resolved conflict task {:?} with profit: {:?} and algorithm: {:?}",
                    task.group.id,
                    best_resolution_result.total_profit,
                    task.algorithm
                );
                Ok((best_resolution_result, vec![]))
            }
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (dex_alpha, dex_lambda) = match task.algorithm {
            Algorithm::DexDirectionBalanced { alpha, lambda } => (Some(alpha), Some(lambda)),
            _ => (None, None),
        };
        inner.map(|(res, ga_records)| {
            let algo_record = AlgoRecord {
                group_id,
                algo,
                order_count,
                elapsed_ms,
                profit_wei: res.total_profit.to_string(),
                sequence_tx_count: res.sequence_of_orders.len(),
                gas_used: res.gas_used,
                dex_alpha,
                dex_lambda,
            };
            (res, algo_record, ga_records)
        })
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
        if new_result.total_profit > best_result.total_profit
            || (new_result.total_profit == best_result.total_profit
                && new_result.gas_used < best_result.gas_used)
        {
            *best_result = new_result;
        }
    }

    fn is_simulation_too_low(&self, original: &SimValue, inplace: &SimValue) -> bool {
        let orig = original.full_profit_info().coinbase_profit();
        let new = inplace.full_profit_info().coinbase_profit();
        new * U256::from(100) < orig * U256::from(95)
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
        &self,
        sequence_of_orders: Vec<usize>,
        task: &ConflictTask,
        state_provider: Arc<dyn StateProvider>,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> Result<(ResolutionResult, BlockState)> {
        let order_id_to_index = self.initialize_order_id_to_index_map(task);
        let full_sequence_of_orders = self.initialize_full_order_ids_vec(&sequence_of_orders, task);

        let (cached_state_option, cached_up_to_index) = self
            .simulation_cache
            .get_cached_state(&full_sequence_of_orders);

        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(&cached_state_option, state_provider);
        if cached_up_to_index == 0 {
            partial_block.pre_block_call(&self.ctx, local_ctx, &mut state)?;
        }
        else {
            if let Some(cached) = &cached_state_option {
                partial_block.space_state = BlockBuildingSpaceState::new(
                    BlockSpace::new(cached.cumulative_gas_used, 0, cached.cumulative_blob_gas_used),
                    BlockSpace::ZERO
                );
                partial_block.coinbase_profit = cached.coinbase_profit;
            }
        }

        // Initialize sequenced_order_result
        let mut sequenced_order_result =
            self.initialize_result_order_sequence(&cached_state_option, &order_id_to_index);

        let mut total_profit = cached_state_option
            .as_ref()
            .map_or(U256::ZERO, |cached| cached.total_profit);

        let mut per_order_profits_and_gas = cached_state_option
            .as_ref()
            .map_or(Vec::new(), |cached| cached.per_order_profits_and_gas.clone());

        // Prepare the sequence of orders to try, skipping already cached orders
        let mut remaining_orders = sequence_of_orders[cached_up_to_index..].to_vec();
        remaining_orders.reverse(); // Use as a stack: pop from the end

        let mut pending_orders: HashMap<(Address, u64), usize> = HashMap::default();

        let mut prefix_ids: Vec<OrderId> = if let Some(c) = &cached_state_option {
            c.per_order_profits_and_gas.iter().map(|(oid, _, _)| oid.clone()).collect()
        } else {
            Vec::with_capacity(sequence_of_orders.len())
        };

        // Processing loop
        while let Some(order_idx) = remaining_orders.pop() {
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            let sim_order = &task.group.orders[order_idx];
            match partial_block.commit_order(
                sim_order,
                &self.ctx,
                local_ctx,
                &mut state,
                &|_| Ok(()),
            )? {
                Ok(res) => {
                    self.handle_successful_commit(
                        res,
                        sim_order,
                        order_idx,
                        &mut pending_orders,
                        &mut remaining_orders,
                        &mut sequenced_order_result,
                        &mut total_profit,
                        &mut per_order_profits_and_gas,
                    );
                    let order_id = sim_order.order.id();
                    prefix_ids.push(order_id.clone());

                    let _inserted = self.simulation_cache.ensure_cached_with(&prefix_ids, || {
                        let bundle_state = state.clone_bundle();
                        CachedSimulationState {
                            bundle_state,
                            total_profit,
                            per_order_profits_and_gas: per_order_profits_and_gas.clone(),
                            cumulative_gas_used: partial_block.space_state.gas_used(),
                            cumulative_blob_gas_used: partial_block.space_state.blob_gas_used(),
                            coinbase_profit: partial_block.coinbase_profit,
                        }
                    });
                }
                Err(err) => self.handle_err(&err, sim_order, &mut pending_orders, order_idx),
            }
        }

        let resolution_result = ResolutionResult::new(
            total_profit,
            partial_block.space_state.gas_used(),
            sequenced_order_result,
        );
        Ok((resolution_result, state))
    }

    /// Helper function to handle a successful commit of an order.
    #[allow(clippy::too_many_arguments)]
    fn handle_successful_commit(
        &self,
        res: ExecutionResult,
        sim_order: &SimulatedOrder,
        order_idx: usize,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        remaining_orders: &mut Vec<usize>,
        sequenced_order_result: &mut Vec<(usize, U256, u64)>,
        total_profit: &mut U256,
        per_order_profits_and_gas: &mut Vec<(OrderId, U256, u64)>,
    ) {
        for (address, nonce) in res.nonces_updated {
            if let Some(pending_order) = pending_orders.remove(&(address, nonce)) {
                remaining_orders.push(pending_order);
            }
        }
        let order_id = sim_order.order.id();
        *total_profit += res.coinbase_profit;
        per_order_profits_and_gas.push((order_id, res.coinbase_profit, res.space_used.gas));
        sequenced_order_result.push((order_idx, res.coinbase_profit, res.space_used.gas));
    }

    /// Helper function to handle an error in committing an order.
    fn handle_err(
        &self,
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
    ) -> Vec<(usize, U256, u64)> {
        if let Some(cached_state) = &cached_state_option {
            cached_state
                .per_order_profits_and_gas
                .iter()
                .filter_map(|(order_id, profit, gas_used)| {
                    order_id_to_index.get(order_id).map(|&idx| (idx, *profit, *gas_used))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    /// Initializes the block state, using a cached state if available.
    fn initialize_block_state(
        &self,
        cached_state_option: &Option<Arc<CachedSimulationState>>,
        state_provider: Arc<dyn StateProvider>,
    ) -> BlockState {
        if let Some(cached_state) = &cached_state_option {
            BlockState::new_arc(state_provider.clone())
                .with_bundle_state(cached_state.bundle_state.clone())
        } else {
            BlockState::new_arc(state_provider)
        }
    }

    fn evaluate_fitness(
        &self,
        seq: &[usize],
        task: &ConflictTask,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> eyre::Result<ResolutionResult> {
        let (res, _state) = self.process_sequence_of_orders(
            seq.to_vec(), task, self.state.clone(), local_ctx)?;
        Ok(res)
    }

    fn eval_individual(
        &self,
        mut ind: Individual,
        task: &ConflictTask,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> eyre::Result<(Individual, ResolutionResult)> {
        let res = self.evaluate_fitness(&ind.nonce_seq, task, local_ctx)?;
        ind.profit = res.total_profit;
        ind.gas = res.gas_used;
        Ok((ind, res))
    }

    fn process_orders_greedy(
        &self,
        task: &ConflictTask,
        state_provider: Arc<dyn StateProvider>,
        local_ctx: &mut ThreadBlockBuildingContext,
        heap_key: GreedyKey
    ) -> Result<(ResolutionResult, BlockState)> {
        let mut heap: BinaryHeap<(U256, Reverse<OrderId>, usize)> = task.group.orders
            .iter()
            .enumerate()
            .filter(|(_, o)| o.sim_value.gas_used() > 0)
            .map(|(i, o)| {
                let key = match heap_key {
                    GreedyKey::Profit => o.sim_value.full_profit_info().coinbase_profit(),
                    GreedyKey::MevGasPrice => o.sim_value.full_profit_info().mev_gas_price(),
                };
                (key, Reverse(o.order.id()), i)
            })
            .collect();

        let mut partial_block = PartialBlock::new(true);
        let mut state = BlockState::new_arc(state_provider);
        partial_block.pre_block_call(&self.ctx, local_ctx, &mut state)?;

        let mut sequenced_order_result: Vec<(usize, U256, u64)> = Vec::new();
        let mut total_profit = U256::ZERO;
        let mut per_order_profits_and_gas: Vec<(OrderId, U256, u64)> = Vec::new();
        let mut prefix_ids: Vec<OrderId> = Vec::new();
        let mut pending_orders: HashMap<(Address, u64), usize> = HashMap::default();
        let mut retry_counts: HashMap<usize, usize> = HashMap::default();
        let mut overridden_sim_values: HashMap<usize, SimValue> = HashMap::default();
        const MAX_RETRIES: usize = 1;

        while let Some((_, _, order_idx)) = heap.pop() {
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            let sim_order = &task.group.orders[order_idx];
            let original_sim_value = overridden_sim_values
                .get(&order_idx)
                .cloned()
                .unwrap_or_else(|| sim_order.sim_value.clone());

            match partial_block.commit_order(
                sim_order,
                &self.ctx,
                local_ctx,
                &mut state,
                &|new_sim_value| {
                    if !sim_order.order.metadata().is_system
                        && self.is_simulation_too_low(&original_sim_value, new_sim_value)
                    {
                        Err(ExecutionError::LowerInsertedValue {
                            before: original_sim_value.clone(),
                            inplace: new_sim_value.clone(),
                        })
                    } else {
                        Ok(())
                    }
                },
            )? {
                Ok(res) => {
                    for (address, nonce) in &res.nonces_updated {
                        if let Some(pending_idx) = pending_orders.remove(&(*address, *nonce)) {
                            let pending_order = &task.group.orders[pending_idx];
                            let pending_profit = pending_order.sim_value.full_profit_info().coinbase_profit();
                            heap.push((pending_profit, Reverse(pending_order.order.id()), pending_idx));
                        }
                    }
                    let order_id = sim_order.order.id();
                    total_profit += res.coinbase_profit;
                    per_order_profits_and_gas.push((order_id.clone(), res.coinbase_profit, res.space_used.gas));
                    sequenced_order_result.push((order_idx, res.coinbase_profit, res.space_used.gas));
                    prefix_ids.push(order_id);

                    let _ = self.simulation_cache.ensure_cached_with(&prefix_ids, || {
                        CachedSimulationState {
                            bundle_state: state.clone_bundle(),
                            total_profit,
                            per_order_profits_and_gas: per_order_profits_and_gas.clone(),
                            cumulative_gas_used: partial_block.space_state.gas_used(),
                            cumulative_blob_gas_used: partial_block.space_state.blob_gas_used(),
                            coinbase_profit: partial_block.coinbase_profit,
                        }
                    });
                }
                Err(ExecutionError::LowerInsertedValue { inplace, .. }) => {
                    let retries = retry_counts.entry(order_idx).or_insert(0);
                    if order_is_worth_executing(&inplace).is_ok() && *retries < MAX_RETRIES {
                        *retries += 1;
                        let inplace_profit = inplace.full_profit_info().coinbase_profit();
                        overridden_sim_values.insert(order_idx, inplace.clone());
                        heap.push((inplace_profit, Reverse(sim_order.order.id()), order_idx));
                    }
                }
                Err(err) => {
                    self.handle_err(&err, sim_order, &mut pending_orders, order_idx);
                }
            }
        }

        let resolution_result = ResolutionResult::new(
            total_profit,
            partial_block.space_state.gas_used(),
            sequenced_order_result,
        );
        Ok((resolution_result, state))
    }

    fn run_genetic(
        &mut self,
        task: &ConflictTask,
        params: GAParams,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> eyre::Result<(ResolutionResult, Vec<GAGenRecord>)> {
        let Some(deps) = build_group_deps(task) else {
            return Ok((ResolutionResult::default(), vec![]));
        };

        if task.group.orders.is_empty() {
            return Ok((ResolutionResult::default(), vec![]));
        }

        let start = Instant::now();
        let deadline = start + std::time::Duration::from_millis(params.time_ms);

        let num_islands = params.num_islands.max(1);
        let migration_interval = params.migration_interval.max(1);
        let population_per_island = (params.population / num_islands).max(4);
        let total_pop = num_islands * population_per_island;

        let all_seed_seqs = generate_seed_sequences(task, params.seed, total_pop, &params);
        let all_seed_inds: Vec<Individual> = all_seed_seqs.into_iter()
            .map(|seq| individual_from_seq(seq, &deps))
            .collect();

        let all_evaluated: Vec<(Individual, ResolutionResult)> = all_seed_inds
            .into_par_iter()
            .map_init(
                || local_ctx.clone(),
                |thread_ctx, ind| {
                    if self.cancellation_token.is_cancelled() {
                        return Err(eyre::eyre!("Cancelled during initial evaluation"));
                    }
                    self.eval_individual(ind, task, thread_ctx)
                },
            )
            .collect::<Result<Vec<_>, _>>()?;

        // Distribute into islands (round-robin)
        let mut islands: Vec<Island> = (0..num_islands)
            .map(|i| Island {
                population: Vec::with_capacity(population_per_island),
                rng: SmallRng::seed_from_u64(params.seed.wrapping_add(1000 + i as u64)),
            })
            .collect();

        let mut best_individual: Option<Individual> = None;
        let mut best_result: ResolutionResult = ResolutionResult::default();

        for (i, (ind, res)) in all_evaluated.into_iter().enumerate() {
            if res.total_profit > best_result.total_profit {
                best_individual = Some(ind.clone());
                best_result = res;
            }
            islands[i % num_islands].population.push(ind);
        }

        let mut ga_records: Vec<GAGenRecord> = Vec::new();

        // Record initial population stats
        let pop_stats = collect_pop_stats(&islands);
        ga_records.push(GAGenRecord {
            group_id: task.group.id,
            is_initial: true,
            generation: 0,
            best_profit_wei: best_result.total_profit.to_string(),
            improved: false,
            num_children: 0, children_evaluated: 0, children_failed: 0,
            children_zero_profit: 0, children_better: 0, children_equal: 0, children_worse: 0,
            child_profit_min_wei: U256::ZERO.to_string(),
            child_profit_median_wei: U256::ZERO.to_string(),
            child_profit_max_wei: U256::ZERO.to_string(),
            dc_replacements: 0, dc_kept: 0,
            gen_time_us: 0, eval_time_us: 0, dc_time_us: 0, migration_time_us: 0,
            total_elapsed_us: start.elapsed().as_micros() as u64,
            pop_size: pop_stats.0, unique_genomes: pop_stats.1, unique_profits: pop_stats.2,
            zero_profit_in_pop: pop_stats.3,
            pop_profit_min_wei: pop_stats.4.to_string(),
            pop_profit_median_wei: pop_stats.5.to_string(),
            pop_profit_max_wei: pop_stats.6.to_string(),
        });

        let mut generation = 0usize;
        let mut gens_without_improvement = 0usize;

        loop {
            if generation >= params.max_generations || Instant::now() >= deadline {
                break;
            }
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            let profit_before = best_result.total_profit;

            // Phase 1: Generate all children
            let t_phase1 = Instant::now();
            let mut all_children: Vec<PendingChild> = Vec::new();
            let mut all_pair_infos: Vec<(usize, Vec<ParentPairInfo>)> = Vec::new();

            for (island_idx, island) in islands.iter_mut().enumerate() {
                let (children, pair_infos) = generate_dc_children(
                    island, island_idx, &params, &deps,
                );
                all_children.extend(children);
                all_pair_infos.push((island_idx, pair_infos));
            }

            let num_children = all_children.len();
            let phase1_dur = t_phase1.elapsed();

            // Phase 2: Evaluate all children in one par_iter
            let t_phase2 = Instant::now();
            let eval_results: Vec<(usize, usize, usize, Option<(Individual, ResolutionResult)>)> = all_children
                .into_par_iter()
                .map_init(
                    || local_ctx.clone(),
                    |thread_ctx, pending| {
                        match self.eval_individual(pending.ind, task, thread_ctx) {
                            Ok((ind, res)) => (pending.island_idx, pending.pair_idx, pending.child_slot, Some((ind, res))),
                            Err(_) => (pending.island_idx, pending.pair_idx, pending.child_slot, None),
                        }
                    },
                )
                .collect();

            let phase2_dur = t_phase2.elapsed();

            // Track best from children + scatter to per-island arrays
            let mut island_children: Vec<Vec<Option<Individual>>> = all_pair_infos
                .iter()
                .map(|(_, infos)| vec![None; infos.len() * 2])
                .collect();

            // Child evaluation diagnostics
            let mut child_profits: Vec<U256> = Vec::new();
            let mut child_better_count = 0usize;
            let mut child_worse_count = 0usize;
            let mut child_equal_count = 0usize;
            let mut child_zero_profit = 0usize;
            let mut eval_fail_count = 0usize;

            for (island_idx, pair_idx, child_slot, eval_opt) in eval_results {
                let island_pos = all_pair_infos
                    .iter()
                    .position(|(idx, _)| *idx == island_idx)
                    .unwrap();

                if let Some((ind, res)) = eval_opt {
                    // Compare child to its corresponding parent
                    let parent_idx = if child_slot == 0 {
                        all_pair_infos[island_pos].1[pair_idx].p1_idx
                    } else {
                        all_pair_infos[island_pos].1[pair_idx].p2_idx
                    };
                    let parent_profit = islands[island_idx].population[parent_idx].profit;

                    if res.total_profit > parent_profit {
                        child_better_count += 1;
                    } else if res.total_profit < parent_profit {
                        child_worse_count += 1;
                    } else {
                        child_equal_count += 1;
                    }
                    if res.total_profit == U256::ZERO {
                        child_zero_profit += 1;
                    }
                    child_profits.push(res.total_profit);

                    if res.total_profit > best_result.total_profit {
                        best_individual = Some(ind.clone());
                        best_result = res;
                    }
                    island_children[island_pos][pair_idx * 2 + child_slot] = Some(ind);
                } else {
                    eval_fail_count += 1;
                }
            }

            child_profits.sort();
            let cn = child_profits.len();
            let child_min = if cn > 0 { child_profits[0] } else { U256::ZERO };
            let child_max = if cn > 0 { child_profits[cn - 1] } else { U256::ZERO };
            let child_median = if cn > 0 { child_profits[cn / 2] } else { U256::ZERO };

            // Snapshot parent profits before DC to measure competition outcomes
            let pre_dc_profits: Vec<Vec<U256>> = islands
                .iter()
                .map(|isl| isl.population.iter().map(|ind| ind.profit).collect())
                .collect();

            // Phase 3: DC competition per island
            let t_phase3 = Instant::now();
            for (island_pos, (island_idx, pair_infos)) in all_pair_infos.iter().enumerate() {
                apply_dc_competition(
                    &mut islands[*island_idx],
                    pair_infos,
                    &mut island_children[island_pos],
                    &deps,
                    params.w_choice,
                );
            }
            let phase3_dur = t_phase3.elapsed();

            let mut dc_replacements = 0usize;
            let mut dc_kept = 0usize;
            for (isl_idx, pre_profits) in pre_dc_profits.iter().enumerate() {
                let post_profits: Vec<U256> = islands[isl_idx].population.iter().map(|ind| ind.profit).collect();
                for (pre, post) in pre_profits.iter().zip(post_profits.iter()) {
                    if pre != post { dc_replacements += 1; } else { dc_kept += 1; }
                }
            }

            // Migration
            let t_migration = Instant::now();
            if generation > 0 && generation % migration_interval == 0 {
                let migrants: Vec<Individual> = islands
                    .iter()
                    .map(|island| {
                        island.population.iter().max_by_key(|ind| ind.profit).unwrap().clone()
                    })
                    .collect();

                for i in 0..num_islands {
                    let source = (i + num_islands - 1) % num_islands;
                    let migrant = migrants[source].clone();
                    if let Some(worst_idx) = islands[i]
                        .population
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, ind)| ind.profit)
                        .map(|(idx, _)| idx)
                    {
                        if migrant.profit > islands[i].population[worst_idx].profit {
                            islands[i].population[worst_idx] = migrant;
                        }
                    }
                }
            }
            let migration_dur = t_migration.elapsed();

            let improved = best_result.total_profit > profit_before;

            let pop_stats = collect_pop_stats(&islands);
            ga_records.push(GAGenRecord {
                group_id: task.group.id,
                is_initial: false,
                generation,
                best_profit_wei: best_result.total_profit.to_string(),
                improved,
                num_children,
                children_evaluated: cn,
                children_failed: eval_fail_count,
                children_zero_profit: child_zero_profit,
                children_better: child_better_count,
                children_equal: child_equal_count,
                children_worse: child_worse_count,
                child_profit_min_wei: child_min.to_string(),
                child_profit_median_wei: child_median.to_string(),
                child_profit_max_wei: child_max.to_string(),
                dc_replacements,
                dc_kept,
                gen_time_us: phase1_dur.as_micros() as u64,
                eval_time_us: phase2_dur.as_micros() as u64,
                dc_time_us: phase3_dur.as_micros() as u64,
                migration_time_us: migration_dur.as_micros() as u64,
                total_elapsed_us: start.elapsed().as_micros() as u64,
                pop_size: pop_stats.0, unique_genomes: pop_stats.1, unique_profits: pop_stats.2,
                zero_profit_in_pop: pop_stats.3,
                pop_profit_min_wei: pop_stats.4.to_string(),
                pop_profit_median_wei: pop_stats.5.to_string(),
                pop_profit_max_wei: pop_stats.6.to_string(),
            });

            if improved {
                gens_without_improvement = 0;
            } else {
                gens_without_improvement += 1;
            }

            if gens_without_improvement >= params.early_stopping_generations {
                break;
            }

            generation += 1;
        }

        Ok((best_result, ga_records))
    }
}

/// Generate seed sequences as biased raw permutations of all N orders.
/// Uses profit-weighted sampling with increasing temperature for diversity.
fn generate_seed_sequences(
    task: &ConflictTask,
    seed: u64,
    count: usize,
    params: &GAParams,
) -> Vec<Vec<usize>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let order_values = profit_values_f64(&task.group.orders);
    let n = task.group.orders.len();

    temperature_schedule(count, params).into_iter()
        .map(|t| sample_weighted_raw_permutation(&order_values, t, &mut rng, n))
        .collect()
}

/// Sample a biased permutation of all N order indices using profit weights.
/// At t=0: deterministic descending-profit sort. At t>0: weighted shuffle.
fn sample_weighted_raw_permutation(
    values: &[f64],
    temperature: f64,
    rng: &mut SmallRng,
    n: usize,
) -> Vec<usize> {
    if temperature == 0.0 {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
        return indices;
    }

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut weights: Vec<f64> = remaining.iter().map(|&i| (values[i] / temperature).exp()).collect();
    let mut result = Vec::with_capacity(n);

    while !remaining.is_empty() {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            result.extend(remaining.iter());
            break;
        }
        let r = rng.gen::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen = remaining.len() - 1;
        for i in 0..remaining.len() {
            cumsum += weights[i];
            if r <= cumsum {
                chosen = i;
                break;
            }
        }
        result.push(remaining[chosen]);
        remaining.swap_remove(chosen);
        weights.swap_remove(chosen);
    }
    result
}

/// Convert order profits to f64 for weighted sampling.
fn profit_values_f64(orders: &[Arc<SimulatedOrder>]) -> Vec<f64> {
    orders.iter()
        .map(|o| {
            let profit = o.sim_value.full_profit_info().coinbase_profit();
            let limbs = profit.as_limbs();
            let scale = u64::MAX as f64 + 1.0;
            limbs[0] as f64
                + limbs[1] as f64 * scale
                + limbs[2] as f64 * scale * scale
                + limbs[3] as f64 * scale * scale * scale
        })
        .collect()
}

/// Generate a temperature schedule parameterized by GAParams.
fn temperature_schedule(count: usize, params: &GAParams) -> Vec<f64> {
    let mut temps = Vec::with_capacity(count);

    for _ in 0..2.min(count) {
        temps.push(0.0);
    }

    let tight_count = ((count as f64 * params.tight_fraction) as usize).min(count.saturating_sub(temps.len()));
    for i in 0..tight_count {
        temps.push(params.temp_tight_low + (params.temp_tight_high - params.temp_tight_low) * (i as f64) / (tight_count.max(1) as f64));
    }

    while temps.len() < count {
        let i = temps.len();
        temps.push(params.temp_broad_low + (params.temp_broad_high - params.temp_broad_low) * (i as f64) / (count.max(1) as f64));
    }

    temps
}

/// Temperature schedule with default values for non-GA callers.
fn temperature_schedule_defaults(count: usize) -> Vec<f64> {
    let mut temps = Vec::with_capacity(count);

    for _ in 0..2.min(count) {
        temps.push(0.0);
    }

    let tight_count = ((count as f64 * 0.6) as usize).min(count.saturating_sub(temps.len()));
    for i in 0..tight_count {
        temps.push(0.5 + 1.5 * (i as f64) / (tight_count.max(1) as f64));
    }

    while temps.len() < count {
        let i = temps.len();
        temps.push(3.0 + 5.0 * (i as f64) / (count.max(1) as f64));
    }

    temps
}

/// Collect key population stats across all islands.
/// Returns (pop_size, unique_genomes, unique_profits, zero_profit_count,
///          min_profit, median_profit, max_profit).
fn collect_pop_stats(islands: &[Island]) -> (usize, usize, usize, usize, U256, U256, U256) {
    let all_inds: Vec<&Individual> = islands.iter().flat_map(|isl| isl.population.iter()).collect();
    let n = all_inds.len();
    if n == 0 {
        return (0, 0, 0, 0, U256::ZERO, U256::ZERO, U256::ZERO);
    }
    let mut profits: Vec<U256> = all_inds.iter().map(|ind| ind.profit).collect();
    profits.sort();
    let unique_profits = profits.iter().collect::<ahash::HashSet<_>>().len();
    let zero_profit_count = profits.iter().filter(|&&p| p == U256::ZERO).count();
    let unique_genomes = all_inds
        .iter()
        .map(|ind| ind.nonce_seq.clone())
        .collect::<ahash::HashSet<_>>()
        .len();
    (n, unique_genomes, unique_profits, zero_profit_count, profits[0], profits[n / 2], profits[n - 1])
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
pub fn generate_sequences_of_orders_to_try(task: &ConflictTask) -> Vec<Vec<usize>> {
    match task.algorithm {
        Algorithm::Greedy => generate_greedy_sequence(task, false),
        Algorithm::GreedyFast => generate_greedy_sequence_with_nonce(task, false),
        Algorithm::GreedyHeap => vec![],
        Algorithm::ReverseGreedy => generate_greedy_sequence(task, true),
        Algorithm::Length => generate_length_based_sequence(task),
        Algorithm::AllPermutations => generate_all_permutations(task),
        Algorithm::Random { seed, count } => generate_random_permutations(task, seed, count),
        Algorithm::Genetic { .. } => vec![],
        Algorithm::RandomImproved { seed, count } => generate_random_permutations_with_nonce(task, seed, count),
        Algorithm::DexDirectionBalanced { alpha, lambda } => generate_dex_marginal_centered_sequence(task, alpha, lambda),
    }
}

fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut indexes: Vec<usize> = (0..task.group.orders.len()).collect();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        out.push(indexes.clone());
    }
    out
}

pub fn generate_random_permutations_with_nonce(
    task: &ConflictTask,
    seed: u64,
    count: usize,
) -> Vec<Vec<usize>> {
    let Some(deps) = build_group_deps(task) else {
        return generate_random_permutations(task, seed, count);
    };

    let mut rng = SmallRng::seed_from_u64(seed);
    let order_values = profit_values_f64(&task.group.orders);

    temperature_schedule_defaults(count).into_iter()
        .map(|t| ordering_with_biased_choices_and_weighted_topo(
            &deps, &order_values, t, t, &mut rng,
        ))
        .collect()
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
    if let Some(deps) = build_group_deps(task) {
        return enumerate_all_with_choices(&deps, ALL_PERMS_CAP);
    }

    let sequences_of_orders = (0..task.group.orders.len()).collect::<Vec<_>>();
    sequences_of_orders
        .into_iter()
        .permutations(task.group.orders.len())
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
fn generate_greedy_sequence_with_nonce(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
    let g = &task.group;

    let allowed_profit =
        allowed_indices_after_nonce_dedup(g, GreedyKey::Profit, reverse);
    let allowed_mgp =
        allowed_indices_after_nonce_dedup(g, GreedyKey::MevGasPrice, reverse);

    let build = |allowed: &Option<HashSet<usize>>,
                 value_extractor: fn(&SimulatedOrder) -> U256| -> Vec<usize> {
        let mut v: Vec<_> = g.orders.iter().enumerate()
            .filter(|(idx, _)| allowed.as_ref().map_or(true, |s| s.contains(idx)))
            .map(|(idx, o)| (idx, value_extractor(o)))
            .collect();

        v.sort_by(|a, b| if reverse { a.1.cmp(&b.1) } else { b.1.cmp(&a.1) });
        v.into_iter().map(|(idx, _)| idx).collect()
    };

    vec![
        build(&allowed_profit, |o| o.sim_value.full_profit_info().coinbase_profit()),
        build(&allowed_mgp, |o| o.sim_value.full_profit_info().mev_gas_price()),
    ]
}

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

/// Score-based DEX ordering: sort orders by `profit_eth - lambda * impact`, where
/// impact is the sum of pool price displacements weighted by pool popularity:
///   `w(pool) = 1 + alpha * ln(1 + touch_count)`.
///
/// Pools touched by many orders in the conflict group get a higher weight, because
/// a displacement there cascades across more downstream orders.
///
/// Orders without a state trace get impact = 0 and are sorted purely by profit.
fn generate_dex_marginal_centered_sequence(task: &ConflictTask, alpha: f64, lambda: f64) -> Vec<Vec<usize>> {
    const WEI_TO_ETH: f64 = 1e-18;

    let n = task.group.orders.len();
    let orders = &task.group.orders;

    // Step 1: count how many orders touch each pool.
    let mut touch_count: HashMap<Address, usize> = HashMap::default();
    for order in orders.iter() {
        if let Some(trace) = &order.used_state_trace {
            for pool in trace.touched_pools.keys() {
                *touch_count.entry(*pool).or_default() += 1;
            }
        }
    }

    // Step 2: compute a continuous score for each order.
    let mut scored: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let order = &orders[i];
            let profit_eth =
                order.sim_value.full_profit_info().coinbase_profit().to::<u128>() as f64
                    * WEI_TO_ETH;

            let (impact, pool_dbg) = if let Some(trace) = &order.used_state_trace {
                let displacements = trace.price_displacement();
                let mut impact = 0.0f64;
                let mut pool_dbg: Vec<String> = Vec::new();
                for (pool, disp) in &displacements {
                    let tc = *touch_count.get(pool).unwrap_or(&1);
                    let w = 1.0 + alpha * (1.0 + tc as f64).ln();
                    impact += w * disp;
                    pool_dbg.push(format!("{pool:?}:disp={disp:.4},w={w:.2},tc={tc}"));
                }
                (impact, pool_dbg)
            } else {
                (0.0, vec![])
            };

            let score = profit_eth - lambda * impact;
            (i, score)
        })
        .collect();

    // Step 3: sort by score descending.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let score_sequence: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();

    vec![score_sequence]
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, TxHash, B256, U256, address};
    use alloy_primitives::Signature;
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use super::*;
    use crate::building::builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority};
    use rbuilder_primitives::{
        Bundle, Metadata, Order, SimValue, SimulatedOrder, MempoolTx,
        TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
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

            Arc::new(SimulatedOrder::new(
                Arc::new(Order::Bundle(bundle)),
                sim_value,
                None,
            ))
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

    fn mk_tx_with(sender: Address, nonce: u64, hash: TxHash) -> Recovered<TransactionSigned> {
        let tx_legacy = TxLegacy { nonce, ..Default::default() };
        Recovered::new_unchecked(
            TransactionSigned::new(
                Transaction::Legacy(tx_legacy),
                Signature::test_signature(),
                hash,
            ),
            sender,
        )
    }

    fn mk_order_single_tx(sender: Address, nonce: u64, profit: u64, dg: &mut DataGenerator) -> Arc<SimulatedOrder> {
        let rec = mk_tx_with(sender, nonce, dg.create_hash());
        let with_blobs = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(rec).unwrap();
        let sim_value = SimValue::new_test(U256::from(profit), U256::from(profit), 0);

        Arc::new(SimulatedOrder {
            order: Arc::new(Order::Tx(MempoolTx { tx_with_blobs: with_blobs })),
            used_state_trace: None,
            sim_value,
        })
    }

    /// Create a conflict group with:
    /// - Sender A: nonce 0 (2 candidates), nonce 1 (1 candidate)
    /// - Sender B: nonce 0 (1 candidate), nonce 1 (2 candidates)
    /// Total slots = 4
    fn make_group_with_duplicate_buckets() -> ConflictGroup {
        let mut dg = DataGenerator::new();

        let a = address!("000000000000000000000000000000000000000a");
        let b = address!("000000000000000000000000000000000000000b");

        // A, nonce 0: two candidates
        let a0_1 = mk_order_single_tx(a, 0, 100, &mut dg);
        let a0_2 = mk_order_single_tx(a, 0, 90, &mut dg);
        // A, nonce 1: one candidate
        let a1   = mk_order_single_tx(a, 1, 80, &mut dg);

        // B, nonce 0: one candidate
        let b0   = mk_order_single_tx(b, 0, 70, &mut dg);
        // B, nonce 1: two candidates
        let b1_1 = mk_order_single_tx(b, 1, 60, &mut dg);
        let b1_2 = mk_order_single_tx(b, 1, 50, &mut dg);

        // Any order; the view code maps by (sender,nonce).
        let orders = vec![a0_1, a0_2, a1, b0, b1_1, b1_2];

        create_mock_order_group(42, orders, HashSet::default())
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
        assert_eq!(sequences.len(), 2);

        // Coinbase profit is the first
        assert_eq!(sequences[0], vec![3, 1, 2, 0]);
        // MEV gas price is the second
        assert_eq!(sequences[1], vec![0, 2, 1, 3]);
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
        assert_eq!(sequences.len(), 2);

        // Coinbase profit is the first
        assert_eq!(sequences[0], vec![0, 2, 1, 3]);
        // MEV gas price is the second
        assert_eq!(sequences[1], vec![3, 1, 2, 0]);
    }

    #[test]
    fn random_is_reproducible() {
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

        let a = generate_random_permutations(&task, 123, 50);
        let b = generate_random_permutations(&task, 123, 50);
        let c = generate_random_permutations(&task, 456, 50);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }


}