use ahash::HashMap;
use alloy_primitives::{Address, U256};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use rand::seq::SliceRandom;
use reth::providers::StateProvider;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use rayon::prelude::*;

use serde::Serialize;
use std::cmp::Reverse;
use std::fs::OpenOptions;
use std::io::Write;


use super::{
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
    nonce_handling::*, genetic_algo::*,
};

use crate::{
    building::{
        BlockBuildingContext, BlockState, ExecutionError, ExecutionResult, PartialBlock,
        ThreadBlockBuildingContext,
    },
    primitives::{OrderId, SimulatedOrder},
};
const ALL_PERMS_INCLUDE_DUPLICATE_NONCE_CHOICES: bool = true;

#[derive(Serialize)]
struct ExhaustiveTopSeq {
    seq: Vec<usize>,
    profit: String,
}

#[derive(Serialize)]
struct ExhaustiveJsonLine {
    block_number: u64,
    group_id: usize,
    terminated_by_time: bool,
    examined: u64,
    baseline_profit: String,
    top: Vec<ExhaustiveTopSeq>,
}

fn append_exhaustive_json_line(
    out_dir: &str,
    block_number: u64,
    payload: &ExhaustiveJsonLine,
) -> eyre::Result<()> {
    let dir = std::path::Path::new(out_dir);
    if !dir.exists() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file_name = format!("exhaustive_block_{:0>8}.ndjson", block_number);
    let path = dir.join(file_name);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(payload)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

#[derive(Serialize)]
struct BestAlgorithmJsonLine {
    group_id: usize,
    best_profit: String,
    algorithm: String,
}

fn append_best_algorithm_json_line(
    out_dir: &str,
    block_number: u64,
    payload: &BestAlgorithmJsonLine,
) -> eyre::Result<()> {
    let dir = std::path::Path::new(out_dir);
    if !dir.exists() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file_name = format!("block_{:0>8}.ndjson", block_number);
    let path = dir.join(file_name);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(payload)?;
    writeln!(file, "{}", line)?;
    Ok(())
}


fn build_nonce_layout(task: &ConflictTask) -> Option<NonceLayout> {
    NonceLayout::from_group(&task.group)
}

fn seed_initial_population(
    task: &ConflictTask,
    layout: &NonceLayout,
    population_size: usize,
    rng: &mut SmallRng,
) -> Vec<Vec<usize>> {
    use ahash::HashSet as AHashSet;

    let mut seeds: Vec<Vec<usize>> = Vec::new();
    let mut seen: AHashSet<Vec<usize>> = AHashSet::default();

    // Greedy and ReverseGreedy
    for &rev in &[false, true] {
        for seq in generate_greedy_sequence(task, rev) {
            let repaired = repair_to_nonce_valid(&seq, layout);
            if repaired.len() == layout.total_steps && seen.insert(repaired.clone()) {
                seeds.push(repaired);
            }
        }
    }

    // let to_add = population_size.saturating_sub(seeds.len());
    // let cap = to_add.min(10);
    // if cap > 0 {
    //     for seq in generate_chain_grouped_sequences(layout, rng, cap) {
    //         if seq.len() == layout.total_steps && seen.insert(seq.clone()) {
    //             seeds.push(seq);
    //             if seeds.len() >= population_size { return seeds; }
    //         }
    //     }
    // }

    // TODO: add some random interleavings but picking candidates with highest mev_gas_price

    // Fill rest with random candidates per step + uniform interleaving
    while seeds.len() < population_size {
        let s = random_interleaving_with_random_choices(layout, rng);
        if s.len() == layout.total_steps && seen.insert(s.clone()) {
            seeds.push(s);
        }
    }

    seeds
}


/// Streaming enumerator of all nonce-valid interleavings with per-slot choices.
/// Memory-light: no per-depth ready vectors are stored.
struct InterleaveStreamer<'a> {
    layout: &'a NonceLayout,
    cursors: Vec<usize>,     // next slot per chain
    seq: Vec<usize>,         // current path (indices)
    stack: Vec<Frame>,       // DFS frames
}

struct Frame {
    next_i: usize,           // which "nth ready choice" to try at this depth
    applied_chain: Option<usize>, // which chain we advanced to reach this depth (for undo)
}

impl<'a> InterleaveStreamer<'a> {
    fn new(layout: &'a NonceLayout) -> Self {
        let mut it = Self {
            layout,
            cursors: vec![0; layout.chains.len()],
            seq: Vec::with_capacity(layout.total_steps),
            stack: Vec::new(),
        };
        // root frame
        it.stack.push(Frame { next_i: 0, applied_chain: None });
        it
    }

    #[inline]
    fn ready_count(&self) -> usize {
        let mut tot = 0usize;
        for c in 0..self.cursors.len() {
            let s = self.cursors[c];
            if s < self.layout.chains[c].steps.len() {
                tot += self.layout.chains[c].steps[s].candidates.len();
            }
        }
        tot
    }

    /// Map nth ready option (0-based) to (chain, cand_pos_in_bucket).
    #[inline]
    fn nth_ready(&self, mut n: usize) -> (usize, usize) {
        for c in 0..self.cursors.len() {
            let s = self.cursors[c];
            if s >= self.layout.chains[c].steps.len() {
                continue;
            }
            let len = self.layout.chains[c].steps[s].candidates.len();
            if n < len {
                return (c, n);
            }
            n -= len;
        }
        unreachable!("nth_ready called with n >= ready_count()");
    }

    fn next(&mut self) -> Option<Vec<usize>> {
        let total_slots = self.layout.total_steps;

        loop {
            // Done?
            if self.stack.is_empty() {
                return None;
            }

            // Take the frame by value so we don't hold a borrow on self.stack
            let mut frame = self.stack.pop().unwrap();

            // These only need &self, and no frame is borrowed now.
            let total_ready = self.ready_count();

            // No more options at this depth → backtrack
            if frame.next_i >= total_ready {
                if let Some(chain) = frame.applied_chain {
                    self.seq.pop();
                    self.cursors[chain] -= 1;
                }
                // do not push this exhausted frame back
                continue;
            }

            // Figure out which ready option to try next (still only &self)
            let (chain, cand_pos) = self.nth_ready(frame.next_i);
            frame.next_i += 1;

            // We will come back to this depth; put the updated frame back first.
            self.stack.push(frame);

            // Apply choice
            let slot = self.cursors[chain];
            let cand = self.layout.chains[chain].steps[slot].candidates[cand_pos];
            self.seq.push(cand);
            self.cursors[chain] += 1;

            if self.seq.len() == total_slots {
                // Leaf: emit and undo immediately
                let out = self.seq.clone();
                self.seq.pop();
                self.cursors[chain] -= 1;
                return Some(out);
            } else {
                // Go deeper
                self.stack.push(Frame { next_i: 0, applied_chain: Some(chain) });
                // continue the loop
            }
        }
    }

}


/// Context for resolving conflicts in merging tasks.

#[derive(Derivative)]
#[derivative(Debug, Clone)]
pub struct ResolverContext {
    #[derivative(Debug = "ignore")]
    pub state: Arc<dyn StateProvider + Send + Sync>,
    pub ctx: BlockBuildingContext,
    pub cancellation_token: CancellationToken,
    pub simulation_cache: Arc<SharedSimulationCache>,
}

/// Runs one generation of the GA using a Deterministic Crowding replacement strategy.
fn run_dc_generation(
    island: &mut Island,
    best_seen: &Option<ResolutionResult>,
    task: &ConflictTask,
    params: &GAParams,
    layout: &NonceLayout,
    evaluator: &ResolverContext,
    local_ctx: &mut ThreadBlockBuildingContext,
) -> eyre::Result<DCGenerationResult> {
    let mut new_best_hits = 0;
    let mut children_who_won = 0;
    let mut evals_this_gen = 0;
    let population = &mut island.population;
    let rng = &mut island.rng;

    let mut next_population = Vec::with_capacity(population.len());
    let offsets = chain_offsets(layout);

    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.shuffle(rng);

    for i in (0..population.len()).step_by(2) {
        if i + 1 >= indices.len() {
            if let Some(idx) = indices.get(i) { next_population.push(population[*idx].clone()); }
            continue;
        }

        let p1_idx = indices[i];
        let p2_idx = indices[i+1];
        let (p1, p2) = (&population[p1_idx], &population[p2_idx]);
        
        let mut c1_seq = adapted_order_crossover(&p1.seq, &p2.seq, layout, rng);
        let mut c2_seq = adapted_order_crossover(&p2.seq, &p1.seq, layout, rng);
        
        let multi_steps = layout.multi_steps();
        if rng.gen::<f64>() < params.mutation_rate {
            mutate(&mut c1_seq, layout, rng, &multi_steps);
        }
        if rng.gen::<f64>() < params.mutation_rate {
            mutate(&mut c2_seq, layout, rng, &multi_steps);
        }

        let (c1, res1) = evaluator.eval_to_individual(c1_seq, task, local_ctx)?;
        let (c2, res2) = evaluator.eval_to_individual(c2_seq, task, local_ctx)?;
        evals_this_gen += 2;

        if best_seen.as_ref().map_or(true, |b| res1.total_profit > b.total_profit) { new_best_hits += 1; }
        if best_seen.as_ref().map_or(true, |b| res2.total_profit > b.total_profit) { new_best_hits += 1; }

        let winner1;
        let winner2;

        let dist_p1c1 = dc_distance(&p1.seq, &c1.seq, layout, &offsets, 0.5, 0.5);
        let dist_p2c2 = dc_distance(&p2.seq, &c2.seq, layout, &offsets, 0.5, 0.5);
        let dist_p1c2 = dc_distance(&p1.seq, &c2.seq, layout, &offsets, 0.5, 0.5);
        let dist_p2c1 = dc_distance(&p2.seq, &c1.seq, layout, &offsets, 0.5, 0.5);

        if dist_p1c1 + dist_p2c2 <= dist_p1c2 + dist_p2c1 {
            winner1 = compete(p1, &c1, rng);
            winner2 = compete(p2, &c2, rng);
        } else {
            winner1 = compete(p1, &c2, rng);
            winner2 = compete(p2, &c1, rng);
        }

        if winner1.seq == c1.seq || winner1.seq == c2.seq { children_who_won += 1; }
        if winner2.seq == c1.seq || winner2.seq == c2.seq { children_who_won += 1; }
        
        next_population.push(winner1);
        next_population.push(winner2);
    }

    *population = next_population;
    
    Ok(DCGenerationResult { new_best_hits, children_who_won, evals: evals_this_gen })
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
    pub fn run_conflict_task(&mut self, task: ConflictTask, local_ctx: &mut ThreadBlockBuildingContext) -> Result<ResolutionResult> {
        trace!(
            "run_conflict_task: {:?} with algorithm {:?}",
            task.group.id,
            task.algorithm
        );

        match task.algorithm {
            Algorithm::Genetic {
                population,
                crossover_rate,
                mutation_rate,
                tourn_k,
                max_generations,
                time_ms,
                seed,
            } => {
                let params = GAParams {
                    population,
                    crossover_rate,
                    mutation_rate,
                    tourn_k,
                    max_generations,
                    time_ms,
                    seed,
                };
                let res = self.run_genetic(&task, params, local_ctx)?;
                trace!(
                    "Resolved GA task {:?} with profit: {:?}",
                    task.group.id,
                    res.total_profit
                );

                if !matches!(task.algorithm, Algorithm::AllPermutations { .. }) {
                    // Save to JSON
                    let out_dir = "performance_testing/best_algorithms_with_genetic";
                    let line = BestAlgorithmJsonLine {
                        group_id: task.group.id,
                        best_profit: res.total_profit.to_string(),
                        algorithm: task.algorithm.display().to_string(),
                    };
                    let _ = append_best_algorithm_json_line(&out_dir, self.ctx.evm_env.block_env.number, &line);
                }

                Ok(res)
            }
            Algorithm::ExhaustiveStreaming { time_ms, top_k } => {
                let res = self.run_exhaustive_streaming(&task, time_ms, top_k, local_ctx)?;
                trace!(
                    "Resolved ExhaustiveStreaming task {:?} with profit: {:?}",
                    task.group.id,
                    res.total_profit
                );
                Ok(res)
            }
            _ => {
                let sequence_to_try = generate_sequences_of_orders_to_try(&task);

                let mut best_resolution_result = ResolutionResult::new(U256::ZERO, 0,  vec![]);

                for sequence_of_orders in sequence_to_try {
                    let (resolution_result, _state) =
                        self.process_sequence_of_orders(sequence_of_orders, &task, self.state.clone(), local_ctx)?;
                    self.update_best_result(resolution_result, &mut best_resolution_result);
                }


                if !matches!(task.algorithm, Algorithm::AllPermutations { .. }) {
                    // Save to JSON
                    let out_dir = "performance_testing/best_algorithms_with_genetic";
                    let line = BestAlgorithmJsonLine {
                        group_id: task.group.id,
                        best_profit: best_resolution_result.total_profit.to_string(),
                        algorithm: task.algorithm.display().to_string(),
                    };
                    let _ = append_best_algorithm_json_line(&out_dir, self.ctx.evm_env.block_env.number, &line);
                }

                trace!(
                    "Resolved conflict task {:?} with profit: {:?} and algorithm: {:?}",
                    task.group.id,
                    best_resolution_result.total_profit,
                    task.algorithm
                );
                Ok(best_resolution_result)
            }
        }
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

        // Check for cached simulation state
        let use_cache = !matches!(task.algorithm, Algorithm::ExhaustiveStreaming { .. });
        let (cached_state_option, cached_up_to_index) = if use_cache {
            self.simulation_cache.get_cached_state(&full_sequence_of_orders)
        } else {
            (None, 0)
        };


        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(&cached_state_option, state_provider);
        if cached_up_to_index == 0 {
            partial_block.pre_block_call(&self.ctx, local_ctx, &mut state)?;
        }
        else {
            if let Some(cached) = &cached_state_option {
                partial_block.gas_used = cached.cumulative_gas_used;
                partial_block.blob_gas_used = cached.cumulative_blob_gas_used;
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
                            cumulative_gas_used: partial_block.gas_used,
                            cumulative_blob_gas_used: partial_block.blob_gas_used,
                            coinbase_profit: partial_block.coinbase_profit,
                        }
                    });
                }
                Err(err) => self.handle_err(&err, sim_order, &mut pending_orders, order_idx),
            }
        }

        // self.store_simulation_state(
        //     &full_sequence_of_orders,
        //     &state,
        //     total_profit,
        //     &per_order_profits,
        // );

        let resolution_result = ResolutionResult::new(
            total_profit,
            partial_block.gas_used,
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
        per_order_profits_and_gas.push((order_id, res.coinbase_profit, res.gas_used));
        sequenced_order_result.push((order_idx, res.coinbase_profit, res.gas_used));
    }

    /// Helper function to handle an error in committing an order.
    fn handle_err(
        &self,
        err: &ExecutionError,
        sim_order: &SimulatedOrder,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        order_idx: usize,
    ) {
        // println!("Order {:?} failed with error: {:?}", sim_order.order.id(), err);
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

    // /// Initializes the block state, using a cached state if available.
    // fn initialize_block_state(&mut self, state_provider: Arc<dyn StateProvider>) -> BlockState {
    //     BlockState::new_arc(state_provider)
    // }

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

    // /// Stores the simulation state in the cache.
    // fn store_simulation_state(
    //     &self,
    //     full_order_ids: &[OrderId],
    //     state: &BlockState,
    //     total_profit: U256,
    //     per_order_profits: &[(OrderId, U256)],
    // ) {
    //     let (bundle_state, _) = state.clone().into_parts();
    //     let cached_simulation_state = CachedSimulationState {
    //         bundle_state,
    //         total_profit,
    //         per_order_profits: per_order_profits.to_owned(),
    //     };
    //     self.simulation_cache
    //         .store_cached_state(full_order_ids, cached_simulation_state);
    // }

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

    /// Evaluate a sequence and build an Individual (+ return its full result if you need it).
    fn eval_to_individual(
        &self,
        seq: Vec<usize>,
        task: &ConflictTask,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> eyre::Result<(Individual, ResolutionResult)> {
        let res = self.evaluate_fitness(&seq, task, local_ctx)?;
        let ind = Individual {
            seq,
            profit: res.total_profit,
            gas: res.gas_used,
            rank: 0,
            crowding: 0.0,
        };
        Ok((ind, res))
    }

    /// Compute baseline profit using current implementation tasks:
    /// Greedy, ReverseGreedy, Length, and a random task.
    fn compute_baseline_profit(&mut self, task: &ConflictTask, local_ctx: &mut ThreadBlockBuildingContext) -> eyre::Result<U256> {
        let mut best = U256::ZERO;

        // Greedy and ReverseGreedy
        for &rev in &[false, true] {
            for seq in generate_greedy_sequence(task, rev) {
                let res = self.evaluate_fitness(&seq, task, local_ctx)?;
                if res.total_profit > best { best = res.total_profit; }
            }
        }

        // Length
        for seq in generate_length_based_sequence(task) {
            let res = self.evaluate_fitness(&seq, task, local_ctx)?;
            if res.total_profit > best { best = res.total_profit; }
        }

        // Random
        let rnd = generate_random_permutations(
            &ConflictTask { algorithm: Algorithm::Random { seed: task.group.id as u64, count: 50 }, ..task.clone() },
            task.group.id as u64,
            50,
        );
        for seq in rnd {
            let res = self.evaluate_fitness(&seq, task, local_ctx)?;
            if res.total_profit > best { best = res.total_profit; }
        }

        Ok(best)
    }


    fn run_exhaustive_streaming(
        &mut self,
        task: &ConflictTask,
        time_ms: u64,
        top_k: usize,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> eyre::Result<ResolutionResult> {
        let Some(layout) = build_nonce_layout(task) else {
            // Fallback when we can’t derive (sender, nonce)
            return Ok(ResolutionResult::new(U256::ZERO, 0, vec![]));
        };

        // Baseline first (so we can compare later)
        let baseline = self.compute_baseline_profit(task, local_ctx)?;

        // Stream all sequences; simulate each immediately.
        let start = Instant::now();
        let deadline = start + std::time::Duration::from_millis(time_ms);
        let mut streamer = InterleaveStreamer::new(&layout);

        // Keep Top-K via a min-heap on profit
        let mut heap: std::collections::BinaryHeap<(Reverse<U256>, Vec<usize>)> =
            std::collections::BinaryHeap::new();
        let mut examined: u64 = 0;
        let mut terminated_by_time = false;

        // Track best ResolutionResult (so we can return a full record with per-order profits)
        let mut best_res: Option<ResolutionResult> = None;

        while let Some(seq) = streamer.next() {
            if Instant::now() >= deadline {
                terminated_by_time = true;
                break;
            }
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            // Evaluate and update Top-K
            let res = self.evaluate_fitness(&seq, task, local_ctx)?;
            examined += 1;

            if heap.len() < top_k {
                heap.push((Reverse(res.total_profit), seq.clone()));
            } else if let Some(&(Reverse(curr_min), _)) = heap.peek() {
                if res.total_profit > curr_min {
                    heap.pop();
                    heap.push((Reverse(res.total_profit), seq.clone()));
                }
            }

            // Track the global best (so we don't need to re-run later)
            match &best_res {
                Some(b) if res.total_profit > b.total_profit => best_res = Some(res),
                None => best_res = Some(res),
                _ => {}
            }
        }

        // Serialize Top-K to JSON (sorted descending)
        let mut top: Vec<(U256, Vec<usize>)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|(Reverse(p), s)| (p, s))
            .collect();
        top.reverse(); // now descending by profit

        let json_top: Vec<ExhaustiveTopSeq> = top
            .iter()
            .map(|(p, s)| ExhaustiveTopSeq {
                seq: s.clone(),
                profit: p.to_string(),
            })
            .collect();

        let payload = ExhaustiveJsonLine {
            block_number: self.ctx.evm_env.block_env.number,
            group_id: task.group.id,
            terminated_by_time,
            examined,
            baseline_profit: baseline.to_string(),
            top: json_top,
        };

        let _ = append_exhaustive_json_line(
            "performance_testing/exhaustive_streaming",
            self.ctx.evm_env.block_env.number,
            &payload,
        );

        // Return the best result we saw (if none, return zero)
        Ok(best_res.unwrap_or(ResolutionResult::new(U256::ZERO, 0, vec![])))
    }

    fn run_genetic(&mut self, task: &ConflictTask, params: GAParams, local_ctx: &mut ThreadBlockBuildingContext) -> eyre::Result<ResolutionResult> {
        let Some(layout) = build_nonce_layout(task) else {
            return Ok(ResolutionResult::default());
        };
        let start = Instant::now();
        let deadline = start + std::time::Duration::from_millis(params.time_ms);
        
        // Island model configuration
        let num_islands = 10;
        let migration_interval = 3; // Migrate every 2 generations
        let population_per_island = (params.population / num_islands).max(2);

        let mut islands: Vec<Island> = Vec::with_capacity(num_islands);
        let mut best_seen: Option<ResolutionResult> = None;
        let mut evals_cum: u64 = 0;

        let initial_islands: Vec<(Vec<Vec<usize>>, SmallRng)> = (0..num_islands)
            .into_iter()
            .map(|i| {
                let mut island_rng = SmallRng::seed_from_u64(params.seed.wrapping_add(i as u64));
                let seed_seqs = seed_initial_population(task, &layout, population_per_island, &mut island_rng);
                
                (seed_seqs, island_rng)
            })
            .collect();
        
        for (seqs, rng) in initial_islands {
            let evaluated_results = seqs
                .into_par_iter()
                .map(|s| {
                    if self.cancellation_token.is_cancelled() {
                        return Err(eyre::eyre!("Cancelled during initial evaluation"));
                    }
                    let mut thread_ctx = ThreadBlockBuildingContext::default();
                    self.eval_to_individual(s, task, &mut thread_ctx)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut population = Vec::with_capacity(population_per_island);
            // Process results sequentially to safely update shared state
            for (ind, res) in evaluated_results {
                evals_cum += 1;
                if best_seen.as_ref().map_or(true, |b| res.total_profit > b.total_profit) {
                    best_seen = Some(res.clone());
                }
                population.push(ind);
            }
            islands.push(Island { population, rng });
        }

        // main generational loop
        let mut generation = 0usize;
        let mut generations_without_improvement = 0usize;
        const EARLY_STOPPING_LIMIT: usize = 10;
        loop{
            if generation >= params.max_generations {
                tracing::info!("Terminating at generation {} due to max generations.", generation);
                break;
            }
            if Instant::now() >= deadline {
                tracing::info!("Terminating at generation {} due to time limit.", generation);
                break;
            }
            if self.cancellation_token.is_cancelled() { return Err(eyre::eyre!("Cancelled")); }

            let profit_before_gen = best_seen.as_ref().map(|b| b.total_profit).unwrap_or_default();

            // Run one generation on each island in parallel
            let gen_results: Vec<DCGenerationResult> = islands
                .par_iter_mut()
                .map(|island| {
                    let mut thread_ctx = ThreadBlockBuildingContext::default();
                    run_dc_generation(
                        island, &best_seen, task, &params, &layout, self, &mut thread_ctx
                    )
                })
                .collect::<Result<_, _>>()?;

            // Aggregate results for logging
            let mut total_evals_this_gen = 0;
            let mut total_new_best_hits = 0;
            let mut total_children_who_won = 0;
            for result in gen_results {
                total_evals_this_gen += result.evals;
                total_new_best_hits += result.new_best_hits;
                total_children_who_won += result.children_who_won;
            }
            evals_cum += total_evals_this_gen;

            if generation > 0 && generation % migration_interval == 0 {
                let mut migrants: Vec<Individual> = Vec::with_capacity(num_islands);
                // Select the best individual from each island to be a migrant
                for island in &islands {
                    let best_migrant = island.population.iter().max_by_key(|ind| ind.profit).unwrap().clone();
                    migrants.push(best_migrant);
                }

                let mut island_indices: Vec<usize> = (0..num_islands).collect();
                island_indices.shuffle(&mut islands[0].rng);

                // Apply the ring topology to the shuffled list of indices
                for i in 0..num_islands {
                    let target_island_idx = island_indices[i];
                    let source_island_idx = island_indices[(i + num_islands - 1) % num_islands];

                    let migrant = migrants[source_island_idx].clone();

                    // Replace the worst individual in the target island
                    if let Some(worst_idx) = islands[target_island_idx]
                        .population
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, ind)| ind.profit)
                        .map(|(idx, _)| idx)
                    {
                        islands[target_island_idx].population[worst_idx] = migrant;
                    }
                }
            }

            for island in &islands {
                if let Some(island_best) = island.population.iter().max_by_key(|ind| ind.profit) {
                    if best_seen.as_ref().map_or(true, |b| island_best.profit > b.total_profit) {
                        let (_, res) = self.eval_to_individual(island_best.seq.clone(), task, local_ctx)?;
                        best_seen = Some(res);
                    }
                }
            }

            let profit_after_gen = best_seen.as_ref().map(|b| b.total_profit).unwrap_or_default();
            if profit_after_gen > profit_before_gen {
                generations_without_improvement = 0;
            } else {
                generations_without_improvement += 1;
            }

            if generations_without_improvement >= EARLY_STOPPING_LIMIT {
                tracing::info!(
                    "Terminating at generation {} due to early stopping ({} gens without improvement).",
                    generation,
                    EARLY_STOPPING_LIMIT
                );
                break;
            }
            
            // logging
            let mut combined_population: Vec<Individual> = islands.iter().flat_map(|i| i.population.clone()).collect();
            let fronts = fast_non_dominated_sort(&mut combined_population);  // taken from nsga2 algorithm
            for f in &fronts { assign_crowding_distance(&mut combined_population, f); }            
            let f1: Vec<usize> = if fronts.is_empty() { vec![] } else { fronts[0].clone() };
            let front_sizes: Vec<usize> = fronts.iter().map(|f| f.len()).collect();
            let pareto_points: Vec<(U256, u64)> = f1.iter().map(|&i| (combined_population[i].profit, combined_population[i].gas)).collect();
            let hv = hv2d(&pareto_points);
            let s  = spacing_s(&pareto_points);
            let (avg_dc, min_dc, max_dc, std_dc) = pairwise_dc_stats(&combined_population, &layout);
            let (uniq_seqs, uniq_slot_orders)   = uniq_counts(&combined_population, &layout);
            let niches = niche_components(&combined_population, &layout, 0.15);

            let mut crowd_f1: Vec<f64> = f1.iter().map(|&i| combined_population[i].crowding).collect();
            crowd_f1.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (cmin, cmed, cmax) = if crowd_f1.is_empty() { (0.0,0.0,0.0) } else {
                let med = crowd_f1[crowd_f1.len()/2];
                (*crowd_f1.first().unwrap(), med, *crowd_f1.last().unwrap())
            };

            let best_profit = best_seen.as_ref().map(|r| r.total_profit.to_string()).unwrap_or_else(|| "0".to_string());
            let best_gas    = best_seen.as_ref().map(|r| r.gas_used).unwrap_or(0);
            
            // Use the new metrics from gen_result for logging
            let child_survival_rate = (total_children_who_won as f64) / (params.population as f64);

            let payload = NSGA2GenLine {
                block_number: self.ctx.evm_env.block_env.number,
                group_id: task.group.id,
                gen: generation,
                elapsed_ms: start.elapsed().as_millis() as u64,
                evals_cum,

                best_profit,
                best_gas,

                pareto_size: pareto_points.len(),
                hv2d: hv,
                spacing_s: s,
                crowding_front1_min: cmin,
                crowding_front1_med: cmed,
                crowding_front1_max: cmax,

                avg_dc, min_dc, max_dc, std_dc,
                uniq_seqs, uniq_slot_orders,
                niche_sizes: niches,

                fronts: front_sizes,
                new_best_hits: total_new_best_hits,
                child_accept_rate: child_survival_rate,
                child_non_dominated_vs_parents: child_survival_rate, // In DC, this is essentially the same metric
            };
            let _ = append_nsga2_debug_line(
                "performance_testing/nsga2_debug",
                self.ctx.evm_env.block_env.number,
                &payload,
            );
            
            generation += 1;
        }

        Ok(best_seen.unwrap_or_default())
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
pub fn generate_sequences_of_orders_to_try(task: &ConflictTask) -> Vec<Vec<usize>> {
    match task.algorithm {
        Algorithm::Greedy => generate_greedy_sequence(task, false),
        Algorithm::ReverseGreedy => generate_greedy_sequence(task, true),
        Algorithm::Length => generate_length_based_sequence(task),
        Algorithm::AllPermutations => generate_all_permutations(task),
        Algorithm::Random { seed, count } => generate_random_permutations(task, seed, count),
        Algorithm::Genetic { .. } => {
            // Genetic algorithm is handled separately in ResolverContext::run_genetic
            vec![]
        }
        Algorithm::ExhaustiveStreaming { .. } => {
            // Exhaustive streaming is handled separately in ResolverContext::run_exhaustive_streaming
            vec![]
        }
        Algorithm::RandomChain { seed, count } => generate_chain_based_sequences(task, seed, count),
        Algorithm::RandomImproved { seed, count } => generate_random_permutations_with_nonce(task, seed, count),
    }
}

/// Generates random permutations of sequences of order indices (with replacement),
/// respecting nonce dependencies. For large spaces (always the case for Random),
/// we sample `count` independent uniform interleavings.
/// Fallback (when no nonce view is available): old shuffle behavior.
/// # Arguments
///
/// * `task` - The current conflict task.
/// * `seed` - Seed for the random number generator.
/// * `count` - Number of random permutations to generate.
///
/// # Returns
///
/// A vector of randomly generated sequences of order indices.
// fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
//     let mut sequences_of_orders = vec![];

//     let order_group = &task.group;
//     let mut indexes = (0..order_group.orders.len()).collect::<Vec<_>>();
//     let mut rng = SmallRng::seed_from_u64(seed);
//     for _ in 0..count {
//         indexes.shuffle(&mut rng);
//         sequences_of_orders.push(indexes.clone());
//     }

//     sequences_of_orders
// }

fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    // if let Some(layout) = build_nonce_layout(task) {
    //     let mut rng = SmallRng::seed_from_u64(seed);
    //     let mut out = Vec::with_capacity(count);
    //     for _ in 0..count {
    //         out.push(random_interleaving_with_random_choices(&layout, &mut rng));
    //     }

    //     return out;
    // }

    // Fallback: bundles/multi-tx orders where we can't derive nonce chains
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut indexes: Vec<usize> = (0..task.group.orders.len()).collect();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        out.push(indexes.clone());
    }
    out
}

fn generate_random_permutations_with_nonce(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    if let Some(layout) = build_nonce_layout(task) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(random_interleaving_with_random_choices(&layout, &mut rng));
        }

        return out;
    }

    // Fallback: bundles/multi-tx orders where we can't derive nonce chains
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut indexes: Vec<usize> = (0..task.group.orders.len()).collect();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        out.push(indexes.clone());
    }
    out
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
    if let Some(layout) = build_nonce_layout(task) {
        if ALL_PERMS_INCLUDE_DUPLICATE_NONCE_CHOICES {
            return enumerate_all_interleavings_with_choices(&layout, ALL_PERMS_CAP);
        } else {
            let per_chain = build_chains_best(&layout, &task.group);
            return enumerate_all_interleavings_best(&per_chain, ALL_PERMS_CAP);
        }
    }

    let sequences_of_orders = (0..task.group.orders.len()).collect::<Vec<_>>();
    sequences_of_orders
        .into_iter()
        .permutations(task.group.orders.len())
        .collect()
}

// / Generates static sequences of order indices based on gas price and coinbase profit.
// /
// / # Arguments
// /
// / * `task` - The current conflict task.
// / * `reverse` - Whether to reverse the sorting order (e.g. sorting by min coinbase profit and mev_gas_price)
// /
// / # Returns
// /
// / A vector of static sequences of order indices, sorted by coinbase profit and mev_gas_price.
// fn generate_greedy_sequence(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
//     let order_group = &task.group;

//     let create_sequence = |value_extractor: fn(&SimulatedOrder) -> U256| {
//         let mut ids_and_value: Vec<_> = order_group
//             .orders
//             .iter()
//             .enumerate()
//             .map(|(idx, order)| (idx, value_extractor(order)))
//             .collect();

//         ids_and_value.sort_by(|a, b| {
//             if reverse {
//                 a.1.cmp(&b.1)
//             } else {
//                 b.1.cmp(&a.1)
//             }
//         });
//         ids_and_value.into_iter().map(|(idx, _)| idx).collect()
//     };

//     vec![
//         create_sequence(|sim_order| sim_order.sim_value.coinbase_profit),
//         create_sequence(|sim_order| sim_order.sim_value.mev_gas_price),
//     ]
// }


fn generate_greedy_sequence(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
    let group = &task.group;

    // Build a single greedy preference list for the given key, filtered by
    // key-aware per-slot dedup.
    let build_for = |key: GreedyKey| {
        // Only keep best-per-slot candidates according to this key+reverse.
        let allowed = allowed_indices_after_nonce_dedup(group, key, reverse);

        // Collect indices with both metrics so we can do a stable secondary tie-break.
        let mut rows: Vec<(usize, U256, U256)> = group
            .orders
            .iter()
            .enumerate()
            .filter(|(idx, _)| allowed.as_ref().map_or(true, |set| set.contains(idx)))
            .map(|(idx, o)| (idx, o.sim_value.coinbase_profit, o.sim_value.mev_gas_price))
            .collect();

        rows.sort_by(|a, b| {
            // a: (idx, profit, gas), b: (idx, profit, gas)
            let (pa, sa) = match key {
                GreedyKey::Profit     => (a.1, a.2),
                GreedyKey::MevGasPrice=> (a.2, a.1),
            };
            let (pb, sb) = match key {
                GreedyKey::Profit     => (b.1, b.2),
                GreedyKey::MevGasPrice=> (b.2, b.1),
            };

            let ord1 = if reverse { pa.cmp(&pb) } else { pb.cmp(&pa) };
            if ord1 != std::cmp::Ordering::Equal { return ord1; }

            let ord2 = if reverse { sa.cmp(&sb) } else { sb.cmp(&sa) };
            if ord2 != std::cmp::Ordering::Equal { return ord2; }

            a.0.cmp(&b.0)
        });

        rows.into_iter().map(|(idx, _, _)| idx).collect::<Vec<_>>()
    };

    vec![
        build_for(GreedyKey::Profit),
        build_for(GreedyKey::MevGasPrice),
    ]
}

// / Generates length based sequences of order indices based on the length of the orders.
// / e.g. prioritizes longer bundles first
// /
// / # Arguments
// /
// / * `task` - The current conflict task.
// /
// / # Returns
// /
// / A vector of length based sequences of order indices.
// fn generate_length_based_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
//     let mut sequences_of_orders = vec![];
//     let order_group = &task.group;

//     let mut order_data: Vec<(usize, usize, U256)> = order_group
//         .orders
//         .iter()
//         .enumerate()
//         .map(|(idx, order)| {
//             (
//                 idx,
//                 order.order.list_txs().len(),
//                 order.sim_value.coinbase_profit,
//             )
//         })
//         .collect();

//     // Sort by length (descending) and then by profit (descending) as a tie-breaker
//     order_data.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));

//     // Extract the sorted indices
//     let length_based_sequence: Vec<usize> = order_data.into_iter().map(|(idx, _, _)| idx).collect();

//     sequences_of_orders.push(length_based_sequence);
//     sequences_of_orders
// }


fn generate_length_based_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let allowed = allowed_indices_after_nonce_dedup(order_group, GreedyKey::Profit, false);

    let mut order_data: Vec<(usize, usize, U256)> = order_group
        .orders
        .iter()
        .enumerate()
        .filter(|(idx, _)| allowed.as_ref().map_or(true, |set| set.contains(idx)))
        .map(|(idx, order)| (idx, order.order.list_txs().len(), order.sim_value.coinbase_profit))
        .collect();

    order_data.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    let seq: Vec<usize> = order_data.into_iter().map(|(idx, _, _)| idx).collect();
    vec![seq]
}

fn generate_chain_based_sequences(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    if let Some(layout) = build_nonce_layout(task) {
        let mut rng = SmallRng::seed_from_u64(seed);
        return generate_chain_grouped_sequences(&layout, &mut rng, count);
    }

    // Fallback: bundles/multi-tx orders where we can't derive nonce chains
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut indexes: Vec<usize> = (0..task.group.orders.len()).collect();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        out.push(indexes.clone());
    }
    out
}

/// Generate sequences where each nonce chain appears as one contiguous block.
/// We permute the chains randomly; within each chain, we append all steps in order,
/// picking a random candidate when a step has multiple candidates.
fn generate_chain_grouped_sequences(
    layout: &NonceLayout,
    rng: &mut SmallRng,
    count: usize,
) -> Vec<Vec<usize>> {
    let k = layout.chains.len();
    if k == 0 || count == 0 { return Vec::new(); }

    let mut out = Vec::with_capacity(count);
    let mut chains: Vec<usize> = (0..k).collect();

    for _ in 0..count {
        chains.shuffle(rng);

        let mut seq = Vec::with_capacity(layout.total_steps);
        for &c in &chains {
            let steps = &layout.chains[c].steps;
            for s in 0..steps.len() {
                let bucket = &steps[s].candidates;
                let pick = if bucket.len() == 1 {
                    bucket[0]
                } else {
                    bucket[rng.gen_range(0..bucket.len())]
                };
                seq.push(pick);
            }
        }
        out.push(seq);
    }

    out
}

pub fn generate_chain_greedy_sequences(task: &ConflictTask) -> Vec<Vec<usize>> {
    let Some(layout) = build_nonce_layout(task) else {
        // Fallback when we can’t derive (sender, nonce): reuse plain greedy.
        return generate_greedy_sequence(task, false);
    };

    fn metrics(task: &ConflictTask, idx: usize) -> (U256, U256) {
        let o = &task.group.orders[idx];
        (o.sim_value.coinbase_profit, o.sim_value.mev_gas_price)
    }

    let build_for = |key: GreedyKey| -> Vec<usize> {
        // For each chain, pick per-step best candidate and sum as chain score.
        let mut chains: Vec<(usize /*chain_id*/, U256 /*score*/, Vec<usize> /*picked*/)> =
            Vec::with_capacity(layout.chains.len());

        for (c_id, chain) in layout.chains.iter().enumerate() {
            let mut picked: Vec<usize> = Vec::with_capacity(chain.steps.len());
            let mut score = U256::ZERO;

            for step in &chain.steps {
                // pick best candidate for this nonce step
                let &best_idx = step
                    .candidates
                    .iter()
                    .max_by(|&&a, &&b| {
                        let (pa, ga) = metrics(task, a);
                        let (pb, gb) = metrics(task, b);

                        // primary / secondary according to `key`
                        let (ka1, ka2) = match key {
                            GreedyKey::Profit => (pa, ga),
                            GreedyKey::MevGasPrice => (ga, pa),
                        };
                        let (kb1, kb2) = match key {
                            GreedyKey::Profit => (pb, gb),
                            GreedyKey::MevGasPrice => (gb, pb),
                        };

                        // Desc on primary, then desc on secondary, then idx asc
                        ka1.cmp(&kb1)
                            .then_with(|| ka2.cmp(&kb2))
                            .then_with(|| a.cmp(&b))
                    })
                    .expect("step must have at least one candidate");

                picked.push(best_idx);

                // add to chain score
                let (p, g) = metrics(task, best_idx);
                score += match key {
                    GreedyKey::Profit => p,
                    GreedyKey::MevGasPrice => g,
                };
            }

            chains.push((c_id, score, picked));
        }

        // Sort chains by score desc (tie-break by chain id for determinism)
        chains.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Concatenate picked steps chain-by-chain (keeps chains contiguous)
        let mut seq = Vec::with_capacity(layout.total_steps);
        for (_, _, picked) in chains {
            seq.extend(picked);
        }
        seq
    };

    vec![
        build_for(GreedyKey::Profit),
        build_for(GreedyKey::MevGasPrice),
    ]
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
    use crate::{
        building::builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority},
        primitives::{
            Bundle, Metadata, Order, SimValue, SimulatedOrder, MempoolTx,
            TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
        },
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
                TransactionSigned::new(
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

            let sim_value = SimValue {
                coinbase_profit,
                mev_gas_price,
                ..Default::default()
            };

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
                version: LAST_BUNDLE_VERSION,
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
        let sim_value = SimValue { coinbase_profit: U256::from(profit), ..Default::default() };

        Arc::new(SimulatedOrder {
            order: Order::Tx(MempoolTx { tx_with_blobs: with_blobs }),
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

    /// Assert sequence is a valid interleaving:
    /// - length == total_slots
    /// - uses exactly one candidate per (chain, slot)
    /// - per-chain slot order strictly increasing by slot index
    fn assert_nonce_valid(seq: &[usize], layout: &NonceLayout) {
        assert_eq!(seq.len(), layout.total_steps, "length mismatch");

        let _k = layout.chains.len();
        let mut expected_slot: Vec<usize> = layout.chains.iter().map(|_ch| 0usize).collect();
        let mut seen: HashSet<usize> = HashSet::default();

        for &idx in seq {
            assert!(seen.insert(idx), "duplicate index in sequence");
            let (c, s) = layout.index_of[&idx];
            // Must be the next expected slot for this chain
            assert_eq!(s, expected_slot[c], "slot order broken for chain {}", c);
            // idx must belong to that bucket
            assert!(layout.chains[c].steps[s].candidates.contains(&idx), "idx not in its bucket");
            expected_slot[c] += 1;
        }

        for (c, exp) in expected_slot.into_iter().enumerate() {
            assert_eq!(exp, layout.chains[c].steps.len(), "did not cover all slots for chain {}", c);
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

    #[test]
    fn test_seed_initial_population_validity_and_uniqueness() {
        let group = make_group_with_duplicate_buckets();
        let task = create_mock_task(0, group.clone(), Algorithm::Greedy, TaskPriority::Low, Instant::now());
        let layout = NonceLayout::from_group(&group).unwrap();

        let mut rng = SmallRng::seed_from_u64(999);
        let population_size = 2;
        let seeds = seed_initial_population(&task, &layout, population_size, &mut rng);

        assert_eq!(seeds.len(), population_size);
        let mut uniq: HashSet<Vec<usize>> = HashSet::default();
        for s in &seeds {
            assert_nonce_valid(s, &layout);
            assert!(uniq.insert(s.clone()), "duplicate initial individual");
        }
    }

}