use ahash::{HashMap, AHashSet};
use alloy_primitives::{Address, U256};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use rand::{seq::SliceRandom, SeedableRng};
use reth::providers::StateProvider;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;
use rand::Rng;
use std::time::Instant;

use super::{
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
    nonce_interleavings::*,
};

use crate::{
    building::{
        BlockBuildingContext, BlockState, ExecutionError, ExecutionResult, PartialBlock,
        ThreadBlockBuildingContext,
    },
    primitives::{OrderId, SimulatedOrder},
};
const ALL_PERMS_INCLUDE_DUPLICATE_NONCE_CHOICES: bool = true;

#[derive(Clone, Copy, Debug)]
struct GAParams {
    population: usize,
    elitism: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    tourn_k: usize,
    max_generations: usize,
    time_ms: u64,
    seed: u64,
}

/// Compact wrapper over SenderNonceView with reverse index lookup.
#[derive(Debug, Clone)]
struct NonceBuckets {
    /// chains_slots[c][s] = candidate order indices at chain c, slot s (same-nonce bucket)
    chains_slots: Vec<Vec<Vec<usize>>>,
    /// index -> (chain, slot)
    idx_to_chain_slot: HashMap<usize, (usize, usize)>,
    /// total number of slots across chains (length of a valid interleaving)
    total_slots: usize,
}

fn build_nonce_buckets(task: &ConflictTask) -> Option<NonceBuckets> {
    let view = build_sender_nonce_view(&task.group)?;
    let mut idx_to_chain_slot: HashMap<usize, (usize, usize)> = HashMap::default();

    let mut total_slots = 0usize;
    for (ci, chain) in view.chains_slots.iter().enumerate() {
        for (si, slot) in chain.iter().enumerate() {
            total_slots += 1;
            for &idx in slot {
                idx_to_chain_slot.insert(idx, (ci, si));
            }
        }
    }

    Some(NonceBuckets {
        chains_slots: view.chains_slots,
        idx_to_chain_slot,
        total_slots,
    })
}

#[inline]
fn repair_to_nonce_valid(preferred: &[usize], buckets: &NonceBuckets) -> Vec<usize> {
    // Project a single preference list onto a valid interleaving.
    // Internally this uses the same ready-set selection as crossover.
    ppx_build_child_from_parents(preferred, preferred, buckets)
}

#[derive(Clone)]
struct Individual {
    seq: Vec<usize>,
    fitness: Option<U256>,
}

fn seed_initial_population(
    task: &ConflictTask,
    buckets: &NonceBuckets,
    population_size: usize,
    rng: &mut rand::rngs::SmallRng,
) -> Vec<Vec<usize>> {
    let mut seeds: Vec<Vec<usize>> = Vec::new();
    let mut seen: AHashSet<Vec<usize>> = AHashSet::default();

    // Generate Greedy and ReverseGreedy sequences
    for &rev in &[false, true] {
        for seq in generate_greedy_sequence(task, rev) {
            // Project the preference ordering into a valid interleaving.
            let repaired = repair_to_nonce_valid(&seq, buckets);
            if repaired.len() == buckets.total_slots && seen.insert(repaired.clone()) {
                seeds.push(repaired);
            }
        }
    }

    // Fill rest with random candidates per slot + uniform interleaving
    let view = build_sender_nonce_view(&task.group).expect("nonce view");
    while seeds.len() < population_size {
        // pick 1 candidate per slot at random
        let mut per_chain: Vec<Vec<usize>> = Vec::with_capacity(view.chains_slots.len());
        for chain in &view.chains_slots {
            let mut best_chain = Vec::with_capacity(chain.len());
            for slot in chain {
                let cand = slot[rng.gen_range(0..slot.len())];
                best_chain.push(cand);
            }
            per_chain.push(best_chain);
        }
        let s = sample_one_uniform_interleaving(&per_chain, rng); // interleave across senders
        if s.len() == buckets.total_slots && seen.insert(s.clone()) {
            seeds.push(s);
        }
    }

    seeds
}

fn tournament_select<'p>(
    population: &'p [Individual],
    k: usize,
    rng: &mut rand::rngs::SmallRng,
) -> usize {
    debug_assert!(population.len() >= k);
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fit = population[best_idx].fitness.expect("fitness must be set");
    for _ in 1..k {
        let i = rng.gen_range(0..population.len());
        let fit = population[i].fitness.expect("fitness must be set");
        if fit > best_fit {
            best_idx = i;
            best_fit = fit;
        }
    }
    best_idx
}

/// Build child interleaving using a ready-set, prioritising by parent ranks.
/// - Each chain contributes exactly one candidate per slot (consumes slot on pick).
/// - Ready set = candidates in current slot for each chain.
/// - Priority = min(rank_in_parent_a, rank_in_parent_b), tie-break by sum then by idx.
/// Build a nonce-valid child by precedence-preserving selection from the
/// ready set (current slot of each chain), guided by ranks in both parents.
fn ppx_build_child_from_parents(
    parent_a: &[usize],
    parent_b: &[usize],
    buckets: &NonceBuckets,
) -> Vec<usize> {
    let total_slots = buckets.total_slots;
    let n_chains = buckets.chains_slots.len();
    let mut next_slot_per_chain = vec![0usize; n_chains];
    let mut child = Vec::with_capacity(total_slots);

    // Rank maps: position in each parent gives priority (lower is better).
    // If a candidate doesn't appear in a parent, assign a large (bad) rank.
    const LARGE_RANK: usize = usize::MAX / 4;
    let mut rank_in_a: HashMap<usize, usize> = HashMap::default();
    let mut rank_in_b: HashMap<usize, usize> = HashMap::default();
    for (i, &idx) in parent_a.iter().enumerate() { rank_in_a.insert(idx, i); }
    for (i, &idx) in parent_b.iter().enumerate() { rank_in_b.insert(idx, i); }

    while child.len() < total_slots {
        // Consider only ready candidates: current slot for each chain.
        let mut best_choice: Option<(usize, (usize, usize, usize))> = None;
        // tuple = (candidate_idx, (best_of_parent_ranks, sum_of_ranks, tie_idx))

        for chain_id in 0..n_chains {
            let slot = next_slot_per_chain[chain_id];
            if slot >= buckets.chains_slots[chain_id].len() {
                continue;
            }
            // All candidates that share this (chain, slot) (same-nonce duplicates).
            for &cand in &buckets.chains_slots[chain_id][slot] {
                let ra = *rank_in_a.get(&cand).unwrap_or(&LARGE_RANK);
                let rb = *rank_in_b.get(&cand).unwrap_or(&LARGE_RANK);
                let key = (ra.min(rb), ra.saturating_add(rb), cand);

                if let Some((_, best_key)) = &best_choice {
                    if key < *best_key {
                        best_choice = Some((cand, key));
                    }
                } else {
                    best_choice = Some((cand, key));
                }
            }
        }

        // Pick and advance its chain.
        let (chosen_idx, _) = best_choice.expect("non-empty ready set until all slots are filled");
        let (chain_id, slot_id) = buckets.idx_to_chain_slot[&chosen_idx];
        debug_assert_eq!(next_slot_per_chain[chain_id], slot_id);

        child.push(chosen_idx);
        next_slot_per_chain[chain_id] += 1;
    }

    child
}

fn mutation_inter_sender_swap(
    seq: &mut Vec<usize>,
    buckets: &NonceBuckets,
    rng: &mut rand::rngs::SmallRng,
) {
    if seq.is_empty() { return; }
    // Sample up to a few tries to find two positions from different chains
    for _ in 0..5 {
        let i = rng.gen_range(0..seq.len());
        let j = rng.gen_range(0..seq.len());
        if i == j { continue; }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let (ca, _sa) = buckets.idx_to_chain_slot[&seq[a]];
        let (cb, _sb) = buckets.idx_to_chain_slot[&seq[b]];
        if ca != cb {
            seq.swap(a, b);
            return;
        }
    }
}

fn mutation_same_nonce_flip(
    seq: &mut Vec<usize>,
    buckets: &NonceBuckets,
    rng: &mut rand::rngs::SmallRng,
) {
    if seq.is_empty() { return; }
    // pick a random position, if bucket has >1 candidates, flip to a different one
    let p = rng.gen_range(0..seq.len());
    let (c, s) = buckets.idx_to_chain_slot[&seq[p]];
    let bucket = &buckets.chains_slots[c][s];
    if bucket.len() <= 1 { return; }

    // pick alternative different from current
    // tiny guard to avoid duplicates in seq (shouldn't exist for same slot)
    let cur = seq[p];
    let alts: Vec<usize> = bucket.iter().copied().filter(|&x| x != cur).collect();
    if alts.is_empty() { return; }
    let alt = alts[rng.gen_range(0..alts.len())];

    // replace
    seq[p] = alt;
}



/// Build one uniformly random nonce-respecting ordering.
/// Weighted choice: at each step pick chain i with probability
/// remaining_i / total_remaining, then take its next tx.
fn sample_one_uniform_interleaving(chains: &[Vec<usize>], rng: &mut rand::rngs::SmallRng) -> Vec<usize> {
    let k = chains.len();
    let total: usize = chains.iter().map(|c| c.len()).sum();
    let mut cursors = vec![0usize; k];
    let mut remains: Vec<usize> = chains.iter().map(|c| c.len()).collect();
    let mut seq: Vec<usize> = Vec::with_capacity(total);

    for _ in 0..total {
        let total_rem: usize = remains.iter().sum();
        debug_assert!(total_rem > 0);

        let mut r = rng.gen_range(0..total_rem);
        let mut chosen = 0usize;
        for i in 0..k {
            let w = remains[i];
            if w == 0 { continue; }
            if r < w {
                chosen = i;
                break;
            }
            r -= w;
        }

        let idx = chains[chosen][cursors[chosen]];
        cursors[chosen] += 1;
        remains[chosen] -= 1;
        seq.push(idx);
    }

    seq
}

/// Context for resolving conflicts in merging tasks.

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ResolverContext<'a> {
    #[derivative(Debug = "ignore")]
    pub state: Arc<dyn StateProvider>,
    pub ctx: BlockBuildingContext,
    pub local_ctx: &'a mut ThreadBlockBuildingContext,
    pub cancellation_token: CancellationToken,
    pub simulation_cache: Arc<SharedSimulationCache>,
}

impl<'a> ResolverContext<'a> {
    /// Creates a new `ResolverContext`.
    ///
    /// # Arguments
    ///
    /// * `provider_factory` - Factory for creating state providers.
    /// * `ctx` - Context for block building.
    /// * `local_ctx` - Context for the current thread.
    /// * `cancellation_token` - Token for cancelling operations.
    /// * `cache` - Optional cached reads for optimization.
    /// * `simulation_cache` - Shared cache for simulation results.
    pub fn new(
        state: Arc<dyn StateProvider>,
        ctx: BlockBuildingContext,
        local_ctx: &'a mut ThreadBlockBuildingContext,
        cancellation_token: CancellationToken,
        simulation_cache: Arc<SharedSimulationCache>,
    ) -> Self {
        ResolverContext {
            state,
            ctx,
            local_ctx,
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

        match task.algorithm {
            Algorithm::Genetic {
                population,
                elitism,
                crossover_rate,
                mutation_rate,
                tourn_k,
                max_generations,
                time_ms,
                seed,
            } => {
                let params = GAParams {
                    population,
                    elitism,
                    crossover_rate,
                    mutation_rate,
                    tourn_k,
                    max_generations,
                    time_ms,
                    seed,
                };
                let res = self.run_genetic(&task, params)?;
                trace!(
                    "Resolved GA task {:?} with profit: {:?}",
                    task.group.id,
                    res.total_profit
                );
                Ok(res)
            }
            _ => {
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
        let order_id_to_index = self.initialize_order_id_to_index_map(task);
        let full_sequence_of_orders = self.initialize_full_order_ids_vec(&sequence_of_orders, task);

        // Check for cached simulation state
        let (cached_state_option, cached_up_to_index) = self
            .simulation_cache
            .get_cached_state(&full_sequence_of_orders);

        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(&cached_state_option, state_provider);
        if cached_up_to_index == 0 {
            partial_block.pre_block_call(&self.ctx, &mut self.local_ctx, &mut state)?;
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

        let mut per_order_profits = cached_state_option
            .as_ref()
            .map_or(Vec::new(), |cached| cached.per_order_profits.clone());

        // Prepare the sequence of orders to try, skipping already cached orders
        let mut remaining_orders = sequence_of_orders[cached_up_to_index..].to_vec();
        remaining_orders.reverse(); // Use as a stack: pop from the end

        let mut pending_orders: HashMap<(Address, u64), usize> = HashMap::default();

        let mut prefix_ids: Vec<OrderId> = if let Some(c) = &cached_state_option {
            c.per_order_profits.iter().map(|(oid, _)| oid.clone()).collect()
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
                &mut self.local_ctx,
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
                        &mut per_order_profits,
                    );
                    let order_id = sim_order.order.id();
                    prefix_ids.push(order_id.clone());

                    let _inserted = self.simulation_cache.ensure_cached_with(&prefix_ids, || {
                    let bundle_state = state.clone_bundle();
                    CachedSimulationState {
                        bundle_state,
                        total_profit,
                        per_order_profits: per_order_profits.clone(),
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

    // /// Initializes the block state, using a cached state if available.
    // fn initialize_block_state(&mut self, state_provider: Arc<dyn StateProvider>) -> BlockState {
    //     BlockState::new_arc(state_provider)
    // }

    /// Initializes the block state, using a cached state if available.
    fn initialize_block_state(
        &mut self,
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
        &mut self,
        seq: &Vec<usize>,
        task: &ConflictTask,
    ) -> eyre::Result<ResolutionResult> {
        let (res, _state) = self.process_sequence_of_orders(seq.clone(), task, self.state.clone())?;
        Ok(res)
    }

    fn run_genetic(&mut self, task: &ConflictTask, params: GAParams) -> eyre::Result<ResolutionResult> {
        if task.group.id == 4 {
            println!("Running Genetic Algorithm with params: {:?}", params);
        }
        // Precompute nonce buckets
        let Some(buckets) = build_nonce_buckets(task) else {
            // Fallback: if we somehow get here without nonce view, evaluate Greedy once
            let greedy = generate_greedy_sequence(task, false).into_iter().next().unwrap_or_default();
            let (res, _) = self.process_sequence_of_orders(greedy, task, self.state.clone())?;
            return Ok(res);
        };

        let start = Instant::now();
        let deadline = start + std::time::Duration::from_millis(params.time_ms);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(params.seed);

        // 1) Seed population
        let seed_seqs = seed_initial_population(task, &buckets, params.population, &mut rng);
        let mut population: Vec<Individual> = seed_seqs.into_iter().map(|seq| Individual { seq, fitness: None }).collect();

        println!("Task {} Initial population sequences:", task.group.id);
        for (i, individual) in population.iter().enumerate() {
            println!("  Population[{}]: {:?}", i, individual.seq);
        }

        // 2) Evaluate initial population
        let mut best: Option<(usize, ResolutionResult)> = None;
        for i in 0..population.len() {
            if self.cancellation_token.is_cancelled() { return Err(eyre::eyre!("Cancelled")); }
            let fit_res = self.evaluate_fitness(&population[i].seq, task)?;
            population[i].fitness = Some(fit_res.total_profit);
            
            println!("  Population[{}] fitness: {}", i, fit_res.total_profit);
            
            if let Some((_bi, bres)) = &best {
                if fit_res.total_profit > bres.total_profit { best = Some((i, fit_res)); }
            } else {
                best = Some((i, fit_res));
            }
            if Instant::now() >= deadline { return Ok(best.unwrap().1.clone()); }
        }

        let mut best_profit = best.as_ref().unwrap().1.total_profit;
        let mut gens_since_improve = 0usize;
        let plateau_patience = 10usize;

        // 3) GA loop
        let mut generation = 0usize;
        while generation < params.max_generations && Instant::now() < deadline {
            if task.group.id == 4 {
                println!("Task {} Generation {}: Population size: {}", task.group.id, generation, population.len());
            }
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            // Sort by fitness desc for elitism
            population.sort_by(|a, b| {
                let fa = a.fitness.unwrap_or(U256::ZERO);
                let fb = b.fitness.unwrap_or(U256::ZERO);
                fb.cmp(&fa)
            });

            println!("  Best sequences after sorting:");
            for i in 0..3.min(population.len()) {
                println!("    Rank[{}]: seq={:?}, fitness={}", 
                            i, population[i].seq, population[i].fitness.unwrap_or(U256::ZERO));
            }

            println!("Worst sequence: {:?}, fitness=[]={}", 
                population[population.len() - 1].seq, 
                population[population.len() - 1].fitness.unwrap_or(U256::ZERO));

            // Next population with elites
            let mut next_population: Vec<Individual> = Vec::with_capacity(params.population);
            let elites = params.elitism.min(population.len());
            for i in 0..elites {
                next_population.push(population[i].clone()); // carry over
            }

            // Fill the rest
            while next_population.len() < params.population && Instant::now() < deadline {
                // Parents
                let p1_idx = tournament_select(&population, params.tourn_k, &mut rng);
                let p2_idx = tournament_select(&population, params.tourn_k, &mut rng);
                let p1 = &population[p1_idx];
                let p2 = &population[p2_idx];

                // Crossover?
                let mut child_seq = if rng.gen::<f64>() < params.crossover_rate {
                    let child = ppx_build_child_from_parents(&p1.seq, &p2.seq, &buckets);

                    println!("    Crossover: P1[{}]={:?} + P2[{}]={:?} -> Child={:?}", 
                                p1_idx, p1.seq, p2_idx, p2.seq, child);

                    child
                } else {
                    // clone a parent (the fitter one)
                    let parent_seq = if p1.fitness.unwrap() >= p2.fitness.unwrap() {
                        p1.seq.clone()
                    } else {
                        p2.seq.clone()
                    };
                    println!("    Clone parent: {:?}", parent_seq);

                    parent_seq
                };

                // Mutations (independent)
                let pre_mutation = child_seq.clone();
                if rng.gen::<f64>() < params.mutation_rate {
                    mutation_inter_sender_swap(&mut child_seq, &buckets, &mut rng);
                }
                if rng.gen::<f64>() < params.mutation_rate {
                    mutation_same_nonce_flip(&mut child_seq, &buckets, &mut rng);
                    child_seq = repair_to_nonce_valid(&child_seq, &buckets);
                }
                println!("    Mutation: {:?} -> {:?}", pre_mutation, child_seq);

                next_population.push(Individual { seq: child_seq, fitness: None });
            }

            // Evaluate children (skip elites that already have fitness)
            for i in 0..next_population.len() {
                if Instant::now() >= deadline { break; }
                if i < elites && next_population[i].fitness.is_some() {
                    continue;
                }
                let fit_res = self.evaluate_fitness(&next_population[i].seq, task)?;
                next_population[i].fitness = Some(fit_res.total_profit);

                // Update global best
                if fit_res.total_profit > best_profit {
                    best_profit = fit_res.total_profit;
                    println!("    NEW BEST in generation {}: seq={:?}, profit={}", 
                                generation, next_population[i].seq, fit_res.total_profit);
                    best = Some((i, fit_res));
                }
            }

            // Early stop on plateau (only after half the budget used)
            if best_profit > population[0].fitness.unwrap_or(U256::ZERO) {
                gens_since_improve = 0;
            } else {
                gens_since_improve += 1;
            }
            let spent = Instant::now().saturating_duration_since(start).as_millis() as u64;
            if gens_since_improve >= plateau_patience && spent >= params.time_ms / 2 {
                println!("Task {} stopping early at generation {} due to plateau", task.group.id, generation);
                println!("Plateau patience: {}, spent {}ms / {}ms", plateau_patience, spent, params.time_ms);
                break;
            }

            population = next_population;
            generation += 1;
        }

        // Return best resolution result (recompute if needed)
        if let Some((_idx, best_res)) = best {
            Ok(best_res)
        } else {
            // Never happens with non-empty population
            Ok(ResolutionResult { total_profit: U256::ZERO, sequence_of_orders: vec![] })
        }
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

// fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
//     let order_group = &task.group;

//     if let Some(view) = build_sender_nonce_view(order_group) {
//         let chains = build_sender_chains_best(&view, order_group);
//         let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
//         let mut out = Vec::with_capacity(count);
//         for _ in 0..count {
//             out.push(sample_one_uniform_interleaving(&chains, &mut rng));
//         }
//         return out;
//     }

//     // Fallback: bundles/multi-tx orders where we can't derive nonce chains
//     let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
//     let mut indexes: Vec<usize> = (0..order_group.orders.len()).collect();
//     let mut out = Vec::with_capacity(count);
//     for _ in 0..count {
//         indexes.shuffle(&mut rng);
//         out.push(indexes.clone());
//     }
//     out
// }


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

    if let Some(view) = build_sender_nonce_view(order_group) {
        if ALL_PERMS_INCLUDE_DUPLICATE_NONCE_CHOICES {
            // Interleave across nonce slots AND branch per-slot over all candidates
            return enumerate_all_interleavings_with_choices(&view, ALL_PERMS_CAP);
        } else {
            // Previous behavior: pick best per slot, then interleave the chains
            let chains = build_sender_chains_best(&view, order_group);
            return enumerate_all_interleavings_best(&chains, ALL_PERMS_CAP);
        }
    }

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
    fn assert_nonce_valid(seq: &[usize], buckets: &NonceBuckets) {
        assert_eq!(seq.len(), buckets.total_slots, "length mismatch");

        let _k = buckets.chains_slots.len();
        let mut expected_slot: Vec<usize> = buckets.chains_slots.iter().map(|_ch| 0usize).collect();
        let mut seen: HashSet<usize> = HashSet::default();

        for &idx in seq {
            assert!(seen.insert(idx), "duplicate index in sequence");
            let (c, s) = buckets.idx_to_chain_slot[&idx];
            // Must be the next expected slot for this chain
            assert_eq!(s, expected_slot[c], "slot order broken for chain {}", c);
            // idx must belong to that bucket
            assert!(buckets.chains_slots[c][s].contains(&idx), "idx not in its bucket");
            expected_slot[c] += 1;
        }

        for (c, exp) in expected_slot.into_iter().enumerate() {
            assert_eq!(exp, buckets.chains_slots[c].len(), "did not cover all slots for chain {}", c);
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
    fn test_nonce_buckets_and_ppx_child_valid() {
        // Build group with duplicate same-nonce choices
        let group = make_group_with_duplicate_buckets();

        // Build a task stub
        let task = create_mock_task(
            0,
            group.clone(),
            Algorithm::Greedy,
            TaskPriority::Low,
            Instant::now(),
        );

        // Buckets
        let buckets = build_nonce_buckets(&task).expect("nonce view present");
        assert_eq!(buckets.total_slots, 4);

        // Create two valid parents manually:
        // First, pick "best per slot" via helper to get one candidate per nonce.
        let view = build_sender_nonce_view(&group).unwrap();
        let chains_best = build_sender_chains_best(&view, &group);

        // Parent A interleaving: A0, B0, A1, B1
        let parent_a = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        // Parent B interleaving: B0, A0, B1, A1
        let parent_b = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]];

        // Child from PPX-style builder
        let child = ppx_build_child_from_parents(&parent_a, &parent_b, &buckets);
        assert_nonce_valid(&child, &buckets);
    }

    #[test]
    fn test_same_nonce_flip_mutation_preserves_validity_and_changes_candidate() {
        let group = make_group_with_duplicate_buckets();
        let task = create_mock_task(0, group.clone(), Algorithm::Greedy, TaskPriority::Low, Instant::now());
        let buckets = build_nonce_buckets(&task).expect("nonce view present");

        // Start from a valid interleaving (best-per-slot, simple A0,B0,A1,B1)
        let view = build_sender_nonce_view(&group).unwrap();
        let chains_best = build_sender_chains_best(&view, &group);
        let seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &buckets);

        // There are at least two slots with multiple candidates (A:nonce0, B:nonce1).
        let mut rng = rand::rngs::SmallRng::seed_from_u64(12345);

        // Try flipping until we observe a change (bounded attempts to avoid flakiness).
        let original = seq.clone();
        let mut changed = false;
        for _ in 0..20 {
            let mut tmp = seq.clone();
            mutation_same_nonce_flip(&mut tmp, &buckets, &mut rng);
            if tmp != original {
                assert_nonce_valid(&tmp, &buckets);
                changed = true;
                break;
            }
        }
        assert!(changed, "same-nonce flip did not change any gene after several tries");
    }

    #[test]
    fn test_inter_sender_swap_mutation_preserves_validity() {
        let group = make_group_with_duplicate_buckets();
        let task = create_mock_task(0, group.clone(), Algorithm::Greedy, TaskPriority::Low, Instant::now());
        let buckets = build_nonce_buckets(&task).expect("nonce view present");

        // Build an initial valid sequence
        let view = build_sender_nonce_view(&group).unwrap();
        let chains_best = build_sender_chains_best(&view, &group);
        let mut seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &buckets);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(777);
        let before = seq.clone();
        mutation_inter_sender_swap(&mut seq, &buckets, &mut rng);

        // Must remain valid
        assert_nonce_valid(&seq, &buckets);
        // It may or may not change (depends on ranks); accept both, but at least it's valid.
        // If you want to enforce change, you could run multiple attempts here.
        let _maybe_changed = seq != before;
    }

    #[test]
    fn test_seed_initial_population_validity_and_uniqueness() {
        let group = make_group_with_duplicate_buckets();
        let task = create_mock_task(0, group.clone(), Algorithm::Greedy, TaskPriority::Low, Instant::now());
        let buckets = build_nonce_buckets(&task).expect("nonce view present");

        let mut rng = rand::rngs::SmallRng::seed_from_u64(999);
        let population_size = 2;
        let seeds = seed_initial_population(&task, &buckets, population_size, &mut rng);

        assert_eq!(seeds.len(), population_size);
        let mut uniq: HashSet<Vec<usize>> = HashSet::default();
        for s in &seeds {
            assert_nonce_valid(s, &buckets);
            assert!(uniq.insert(s.clone()), "duplicate initial individual");
        }
    }
}
