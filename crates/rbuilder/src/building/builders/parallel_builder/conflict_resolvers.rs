use ahash::{HashMap, AHashSet};
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

use serde::Serialize;
use std::cmp::Reverse;
use std::fs::OpenOptions;
use std::io::Write;


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

fn build_nonce_layout(task: &ConflictTask) -> Option<NonceLayout> {
    NonceLayout::from_group(&task.group)
}

#[inline]
fn repair_to_nonce_valid(preferred: &[usize], layout: &NonceLayout) -> Vec<usize> {
    // Project a single preference list onto a valid interleaving.
    // Internally this uses the same ready-set selection as crossover.
    ppx_build_child_from_parents_greedy(preferred, preferred, layout)
}

#[derive(Clone)]
struct Individual {
    seq: Vec<usize>,
    fitness: Option<U256>,
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

    // Fill rest with random candidates per step + uniform interleaving
    while seeds.len() < population_size {
        let s = random_interleaving_with_random_choices(layout, rng);
        if s.len() == layout.total_steps && seen.insert(s.clone()) {
            seeds.push(s);
        }
    }

    seeds
}

fn tournament_select<'p>(
    population: &'p [Individual],
    k: usize,
    rng: &mut SmallRng,
) -> usize {
    debug_assert!(population.len() >= k);
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fit = population[best_idx].fitness.expect("fitness must be set");
    for _ in 1..k {
        let i = rng.gen_range(0..population.len());
        let fit = population[i].fitness.expect("fitness must be set");
        // Tie-break by random chance
        if fit > best_fit || (fit == best_fit && rng.gen_bool(0.5)) {
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
fn ppx_build_child_from_parents_greedy(
    parent_a: &[usize],
    parent_b: &[usize],
    layout: &NonceLayout,
) -> Vec<usize> {
    let total_steps = layout.total_steps;
    let n_chains = layout.chains.len();
    let mut next_step_per_chain = vec![0usize; n_chains];
    let mut child = Vec::with_capacity(total_steps);

    const LARGE_RANK: usize = usize::MAX / 4;
    let mut rank_in_a: HashMap<usize, usize> = HashMap::default();
    let mut rank_in_b: HashMap<usize, usize> = HashMap::default();
    for (i, &idx) in parent_a.iter().enumerate() { rank_in_a.insert(idx, i); }
    for (i, &idx) in parent_b.iter().enumerate() { rank_in_b.insert(idx, i); }

    while child.len() < total_steps {
        let mut best_choice: Option<(usize, (usize, usize, usize))> = None;
        for chain_id in 0..n_chains {
            let step = next_step_per_chain[chain_id];
            if step >= layout.chains[chain_id].steps.len() { continue; }
            for &cand in &layout.chains[chain_id].steps[step].candidates {
                let ra = *rank_in_a.get(&cand).unwrap_or(&LARGE_RANK);
                let rb = *rank_in_b.get(&cand).unwrap_or(&LARGE_RANK);
                let key = (ra.min(rb), ra.saturating_add(rb), cand);
                if let Some((_, best_key)) = &best_choice {
                    if key < *best_key { best_choice = Some((cand, key)); }
                } else {
                    best_choice = Some((cand, key));
                }
            }
        }
        let (chosen_idx, _) = best_choice.expect("ready set non-empty");
        let (chain_id, step_id) = layout.index_of[&chosen_idx];
        debug_assert_eq!(next_step_per_chain[chain_id], step_id);
        child.push(chosen_idx);
        next_step_per_chain[chain_id] += 1;
    }
    child
}

/// Same as `ppx_build_child_from_parents_greedy`, but is not deterministic.
fn ppx_build_child_from_parents(
    parent_a: &[usize],
    parent_b: &[usize],
    layout: &NonceLayout,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let total_steps = layout.total_steps;
    let n_chains = layout.chains.len();
    let mut next_step_per_chain = vec![0usize; n_chains];
    let mut child = Vec::with_capacity(total_steps);

    // Rank maps: position in each parent gives priority (lower is better).
    // If a candidate doesn't appear in a parent, assign a large (bad) rank.
    const LARGE_RANK: usize = usize::MAX / 4;
    let mut rank_in_a: HashMap<usize, usize> = HashMap::default();
    let mut rank_in_b: HashMap<usize, usize> = HashMap::default();
    for (i, &idx) in parent_a.iter().enumerate() { rank_in_a.insert(idx, i); }
    for (i, &idx) in parent_b.iter().enumerate() { rank_in_b.insert(idx, i); }

    let alpha = 1.0;
    let beta = 0.25;

    while child.len() < total_steps {
        // Build ready set and compute a score for each candidate
        let mut cands: Vec<(usize, f64)> = Vec::new();
        for chain_id in 0..n_chains {
            let step = next_step_per_chain[chain_id];
            if step >= layout.chains[chain_id].steps.len() { continue; }
            for &cand in &layout.chains[chain_id].steps[step].candidates {
                let ra = *rank_in_a.get(&cand).unwrap_or(&LARGE_RANK);
                let rb = *rank_in_b.get(&cand).unwrap_or(&LARGE_RANK);

                // TODO: Tweak alpha and beta
                let score = alpha * (ra.min(rb) as f64) + beta * ((ra + rb) as f64);
                cands.push((cand, score));
            }
        }

        let chosen_idx = if cands.len() == 1 {
            cands[0].0
        } else {
            // Convert scores to probabilities via softmax
            let temperature = 0.5_f64; // higher = more random, lower = greedier
            // Numerical stability: subtract max of (-score/T)
            let max_term = cands.iter().map(|(_,s)| -s / temperature).fold(f64::NEG_INFINITY, f64::max);

            let mut weights: Vec<f64> = cands.iter()
                .map(|(_,s)| ((-s / temperature) - max_term).exp())
                .collect();

            // normalize
            let sumw: f64 = weights.iter().sum();
            if sumw <= 0.0 {
                // fallback to uniform if all weights underflowed
                let i = rng.gen_range(0..cands.len());
                cands[i].0
            } else {
                for w in &mut weights { *w /= sumw; }

                // Sample one candidate from the categorical distribution
                let r = rng.gen::<f64>();
                let mut acc = 0.0;
                let mut pick = cands[0].0;
                for ((cand, _), w) in cands.into_iter().zip(weights.into_iter()) {
                    acc += w;
                    if r <= acc { pick = cand; break; }
                }
                pick
            }
        };

        // Place the chosen index and advance its chain
        let (chain_id, step_id) = layout.index_of[&chosen_idx];
        debug_assert_eq!(next_step_per_chain[chain_id], step_id);
        child.push(chosen_idx);
        next_step_per_chain[chain_id] += 1;
    }

    child
}


// Mutation functions
fn mut_adjacent_interchain_swap(seq: &mut [usize], layout: &NonceLayout, rng: &mut SmallRng) -> bool {
    if seq.len() < 2 { return false; }
    let mut edges: Vec<usize> = Vec::new();
    for i in 0..seq.len()-1 {
        let (c1, _) = layout.index_of[&seq[i]];
        let (c2, _) = layout.index_of[&seq[i+1]];
        if c1 != c2 { edges.push(i); }
    }
    if edges.is_empty() { return false; }
    let i = rng.gen_range(0..edges.len());
    let j = edges[i];
    seq.swap(j, j+1);
    true
}

fn mut_bubble_move(seq: &mut [usize], layout: &NonceLayout, rng: &mut SmallRng, max_steps: usize) -> bool {
    if seq.is_empty() { return false; }
    use rand::Rng;
    let mut p = rng.gen_range(0..seq.len());
    let left = rng.gen_bool(0.5);
    let steps = rng.gen_range(1..=max_steps);
    let mut changed = false;
    for _ in 0..steps {
        if left {
            if p == 0 { break; }
            let (c1, _) = layout.index_of[&seq[p-1]];
            let (c2, _) = layout.index_of[&seq[p]];
            if c1 == c2 { break; }
            seq.swap(p-1, p);
            p -= 1;
        } else {
            if p+1 >= seq.len() { break; }
            let (c1, _) = layout.index_of[&seq[p]];
            let (c2, _) = layout.index_of[&seq[p+1]];
            if c1 == c2 { break; }
            seq.swap(p, p+1);
            p += 1;
        }
        changed = true;
    }
    changed
}

fn mutation_inter_sender_swap(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng) {
    if seq.is_empty() { return; }
    use rand::Rng;
    for _ in 0..5 {
        let i = rng.gen_range(0..seq.len());
        let j = rng.gen_range(0..seq.len());
        if i == j { continue; }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let (ca, _) = layout.index_of[&seq[a]];
        let (cb, _) = layout.index_of[&seq[b]];
        if ca != cb { seq.swap(a, b); return; }
    }
}

fn mutation_same_nonce_flip(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng) {
    if seq.is_empty() { return; }
    use rand::Rng;
    // pick a random position, if that (chain, step) has >1 candidates, flip to a different one
    let p = rng.gen_range(0..seq.len());
    let (c, s) = layout.index_of[&seq[p]];
    let bucket = &layout.chains[c].steps[s].candidates;
    if bucket.len() <= 1 { return; }

    let cur = seq[p];
    let alts: Vec<usize> = bucket.iter().copied().filter(|&x| x != cur).collect();
    if alts.is_empty() { return; }
    let alt = alts[rng.gen_range(0..alts.len())];
    seq[p] = alt;
}

fn mut_same_nonce_flip(seq: &mut [usize], layout: &NonceLayout, multi: &[(usize,usize)], rng: &mut SmallRng) -> bool {
    if multi.is_empty() { return false; }
    use rand::Rng;
    let (c, s) = multi[rng.gen_range(0..multi.len())];
    let bucket = &layout.chains[c].steps[s].candidates;
    if bucket.len() <= 1 { return false; }
    // find current gene for this (c,s)
    let cur_idx = *bucket.iter().find(|&&idx| seq.contains(&idx)).expect("must exist");
    let pos = seq.iter().position(|&x| x == cur_idx).unwrap();
    let alts: Vec<usize> = bucket.iter().copied().filter(|&x| x != cur_idx).collect();
    let alt = alts[rng.gen_range(0..alts.len())];
    seq[pos] = alt;
    true
}

fn mutate(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng, multi: &[(usize,usize)]) {
    use rand::Rng;
    for _ in 0..5 {
        let picked = rng.gen_range(0..100);
        let changed = if !multi.is_empty() && picked < 35 {
            mut_same_nonce_flip(seq, layout, multi, rng)
        } else if picked < 65 {
            mut_adjacent_interchain_swap(seq, layout, rng)
        } else {
            let max_steps = rng.gen_range(2..5);
            mut_bubble_move(seq, layout, rng, max_steps)
        };
        if changed { return; }
    }
    // last resort: uniform fresh random
    *seq = random_interleaving_with_random_choices(layout, rng);
}


/// Build one uniformly random nonce-respecting ordering.
/// Weighted choice: at each step pick chain i with probability
/// remaining_i / total_remaining, then take its next tx.
fn sample_one_uniform_interleaving(chains: &[Vec<usize>], rng: &mut SmallRng) -> Vec<usize> {
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
            Algorithm::ExhaustiveStreaming { time_ms, top_k } => {
                let res = self.run_exhaustive_streaming(&task, time_ms, top_k)?;
                trace!(
                    "Resolved ExhaustiveStreaming task {:?} with profit: {:?}",
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

        // let mut prefix_ids: Vec<OrderId> = if let Some(c) = &cached_state_option {
        //     c.per_order_profits.iter().map(|(oid, _)| oid.clone()).collect()
        // } else {
        //     Vec::with_capacity(sequence_of_orders.len())
        // };

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
                    // let order_id = sim_order.order.id();
                    // prefix_ids.push(order_id.clone());

                    // let _inserted = self.simulation_cache.ensure_cached_with(&prefix_ids, || {
                        // let bundle_state = state.clone_bundle();
                        // CachedSimulationState {
                        //     bundle_state,
                        //     total_profit,
                        //     per_order_profits: per_order_profits.clone(),
                        //     cumulative_gas_used: partial_block.gas_used,
                        //     cumulative_blob_gas_used: partial_block.blob_gas_used,
                        //     coinbase_profit: partial_block.coinbase_profit,
                        // }
                    // });
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
        seq: &[usize],
        task: &ConflictTask,
    ) -> eyre::Result<ResolutionResult> {
        let (res, _state) = self.process_sequence_of_orders(seq.to_vec(), task, self.state.clone())?;
        Ok(res)
    }

    /// Compute baseline profit using current implementation tasks:
    /// Greedy, ReverseGreedy, Length, and a random task.
    fn compute_baseline_profit(&mut self, task: &ConflictTask) -> eyre::Result<U256> {
        let mut best = U256::ZERO;

        // Greedy and ReverseGreedy
        for &rev in &[false, true] {
            for seq in generate_greedy_sequence(task, rev) {
                let res = self.evaluate_fitness(&seq, task)?;
                if res.total_profit > best { best = res.total_profit; }
            }
        }

        // Length
        for seq in generate_length_based_sequence(task) {
            let res = self.evaluate_fitness(&seq, task)?;
            if res.total_profit > best { best = res.total_profit; }
        }

        // Random
        let rnd = generate_random_permutations(
            &ConflictTask { algorithm: Algorithm::Random { seed: task.group.id as u64, count: 50 }, ..task.clone() },
            task.group.id as u64,
            50,
        );
        for seq in rnd {
            let res = self.evaluate_fitness(&seq, task)?;
            if res.total_profit > best { best = res.total_profit; }
        }

        Ok(best)
    }


    fn run_exhaustive_streaming(
        &mut self,
        task: &ConflictTask,
        time_ms: u64,
        top_k: usize,
    ) -> eyre::Result<ResolutionResult> {
        let Some(layout) = build_nonce_layout(task) else {
            // Fallback when we can’t derive (sender, nonce)
            return Ok(ResolutionResult { total_profit: U256::ZERO, sequence_of_orders: vec![] });
        };

        // Baseline first (so we can compare later)
        let baseline = self.compute_baseline_profit(task)?;

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
            let res = self.evaluate_fitness(&seq, task)?;
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
        Ok(best_res.unwrap_or(ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        }))
    }

    fn run_genetic(&mut self, task: &ConflictTask, params: GAParams) -> eyre::Result<ResolutionResult> {
        let Some(layout) = build_nonce_layout(task) else {
            // Fallback when we can’t derive (sender, nonce)
            return Ok(ResolutionResult { total_profit: U256::ZERO, sequence_of_orders: vec![] });
        };

        // Identify multi-candidate buckets for mutation
        let multi_steps: Vec<(usize, usize)> = layout.multi_steps();

        let start = Instant::now();
        let deadline = start + std::time::Duration::from_millis(params.time_ms);
        let mut rng = SmallRng::seed_from_u64(params.seed);

        // 1) Seed population
        let seed_seqs = seed_initial_population(task, &layout, params.population, &mut rng);
        let mut population: Vec<Individual> = seed_seqs.into_iter().map(|seq| Individual { seq, fitness: None }).collect();

        println!("Group {} Initial population sequences:", task.group.id);
        for (i, individual) in population.iter().enumerate() {
            println!("  Population[{}]: {:?}", i, individual.seq);
        }

        // 2) Evaluate initial population
        let mut best: Option<(usize, ResolutionResult)> = None;
        for i in 0..population.len() {
            if self.cancellation_token.is_cancelled() { return Err(eyre::eyre!("Cancelled")); }
            let fit_res = self.evaluate_fitness(&population[i].seq, task)?;
            population[i].fitness = Some(fit_res.total_profit);
            
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
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            // Sort by fitness desc for elitism
            population.sort_by(|a, b| {
                let fa = a.fitness.unwrap_or(U256::ZERO);
                let fb = b.fitness.unwrap_or(U256::ZERO);
                fb.cmp(&fa)
            });

            // println!("  Best sequences after sorting:");
            // for i in 0..3.min(population.len()) {
            //     println!("    Rank[{}]: seq={:?}, fitness={}", 
            //                 i, population[i].seq, population[i].fitness.unwrap_or(U256::ZERO));
            // }

            // println!("Worst sequence: {:?}, fitness=[]={}", 
            //     population[population.len() - 1].seq, 
            //     population[population.len() - 1].fitness.unwrap_or(U256::ZERO));

            // Next population with elites
            let mut next_population: Vec<Individual> = Vec::with_capacity(params.population);
            let elites = params.elitism.min(population.len());
            for i in 0..elites {
                next_population.push(population[i].clone()); // carry over
            }

            // Track seen sequences to avoid duplicates
            let mut seen_next: AHashSet<Vec<usize>> = AHashSet::default();
            // mark elites
            for i in 0..elites {
                seen_next.insert(population[i].seq.clone());
            }

            // Fill the rest
            while next_population.len() < params.population && Instant::now() < deadline {
                // Parents
                let p1_idx = tournament_select(&population, params.tourn_k, &mut rng);
                let mut p2_idx = tournament_select(&population, params.tourn_k, &mut rng);

                // avoid identical parents (by index or by sequence)
                for _ in 0..4 {
                    if p2_idx != p1_idx && population[p2_idx].seq != population[p1_idx].seq {
                        break;
                    }
                    p2_idx = tournament_select(&population, params.tourn_k, &mut rng);
                }
                let p1 = &population[p1_idx];
                let p2 = &population[p2_idx];

                // Crossover
                let mut child_seq = if rng.gen::<f64>() < params.crossover_rate {
                    let child = ppx_build_child_from_parents(&p1.seq, &p2.seq, &layout, &mut rng);

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

                    parent_seq
                };

                // Mutations
                let pre_mutation = child_seq.clone();
                if rng.gen::<f64>() < params.mutation_rate {
                    mutate(&mut child_seq, &layout, &mut rng, &multi_steps);
                    // println!("    Mutated: {:?} -> {:?}", pre_mutation, child_seq);
                }

                // ensure uniqueness in next_population
                if !seen_next.insert(child_seq.clone()) {
                    // try mutating again a couple of times
                    let mut accepted = false;
                    for _ in 0..2 {
                        let mut candidate_child = child_seq.clone();
                        mutate(&mut candidate_child, &layout, &mut rng, &multi_steps);
                        if seen_next.insert(candidate_child.clone()) {
                            child_seq = candidate_child;
                            accepted = true;
                            break;
                        }
                    }
                    if !accepted {
                        // last resort: insert a new random individual
                        child_seq = random_interleaving_with_random_choices(&layout, &mut rng);
                        // extremely unlikely to still collide, but just in case:
                        if !seen_next.insert(child_seq.clone()) {
                            continue; // skip and let the while loop create another child
                        }
                    }
                }

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
                println!("Group {} stopping early at generation {} due to plateau", task.group.id, generation);
                println!("Plateau patience: {}, spent {}ms / {}ms", plateau_patience, spent, params.time_ms);
                break;
            }

            population = next_population;
            generation += 1;
        }

        println!("GA finished after {} generations for group {}", generation, task.group.id);
        for individual in &population {
            println!("Final individual: seq={:?}, fitness={:?}", individual.seq, individual.fitness);
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
        Algorithm::ExhaustiveStreaming { .. } => {
            // Exhaustive streaming is handled separately in ResolverContext::run_exhaustive_streaming
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
    // if let Some(layout) = build_nonce_layout(task) {
    //     if ALL_PERMS_INCLUDE_DUPLICATE_NONCE_CHOICES {
    //         return enumerate_all_interleavings_with_choices(&layout, ALL_PERMS_CAP);
    //     } else {
    //         let per_chain = build_chains_best(&layout, &task.group);
    //         return enumerate_all_interleavings_best(&per_chain, ALL_PERMS_CAP);
    //     }
    // }

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
        create_sequence(|sim_order| sim_order.sim_value.coinbase_profit),
        create_sequence(|sim_order| sim_order.sim_value.mev_gas_price),
    ]
}


// fn generate_greedy_sequence(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
//     let group = &task.group;

//     // Build a single greedy preference list for the given key, filtered by
//     // key-aware per-slot dedup.
//     let build_for = |key: GreedyKey| {
//         // Only keep best-per-slot candidates according to this key+reverse.
//         let allowed = allowed_indices_after_nonce_dedup(group, key, reverse);

//         // Collect indices with both metrics so we can do a stable secondary tie-break.
//         let mut rows: Vec<(usize, U256, U256)> = group
//             .orders
//             .iter()
//             .enumerate()
//             .filter(|(idx, _)| allowed.as_ref().map_or(true, |set| set.contains(idx)))
//             .map(|(idx, o)| (idx, o.sim_value.coinbase_profit, o.sim_value.mev_gas_price))
//             .collect();

//         rows.sort_by(|a, b| {
//             // a: (idx, profit, gas), b: (idx, profit, gas)
//             let (pa, sa) = match key {
//                 GreedyKey::Profit     => (a.1, a.2),
//                 GreedyKey::MevGasPrice=> (a.2, a.1),
//             };
//             let (pb, sb) = match key {
//                 GreedyKey::Profit     => (b.1, b.2),
//                 GreedyKey::MevGasPrice=> (b.2, b.1),
//             };

//             let ord1 = if reverse { pa.cmp(&pb) } else { pb.cmp(&pa) };
//             if ord1 != std::cmp::Ordering::Equal { return ord1; }

//             let ord2 = if reverse { sa.cmp(&sb) } else { sb.cmp(&sa) };
//             if ord2 != std::cmp::Ordering::Equal { return ord2; }

//             a.0.cmp(&b.0)
//         });

//         rows.into_iter().map(|(idx, _, _)| idx).collect::<Vec<_>>()
//     };

//     vec![
//         build_for(GreedyKey::Profit),
//         build_for(GreedyKey::MevGasPrice),
//     ]
// }

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
                order.sim_value.coinbase_profit,
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


// fn generate_length_based_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
//     let order_group = &task.group;
//     let allowed = allowed_indices_after_nonce_dedup(order_group, GreedyKey::Profit, false);

//     let mut order_data: Vec<(usize, usize, U256)> = order_group
//         .orders
//         .iter()
//         .enumerate()
//         .filter(|(idx, _)| allowed.as_ref().map_or(true, |set| set.contains(idx)))
//         .map(|(idx, order)| (idx, order.order.list_txs().len(), order.sim_value.coinbase_profit))
//         .collect();

//     order_data.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
//     let seq: Vec<usize> = order_data.into_iter().map(|(idx, _, _)| idx).collect();
//     vec![seq]
// }


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
        let layout = build_nonce_layout(&task).expect("nonce view present");
        assert_eq!(layout.total_steps, 4);

        // Create two valid parents manually:
        // First, pick "best per slot" via helper to get one candidate per nonce.
        let chains_best = build_chains_best(&layout, &group);

        // Parent A interleaving: A0, B0, A1, B1
        let parent_a = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        // Parent B interleaving: B0, A0, B1, A1
        let parent_b = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]];

        // Child from PPX-style builder
        let child = ppx_build_child_from_parents_greedy(&parent_a, &parent_b, &layout);
        assert_nonce_valid(&child, &layout);
    }

    #[test]
    fn test_same_nonce_flip_mutation_preserves_validity_and_changes_candidate() {
        let group = make_group_with_duplicate_buckets();
        let task = create_mock_task(0, group.clone(), Algorithm::Greedy, TaskPriority::Low, Instant::now());

        // Start from a valid interleaving (best-per-slot, simple A0,B0,A1,B1)
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);
        let seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &layout);

        // There are at least two slots with multiple candidates (A:nonce0, B:nonce1).
        let mut rng = SmallRng::seed_from_u64(12345);

        // Try flipping until we observe a change (bounded attempts to avoid flakiness).
        let original = seq.clone();
        let mut changed = false;
        for _ in 0..20 {
            let mut tmp = seq.clone();
            mutation_same_nonce_flip(&mut tmp, &layout, &mut rng);
            if tmp != original {
                assert_nonce_valid(&tmp, &layout);
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
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        let mut seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &layout);

        let mut rng = SmallRng::seed_from_u64(777);
        let before = seq.clone();
        mutation_inter_sender_swap(&mut seq, &layout, &mut rng);

        // Must remain valid
        assert_nonce_valid(&seq, &layout);
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
