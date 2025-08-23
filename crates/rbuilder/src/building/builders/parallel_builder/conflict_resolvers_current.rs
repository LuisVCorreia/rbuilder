use ahash::{HashMap, HashSet};
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
            Algorithm::Beam { beam_width, max_expansions } => {
                let res = self.run_beam_search(&task, beam_width, max_expansions)?;
                trace!(
                    "Resolved conflict task {:?} with profit: {:?} and algorithm: Beam({},{})",
                    task.group.id, res.total_profit, beam_width, max_expansions
                );
                return Ok(res);
            }
            _ => {}
        }

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

    /// Nonce-aware best-first beam search (no inadmissible pruning).
    /// We rank nodes by a heuristic "optimistic" score but we DO NOT prune on it.
    /// Each expansion commits exactly one more tx vs its parent (cache-friendly).
    fn run_beam_search(
        &mut self,
        task: &ConflictTask,
        beam_width: usize,
        max_expansions: usize,
    ) -> eyre::Result<ResolutionResult> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        use rand::SeedableRng;

        // Build nonce chains; fallback to Greedy if not possible.
        let chains = if let Some(view) = build_sender_nonce_view(&task.group) {
            build_sender_chains_best(&view, &task.group)
        } else {
            let greedy = generate_greedy_sequence(task, false)
                .into_iter()
                .next()
                .unwrap_or_default();
            let (res, _st) = self.process_sequence_of_orders(greedy, task, self.state.clone())?;
            return Ok(res);
        };

        let k = chains.len();
        let total_len: usize = chains.iter().map(|c| c.len()).sum();
        if total_len == 0 {
            return Ok(ResolutionResult {
                total_profit: U256::ZERO,
                sequence_of_orders: vec![],
            });
        }

        // Map order idx -> chain id (for greedy completion convenience)
        let mut idx_to_chain: Vec<Option<usize>> = vec![None; task.group.orders.len()];
        for (ci, ch) in chains.iter().enumerate() {
            for &idx in ch {
                idx_to_chain[idx] = Some(ci);
            }
        }

        // Per-order heuristics
        let profits: Vec<U256> = task
            .group
            .orders
            .iter()
            .map(|o| o.sim_value.coinbase_profit)
            .collect();
        let gas_price: Vec<U256> = task
            .group
            .orders
            .iter()
            .map(|o| o.sim_value.mev_gas_price)
            .collect();

        let total_sim_sum: U256 = profits.iter().fold(U256::ZERO, |acc, &p| acc + p);

        #[derive(Clone)]
        struct Node {
            prefix: Vec<usize>,
            used: Vec<usize>,
            realized: U256,
            sum_sim_prefix: U256,
            // Heuristic score used ONLY for ordering (may be over-optimistic).
            score: U256,
        }

        #[derive(Clone)]
        struct HeapItem(Node);

        // Max-heap by `score`
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.score.cmp(&other.0.score)
            }
        }
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.0.score == other.0.score
            }
        }
        impl Eq for HeapItem {}

        // Heads (next unlocked candidates)
        let mut heads = |used: &Vec<usize>| -> Vec<(usize, usize)> {
            let mut out = Vec::with_capacity(k);
            for ci in 0..k {
                if used[ci] < chains[ci].len() {
                    out.push((ci, chains[ci][used[ci]]));
                }
            }
            out
        };

        // Sort heads by a simple heuristic: coinbase, then remaining chain length, then mev_gas_price
        let mut sort_heads =
            |mut hs: Vec<(usize, usize)>, used: &Vec<usize>| -> Vec<(usize, usize)> {
                hs.sort_by(|&(ci_a, idx_a), &(ci_b, idx_b)| {
                    let pa = profits[idx_a];
                    let pb = profits[idx_b];
                    if pa != pb {
                        return pb.cmp(&pa); // higher coinbase first
                    }
                    let rem_a = chains[ci_a].len() - used[ci_a];
                    let rem_b = chains[ci_b].len() - used[ci_b];
                    if rem_a != rem_b {
                        return rem_b.cmp(&rem_a); // longer remaining chain first
                    }
                    let ga = gas_price[idx_a];
                    let gb = gas_price[idx_b];
                    gb.cmp(&ga) // higher mev_gas_price first
                });
                hs
            };

        // Greedy completion from a node (still nonce-respecting)
        let greedy_complete = |node: &Node| -> Vec<usize> {
            let mut used = node.used.clone();
            let mut seq = node.prefix.clone();
            while seq.len() < total_len {
                let mut hs = heads(&used);
                if hs.is_empty() {
                    break;
                }
                hs = sort_heads(hs, &used);
                let (_ci, idx) = hs[0];
                seq.push(idx);
                let ci = idx_to_chain[idx].expect("idx must map to a chain");
                used[ci] += 1;
            }
            seq
        };

        // Random completion from a node (uniform interleaving of remaining chains)
        let mut rng = rand::rngs::SmallRng::seed_from_u64(task.group.id as u64 ^ 0xBEEF_F00D_u64);
        let mut random_complete = |node: &Node| -> Vec<usize> {
            let mut seq = node.prefix.clone();
            // Build remaining chains
            let remaining: Vec<Vec<usize>> = (0..k)
                .map(|ci| chains[ci][node.used[ci]..].to_vec())
                .collect();
            if !remaining.iter().all(|c| c.is_empty()) {
                let tail = sample_one_uniform_interleaving(&remaining, &mut rng);
                seq.extend(tail);
            }
            seq
        };

        // Heuristic score (used ONLY for ranking). We purposely allow this to be over-optimistic.
        let heuristic_score = |realized: U256, sum_sim_prefix: U256| -> U256 {
            // Realized so far + optimistic remaining (sum of standalone profits of remaining).
            // This CAN be < true max (because synergies), but we only use it to order the heap,
            // never to prune.
            realized + (total_sim_sum - sum_sim_prefix)
        };

        // --------- Stats ----------
        let mut nodes_popped: usize = 0;
        let mut sims_run: usize = 0;
        let mut completed_evals: usize = 0;
        let mut forced_completions: usize = 0;
        let mut max_heap: usize = 0;
        let mut trimmed_once: bool = false;

        // Branching cap per pop: keep it controlled
        let branch_cap = std::cmp::min(std::cmp::max(2, beam_width / 2), k.max(1));

        // Warm-start incumbent with Greedy
        let greedy_seed = generate_greedy_sequence(task, false)
            .into_iter()
            .next()
            .unwrap_or_else(|| (0..task.group.orders.len()).collect());
        let (greedy_res, _st) =
            self.process_sequence_of_orders(greedy_seed, task, self.state.clone())?;
        sims_run += 1;
        let mut best_result: ResolutionResult = greedy_res;

        // Beam DS
        let mut seen_prefixes: ahash::HashSet<Vec<usize>> = ahash::HashSet::default();
        let mut heap = BinaryHeap::new();

        // Seed node
        let seed = Node {
            prefix: Vec::with_capacity(total_len),
            used: vec![0; k],
            realized: U256::ZERO,
            sum_sim_prefix: U256::ZERO,
            score: total_sim_sum, // optimistic: "we could get everything"
        };
        seen_prefixes.insert(Vec::new());
        heap.push(HeapItem(seed));
        max_heap = max_heap.max(heap.len());

        // periodic forced completion cadence
        let mut since_last_forced = 0usize;
        let forced_every = 100usize.max(beam_width); // roughly every beam_width pops

        // Main loop
        while let Some(HeapItem(node)) = heap.pop() {
            if self.cancellation_token.is_cancelled() || sims_run >= max_expansions {
                break;
            }
            nodes_popped += 1;
            since_last_forced += 1;

            // If this is a full sequence, evaluate and maybe update best
            if node.prefix.len() == total_len {
                let (res, _st) =
                    self.process_sequence_of_orders(node.prefix.clone(), task, self.state.clone())?;
                sims_run += 1;
                completed_evals += 1;
                if res.total_profit > best_result.total_profit {
                    best_result = res;
                }
                continue;
            }

            // Expand by ONE unlocked tx (nonce-respecting), prioritize better heads
            let mut hs = heads(&node.used);
            if !hs.is_empty() {
                hs = sort_heads(hs, &node.used);
                if hs.len() > branch_cap {
                    hs.truncate(branch_cap);
                }

                for (ci, idx) in hs {
                    if self.cancellation_token.is_cancelled() || sims_run >= max_expansions {
                        break;
                    }
                    // Child prefix
                    let mut child_prefix = node.prefix.clone();
                    child_prefix.push(idx);
                    if !seen_prefixes.insert(child_prefix.clone()) {
                        continue;
                    }

                    // Simulate child prefix; cache will reuse the parent prefix
                    let (res, _st) =
                        self.process_sequence_of_orders(child_prefix.clone(), task, self.state.clone())?;
                    sims_run += 1;

                    // Build child node
                    let mut child_used = node.used.clone();
                    child_used[ci] += 1;

                    let child_realized = res.total_profit;
                    let child_sum = node.sum_sim_prefix + profits[idx];
                    let child_score = heuristic_score(child_realized, child_sum);

                    heap.push(HeapItem(Node {
                        prefix: child_prefix,
                        used: child_used,
                        realized: child_realized,
                        sum_sim_prefix: child_sum,
                        score: child_score,
                    }));
                }
            }

            // Soft heap control: keep only top ~2*beam_width
            if heap.len() > 4 * beam_width {
                let mut kept: Vec<HeapItem> = Vec::with_capacity(2 * beam_width);
                for _ in 0..(2 * beam_width) {
                    if let Some(it) = heap.pop() {
                        kept.push(it);
                    } else {
                        break;
                    }
                }
                heap.clear();
                for it in kept {
                    heap.push(it);
                }
                trimmed_once = true;
            }
            max_heap = max_heap.max(heap.len());

            // Periodically force completions from the current best frontier node:
            // - Greedy completion
            // - Random completion
            if since_last_forced >= forced_every {
                since_last_forced = 0;

                if let Some(best_item) = heap.peek().cloned() {
                    // Greedy
                    if sims_run < max_expansions {
                        let seq_g = greedy_complete(&best_item.0);
                        let (res_g, _st) =
                            self.process_sequence_of_orders(seq_g, task, self.state.clone())?;
                        sims_run += 1;
                        forced_completions += 1;
                        if res_g.total_profit > best_result.total_profit {
                            best_result = res_g;
                        }
                    }

                    // Random
                    if sims_run < max_expansions {
                        let seq_r = random_complete(&best_item.0);
                        let (res_r, _st) =
                            self.process_sequence_of_orders(seq_r, task, self.state.clone())?;
                        sims_run += 1;
                        forced_completions += 1;
                        if res_r.total_profit > best_result.total_profit {
                            best_result = res_r;
                        }
                    }
                }
            }
        }

        // Final safety: if we somehow never evaluated a full sequence through forced completions,
        // complete the best frontier node greedily and evaluate it.
        if best_result.sequence_of_orders.is_empty() {
            if let Some(best_item) = heap.peek().cloned() {
                let seq = greedy_complete(&best_item.0);
                let (res, _st) = self.process_sequence_of_orders(seq, task, self.state.clone())?;
                sims_run += 1;
                if res.total_profit > best_result.total_profit {
                    best_result = res;
                }
            }
        }

        // "Exactly fully explored" means: we didn't have to trim the heap AND
        // we didn't hit the expansion cap AND search naturally exhausted.
        let fully_explored_exact =
            !trimmed_once && sims_run < max_expansions && heap.is_empty() && !self.cancellation_token.is_cancelled();

        println!(
            "BeamSearch stats: group={:?} popped={} sims={} complete_evals={} forced_completions={} \
            heap_max={} fully_explored_exact={}",
            task.group.id,
            nodes_popped,
            sims_run,
            completed_evals,
            forced_completions,
            max_heap,
            fully_explored_exact
        );

        Ok(best_result)
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
        Algorithm::Beam { .. } => {
            // Beam search is handled in ResolverContext::run_beam_search
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
//     let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
//     for _ in 0..count {
//         indexes.shuffle(&mut rng);
//         sequences_of_orders.push(indexes.clone());
//     }

//     sequences_of_orders
// }

fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    let order_group = &task.group;

    if let Some(view) = build_sender_nonce_view(order_group) {
        let chains = build_sender_chains_best(&view, order_group);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(sample_one_uniform_interleaving(&chains, &mut rng));
        }
        return out;
    }

    // Fallback: bundles/multi-tx orders where we can't derive nonce chains
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut indexes: Vec<usize> = (0..order_group.orders.len()).collect();
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

    println!("Falling back on naive permutations for legacy/bundles");
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
    let order_group = &task.group;
    let allowed = allowed_indices_after_nonce_dedup(order_group);

    let create_sequence = |value_extractor: fn(&SimulatedOrder) -> U256| {
        let mut ids_and_value: Vec<_> = order_group
            .orders
            .iter()
            .enumerate()
            .filter(|(idx, _)| allowed.as_ref().map_or(true, |set| set.contains(idx)))
            .map(|(idx, order)| (idx, value_extractor(order)))
            .collect();

        ids_and_value.sort_by(|a, b| {
            if reverse { a.1.cmp(&b.1) } else { b.1.cmp(&a.1) }
        });

        ids_and_value.into_iter().map(|(idx, _)| idx).collect()
    };

    vec![
        create_sequence(|sim_order| sim_order.sim_value.coinbase_profit),
        create_sequence(|sim_order| sim_order.sim_value.mev_gas_price),
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
    let order_group = &task.group;
    let allowed = allowed_indices_after_nonce_dedup(order_group);

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
    use alloy_primitives::{Address, TxHash, B256, U256};
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use super::*;
    use crate::{
        building::builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority},
        primitives::{
            Bundle, Metadata, Order, SimValue, SimulatedOrder,
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
}
