//! Genetic algorithm operators for conflict resolution.
//!
//! All operators in this module produce **nonce-valid** orderings by working
//! with the [`DependencyDag`] from `nonce_handling`.  The DAG encodes which
//! orders must precede which; every sequence this module produces is a valid
//! topological sort of that DAG.
//!
//! The key abstraction is the **ready set**: at any point during sequence
//! construction, the set of orders whose predecessors have all been placed.
//! Every operator builds sequences by repeatedly choosing from the ready set,
//! which guarantees validity for both single-tx orders and bundles.

use ahash::HashMap;
use alloy_primitives::U256;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

use super::nonce_handling::{DependencyDag, GroupDeps, random_ordering_with_random_choices};

// ─── GA parameters and individuals ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct GAParams {
    pub population: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub tourn_k: usize,
    pub max_generations: usize,
    pub time_ms: u64,
    pub seed: u64,
}

#[derive(Clone)]
pub struct Individual {
    pub seq: Vec<usize>,
    pub profit: U256,
    pub gas: u64,
}

pub fn dominates(a: &Individual, b: &Individual) -> bool {
    a.profit > b.profit || (a.profit == b.profit && a.gas < b.gas)
}

// ─── Ready-set builder ──────────────────────────────────────────────────
//
// This is the core primitive shared by crossover, mutation repair, and
// distance computation.  It tracks which DAG nodes have been placed and
// maintains a live in-degree vector so we can efficiently query "what is
// ready to be placed next?"

/// Mutable state for incrementally building a valid topological ordering.
///
/// Used internally by crossover and mutation operators.  Call [`place`] to
/// add the next order, and [`is_ready`] / [`ready_set`] to query what can
/// be placed next.
struct ReadyTracker<'a> {
    dag: &'a DependencyDag,
    in_deg: Vec<usize>,
    placed: Vec<bool>,
    n: usize,
}

impl<'a> ReadyTracker<'a> {
    fn new(dag: &'a DependencyDag) -> Self {
        let n = dag.len();
        Self {
            dag,
            in_deg: dag.in_degree.clone(),
            placed: vec![false; n],
            n,
        }
    }

    /// Is DAG node `ni` ready to be placed (all predecessors done)?
    #[inline]
    fn is_ready_node(&self, ni: usize) -> bool {
        !self.placed[ni] && self.in_deg[ni] == 0
    }

    /// Is original order index `oi` ready?
    #[inline]
    fn is_ready_order(&self, oi: usize) -> bool {
        if let Some(&ni) = self.dag.node_of.get(&oi) {
            self.is_ready_node(ni)
        } else {
            false
        }
    }

    /// Collect all ready DAG node indices.
    fn ready_nodes(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&ni| self.is_ready_node(ni))
            .collect()
    }

    /// Collect all ready original order indices.
    fn ready_orders(&self) -> Vec<usize> {
        self.ready_nodes()
            .into_iter()
            .map(|ni| self.dag.nodes[ni])
            .collect()
    }

    /// Place a node (by original order index), updating in-degrees.
    fn place(&mut self, oi: usize) {
        let ni = self.dag.node_of[&oi];
        debug_assert!(
            self.is_ready_node(ni),
            "Tried to place order {} (node {}) which is not ready",
            oi,
            ni
        );
        self.placed[ni] = true;
        for &succ in &self.dag.successors[ni] {
            self.in_deg[succ] -= 1;
        }
    }

    /// Has this original order index already been placed?
    #[inline]
    fn is_placed_order(&self, oi: usize) -> bool {
        if let Some(&ni) = self.dag.node_of.get(&oi) {
            self.placed[ni]
        } else {
            true // not in DAG = treat as done
        }
    }
}

// ─── Distance metric ────────────────────────────────────────────────────

/// Normalized positional distance for deterministic crowding.
///
/// Counts the fraction of positions where the two sequences differ.
/// Both sequences must be valid orderings over the same DAG.
pub fn dc_distance(a: &[usize], b: &[usize], _deps: &GroupDeps) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 1.0;
    }
    let diffs = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    diffs as f64 / a.len() as f64
}

// ─── Crossover: precedence-preserving (PPX) ─────────────────────────────

/// Build a nonce-valid child guided by rank in both parents.
///
/// At each step, every ready order gets a score based on its position in
/// each parent (lower = earlier = better).  The order with the best
/// combined score is placed next.  This is deterministic given the parents.
///
/// Works for any DAG topology (chains, bundles, diamonds, etc.).
pub fn ppx_build_child_from_parents_greedy(
    parent_a: &[usize],
    parent_b: &[usize],
    dag: &DependencyDag,
) -> Vec<usize> {
    let n = dag.len();
    if n == 0 {
        return Vec::new();
    }

    let rank_a = build_rank_map(parent_a);
    let rank_b = build_rank_map(parent_b);
    let mut tracker = ReadyTracker::new(dag);
    let mut child = Vec::with_capacity(n);

    for _ in 0..n {
        let ready = tracker.ready_orders();
        debug_assert!(!ready.is_empty());

        let best = ready
            .into_iter()
            .min_by_key(|&oi| {
                let ra = rank_a.get(&oi).copied().unwrap_or(usize::MAX / 4);
                let rb = rank_b.get(&oi).copied().unwrap_or(usize::MAX / 4);
                (ra.min(rb), ra.saturating_add(rb), oi)
            })
            .unwrap();

        child.push(best);
        tracker.place(best);
    }

    child
}

/// Stochastic PPX: same idea but with softmax sampling over scores.
pub fn ppx_build_child_from_parents(
    parent_a: &[usize],
    parent_b: &[usize],
    dag: &DependencyDag,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = dag.len();
    if n == 0 {
        return Vec::new();
    }

    let rank_a = build_rank_map(parent_a);
    let rank_b = build_rank_map(parent_b);
    let mut tracker = ReadyTracker::new(dag);
    let mut child = Vec::with_capacity(n);

    let alpha = 1.0;
    let beta = 0.25;
    let temperature = 0.5;

    for _ in 0..n {
        let ready = tracker.ready_orders();

        let chosen = if ready.len() == 1 {
            ready[0]
        } else {
            // Compute scores (lower = better).
            let scores: Vec<f64> = ready
                .iter()
                .map(|&oi| {
                    let ra = rank_a.get(&oi).copied().unwrap_or(usize::MAX / 4);
                    let rb = rank_b.get(&oi).copied().unwrap_or(usize::MAX / 4);
                    alpha * (ra.min(rb) as f64) + beta * ((ra + rb) as f64)
                })
                .collect();

            softmax_sample(&ready, &scores, temperature, rng)
        };

        child.push(chosen);
        tracker.place(chosen);
    }

    child
}

// ─── Crossover: adapted order crossover (OX1) ──────────────────────────

/// Order crossover adapted for DAG constraints.
///
/// Picks a random contiguous slice from parent_a. Builds the child one
/// element at a time from the ready set:
///   - If a **slice element** is ready, place it (preserving slice order).
///   - Otherwise, place the earliest ready element from **parent_b's order**
///     (this naturally pulls in prerequisites that the slice needs).
///
/// This always terminates and always produces a valid topological sort,
/// regardless of DAG depth or structure.
pub fn adapted_order_crossover(
    parent_a: &[usize],
    parent_b: &[usize],
    dag: &DependencyDag,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = parent_a.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![parent_a[0]];
    }

    // Pick random slice bounds from parent_a.
    let p1 = rng.gen_range(0..n);
    let mut p2 = rng.gen_range(0..n);
    while p1 == p2 {
        p2 = rng.gen_range(0..n);
    }
    let (start, end) = (p1.min(p2), p1.max(p2));

    // Slice elements in parent_a's order (queue: front = highest priority).
    let slice_set: HashSet<usize> = parent_a[start..=end].iter().copied().collect();
    let mut slice_queue: std::collections::VecDeque<usize> =
        parent_a[start..=end].iter().copied().collect();

    // Fill elements: everything from parent_b that's not in the slice,
    // preserving parent_b's relative order.
    let mut fill_queue: std::collections::VecDeque<usize> = parent_b
        .iter()
        .copied()
        .filter(|oi| !slice_set.contains(oi))
        .collect();

    let mut tracker = ReadyTracker::new(dag);
    let mut child = Vec::with_capacity(n);

    while child.len() < n {
        // Try to place the next slice element if it's ready.
        let placed_from_slice = place_next_ready(&mut slice_queue, &mut tracker, &mut child);

        if !placed_from_slice {
            // Slice front is blocked (or empty). Place from fill queue instead.
            // This pulls in prerequisites the slice needs, or fills remaining
            // positions after the slice is exhausted.
            let placed_from_fill = place_next_ready(&mut fill_queue, &mut tracker, &mut child);

            if !placed_from_fill {
                // Neither queue has a ready element at its front. This means
                // the front elements of both queues are blocked by something
                // deeper in one of the queues. Place ANY ready node to make
                // progress.
                let ready = tracker.ready_orders();
                if let Some(&fallback) = ready.first() {
                    child.push(fallback);
                    tracker.place(fallback);
                    // Remove from whichever queue contains it.
                    remove_from_deque(&mut slice_queue, fallback);
                    remove_from_deque(&mut fill_queue, fallback);
                } else {
                    // No ready nodes and child isn't complete — shouldn't
                    // happen with a valid DAG and correct inputs.
                    debug_assert!(false, "No ready nodes but child is incomplete");
                    break;
                }
            }
        }
    }

    child
}

/// Try to place the front element of `queue` if it's ready.
/// Skips over elements already placed. Returns true if something was placed.
fn place_next_ready(
    queue: &mut std::collections::VecDeque<usize>,
    tracker: &mut ReadyTracker<'_>,
    child: &mut Vec<usize>,
) -> bool {
    // Skip elements that were already placed (e.g., by the fallback path).
    while let Some(&front) = queue.front() {
        if tracker.is_placed_order(front) {
            queue.pop_front();
        } else {
            break;
        }
    }

    if let Some(&front) = queue.front() {
        if tracker.is_ready_order(front) {
            queue.pop_front();
            child.push(front);
            tracker.place(front);
            return true;
        }
    }

    false
}

/// Remove a specific value from a deque (used for fallback cleanup).
fn remove_from_deque(deque: &mut std::collections::VecDeque<usize>, val: usize) {
    if let Some(pos) = deque.iter().position(|&x| x == val) {
        deque.remove(pos);
    }
}

// ─── Mutation operators ─────────────────────────────────────────────────

/// Swap two adjacent orders from different dependency chains.
///
/// This is the simplest nonce-safe mutation: swapping neighbors that have no
/// dependency between them always produces a valid topo sort.
pub fn mut_adjacent_swap(seq: &mut [usize], dag: &DependencyDag, rng: &mut SmallRng) -> bool {
    if seq.len() < 2 {
        return false;
    }

    // Find swappable pairs: adjacent elements with no edge between them.
    let mut swappable: Vec<usize> = Vec::new();
    for i in 0..seq.len() - 1 {
        if can_swap_adjacent(seq, i, dag) {
            swappable.push(i);
        }
    }

    if swappable.is_empty() {
        return false;
    }

    let &i = swappable.choose(rng).unwrap();
    seq.swap(i, i + 1);
    true
}

/// Bubble an element left or right by repeated adjacent swaps.
///
/// Stops when it would violate a dependency or reaches the sequence boundary.
pub fn mut_bubble_move(
    seq: &mut [usize],
    dag: &DependencyDag,
    rng: &mut SmallRng,
    max_steps: usize,
) -> bool {
    if seq.is_empty() {
        return false;
    }

    let mut pos = rng.gen_range(0..seq.len());
    let go_left = rng.gen_bool(0.5);
    let steps = rng.gen_range(1..=max_steps);
    let mut changed = false;

    for _ in 0..steps {
        if go_left {
            if pos == 0 {
                break;
            }
            if !can_swap_adjacent(seq, pos - 1, dag) {
                break;
            }
            seq.swap(pos - 1, pos);
            pos -= 1;
        } else {
            if pos + 1 >= seq.len() {
                break;
            }
            if !can_swap_adjacent(seq, pos, dag) {
                break;
            }
            seq.swap(pos, pos + 1);
            pos += 1;
        }
        changed = true;
    }

    changed
}

/// Segment reversal that preserves DAG validity.
///
/// Pick a random segment, reverse it, then repair any violations by
/// re-sorting the segment as a valid topo sort.
pub fn mut_segment_shuffle(
    seq: &mut [usize],
    dag: &DependencyDag,
    rng: &mut SmallRng,
) -> bool {
    let n = seq.len();
    if n < 2 {
        return false;
    }

    let i = rng.gen_range(0..n);
    let j = rng.gen_range(0..n);
    let (start, end) = (i.min(j), i.max(j));
    if start == end {
        return false;
    }

    // Shuffle the segment.
    seq[start..=end].shuffle(rng);

    // Repair: re-sort the segment to be a valid sub-topo-sort.
    // We only need to fix ordering within the segment; the rest of the
    // sequence is untouched.
    repair_segment(seq, start, end, dag);

    true
}

/// Top-level mutation dispatcher.
pub fn mutate(seq: &mut Vec<usize>, deps: &GroupDeps, dag: &DependencyDag, rng: &mut SmallRng) {
    for _ in 0..5 {
        let roll = rng.gen_range(0..100);
        let changed = if roll < 40 {
            mut_adjacent_swap(seq, dag, rng)
        } else if roll < 70 {
            let max_steps = rng.gen_range(2..10);
            mut_bubble_move(seq, dag, rng, max_steps)
        } else {
            mut_segment_shuffle(seq, dag, rng)
        };
        if changed {
            return;
        }
    }
    // Last resort: generate a fresh random valid ordering.
    *seq = dag.sample_random_topo_sort(rng);
}

// ─── Competition ────────────────────────────────────────────────────────

pub fn compete(parent: &Individual, child: &Individual, rng: &mut SmallRng) -> Individual {
    if dominates(child, parent) {
        child.clone()
    } else if dominates(parent, child) {
        parent.clone()
    } else if rng.gen_bool(0.5) {
        child.clone()
    } else {
        parent.clone()
    }
}

// ─── Island model ───────────────────────────────────────────────────────

pub struct Island {
    pub population: Vec<Individual>,
    pub rng: SmallRng,
}

pub struct DCGenerationResult {
    pub new_best_hits: u32,
    pub children_who_won: usize,
    pub evals: u64,
}

// ─── Internal helpers ───────────────────────────────────────────────────

/// Build a map from order index to its position in a sequence.
fn build_rank_map(seq: &[usize]) -> HashMap<usize, usize> {
    let mut map = HashMap::default();
    for (i, &oi) in seq.iter().enumerate() {
        map.insert(oi, i);
    }
    map
}

/// Check whether `seq[i]` and `seq[i+1]` can be swapped without breaking
/// any DAG edge.
fn can_swap_adjacent(seq: &[usize], i: usize, dag: &DependencyDag) -> bool {
    let oi_a = seq[i];
    let oi_b = seq[i + 1];

    let ni_a = match dag.node_of.get(&oi_a) {
        Some(&ni) => ni,
        None => return true,
    };
    let ni_b = match dag.node_of.get(&oi_b) {
        Some(&ni) => ni,
        None => return true,
    };

    // Can't swap if there's a direct edge a→b (b depends on a).
    // Edge b→a would mean b is before a in the DAG, but b is *after* a in
    // the sequence, so that edge can't exist in a valid topo sort.
    !dag.successors[ni_a].contains(&ni_b)
}

/// Repair a segment `seq[start..=end]` to be a valid sub-topo-sort.
///
/// The elements outside the segment are fixed. We re-order the segment
/// elements to respect DAG edges among themselves, while keeping their
/// relative order with respect to elements outside the segment.
fn repair_segment(seq: &mut [usize], start: usize, end: usize, dag: &DependencyDag) {
    let segment: Vec<usize> = seq[start..=end].to_vec();
    let segment_set: HashSet<usize> = segment.iter().copied().collect();

    // Build a local precedence: for elements in the segment, which must
    // come before which? We only care about edges where both endpoints are
    // in the segment.
    let mut local_succs: HashMap<usize, Vec<usize>> = HashMap::default();
    let mut local_in_deg: HashMap<usize, usize> = HashMap::default();

    for &oi in &segment {
        local_in_deg.entry(oi).or_insert(0);
    }

    for &oi in &segment {
        if let Some(&ni) = dag.node_of.get(&oi) {
            for &succ_ni in &dag.successors[ni] {
                let succ_oi = dag.nodes[succ_ni];
                if segment_set.contains(&succ_oi) {
                    local_succs.entry(oi).or_default().push(succ_oi);
                    *local_in_deg.entry(succ_oi).or_insert(0) += 1;
                }
            }
        }
    }

    // Topo-sort the segment elements using Kahn's algorithm.
    // Use the shuffled order as tie-breaker (this preserves the mutation effect).
    let mut queue: Vec<usize> = segment
        .iter()
        .copied()
        .filter(|oi| local_in_deg.get(oi).copied().unwrap_or(0) == 0)
        .collect();

    let mut sorted = Vec::with_capacity(segment.len());
    while let Some(oi) = queue.pop() {
        sorted.push(oi);
        if let Some(succs) = local_succs.get(&oi) {
            for &s in succs {
                let deg = local_in_deg.get_mut(&s).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(s);
                }
            }
        }
    }

    // Write back.
    for (i, oi) in sorted.into_iter().enumerate() {
        seq[start + i] = oi;
    }
}

/// Softmax-weighted random selection. Lower scores are better (we negate).
fn softmax_sample(candidates: &[usize], scores: &[f64], temperature: f64, rng: &mut SmallRng) -> usize {
    // Compute weights: exp(-score / temperature), with numerical stability.
    let max_term = scores
        .iter()
        .map(|s| -s / temperature)
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = scores
        .iter()
        .map(|s| ((-s / temperature) - max_term).exp())
        .collect();

    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        // Fallback to uniform.
        return candidates[rng.gen_range(0..candidates.len())];
    }

    let r = rng.gen::<f64>() * sum;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if r <= acc {
            return candidates[i];
        }
    }
    *candidates.last().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{address, Address, Signature, TxHash, B256, U256};
    use rand::SeedableRng;
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use crate::building::builders::parallel_builder::ConflictGroup;
    use rbuilder_primitives::{
        Bundle, MempoolTx, Metadata, Order, SimValue, SimulatedOrder,
        TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
    };

    const SENDER_A: Address = address!("0x000000000000000000000000000000000000000a");
    const SENDER_B: Address = address!("0x000000000000000000000000000000000000000b");
    const SENDER_C: Address = address!("0x000000000000000000000000000000000000000c");

    struct IdGen(u64);
    impl IdGen {
        fn new() -> Self {
            Self(0)
        }
        fn next_hash(&mut self) -> TxHash {
            self.0 += 1;
            TxHash::from(U256::from(self.0))
        }
    }

    fn mk_tx(sender: Address, nonce: u64, gen: &mut IdGen) -> Recovered<TransactionSigned> {
        let tx_legacy = TxLegacy {
            nonce,
            ..Default::default()
        };
        Recovered::new_unchecked(
            TransactionSigned::new(
                Transaction::Legacy(tx_legacy),
                Signature::test_signature(),
                gen.next_hash(),
            ),
            sender,
        )
    }

    fn mk_single_tx_order(
        sender: Address,
        nonce: u64,
        profit: u64,
        gen: &mut IdGen,
    ) -> Arc<SimulatedOrder> {
        let rec = mk_tx(sender, nonce, gen);
        let with_blobs = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(rec).unwrap();
        Arc::new(SimulatedOrder {
            order: Arc::new(Order::Tx(MempoolTx {
                tx_with_blobs: with_blobs,
            })),
            used_state_trace: None,
            sim_value: SimValue::new_test(U256::from(profit), U256::from(profit), 0),
        })
    }

    fn mk_bundle_order(
        tx_specs: &[(Address, u64)],
        profit: u64,
        gen: &mut IdGen,
    ) -> Arc<SimulatedOrder> {
        let txs: Vec<_> = tx_specs
            .iter()
            .map(|&(sender, nonce)| {
                TransactionSignedEcRecoveredWithBlobs::new_no_blobs(mk_tx(sender, nonce, gen))
                    .unwrap()
            })
            .collect();
        let bundle = Bundle {
            version: LAST_BUNDLE_VERSION,
            block: Some(0),
            min_timestamp: None,
            max_timestamp: None,
            txs,
            reverting_tx_hashes: Vec::new(),
            dropping_tx_hashes: Vec::new(),
            hash: B256::ZERO,
            uuid: Uuid::new_v4(),
            replacement_data: None,
            signer: None,
            refund_identity: None,
            metadata: Metadata::default(),
            refund: None,
            external_hash: None,
        };
        Arc::new(SimulatedOrder {
            order: Arc::new(Order::Bundle(bundle)),
            used_state_trace: None,
            sim_value: SimValue::new_test(U256::from(profit), U256::from(profit), 0),
        })
    }

    fn mk_group(orders: Vec<Arc<SimulatedOrder>>) -> ConflictGroup {
        ConflictGroup {
            id: 0,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(HashSet::default()),
        }
    }

    /// Verify that a sequence is a valid topological sort of the DAG.
    fn assert_valid_topo_sort(ordering: &[usize], dag: &DependencyDag) {
        assert_eq!(ordering.len(), dag.len());
        let pos: HashMap<usize, usize> = ordering
            .iter()
            .enumerate()
            .map(|(p, &oi)| (oi, p))
            .collect();
        for (ni, succ_list) in dag.successors.iter().enumerate() {
            let from_oi = dag.nodes[ni];
            for &si in succ_list {
                let to_oi = dag.nodes[si];
                assert!(
                    pos[&from_oi] < pos[&to_oi],
                    "Order {} must come before {} but doesn't in {:?}",
                    from_oi,
                    to_oi,
                    ordering
                );
            }
        }
    }

    // ── Helper to build deps + dag for a group ──────────────────────

    fn deps_and_dag(group: &ConflictGroup) -> (GroupDeps, DependencyDag) {
        let deps = GroupDeps::from_group(group).unwrap();
        let dag = deps.build_dag_all();
        (deps, dag)
    }

    // ═════════════════════════════════════════════════════════════════
    // PPX crossover
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn ppx_greedy_produces_valid_ordering_single_txs() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);

        // Two different valid parents.
        let parent_a = vec![0, 2, 1, 3]; // A0, B0, A1, B1
        let parent_b = vec![2, 0, 3, 1]; // B0, A0, B1, A1

        let child = ppx_build_child_from_parents_greedy(&parent_a, &parent_b, &dag);
        assert_valid_topo_sort(&child, &dag);
    }

    #[test]
    fn ppx_greedy_produces_valid_ordering_with_bundles() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // idx 0
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // idx 1
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 300, &mut gen), // idx 2
            mk_single_tx_order(SENDER_C, 0, 200, &mut gen),                  // idx 3
        ]);
        let (_, dag) = deps_and_dag(&group);

        let parent_a = vec![0, 1, 2, 3];
        let parent_b = vec![3, 1, 0, 2];

        let child = ppx_build_child_from_parents_greedy(&parent_a, &parent_b, &dag);
        assert_valid_topo_sort(&child, &dag);

        // idx 2 must come after 0 and 1.
        let pos = |oi: usize| child.iter().position(|&x| x == oi).unwrap();
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(2));
    }

    #[test]
    fn ppx_stochastic_always_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let parent_a = vec![0, 2, 4, 1, 3];
        let parent_b = vec![4, 2, 0, 3, 1];

        for _ in 0..50 {
            let child = ppx_build_child_from_parents(&parent_a, &parent_b, &dag, &mut rng);
            assert_valid_topo_sort(&child, &dag);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Adapted order crossover
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn adapted_ox_produces_valid_ordering() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(1337);

        let parent_a = vec![0, 2, 1, 3];
        let parent_b = vec![2, 0, 3, 1];

        for _ in 0..100 {
            let child = adapted_order_crossover(&parent_a, &parent_b, &dag, &mut rng);
            assert_valid_topo_sort(&child, &dag);
            assert_eq!(child.len(), 4);

            // Must contain all orders.
            let set: HashSet<usize> = child.iter().copied().collect();
            assert_eq!(set.len(), 4);
        }
    }

    #[test]
    fn adapted_ox_with_bundles() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // 1
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 300, &mut gen), // 2
            mk_single_tx_order(SENDER_A, 2, 400, &mut gen),                  // 3
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let parent_a = vec![0, 1, 2, 3];
        let parent_b = vec![1, 0, 2, 3];

        for _ in 0..100 {
            let child = adapted_order_crossover(&parent_a, &parent_b, &dag, &mut rng);
            assert_valid_topo_sort(&child, &dag);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Mutation operators
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn adjacent_swap_preserves_validity() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(99);

        let mut seq = vec![0, 2, 1, 3];
        for _ in 0..50 {
            mut_adjacent_swap(&mut seq, &dag, &mut rng);
            assert_valid_topo_sort(&seq, &dag);
        }
    }

    #[test]
    fn bubble_move_preserves_validity() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(4242);

        let mut seq = vec![0, 2, 1, 3];
        for _ in 0..50 {
            mut_bubble_move(&mut seq, &dag, &mut rng, 3);
            assert_valid_topo_sort(&seq, &dag);
        }
    }

    #[test]
    fn segment_shuffle_preserves_validity() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);
        let (_, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(7777);

        let mut seq = vec![0, 2, 4, 1, 3];
        for _ in 0..50 {
            mut_segment_shuffle(&mut seq, &dag, &mut rng);
            assert_valid_topo_sort(&seq, &dag);
        }
    }

    #[test]
    fn mutate_with_bundles_preserves_validity() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // 1
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 300, &mut gen), // 2
            mk_single_tx_order(SENDER_C, 0, 200, &mut gen),                  // 3
        ]);
        let (deps, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(555);

        let mut seq = vec![0, 1, 2, 3];
        for _ in 0..100 {
            mutate(&mut seq, &deps, &dag, &mut rng);
            assert_valid_topo_sort(&seq, &dag);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Compete
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn compete_picks_dominant() {
        let mut rng = SmallRng::seed_from_u64(1);
        let better = Individual {
            seq: vec![0, 1],
            profit: U256::from(100),
            gas: 50,
        };
        let worse = Individual {
            seq: vec![1, 0],
            profit: U256::from(50),
            gas: 100,
        };

        for _ in 0..20 {
            let winner = compete(&worse, &better, &mut rng);
            assert_eq!(winner.profit, U256::from(100));
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Distance metric
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dc_distance_identity_is_zero() {
        let deps = GroupDeps {
            order_deps: vec![],
            slot_providers: HashMap::default(),
            n: 0,
        };
        let a = vec![0, 1, 2, 3];
        assert!((dc_distance(&a, &a, &deps) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn dc_distance_fully_different() {
        let deps = GroupDeps {
            order_deps: vec![],
            slot_providers: HashMap::default(),
            n: 0,
        };
        let a = vec![0, 1, 2, 3];
        let b = vec![3, 2, 1, 0];
        let d = dc_distance(&a, &b, &deps);
        assert!(d > 0.0);
        assert!(d <= 1.0);
    }

    #[test]
    fn dc_distance_symmetric() {
        let deps = GroupDeps {
            order_deps: vec![],
            slot_providers: HashMap::default(),
            n: 0,
        };
        let a = vec![0, 1, 2, 3];
        let b = vec![2, 0, 3, 1];
        assert!((dc_distance(&a, &b, &deps) - dc_distance(&b, &a, &deps)).abs() < 1e-12);
    }

    // ═════════════════════════════════════════════════════════════════
    // Diamond DAG stress test
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn operators_valid_on_diamond_dag() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0: root
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen), // 1: depends on 0
            mk_bundle_order(&[(SENDER_A, 2), (SENDER_C, 0)], 300, &mut gen), // 2: depends on 1
            mk_single_tx_order(SENDER_B, 1, 150, &mut gen),                  // 3: depends on 1
            mk_bundle_order(&[(SENDER_B, 2), (SENDER_C, 1)], 250, &mut gen), // 4: depends on 2,3
        ]);
        let (deps, dag) = deps_and_dag(&group);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Only 2 valid orderings: [0,1,2,3,4] and [0,1,3,2,4]
        let parent_a = vec![0, 1, 2, 3, 4];
        let parent_b = vec![0, 1, 3, 2, 4];

        for _ in 0..50 {
            let c1 = ppx_build_child_from_parents(&parent_a, &parent_b, &dag, &mut rng);
            assert_valid_topo_sort(&c1, &dag);

            let c2 = adapted_order_crossover(&parent_a, &parent_b, &dag, &mut rng);
            assert_valid_topo_sort(&c2, &dag);

            let mut seq = parent_a.clone();
            mutate(&mut seq, &deps, &dag, &mut rng);
            assert_valid_topo_sort(&seq, &dag);
        }
    }
}
