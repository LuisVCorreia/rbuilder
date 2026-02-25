use ahash::{HashMap, HashSet as AHashSet};
use alloy_primitives::U256;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;

use super::nonce_handling::{DependencyDag, GroupDeps, GreedyKey};
use crate::building::sim::NonceKey;

// ─── Individual representation ──────────────────────────────────────────

/// Maps each conflicting slot to the chosen candidate order index.
/// Non-conflicting slots (single provider) don't need entries here — they're
/// always included.
pub type CandidateChoices = HashMap<NonceKey, usize>;

#[derive(Clone)]
pub struct Individual {
    /// Raw GA genome: permutation of ALL orders (0..deps.n).
    /// Used for crossover + mutation.
    pub raw_seq: Vec<usize>,

    /// Decoded, valid execution order (conflict-free + topo-valid) derived from raw_seq.
    /// Used for fitness eval + distance.
    pub seq: Vec<usize>,

    /// Active set of orders included by decoding (subset of 0..deps.n).
    pub active: AHashSet<usize>,

    /// For each conflicting slot, which candidate was chosen (derived from decoded seq).
    pub choices: CandidateChoices,

    /// Fitness values (filled after evaluation).
    pub profit: U256,
    pub gas: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct GAParams {
    pub population: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub tourn_k: usize,
    pub max_generations: usize,
    pub time_ms: u64,
    pub seed: u64,
    pub num_islands: usize,
    pub migration_interval: usize,
}

pub struct Island {
    pub population: Vec<Individual>,
    pub rng: SmallRng,
}

pub fn dominates(a: &Individual, b: &Individual) -> bool {
    a.profit > b.profit || (a.profit == b.profit && a.gas < b.gas)
}

// ─── Choice resolution helpers ──────────────────────────────────────────

/// Given candidate choices, compute the full active set.
/// This includes all non-conflicting orders plus chosen candidates,
/// minus anything excluded by the chosen candidates' slot claims.
pub fn active_set_from_choices(
    deps: &GroupDeps,
    choices: &CandidateChoices,
) -> AHashSet<usize> {
    // Start with all orders.
    let mut excluded: AHashSet<usize> = AHashSet::default();

    // For each conflicting slot, exclude non-chosen candidates.
    for (slot, providers) in &deps.slot_providers {
        if providers.len() <= 1 {
            continue;
        }
        if let Some(&chosen) = choices.get(slot) {
            for &other in providers {
                if other != chosen {
                    excluded.insert(other);
                }
            }
        }
    }

    // Also handle bundle atomicity: if a chosen candidate provides multiple
    // slots, all rival providers for ALL those slots must be excluded.
    // (This is already handled above if choices is consistent, but let's
    // be defensive.)
    let chosen_orders: AHashSet<usize> = choices.values().copied().collect();
    for &chosen_idx in &chosen_orders {
        for slot in &deps.order_deps[chosen_idx].provides {
            if let Some(providers) = deps.slot_providers.get(slot) {
                for &other in providers {
                    if other != chosen_idx {
                        excluded.insert(other);
                    }
                }
            }
        }
    }

    (0..deps.n).filter(|i| !excluded.contains(i)).collect()
}

pub fn choices_from_decoded_seq(seq: &[usize], deps: &GroupDeps) -> CandidateChoices {
    let mut choices = CandidateChoices::default();

    for &oi in seq {
        for slot in &deps.order_deps[oi].provides {
            if deps
                .slot_providers
                .get(slot)
                .map_or(false, |p| p.len() > 1)
            {
                // First provider encountered in execution order wins.
                choices.entry(slot.clone()).or_insert(oi);
            }
        }
    }

    choices
}

/// Decode a raw permutation into a valid schedule:
/// 1) Greedy "packing" pass: include an order iff none of its provided slots are already taken.
///    - This resolves duplicate nonces and enforces bundle atomicity naturally.
/// 2) Build DAG over included orders.
/// 3) Repair topo sort biased by the packed order (preserve raw priorities where possible).
pub fn decode_raw_sequence(raw: &[usize], deps: &GroupDeps) -> (AHashSet<usize>, Vec<usize>, CandidateChoices) {
    let mut taken: AHashSet<NonceKey> = AHashSet::default();
    let mut packed: Vec<usize> = Vec::new();

    for &oi in raw {
        let provides = &deps.order_deps[oi].provides;

        // If any slot already taken, we cannot include this order.
        if provides.iter().any(|s| taken.contains(s)) {
            continue;
        }

        // Include it; claim all its slots.
        for s in provides {
            taken.insert(s.clone());
        }
        packed.push(oi);
    }

    let active: AHashSet<usize> = packed.iter().copied().collect();
    if active.is_empty() {
        return (active, Vec::new(), CandidateChoices::default());
    }

    let dag = deps.build_dag(&active);
    let decoded = repair_topo_sort(&packed, &dag);
    let choices = choices_from_decoded_seq(&decoded, deps);

    (active, decoded, choices)
}

pub fn individual_from_raw_seq(raw_seq: Vec<usize>, deps: &GroupDeps) -> Individual {
    let (active, seq, choices) = decode_raw_sequence(&raw_seq, deps);

    Individual {
        raw_seq,
        seq,
        active,
        choices,
        profit: U256::ZERO,
        gas: 0,
    }
}

pub fn crossover_simple_ox(
    parent_a: &Individual,
    parent_b: &Individual,
    deps: &GroupDeps,
    rng: &mut SmallRng,
) -> Individual {
    let n = deps.n;
    debug_assert_eq!(parent_a.raw_seq.len(), n);
    debug_assert_eq!(parent_b.raw_seq.len(), n);

    if n <= 1 {
        return individual_from_raw_seq(parent_a.raw_seq.clone(), deps);
    }

    let lo = rng.gen_range(0..n);
    let hi = rng.gen_range(lo..=n);

    // Core segment from A
    let mut child = vec![usize::MAX; n];
    let mut in_child: AHashSet<usize> = AHashSet::default();

    for i in lo..hi {
        let g = parent_a.raw_seq[i];
        child[i] = g;
        in_child.insert(g);
    }

    // Fill remaining positions in order from B, skipping already present genes.
    let mut write = hi % n;
    for &g in &parent_b.raw_seq {
        if in_child.contains(&g) {
            continue;
        }
        // Find next empty slot
        while child[write] != usize::MAX {
            write = (write + 1) % n;
        }
        child[write] = g;
        write = (write + 1) % n;
    }

    // Decode
    individual_from_raw_seq(child, deps)
}

pub fn mutate_simple_swap(
    ind: &mut Individual,
    deps: &GroupDeps,
    rng: &mut SmallRng,
    mutation_rate: f64,
) {
    let n = deps.n;
    if n <= 1 {
        return;
    }

    // Number of swaps ~ mutation_rate * n (at least 1 sometimes).
    let expected = mutation_rate * (n as f64);
    let num_swaps = if expected < 1.0 {
        if rng.gen::<f64>() < expected { 1 } else { 0 }
    } else {
        expected.round() as usize
    };

    for _ in 0..num_swaps {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        ind.raw_seq.swap(i, j);
    }

    // Re-decode after mutation
    let (active, seq, choices) = decode_raw_sequence(&ind.raw_seq, deps);
    ind.active = active;
    ind.seq = seq;
    ind.choices = choices;
}

// /// Build a valid Individual from just a set of candidate choices.
// /// Constructs the DAG and produces a random topo sort.
// pub fn individual_from_choices(
//     choices: CandidateChoices,
//     deps: &GroupDeps,
//     rng: &mut SmallRng,
// ) -> Individual {
//     let active = active_set_from_choices(deps, &choices);
//     let dag = deps.build_dag(&active);
//     let seq = dag.sample_random_topo_sort(rng);

//     Individual {
//         choices,
//         active,
//         seq,
//         profit: U256::ZERO,
//         gas: 0,
//     }
// }

// /// Build a valid Individual from a sequence (extracts choices from it).
// pub fn individual_from_seq(seq: Vec<usize>, deps: &GroupDeps) -> Individual {
//     let choices = choices_from_seq(&seq, deps);
//     let active = active_set_from_choices(deps, &choices);

//     Individual {
//         choices,
//         active,
//         seq,
//         profit: U256::ZERO,
//         gas: 0,
//     }
// }

/// Validate that an individual's sequence is a valid topo sort of its DAG.
/// Used in debug/test builds.
#[cfg(debug_assertions)]
pub fn validate_individual(ind: &Individual, deps: &GroupDeps) -> bool {
    // Check seq contains exactly the active set.
    let seq_set: AHashSet<usize> = ind.seq.iter().copied().collect();
    if seq_set != ind.active {
        return false;
    }

    // Check it's a valid topo sort.
    let dag = deps.build_dag(&ind.active);
    let pos: HashMap<usize, usize> = ind.seq.iter().enumerate()
        .map(|(p, &oi)| (oi, p)).collect();

    for (ni, succs) in dag.successors.iter().enumerate() {
        let from = dag.nodes[ni];
        for &si in succs {
            let to = dag.nodes[si];
            if pos.get(&from).copied().unwrap_or(0) >= pos.get(&to).copied().unwrap_or(0) {
                return false;
            }
        }
    }

    // Check no two orders in seq compete for the same slot.
    for (slot, providers) in &deps.slot_providers {
        if providers.len() <= 1 {
            continue;
        }
        let count = providers.iter().filter(|idx| seq_set.contains(idx)).count();
        if count > 1 {
            return false;
        }
    }

    true
}

// ─── ReadyTracker (reused from before, but now always on a correctly-built DAG) ─

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

    #[inline]
    fn is_ready_node(&self, ni: usize) -> bool {
        !self.placed[ni] && self.in_deg[ni] == 0
    }

    #[inline]
    fn is_ready_order(&self, oi: usize) -> bool {
        if let Some(&ni) = self.dag.node_of.get(&oi) {
            self.is_ready_node(ni)
        } else {
            false
        }
    }

    fn ready_orders(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&ni| self.is_ready_node(ni))
            .map(|ni| self.dag.nodes[ni])
            .collect()
    }

    fn place(&mut self, oi: usize) {
        let ni = self.dag.node_of[&oi];
        debug_assert!(self.is_ready_node(ni), "placing non-ready order {} (node {})", oi, ni);
        self.placed[ni] = true;
        for &succ in &self.dag.successors[ni] {
            self.in_deg[succ] -= 1;
        }
    }

    #[inline]
    fn is_placed_order(&self, oi: usize) -> bool {
        if let Some(&ni) = self.dag.node_of.get(&oi) {
            self.placed[ni]
        } else {
            true // not in DAG, treat as already placed
        }
    }
}

// // ─── Crossover ──────────────────────────────────────────────────────────
// //
// // The crossover works in two phases:
// //   Phase 1: Merge candidate choices from both parents (nonce conflict resolution).
// //   Phase 2: Two-point Order Crossover (OX) on the ordering, repaired to be
// //            a valid topological sort.

// /// Main crossover operator.
// ///
// /// 1. Merge candidate choices (randomly pick from parent A or B for each slot).
// /// 2. Build the child's active set and DAG from the merged choices.
// /// 3. Apply two-point OX on both parents' filtered orderings, then repair the
// ///    result into a valid topological sort (using repair_topo_sort).
// pub fn crossover(
//     parent_a: &Individual,
//     parent_b: &Individual,
//     deps: &GroupDeps,
//     rng: &mut SmallRng,
// ) -> Individual {
//     // Phase 1: Merge candidate choices.
//     let choices = merge_choices(
//         &parent_a.choices,
//         &parent_b.choices,
//         deps,
//         rng,
//     );

//     let active = active_set_from_choices(deps, &choices);
//     let dag = deps.build_dag(&active);

//     if dag.is_empty() {
//         return Individual {
//             choices,
//             active,
//             seq: Vec::new(),
//             profit: U256::ZERO,
//             gas: 0,
//         };
//     }

//     // Phase 2: Two-point Order Crossover (OX) on filtered sequences.
//     //
//     // Filter both parents' sequences to only the orders in the child's active set.
//     // Every child-active order appears in at least one parent's filtered sequence
//     // (since each choice came from either parent A or parent B).
//     let pa: Vec<usize> = parent_a.seq.iter().copied()
//         .filter(|oi| active.contains(oi))
//         .collect();
//     let pb: Vec<usize> = parent_b.seq.iter().copied()
//         .filter(|oi| active.contains(oi))
//         .collect();

//     let n = active.len();
//     let seq = if n == 0 || pa.is_empty() {
//         // Parent A contributed no ordering info — use parent B's ordering.
//         if pb.is_empty() {
//             dag.sample_random_topo_sort(rng)
//         } else {
//             repair_topo_sort(&pb, &dag)
//         }
//     } else {
//         // Pick two crossover points within parent A's filtered sequence.
//         let lo = rng.gen_range(0..pa.len());
//         let hi = rng.gen_range(lo..=pa.len());

//         // Core: the segment [lo, hi) from parent A.
//         let core: Vec<usize> = pa[lo..hi].to_vec();
//         let core_set: AHashSet<usize> = core.iter().copied().collect();

//         // Remainder: orders not in core, taken from parent B first (in B's order),
//         // then any active orders not covered by either (from parent A, in A's order).
//         let mut remainder: Vec<usize> = Vec::with_capacity(n.saturating_sub(core.len()));
//         let mut covered: AHashSet<usize> = core_set.clone();

//         for &oi in &pb {
//             if !core_set.contains(&oi) {
//                 remainder.push(oi);
//                 covered.insert(oi);
//             }
//         }
//         for &oi in &pa {
//             if !covered.contains(&oi) {
//                 remainder.push(oi);
//             }
//         }

//         // Classic OX layout: remainder[..lo] | core | remainder[lo..]
//         let split = lo.min(remainder.len());
//         let mut proposed = Vec::with_capacity(n);
//         proposed.extend_from_slice(&remainder[..split]);
//         proposed.extend_from_slice(&core);
//         proposed.extend_from_slice(&remainder[split..]);

//         // Repair to a valid topological sort while preserving proposed order.
//         repair_topo_sort(&proposed, &dag)
//     };

//     Individual {
//         choices,
//         active,
//         seq,
//         profit: U256::ZERO,
//         gas: 0,
//     }
// }

// /// Merge candidate choices from two parents.
// ///
// /// For each conflicting slot:
// /// - If both parents chose the same candidate, keep it.
// /// - If they differ, randomly pick one (50/50).
// /// - If only one parent has a choice (the other excluded it), use that one.
// ///
// /// After initial merge, we need to fix inconsistencies from bundle atomicity:
// /// if choosing candidate X for slot S forces candidate X for slot T (because X
// /// is a bundle providing both S and T), we must respect that.
// fn merge_choices(
//     choices_a: &CandidateChoices,
//     choices_b: &CandidateChoices,
//     deps: &GroupDeps,
//     rng: &mut SmallRng,
// ) -> CandidateChoices {
//     let mut merged = CandidateChoices::default();

//     // Collect all conflicting slots.
//     let all_slots: AHashSet<NonceKey> = deps.slot_providers.iter()
//         .filter(|(_, providers)| providers.len() > 1)
//         .map(|(slot, _)| slot.clone())
//         .collect();

//     // Slots already decided (by bundle atomicity propagation).
//     let mut decided_slots: AHashSet<NonceKey> = AHashSet::default();
//     // Orders already chosen (to detect conflicts).
//     let mut chosen_orders: AHashSet<usize> = AHashSet::default();
//     // Orders excluded by chosen bundles.
//     let mut excluded_orders: AHashSet<usize> = AHashSet::default();

//     // Process slots in random order to avoid bias.
//     let mut slot_list: Vec<NonceKey> = all_slots.into_iter().collect();
//     slot_list.shuffle(rng);

//     for slot in &slot_list {
//         if decided_slots.contains(slot) {
//             continue;
//         }

//         let ca = choices_a.get(slot).copied();
//         let cb = choices_b.get(slot).copied();

//         // Filter out already-excluded candidates.
//         let ca = ca.filter(|&c| !excluded_orders.contains(&c));
//         let cb = cb.filter(|&c| !excluded_orders.contains(&c));

//         let chosen = match (ca, cb) {
//             (Some(a), Some(b)) if a == b => Some(a),
//             (Some(a), Some(b)) => {
//                 // Both valid but different — pick randomly.
//                 Some(if rng.gen_bool(0.5) { a } else { b })
//             }
//             (Some(a), None) => Some(a),
//             (None, Some(b)) => Some(b),
//             (None, None) => {
//                 // Neither parent had a valid choice. Pick from available providers.
//                 if let Some(providers) = deps.slot_providers.get(slot) {
//                     let eligible: Vec<usize> = providers.iter()
//                         .copied()
//                         .filter(|&p| !excluded_orders.contains(&p))
//                         .collect();
//                     eligible.choose(rng).copied()
//                 } else {
//                     None
//                 }
//             }
//         };

//         if let Some(chosen_idx) = chosen {
//             // Record this choice and propagate bundle atomicity.
//             chosen_orders.insert(chosen_idx);

//             // For every slot this order provides, mark it as decided
//             // and exclude rival candidates.
//             for provided_slot in &deps.order_deps[chosen_idx].provides {
//                 if let Some(providers) = deps.slot_providers.get(provided_slot) {
//                     if providers.len() > 1 {
//                         merged.insert(provided_slot.clone(), chosen_idx);
//                         decided_slots.insert(provided_slot.clone());

//                         for &rival in providers {
//                             if rival != chosen_idx {
//                                 excluded_orders.insert(rival);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     merged
// }

/// Repair a proposed sequence to produce a valid topological sort of `dag`.
///
/// At each step, among all "ready" orders (predecessors already placed),
/// picks the one appearing earliest in `proposed`. This maximises preservation
/// of the proposed ordering while guaranteeing topological validity.
///
/// Orders in the DAG but absent from `proposed` are treated as having the
/// lowest priority (placed as late as valid constraints allow).
fn repair_topo_sort(proposed: &[usize], dag: &DependencyDag) -> Vec<usize> {
    let n = dag.len();
    if n == 0 {
        return Vec::new();
    }

    let mut tracker = ReadyTracker::new(dag);
    let mut result = Vec::with_capacity(n);

    // Position map: order index → position in proposed (absent = usize::MAX).
    let pos: HashMap<usize, usize> = proposed.iter().enumerate()
        .map(|(i, &oi)| (oi, i))
        .collect();

    while result.len() < n {
        let ready = tracker.ready_orders();
        debug_assert!(!ready.is_empty(), "No ready orders; DAG has a cycle");

        // Among ready orders, pick the one appearing earliest in proposed.
        let best = *ready.iter()
            .min_by_key(|&&oi| pos.get(&oi).copied().unwrap_or(usize::MAX))
            .unwrap();

        result.push(best);
        tracker.place(best);
    }

    result
}

// ─── Mutation operators ─────────────────────────────────────────────────

/// Per-gene mutation that operates at both levels:
/// - Candidate-level: flip a conflicting slot to a different candidate
/// - Order-level: adjacent swap or bubble move
///
/// After any candidate flip, the DAG changes, so we rebuild and repair.
pub fn mutate(
    ind: &mut Individual,
    deps: &GroupDeps,
    rng: &mut SmallRng,
    mutation_rate: f64,
) {
    let n = ind.seq.len();
    if n == 0 {
        return;
    }

    let expected = (mutation_rate * n as f64).max(1.0);
    let num_mutations = if expected < 1.0 {
        if rng.gen::<f64>() < expected { 1 } else { 0 }
    } else {
        expected.round() as usize
    };

    let has_conflicts = deps.has_conflicts();

    for _ in 0..num_mutations {
        let roll = rng.gen_range(0..100);

        if has_conflicts && roll < 25 {
            // Candidate-level mutation: flip a conflicting slot.
            mut_candidate_flip(ind, deps, rng);
        } else if roll < 55 {
            // Order-level mutation: adjacent swap.
            let dag = deps.build_dag(&ind.active);
            mut_adjacent_swap(&mut ind.seq, &dag, rng);
        } else {
            // Order-level mutation: bubble move.
            let dag = deps.build_dag(&ind.active);
            let max_steps = rng.gen_range(2..=(n / 4).max(3));
            mut_bubble_move(&mut ind.seq, &dag, rng, max_steps);
        }
    }
}

/// Flip a candidate at a conflicting slot and rebuild the sequence.
fn mut_candidate_flip(
    ind: &mut Individual,
    deps: &GroupDeps,
    rng: &mut SmallRng,
) {
    let conflicts = deps.conflicting_slots();
    if conflicts.is_empty() {
        return;
    }

    let (slot, candidates) = &conflicts[rng.gen_range(0..conflicts.len())];

    let current = ind.choices.get(slot).copied();
    let alts: Vec<usize> = candidates.iter().copied()
        .filter(|&c| Some(c) != current)
        .collect();

    if alts.is_empty() {
        return;
    }

    let new_choice = alts[rng.gen_range(0..alts.len())];

    // Build updated choices:
    let mut new_choices = ind.choices.clone();

    // 1. Remove every slot claimed by the old candidate (if any).
    if let Some(old) = current {
        for provided_slot in &deps.order_deps[old].provides {
            if deps.slot_providers.get(provided_slot).map_or(false, |p| p.len() > 1) {
                new_choices.remove(provided_slot);
            }
        }
    }

    // 2. For each slot the new candidate provides: if another order currently
    //    claims that slot, evict it (remove all of its claimed slots) first,
    //    then claim the slot for new_choice.
    //    This correctly handles bundles that provide multiple slots.
    for provided_slot in &deps.order_deps[new_choice].provides {
        if deps.slot_providers.get(provided_slot).map_or(false, |p| p.len() > 1) {
            if let Some(&existing) = new_choices.get(provided_slot) {
                if existing != new_choice {
                    // Evict the conflicting order's entire claim set.
                    let existing_provides = deps.order_deps[existing].provides.clone();
                    for ep in &existing_provides {
                        if deps.slot_providers.get(ep).map_or(false, |p| p.len() > 1) {
                            new_choices.remove(ep);
                        }
                    }
                }
            }
            new_choices.insert(provided_slot.clone(), new_choice);
        }
    }

    // Rebuild active set and repair the sequence to match the new DAG.
    ind.choices = new_choices;
    ind.active = active_set_from_choices(deps, &ind.choices);
    let dag = deps.build_dag(&ind.active);
    ind.seq = repair_topo_sort(&ind.seq, &dag);
}


/// Swap two adjacent orders that have no dependency between them.
fn mut_adjacent_swap(seq: &mut [usize], dag: &DependencyDag, rng: &mut SmallRng) -> bool {
    if seq.len() < 2 {
        return false;
    }
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
fn mut_bubble_move(
    seq: &mut [usize],
    dag: &DependencyDag,
    rng: &mut SmallRng,
    max_steps: usize,
) -> bool {
    if seq.len() < 2 {
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
    !dag.successors[ni_a].contains(&ni_b)
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

// ─── Distance metrics ───────────────────────────────────────────────────

/// Normalized Kendall-tau distance for sequences that may have different elements.
/// Only counts inversions among shared elements.
fn kendall_tau_normalized(a: &[usize], b: &[usize]) -> f64 {
    // Find shared elements.
    let a_set: AHashSet<usize> = a.iter().copied().collect();
    let b_set: AHashSet<usize> = b.iter().copied().collect();
    let shared: Vec<usize> = a.iter().copied().filter(|x| b_set.contains(x)).collect();

    let n = shared.len();
    if n <= 1 {
        return 0.0;
    }

    // Build position map for b (among shared elements only).
    let shared_set: AHashSet<usize> = shared.iter().copied().collect();
    let mut pos_in_b = HashMap::default();
    let mut rank = 0usize;
    for &oi in b {
        if shared_set.contains(&oi) {
            pos_in_b.insert(oi, rank);
            rank += 1;
        }
    }

    // Map a's shared elements to their position in b.
    let mut mapped: Vec<usize> = Vec::with_capacity(n);
    for &oi in a {
        if let Some(&pos) = pos_in_b.get(&oi) {
            mapped.push(pos);
        }
    }

    let mut buf = vec![0usize; n];
    let inv = count_inversions(&mut mapped, &mut buf) as f64;
    let max_inv = (n as f64) * ((n as f64) - 1.0) / 2.0;
    if max_inv == 0.0 { 0.0 } else { inv / max_inv }
}

fn count_inversions(arr: &mut [usize], buf: &mut [usize]) -> u64 {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let (left, right) = arr.split_at_mut(mid);
    let (buf_left, buf_right) = buf.split_at_mut(mid);

    let inv_left = count_inversions(left, buf_left);
    let inv_right = count_inversions(right, buf_right);

    buf[..mid].copy_from_slice(left);
    buf[mid..].copy_from_slice(right);

    let mut i = 0;
    let mut j = mid;
    let mut k = 0;
    let mut inv = inv_left + inv_right;

    while i < mid && j < n {
        if buf[i] <= buf[j] {
            arr[k] = buf[i];
            i += 1;
        } else {
            arr[k] = buf[j];
            j += 1;
            inv += (mid - i) as u64;
        }
        k += 1;
    }
    while i < mid {
        arr[k] = buf[i];
        i += 1;
        k += 1;
    }
    while j < n {
        arr[k] = buf[j];
        j += 1;
        k += 1;
    }

    inv
}

/// Fraction of conflicting slots where a and b chose different candidates.
fn candidate_mismatch_rate(a: &Individual, b: &Individual, deps: &GroupDeps) -> f64 {
    if !deps.has_conflicts() {
        return 0.0;
    }

    let mut mismatches = 0usize;
    let mut total = 0usize;

    for (slot, providers) in &deps.slot_providers {
        if providers.len() <= 1 {
            continue;
        }
        total += 1;
        let pick_a = a.choices.get(slot);
        let pick_b = b.choices.get(slot);
        if pick_a != pick_b {
            mismatches += 1;
        }
    }

    if total == 0 { 0.0 } else { mismatches as f64 / total as f64 }
}

/// Combined distance for deterministic crowding.
pub fn dc_distance(a: &Individual, b: &Individual, deps: &GroupDeps) -> f64 {
    if a.seq.is_empty() || b.seq.is_empty() {
        return 1.0;
    }
    let w_order = 0.7;
    let w_choice = 0.3;
    let tau = kendall_tau_normalized(&a.seq, &b.seq);
    let idm = candidate_mismatch_rate(a, b, deps);
    w_order * tau + w_choice * idm
}

// ─── Deterministic Crowding generation step ─────────────────────────────

/// An unevaluated child with metadata for DC competition.
pub struct PendingChild {
    pub island_idx: usize,
    pub pair_idx: usize,
    pub child_slot: usize,
    pub ind: Individual,  // has seq but profit/gas not yet filled
}

/// Metadata about a parent pair for DC competition.
pub struct ParentPairInfo {
    pub p1_idx: usize,
    pub p2_idx: usize,
}

/// Generate all children for one island without evaluating them.
pub fn generate_dc_children(
    island: &mut Island,
    island_idx: usize,
    params: &GAParams,
    deps: &GroupDeps,
) -> (Vec<PendingChild>, Vec<ParentPairInfo>) {
    let population = &island.population;
    let rng = &mut island.rng;

    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.shuffle(rng);

    let mut children = Vec::new();
    let mut pair_infos = Vec::new();

    for pair_idx in 0..(population.len() / 2) {
        let i = pair_idx * 2;
        if i + 1 >= indices.len() {
            break;
        }

        let p1_idx = indices[i];
        let p2_idx = indices[i + 1];
        let (p1, p2) = (&population[p1_idx], &population[p2_idx]);

        let (mut c1, mut c2) = if rng.gen::<f64>() < params.crossover_rate {
            (
                crossover_simple_ox(p1, p2, deps, rng),
                crossover_simple_ox(p2, p1, deps, rng),
            )
        } else {
            (p1.clone(), p2.clone())
        };

        mutate_simple_swap(&mut c1, deps, rng, params.mutation_rate);
        mutate_simple_swap(&mut c2, deps, rng, params.mutation_rate);

        children.push(PendingChild { island_idx, pair_idx, child_slot: 0, ind: c1 });
        children.push(PendingChild { island_idx, pair_idx, child_slot: 1, ind: c2 });
        pair_infos.push(ParentPairInfo { p1_idx, p2_idx });
    }

    (children, pair_infos)
}

/// Apply DC competition using evaluated children.
pub fn apply_dc_competition(
    island: &mut Island,
    pair_infos: &[ParentPairInfo],
    evaluated_children: &mut Vec<Option<Individual>>,
    deps: &GroupDeps,
) {
    let population = &island.population;
    let rng = &mut island.rng;
    let mut next_population = Vec::with_capacity(population.len());

    for (pair_idx, info) in pair_infos.iter().enumerate() {
        let c1 = evaluated_children[pair_idx * 2].take();
        let c2 = evaluated_children[pair_idx * 2 + 1].take();

        let (c1, c2) = match (c1, c2) {
            (Some(c1), Some(c2)) => (c1, c2),
            _ => {
                next_population.push(population[info.p1_idx].clone());
                next_population.push(population[info.p2_idx].clone());
                continue;
            }
        };

        let p1 = &population[info.p1_idx];
        let p2 = &population[info.p2_idx];

        let dist_p1c1 = dc_distance(p1, &c1, deps);
        let dist_p2c2 = dc_distance(p2, &c2, deps);
        let dist_p1c2 = dc_distance(p1, &c2, deps);
        let dist_p2c1 = dc_distance(p2, &c1, deps);

        let (winner1, winner2) = if dist_p1c1 + dist_p2c2 <= dist_p1c2 + dist_p2c1 {
            (compete(p1, &c1, rng), compete(p2, &c2, rng))
        } else {
            (compete(p1, &c2, rng), compete(p2, &c1, rng))
        };

        next_population.push(winner1);
        next_population.push(winner2);
    }

    if population.len() % 2 == 1 {
        next_population.push(population.last().unwrap().clone());
    }

    island.population = next_population;
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::sync::Arc;

//     use ahash::HashSet;
//     use alloy_consensus::TxLegacy;
//     use alloy_primitives::{address, Address, Signature, TxHash, B256, U256};
//     use rand::SeedableRng;
//     use reth::primitives::TransactionSigned;
//     use reth_primitives::{Recovered, Transaction};
//     use uuid::Uuid;

//     // Adjust these imports to match your actual crate structure:
//     use crate::building::builders::parallel_builder::ConflictGroup;
//     use rbuilder_primitives::{
//         Bundle, MempoolTx, Metadata, Order, SimValue, SimulatedOrder,
//         TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
//     };
//     use crate::building::sim::NonceKey;

//     const SENDER_A: Address = address!("0x000000000000000000000000000000000000000a");
//     const SENDER_B: Address = address!("0x000000000000000000000000000000000000000b");
//     const SENDER_C: Address = address!("0x000000000000000000000000000000000000000c");

//     struct IdGen(u64);
//     impl IdGen {
//         fn new() -> Self { Self(0) }
//         fn next_hash(&mut self) -> TxHash {
//             self.0 += 1;
//             TxHash::from(U256::from(self.0))
//         }
//     }

//     fn mk_tx(sender: Address, nonce: u64, gen: &mut IdGen) -> Recovered<TransactionSigned> {
//         let tx_legacy = TxLegacy { nonce, ..Default::default() };
//         Recovered::new_unchecked(
//             TransactionSigned::new(
//                 Transaction::Legacy(tx_legacy),
//                 Signature::test_signature(),
//                 gen.next_hash(),
//             ),
//             sender,
//         )
//     }

//     fn mk_single_tx_order(
//         sender: Address,
//         nonce: u64,
//         profit: u64,
//         gen: &mut IdGen,
//     ) -> Arc<SimulatedOrder> {
//         let rec = mk_tx(sender, nonce, gen);
//         let with_blobs = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(rec).unwrap();
//         Arc::new(SimulatedOrder {
//             order: Arc::new(Order::Tx(MempoolTx { tx_with_blobs: with_blobs })),
//             used_state_trace: None,
//             sim_value: SimValue::new_test(U256::from(profit), U256::from(profit), 0),
//         })
//     }

//     fn mk_bundle_order(
//         tx_specs: &[(Address, u64)],
//         profit: u64,
//         gen: &mut IdGen,
//     ) -> Arc<SimulatedOrder> {
//         let txs: Vec<_> = tx_specs.iter()
//             .map(|&(sender, nonce)| {
//                 TransactionSignedEcRecoveredWithBlobs::new_no_blobs(mk_tx(sender, nonce, gen)).unwrap()
//             })
//             .collect();
//         let bundle = Bundle {
//             version: LAST_BUNDLE_VERSION,
//             block: Some(0),
//             min_timestamp: None,
//             max_timestamp: None,
//             txs,
//             reverting_tx_hashes: Vec::new(),
//             dropping_tx_hashes: Vec::new(),
//             hash: B256::ZERO,
//             uuid: Uuid::new_v4(),
//             replacement_data: None,
//             signer: None,
//             refund_identity: None,
//             metadata: Metadata::default(),
//             refund: None,
//             external_hash: None,
//         };
//         Arc::new(SimulatedOrder {
//             order: Arc::new(Order::Bundle(bundle)),
//             used_state_trace: None,
//             sim_value: SimValue::new_test(U256::from(profit), U256::from(profit), 0),
//         })
//     }

//     fn mk_group(orders: Vec<Arc<SimulatedOrder>>) -> ConflictGroup {
//         ConflictGroup {
//             id: 0,
//             orders: Arc::new(orders),
//             conflicting_group_ids: Arc::new(HashSet::default()),
//         }
//     }

//     fn assert_valid_individual(ind: &Individual, deps: &GroupDeps) {
//         // 1. Seq contains exactly the active set.
//         let seq_set: AHashSet<usize> = ind.seq.iter().copied().collect();
//         assert_eq!(seq_set, ind.active,
//             "Seq elements {:?} don't match active set {:?}", seq_set, ind.active);
//         assert_eq!(ind.seq.len(), ind.active.len(),
//             "Seq has duplicates: {:?}", ind.seq);

//         // 2. No two orders in seq compete for the same slot.
//         for (slot, providers) in &deps.slot_providers {
//             if providers.len() <= 1 { continue; }
//             let count = providers.iter().filter(|idx| seq_set.contains(idx)).count();
//             assert!(count <= 1,
//                 "Slot {:?} has {} providers in seq: {:?}", slot, count, ind.seq);
//         }

//         // 3. Valid topo sort.
//         let dag = deps.build_dag(&ind.active);
//         let pos: HashMap<usize, usize> = ind.seq.iter().enumerate()
//             .map(|(p, &oi)| (oi, p)).collect();
//         for (ni, succs) in dag.successors.iter().enumerate() {
//             let from = dag.nodes[ni];
//             for &si in succs {
//                 let to = dag.nodes[si];
//                 assert!(pos[&from] < pos[&to],
//                     "{} must come before {} in {:?}", from, to, ind.seq);
//             }
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // Basic: no conflicts
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn crossover_no_conflicts_preserves_validity() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
//             mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
//             mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
//             mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(42);

//         let pa = individual_from_seq(vec![0, 2, 1, 3], &deps);
//         let pb = individual_from_seq(vec![2, 0, 3, 1], &deps);

//         for _ in 0..100 {
//             let child = crossover(&pa, &pb, &deps, &mut rng);
//             assert_valid_individual(&child, &deps);
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // With conflicts: different candidates in parents
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn crossover_with_conflicts_always_valid() {
//         let mut gen = IdGen::new();
//         // Two candidates for A@0, plus A@1 and B@0.
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
//             mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1 (conflicts with 0)
//             mk_single_tx_order(SENDER_A, 1, 300, &mut gen), // idx 2
//             mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // idx 3
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(42);

//         // Parent A chose idx 0 for A@0.
//         let pa = individual_from_seq(vec![0, 3, 2], &deps);
//         // Parent B chose idx 1 for A@0.
//         let pb = individual_from_seq(vec![3, 1, 2], &deps);

//         assert_eq!(pa.seq.len(), 3);
//         assert_eq!(pb.seq.len(), 3);

//         for _ in 0..200 {
//             let child = crossover(&pa, &pb, &deps, &mut rng);
//             assert_valid_individual(&child, &deps);
//             // Child must have exactly one of {0, 1}.
//             let has_0 = child.seq.contains(&0);
//             let has_1 = child.seq.contains(&1);
//             assert!(has_0 ^ has_1, "Child must have exactly one A@0 candidate: {:?}", child.seq);
//             // Must always have 2 and 3.
//             assert!(child.seq.contains(&2));
//             assert!(child.seq.contains(&3));
//         }
//     }

//     #[test]
//     fn crossover_with_bundle_conflicts() {
//         let mut gen = IdGen::new();
//         // idx0: A@0 (profit 400)
//         // idx1: B@0 (profit 300)
//         // idx2: Bundle[A@0, B@0] (profit 200) — conflicts with both idx0 and idx1
//         // idx3: A@1
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 400, &mut gen),
//             mk_single_tx_order(SENDER_B, 0, 300, &mut gen),
//             mk_bundle_order(&[(SENDER_A, 0), (SENDER_B, 0)], 200, &mut gen),
//             mk_single_tx_order(SENDER_A, 1, 100, &mut gen),
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(99);

//         // Parent A: chose individual txs (idx0 + idx1).
//         let pa = individual_from_seq(vec![0, 1, 3], &deps);
//         // Parent B: chose the bundle (idx2).
//         let pb = individual_from_seq(vec![2, 3], &deps);

//         for _ in 0..200 {
//             let child = crossover(&pa, &pb, &deps, &mut rng);
//             assert_valid_individual(&child, &deps);

//             // If bundle (2) is chosen, neither 0 nor 1 should be present.
//             if child.seq.contains(&2) {
//                 assert!(!child.seq.contains(&0), "Bundle chosen but idx0 present: {:?}", child.seq);
//                 assert!(!child.seq.contains(&1), "Bundle chosen but idx1 present: {:?}", child.seq);
//             }
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // Mutation preserves validity
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn mutation_preserves_validity_no_conflicts() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
//             mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
//             mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
//             mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(123);

//         let mut ind = individual_from_seq(vec![0, 2, 1, 3], &deps);
//         for _ in 0..200 {
//             mutate(&mut ind, &deps, &mut rng, 0.3);
//             assert_valid_individual(&ind, &deps);
//         }
//     }

//     #[test]
//     fn mutation_with_candidate_flip_preserves_validity() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
//             mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1
//             mk_single_tx_order(SENDER_A, 1, 300, &mut gen), // idx 2
//             mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // idx 3
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(456);

//         let mut ind = individual_from_seq(vec![0, 3, 2], &deps);
//         for _ in 0..200 {
//             mutate(&mut ind, &deps, &mut rng, 0.5); // high rate to trigger flips
//             assert_valid_individual(&ind, &deps);
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // Diamond DAG with bundles
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn operators_valid_on_diamond_dag() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
//             mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen), // 1
//             mk_bundle_order(&[(SENDER_A, 2), (SENDER_C, 0)], 300, &mut gen), // 2
//             mk_single_tx_order(SENDER_B, 1, 150, &mut gen),                  // 3
//             mk_bundle_order(&[(SENDER_B, 2), (SENDER_C, 1)], 250, &mut gen), // 4
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(12345);

//         let pa = individual_from_seq(vec![0, 1, 2, 3, 4], &deps);
//         let pb = individual_from_seq(vec![0, 1, 3, 2, 4], &deps);

//         for _ in 0..100 {
//             let child = crossover(&pa, &pb, &deps, &mut rng);
//             assert_valid_individual(&child, &deps);

//             let mut ind = pa.clone();
//             mutate(&mut ind, &deps, &mut rng, 0.3);
//             assert_valid_individual(&ind, &deps);
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // DC distance
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn dc_distance_same_individual_is_zero() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
//             mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let ind = individual_from_seq(vec![0, 1], &deps);
//         assert!((dc_distance(&ind, &ind, &deps)).abs() < 1e-12);
//     }

//     #[test]
//     fn dc_distance_different_candidates_nonzero() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // 0
//             mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // 1
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let a = individual_from_seq(vec![0], &deps);
//         let b = individual_from_seq(vec![1], &deps);
//         let dist = dc_distance(&a, &b, &deps);
//         assert!(dist > 0.0, "Different candidates should have nonzero distance");
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // Compete
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn compete_picks_dominant() {
//         let mut rng = SmallRng::seed_from_u64(1);
//         let deps = GroupDeps { order_deps: vec![], slot_providers: HashMap::default(), n: 0 };

//         let better = Individual {
//             choices: CandidateChoices::default(),
//             active: AHashSet::default(),
//             seq: vec![0, 1],
//             profit: U256::from(100),
//             gas: 50,
//         };
//         let worse = Individual {
//             choices: CandidateChoices::default(),
//             active: AHashSet::default(),
//             seq: vec![1, 0],
//             profit: U256::from(50),
//             gas: 100,
//         };
//         for _ in 0..20 {
//             assert_eq!(compete(&worse, &better, &mut rng).profit, U256::from(100));
//         }
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // individual_from_seq correctly extracts choices
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn individual_from_seq_extracts_choices() {
//         let mut gen = IdGen::new();
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // 0
//             mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // 1
//             mk_single_tx_order(SENDER_A, 1, 300, &mut gen), // 2
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();

//         let ind = individual_from_seq(vec![0, 2], &deps);
//         let slot = NonceKey { address: SENDER_A, nonce: 0 };
//         assert_eq!(ind.choices.get(&slot), Some(&0));
//         assert!(ind.active.contains(&0));
//         assert!(!ind.active.contains(&1)); // excluded
//         assert!(ind.active.contains(&2));
//     }

//     // ═══════════════════════════════════════════════════════════════
//     // Multiple conflict slots
//     // ═══════════════════════════════════════════════════════════════

//     #[test]
//     fn crossover_multiple_conflict_slots() {
//         let mut gen = IdGen::new();
//         // A@0: two candidates, B@0: two candidates
//         let group = mk_group(vec![
//             mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // 0
//             mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // 1
//             mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // 2
//             mk_single_tx_order(SENDER_B, 0, 250, &mut gen), // 3
//         ]);
//         let deps = GroupDeps::from_group(&group).unwrap();
//         let mut rng = SmallRng::seed_from_u64(77);

//         // Parent A: chose 0 for A@0, 2 for B@0
//         let pa = individual_from_seq(vec![0, 2], &deps);
//         // Parent B: chose 1 for A@0, 3 for B@0
//         let pb = individual_from_seq(vec![1, 3], &deps);

//         for _ in 0..200 {
//             let child = crossover(&pa, &pb, &deps, &mut rng);
//             assert_valid_individual(&child, &deps);
//             assert_eq!(child.seq.len(), 2);
//         }
//     }
// }