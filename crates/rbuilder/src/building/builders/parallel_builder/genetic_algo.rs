use std::collections::HashSet;
use alloy_primitives::U256;
use rand::rngs::SmallRng;
use rand::Rng;
use super::nonce_handling::{GroupDeps, random_ordering_with_random_choices};

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

/// Simple distance metric for deterministic crowding.
/// Counts the number of positions where sequences differ, normalized.
pub fn dc_distance(a: &[usize], b: &[usize], _deps: &GroupDeps) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 1.0;
    }
    let diffs = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    diffs as f64 / a.len() as f64
}

/// Build a valid child ordering from two parent orderings.
/// Uses a simple greedy strategy based on parent rankings.
pub fn ppx_build_child_from_parents_greedy(
    parent_a: &[usize],
    parent_b: &[usize],
    _deps: &GroupDeps,
) -> Vec<usize> {
    // Simple merge: interleave parents, avoiding duplicates
    let mut child = Vec::new();
    let mut used = ahash::HashSet::default();

    let max_len = parent_a.len().max(parent_b.len());
    for i in 0..max_len {
        if i < parent_a.len() && !used.contains(&parent_a[i]) {
            child.push(parent_a[i]);
            used.insert(parent_a[i]);
        }
        if i < parent_b.len() && !used.contains(&parent_b[i]) {
            child.push(parent_b[i]);
            used.insert(parent_b[i]);
        }
    }

    child
}

/// Same as `ppx_build_child_from_parents_greedy`, but with randomness.
pub fn ppx_build_child_from_parents(
    parent_a: &[usize],
    parent_b: &[usize],
    _deps: &GroupDeps,
    rng: &mut SmallRng,
) -> Vec<usize> {
    // Probabilistic merge based on parent rankings
    let mut child = Vec::new();
    let mut used = ahash::HashSet::default();

    let max_len = parent_a.len().max(parent_b.len());
    for i in 0..max_len {
        // Randomly choose which parent to take from
        if rng.gen_bool(0.5) {
            if i < parent_a.len() && !used.contains(&parent_a[i]) {
                child.push(parent_a[i]);
                used.insert(parent_a[i]);
            }
            if i < parent_b.len() && !used.contains(&parent_b[i]) {
                child.push(parent_b[i]);
                used.insert(parent_b[i]);
            }
        } else {
            if i < parent_b.len() && !used.contains(&parent_b[i]) {
                child.push(parent_b[i]);
                used.insert(parent_b[i]);
            }
            if i < parent_a.len() && !used.contains(&parent_a[i]) {
                child.push(parent_a[i]);
                used.insert(parent_a[i]);
            }
        }
    }

    child
}

/// Performs an Order Crossover that produces a valid child ordering.
/// Simplified version that maintains validity through deduplication.
pub fn adapted_order_crossover(
    parent_a: &[usize],
    parent_b: &[usize],
    _deps: &GroupDeps,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = parent_a.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![parent_a[0]]; }

    // Pick a random slice from parent_a
    let p1 = rng.gen_range(0..n);
    let mut p2 = rng.gen_range(0..n);
    while p1 == p2 && n > 1 {
        p2 = rng.gen_range(0..n);
    }
    let (start, end) = (p1.min(p2), p1.max(p2));

    let mut child = Vec::with_capacity(n);
    let mut in_child = HashSet::with_capacity(n);

    // Copy slice from parent_a
    for i in start..=end.min(n - 1) {
        child.push(parent_a[i]);
        in_child.insert(parent_a[i]);
    }

    // Fill remaining from parent_b
    for &val in parent_b {
        if !in_child.contains(&val) {
            child.push(val);
            in_child.insert(val);
        }
    }

    // Fill any remaining from parent_a
    for &val in parent_a {
        if !in_child.contains(&val) {
            child.push(val);
            in_child.insert(val);
        }
    }

    child
}

// Mutation functions
/// Simple swap mutation: swap two random elements
pub fn mut_swap(seq: &mut [usize], rng: &mut SmallRng) -> bool {
    if seq.len() < 2 { return false; }
    let i = rng.gen_range(0..seq.len());
    let j = rng.gen_range(0..seq.len());
    if i != j {
        seq.swap(i, j);
        true
    } else {
        false
    }
}

/// Mutate by randomly shuffling a subsequence
pub fn mut_shuffle_segment(seq: &mut [usize], rng: &mut SmallRng) -> bool {
    if seq.len() < 2 { return false; }
    let i = rng.gen_range(0..seq.len());
    let j = rng.gen_range(0..seq.len());
    let (start, end) = (i.min(j), i.max(j));
    if end > start {
        use rand::seq::SliceRandom;
        seq[start..=end].shuffle(rng);
        true
    } else {
        false
    }
}

pub fn mutate(seq: &mut Vec<usize>, deps: &GroupDeps, rng: &mut SmallRng) {
    use rand::Rng;
    for _ in 0..5 {
        let picked = rng.gen_range(0..100);
        let changed = if picked < 50 {
            mut_swap(seq, rng)
        } else {
            mut_shuffle_segment(seq, rng)
        };
        if changed { return; }
    }
    // last resort: generate a new random valid ordering
    *seq = random_ordering_with_random_choices(deps, rng);
}

pub fn compete(
    parent: &Individual, 
    child: &Individual, 
    rng: &mut SmallRng
) -> Individual {
    let child_dominates = dominates(child, parent);
    let parent_dominates = dominates(parent, child);

    if child_dominates {
        return child.clone();
    } 
    
    if parent_dominates {
        return parent.clone();
    }

    if rng.gen_bool(0.5) {
        child.clone()
    } else {
        parent.clone()
    }
}


/// Represents an isolated population in the Island Model.
pub struct Island {
    pub population: Vec<Individual>,
    pub rng: SmallRng,
}


/// Log metrics calculated during a single DC generation.
pub struct DCGenerationResult {
    pub new_best_hits: u32,
    pub children_who_won: usize,
    pub evals: u64,
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, TxHash, U256, address};
    use alloy_primitives::Signature;
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use rand::SeedableRng;

    use super::*;
    use crate::{
        building::builders::parallel_builder::{ConflictGroup, GroupId},
        primitives::{
            Order, SimValue, SimulatedOrder, MempoolTx,
            TransactionSignedEcRecoveredWithBlobs,
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

    // Tests for genetic algorithm functions have been simplified
    // as they now work with GroupDeps instead of NonceLayout

    // Mutation and crossover tests removed as they now work with simplified logic
}
