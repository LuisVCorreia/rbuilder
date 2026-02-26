use ahash::{HashMap, HashSet as AHashSet};
use alloy_primitives::U256;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;

use super::nonce_handling::DependencyDag;
#[derive(Clone)]
pub struct Individual {
    /// A valid topological sort of the shared DAG. Contains original order indices.
    pub seq: Vec<usize>,
    /// Fitness values (filled after evaluation)
    pub profit: U256,
    pub gas: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct GAParams {
    pub population: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
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

pub fn individual_from_seq(seq: Vec<usize>) -> Individual {
    Individual {
        seq,
        profit: U256::ZERO,
        gas: 0,
    }
}

/// Validate that an individual's sequence is a valid topo sort of the DAG.
#[cfg(debug_assertions)]
pub fn validate_individual(ind: &Individual, dag: &DependencyDag) -> bool {
    let seq_set: AHashSet<usize> = ind.seq.iter().copied().collect();
    let dag_set: AHashSet<usize> = dag.nodes.iter().copied().collect();
    if seq_set != dag_set {
        return false;
    }

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

    true
}

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
}

// Two-point Order Crossover (OX):
//   1. Copy a random contiguous slice from parent A.
//   2. Fill remaining positions with orders from parent B (preserving B's
//      relative order), skipping any already in the slice.
//   3. Repair the result into a valid topological sort.
//
// All individuals share the same active set and DAG, so no candidate-level
// merging is needed.

pub fn crossover(
    parent_a: &Individual,
    parent_b: &Individual,
    dag: &DependencyDag,
    rng: &mut SmallRng,
) -> Individual {
    let n = parent_a.seq.len();
    if n <= 1 {
        return Individual {
            seq: parent_a.seq.clone(),
            profit: U256::ZERO,
            gas: 0,
        };
    }

    // Pick two crossover points
    let lo = rng.gen_range(0..n);
    let hi = rng.gen_range(lo..=n);

    // Core segment from parent A
    let core: Vec<usize> = parent_a.seq[lo..hi].to_vec();
    let core_set: AHashSet<usize> = core.iter().copied().collect();

    // Remainder from parent B, preserving B's relative order
    let remainder: Vec<usize> = parent_b.seq.iter().copied()
        .filter(|oi| !core_set.contains(oi))
        .collect();

    let split = lo.min(remainder.len());
    let mut proposed = Vec::with_capacity(n);
    proposed.extend_from_slice(&remainder[..split]);
    proposed.extend_from_slice(&core);
    proposed.extend_from_slice(&remainder[split..]);

    // Repair to a valid topological sort
    let seq = repair_topo_sort(&proposed, dag);

    Individual {
        seq,
        profit: U256::ZERO,
        gas: 0,
    }
}

/// Repair a proposed sequence into a valid topological sort of `dag`.
///
/// At each step, among all "ready" orders (predecessors already placed),
/// picks the one appearing earliest in `proposed`. This maximises preservation
/// of the proposed ordering while guaranteeing topological validity.
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

        let best = *ready.iter()
            .min_by_key(|&&oi| pos.get(&oi).copied().unwrap_or(usize::MAX))
            .unwrap();

        result.push(best);
        tracker.place(best);
    }

    result
}

// Mutation operators

/// Per-gene mutation: adjacent swap or bubble move, respecting DAG constraints.
pub fn mutate(
    ind: &mut Individual,
    dag: &DependencyDag,
    rng: &mut SmallRng,
    mutation_rate: f64,
) {
    let n = ind.seq.len();
    if n < 2 {
        return;
    }

    let expected = (mutation_rate * n as f64).max(1.0);
    let num_mutations = if expected < 1.0 {
        if rng.gen::<f64>() < expected { 1 } else { 0 }
    } else {
        expected.round() as usize
    };

    for _ in 0..num_mutations {
        if rng.gen_bool(0.5) {
            mut_adjacent_swap(&mut ind.seq, dag, rng);
        } else {
            let max_steps = rng.gen_range(2..=(n / 4).max(3));
            mut_bubble_move(&mut ind.seq, dag, rng, max_steps);
        }
    }
}

/// Swap two adjacent orders that have no dependency between them
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

/// Bubble an element left or right by repeated adjacent swaps
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
            if pos == 0 || !can_swap_adjacent(seq, pos - 1, dag) {
                break;
            }
            seq.swap(pos - 1, pos);
            pos -= 1;
        } else {
            if pos + 1 >= seq.len() || !can_swap_adjacent(seq, pos, dag) {
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

/// Normalized Kendall-tau distance between two sequences.
/// Since all individuals share the same active set, sequences contain the
/// same elements and we count inversions directly.
pub fn kendall_tau_distance(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    if n <= 1 {
        return 0.0;
    }

    // Build position map for b.
    let mut pos_in_b: HashMap<usize, usize> = HashMap::default();
    for (i, &oi) in b.iter().enumerate() {
        pos_in_b.insert(oi, i);
    }

    // Map a's elements to their position in b.
    let mut mapped: Vec<usize> = a.iter()
        .map(|&oi| pos_in_b.get(&oi).copied().unwrap_or(0))
        .collect();

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

// Deterministic Crowding (DC) for maintaining diversity in each island
/// An unevaluated child with metadata for DC competition.
pub struct PendingChild {
    pub island_idx: usize,
    pub pair_idx: usize,
    pub child_slot: usize,
    pub ind: Individual,
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
    dag: &DependencyDag,
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
                crossover(p1, p2, dag, rng),
                crossover(p2, p1, dag, rng),
            )
        } else {
            (p1.clone(), p2.clone())
        };

        mutate(&mut c1, dag, rng, params.mutation_rate);
        mutate(&mut c2, dag, rng, params.mutation_rate);

        children.push(PendingChild {
            island_idx,
            pair_idx,
            child_slot: 0,
            ind: c1,
        });
        children.push(PendingChild {
            island_idx,
            pair_idx,
            child_slot: 1,
            ind: c2,
        });
        pair_infos.push(ParentPairInfo { p1_idx, p2_idx });
    }

    (children, pair_infos)
}

/// Apply DC competition using evaluated children
pub fn apply_dc_competition(
    island: &mut Island,
    pair_infos: &[ParentPairInfo],
    evaluated_children: &mut Vec<Option<Individual>>,
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

        // Match children to most similar parents
        let dist_p1c1 = kendall_tau_distance(&p1.seq, &c1.seq);
        let dist_p2c2 = kendall_tau_distance(&p2.seq, &c2.seq);
        let dist_p1c2 = kendall_tau_distance(&p1.seq, &c2.seq);
        let dist_p2c1 = kendall_tau_distance(&p2.seq, &c1.seq);

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use ahash::HashSet as AHashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{address, Address, Signature, TxHash, B256, U256};
    use rand::SeedableRng;
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use crate::building::builders::parallel_builder::ConflictGroup;
    use crate::building::builders::parallel_builder::nonce_handling::GroupDeps;
    use rbuilder_primitives::{
        Bundle, MempoolTx, Metadata, Order, SimValue, SimulatedOrder,
        TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
    };

    const SENDER_A: Address = address!("0x000000000000000000000000000000000000000a");
    const SENDER_B: Address = address!("0x000000000000000000000000000000000000000b");
    const SENDER_C: Address = address!("0x000000000000000000000000000000000000000c");

    struct IdGen(u64);
    impl IdGen {
        fn new() -> Self { Self(0) }
        fn next_hash(&mut self) -> TxHash {
            self.0 += 1;
            TxHash::from(U256::from(self.0))
        }
    }

    fn mk_tx(sender: Address, nonce: u64, gen: &mut IdGen) -> Recovered<TransactionSigned> {
        let tx_legacy = TxLegacy { nonce, ..Default::default() };
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
            order: Arc::new(Order::Tx(MempoolTx { tx_with_blobs: with_blobs })),
            used_state_trace: None,
            sim_value: SimValue::new_test(U256::from(profit), U256::from(profit), 0),
        })
    }

    fn mk_bundle_order(
        tx_specs: &[(Address, u64)],
        profit: u64,
        gen: &mut IdGen,
    ) -> Arc<SimulatedOrder> {
        let txs: Vec<_> = tx_specs.iter()
            .map(|&(sender, nonce)| {
                TransactionSignedEcRecoveredWithBlobs::new_no_blobs(mk_tx(sender, nonce, gen)).unwrap()
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
            conflicting_group_ids: Arc::new(AHashSet::default()),
        }
    }

    /// Build a DAG with greedy dedup (matching the simplified GA approach).
    fn build_deduped_dag(group: &ConflictGroup) -> (GroupDeps, DependencyDag) {
        use crate::building::builders::parallel_builder::nonce_handling::GreedyKey;
        let deps = GroupDeps::from_group(group).unwrap();
        let dag = if deps.has_conflicts() {
            let active = deps.dedup_best(group, GreedyKey::Profit, false);
            deps.build_dag(&active)
        } else {
            deps.build_dag_all()
        };
        (deps, dag)
    }

    /// Assert the sequence is a valid topo sort of the given DAG.
    fn assert_valid_topo_sort(seq: &[usize], dag: &DependencyDag) {
        let seq_set: AHashSet<usize> = seq.iter().copied().collect();
        let dag_set: AHashSet<usize> = dag.nodes.iter().copied().collect();
        assert_eq!(seq_set, dag_set,
            "Seq elements {:?} don't match DAG nodes {:?}", seq_set, dag_set);
        assert_eq!(seq.len(), dag.nodes.len(),
            "Seq has duplicates: {:?}", seq);

        let pos: HashMap<usize, usize> = seq.iter().enumerate()
            .map(|(p, &oi)| (oi, p)).collect();
        for (ni, succs) in dag.successors.iter().enumerate() {
            let from = dag.nodes[ni];
            for &si in succs {
                let to = dag.nodes[si];
                assert!(pos[&from] < pos[&to],
                    "{} must come before {} in {:?}", from, to, seq);
            }
        }
    }

    // individual_from_seq

    #[test]
    fn individual_from_seq_creates_valid_individual() {
        let ind = individual_from_seq(vec![3, 1, 0, 2]);
        assert_eq!(ind.seq, vec![3, 1, 0, 2]);
        assert_eq!(ind.profit, U256::ZERO);
        assert_eq!(ind.gas, 0);
    }

    #[test]
    fn individual_from_seq_empty() {
        let ind = individual_from_seq(vec![]);
        assert!(ind.seq.is_empty());
    }

    // dominates
    #[test]
    fn dominates_higher_profit() {
        let a = Individual { seq: vec![], profit: U256::from(100), gas: 50 };
        let b = Individual { seq: vec![], profit: U256::from(50), gas: 50 };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_same_profit_lower_gas() {
        let a = Individual { seq: vec![], profit: U256::from(100), gas: 30 };
        let b = Individual { seq: vec![], profit: U256::from(100), gas: 50 };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_neither() {
        let a = Individual { seq: vec![], profit: U256::from(100), gas: 50 };
        let b = Individual { seq: vec![], profit: U256::from(100), gas: 50 };
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    // compete
    #[test]
    fn compete_picks_dominant() {
        let mut rng = SmallRng::seed_from_u64(1);
        let better = Individual { seq: vec![0, 1], profit: U256::from(100), gas: 50 };
        let worse = Individual { seq: vec![1, 0], profit: U256::from(50), gas: 100 };
        for _ in 0..20 {
            let winner = compete(&worse, &better, &mut rng);
            assert_eq!(winner.profit, U256::from(100));
        }
    }

    #[test]
    fn compete_equal_is_random() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = Individual { seq: vec![0, 1], profit: U256::from(100), gas: 50 };
        let b = Individual { seq: vec![1, 0], profit: U256::from(100), gas: 50 };
        let mut a_wins = 0;
        let mut b_wins = 0;
        for _ in 0..200 {
            let winner = compete(&a, &b, &mut rng);
            if winner.seq == a.seq { a_wins += 1; } else { b_wins += 1; }
        }
        // Both should win a reasonable number of times.
        assert!(a_wins > 50 && b_wins > 50,
            "Expected roughly even split, got a={} b={}", a_wins, b_wins);
    }

    // Crossover: no dependencies (independent orders)
    #[test]
    fn crossover_independent_orders_preserves_elements() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 1, 2]);
        let pb = individual_from_seq(vec![2, 0, 1]);

        for _ in 0..100 {
            let child = crossover(&pa, &pb, &dag, &mut rng);
            assert_valid_topo_sort(&child.seq, &dag);
            assert_eq!(child.profit, U256::ZERO);
            assert_eq!(child.gas, 0);
        }
    }

    // Crossover: with nonce chain dependencies
    #[test]
    fn crossover_nonce_chain_preserves_topo_order() {
        let mut gen = IdGen::new();
        // A: nonce 0 -> 1 -> 2, B: nonce 0 -> 1
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // 0
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen), // 1
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen), // 2
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // 3
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen), // 4
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 3, 1, 4, 2]);
        let pb = individual_from_seq(vec![3, 0, 4, 1, 2]);

        for _ in 0..200 {
            let child = crossover(&pa, &pb, &dag, &mut rng);
            assert_valid_topo_sort(&child.seq, &dag);
        }
    }

    #[test]
    fn crossover_single_element() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0]);
        let pb = individual_from_seq(vec![0]);
        let child = crossover(&pa, &pb, &dag, &mut rng);
        assert_eq!(child.seq, vec![0]);
    }

    #[test]
    fn crossover_with_bundles_preserves_dag() {
        let mut gen = IdGen::new();
        // idx0: A@0, idx1: bundle(A@1, B@0), idx2: A@2, idx3: B@1
        // DAG: 0 -> 1 -> 2, 1 -> 3
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen), // 1
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),                  // 2
            mk_single_tx_order(SENDER_B, 1, 150, &mut gen),                  // 3
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 1, 2, 3]);
        let pb = individual_from_seq(vec![0, 1, 3, 2]);

        for _ in 0..200 {
            let child = crossover(&pa, &pb, &dag, &mut rng);
            assert_valid_topo_sort(&child.seq, &dag);
        }
    }

    // Crossover: produces variation
    #[test]
    fn crossover_produces_diverse_children() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 1, 2]);
        let pb = individual_from_seq(vec![2, 1, 0]);

        let mut unique_seqs: AHashSet<Vec<usize>> = AHashSet::default();
        for _ in 0..100 {
            let child = crossover(&pa, &pb, &dag, &mut rng);
            unique_seqs.insert(child.seq);
        }
        assert!(unique_seqs.len() > 1,
            "Crossover should produce diverse children, got {} unique", unique_seqs.len());
    }

    // Mutation: preserves topo sort
    #[test]
    fn mutation_preserves_topo_order_no_deps() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(123);

        let mut ind = individual_from_seq(vec![0, 1, 2]);
        for _ in 0..200 {
            mutate(&mut ind, &dag, &mut rng, 0.3);
            assert_valid_topo_sort(&ind.seq, &dag);
        }
    }

    #[test]
    fn mutation_preserves_topo_order_with_chain() {
        let mut gen = IdGen::new();
        // Strict chain: A@0 -> A@1 -> A@2
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(456);

        let mut ind = individual_from_seq(vec![0, 1, 3, 2]);
        for _ in 0..200 {
            mutate(&mut ind, &dag, &mut rng, 0.5);
            assert_valid_topo_sort(&ind.seq, &dag);
            // A@0 must always precede A@1 must always precede A@2
            let pos: HashMap<usize, usize> = ind.seq.iter().enumerate()
                .map(|(p, &oi)| (oi, p)).collect();
            assert!(pos[&0] < pos[&1], "A@0 must come before A@1");
            assert!(pos[&1] < pos[&2], "A@1 must come before A@2");
        }
    }

    #[test]
    fn mutation_produces_changes() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(789);

        let original = vec![0, 1, 2];
        let mut changed = false;
        for _ in 0..50 {
            let mut ind = individual_from_seq(original.clone());
            mutate(&mut ind, &dag, &mut rng, 0.5);
            if ind.seq != original {
                changed = true;
                break;
            }
        }
        assert!(changed, "Mutation should eventually change the sequence");
    }

    #[test]
    fn mutation_no_change_on_fully_constrained_chain() {
        let mut gen = IdGen::new();
        // Fully constrained: A@0 -> A@1 -> A@2, no independent orders.
        // No valid adjacent swaps exist.
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let original = vec![0, 1, 2];
        let mut ind = individual_from_seq(original.clone());
        for _ in 0..50 {
            mutate(&mut ind, &dag, &mut rng, 1.0);
        }
        // Only valid ordering is [0, 1, 2], mutation can't change it.
        assert_eq!(ind.seq, original);
    }

    // Mutation: diamond DAG with bundles
    #[test]
    fn mutation_diamond_dag_preserves_validity() {
        let mut gen = IdGen::new();
        // 0: A@0
        // 1: bundle(A@1, B@0) — depends on 0
        // 2: bundle(A@2, C@0) — depends on 1
        // 3: B@1 — depends on 1
        // 4: bundle(B@2, C@1) — depends on 2 and 3
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen), // 1
            mk_bundle_order(&[(SENDER_A, 2), (SENDER_C, 0)], 300, &mut gen), // 2
            mk_single_tx_order(SENDER_B, 1, 150, &mut gen),                  // 3
            mk_bundle_order(&[(SENDER_B, 2), (SENDER_C, 1)], 250, &mut gen), // 4
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(12345);

        let mut ind = individual_from_seq(vec![0, 1, 2, 3, 4]);
        for _ in 0..200 {
            mutate(&mut ind, &dag, &mut rng, 0.3);
            assert_valid_topo_sort(&ind.seq, &dag);
        }
    }

    // Kendall-tau distance
    #[test]
    fn kendall_tau_identical_is_zero() {
        let a = vec![0, 1, 2, 3];
        assert!((kendall_tau_distance(&a, &a)).abs() < 1e-12);
    }

    #[test]
    fn kendall_tau_reversed_is_one() {
        let a = vec![0, 1, 2, 3];
        let b = vec![3, 2, 1, 0];
        let d = kendall_tau_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-12, "Reversed should be 1.0, got {}", d);
    }

    #[test]
    fn kendall_tau_single_swap() {
        let a = vec![0, 1, 2, 3];
        let b = vec![0, 2, 1, 3]; // one adjacent swap
        let d = kendall_tau_distance(&a, &b);
        // max inversions = 4*3/2 = 6, one inversion => 1/6
        let expected = 1.0 / 6.0;
        assert!((d - expected).abs() < 1e-12, "Expected {}, got {}", expected, d);
    }

    #[test]
    fn kendall_tau_single_element() {
        assert!((kendall_tau_distance(&[0], &[0])).abs() < 1e-12);
    }

    #[test]
    fn kendall_tau_empty() {
        let empty: Vec<usize> = vec![];
        assert!((kendall_tau_distance(&empty, &empty)).abs() < 1e-12);
    }

    #[test]
    fn kendall_tau_symmetric() {
        let a = vec![0, 2, 1, 3, 4];
        let b = vec![4, 0, 3, 1, 2];
        let d1 = kendall_tau_distance(&a, &b);
        let d2 = kendall_tau_distance(&b, &a);
        assert!((d1 - d2).abs() < 1e-12, "Should be symmetric: {} vs {}", d1, d2);
    }

    // validate_individual (debug only)
    #[test]
    #[cfg(debug_assertions)]
    fn validate_individual_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let ind = individual_from_seq(vec![0, 2, 1]);
        assert!(validate_individual(&ind, &dag));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_individual_wrong_order_invalid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        // A@1 before A@0 violates the dependency.
        let ind = individual_from_seq(vec![1, 0]);
        assert!(!validate_individual(&ind, &dag));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_individual_missing_element_invalid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);
        // Missing order 1.
        let ind = individual_from_seq(vec![0]);
        assert!(!validate_individual(&ind, &dag));
    }

    // DC children generation
    #[test]
    fn generate_dc_children_produces_correct_count() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);

        let params = GAParams {
            population: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            max_generations: 10,
            time_ms: 1000,
            seed: 42,
            num_islands: 1,
            migration_interval: 5,
        };

        let pop: Vec<Individual> = (0..10).map(|_| {
            individual_from_seq(vec![0, 1, 2])
        }).collect();

        let mut island = Island {
            population: pop,
            rng: SmallRng::seed_from_u64(42),
        };

        let (children, pair_infos) = generate_dc_children(&mut island, 0, &params, &dag);

        // 10 individuals => 5 pairs => 10 children
        assert_eq!(children.len(), 10);
        assert_eq!(pair_infos.len(), 5);

        // All children should be valid topo sorts.
        for child in &children {
            assert_valid_topo_sort(&child.ind.seq, &dag);
        }
    }

    #[test]
    fn generate_dc_children_odd_population() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);

        let params = GAParams {
            population: 7,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            max_generations: 10,
            time_ms: 1000,
            seed: 42,
            num_islands: 1,
            migration_interval: 5,
        };

        let pop: Vec<Individual> = (0..7).map(|_| {
            individual_from_seq(vec![0, 1])
        }).collect();

        let mut island = Island {
            population: pop,
            rng: SmallRng::seed_from_u64(42),
        };

        let (children, pair_infos) = generate_dc_children(&mut island, 0, &params, &dag);

        // 7 individuals => 3 pairs => 6 children (last one unpaired)
        assert_eq!(children.len(), 6);
        assert_eq!(pair_infos.len(), 3);
    }

    // DC competition
    #[test]
    fn dc_competition_replaces_with_better_child() {
        let p1 = Individual { seq: vec![0, 1], profit: U256::from(50), gas: 100 };
        let p2 = Individual { seq: vec![1, 0], profit: U256::from(60), gas: 90 };

        // Child that dominates p1 (higher profit, same seq for distance matching).
        let c1 = Individual { seq: vec![0, 1], profit: U256::from(200), gas: 50 };
        let c2 = Individual { seq: vec![1, 0], profit: U256::from(10), gas: 200 };

        let mut island = Island {
            population: vec![p1, p2],
            rng: SmallRng::seed_from_u64(42),
        };

        let pair_infos = vec![ParentPairInfo { p1_idx: 0, p2_idx: 1 }];
        let mut evaluated = vec![Some(c1), Some(c2)];

        apply_dc_competition(&mut island, &pair_infos, &mut evaluated);

        assert_eq!(island.population.len(), 2);
        // c1 dominates p1 so should replace it.
        assert_eq!(island.population[0].profit, U256::from(200));
        // p2 dominates c2 so p2 should be kept.
        assert_eq!(island.population[1].profit, U256::from(60));
    }

    #[test]
    fn dc_competition_handles_failed_eval() {
        let p1 = Individual { seq: vec![0, 1], profit: U256::from(50), gas: 100 };
        let p2 = Individual { seq: vec![1, 0], profit: U256::from(60), gas: 90 };

        let mut island = Island {
            population: vec![p1, p2],
            rng: SmallRng::seed_from_u64(42),
        };

        let pair_infos = vec![ParentPairInfo { p1_idx: 0, p2_idx: 1 }];
        // Both children failed evaluation.
        let mut evaluated: Vec<Option<Individual>> = vec![None, None];

        apply_dc_competition(&mut island, &pair_infos, &mut evaluated);

        // Parents should be preserved.
        assert_eq!(island.population.len(), 2);
        assert_eq!(island.population[0].profit, U256::from(50));
        assert_eq!(island.population[1].profit, U256::from(60));
    }

    // can_swap_adjacent
    #[test]
    fn can_swap_independent_orders() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);

        let seq = vec![0, 1];
        assert!(can_swap_adjacent(&seq, 0, &dag));
    }

    #[test]
    fn cannot_swap_dependent_orders() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
        ]);
        let (_, dag) = build_deduped_dag(&group);

        let seq = vec![0, 1];
        assert!(!can_swap_adjacent(&seq, 0, &dag));
    }

    // Crossover + mutation combined stress test
    #[test]
    fn stress_crossover_mutation_interleaved_chains() {
        let mut gen = IdGen::new();
        // Two interleaved chains with bundles:
        // A@0, A@1, A@2, B@0, B@1, bundle(A@3, B@2)
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),                  // 1
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),                  // 2
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // 3
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),                  // 4
            mk_bundle_order(&[(SENDER_A, 3), (SENDER_B, 2)], 500, &mut gen), // 5
        ]);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(54321);

        let pa = individual_from_seq(vec![0, 3, 1, 4, 2, 5]);
        let pb = individual_from_seq(vec![3, 0, 4, 1, 2, 5]);

        for _ in 0..500 {
            let mut child = crossover(&pa, &pb, &dag, &mut rng);
            assert_valid_topo_sort(&child.seq, &dag);

            mutate(&mut child, &dag, &mut rng, 0.3);
            assert_valid_topo_sort(&child.seq, &dag);

            // Also cross the child back with a parent.
            let grandchild = crossover(&child, &pa, &dag, &mut rng);
            assert_valid_topo_sort(&grandchild.seq, &dag);
        }
    }

    #[test]
    fn stress_many_independent_orders() {
        let mut gen = IdGen::new();
        // 20 independent orders (different senders, nonce 0).
        let orders: Vec<_> = (0..20).map(|i| {
            let sender = Address::from_word(U256::from(i + 100).into());
            mk_single_tx_order(sender, 0, (i + 1) as u64 * 10, &mut gen)
        }).collect();
        let group = mk_group(orders);
        let (_, dag) = build_deduped_dag(&group);
        let mut rng = SmallRng::seed_from_u64(99999);

        let seq_a: Vec<usize> = (0..20).collect();
        let seq_b: Vec<usize> = (0..20).rev().collect();
        let pa = individual_from_seq(seq_a);
        let pb = individual_from_seq(seq_b);

        for _ in 0..200 {
            let mut child = crossover(&pa, &pb, &dag, &mut rng);
            assert_valid_topo_sort(&child.seq, &dag);
            mutate(&mut child, &dag, &mut rng, 0.15);
            assert_valid_topo_sort(&child.seq, &dag);
        }
    }
}
