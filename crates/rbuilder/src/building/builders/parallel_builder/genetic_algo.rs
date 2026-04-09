use ahash::{HashMap, HashSet as AHashSet};
use alloy_primitives::{Address, U256};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::nonce_handling::{DependencyDag, GroupDeps};

#[derive(Clone)]
pub struct Individual {
    pub raw_seq: Vec<usize>,
    pub nonce_seq: Vec<usize>,
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
    pub w_choice: f64,
    pub early_stopping_generations: usize,
    pub temp_tight_low: f64,
    pub temp_tight_high: f64,
    pub temp_broad_low: f64,
    pub temp_broad_high: f64,
    pub tight_fraction: f64,
}

pub struct Island {
    pub population: Vec<Individual>,
    pub rng: SmallRng,
}

pub fn dominates(a: &Individual, b: &Individual) -> bool {
    a.profit > b.profit || (a.profit == b.profit && a.gas < b.gas)
}

pub fn individual_from_seq(raw_seq: Vec<usize>, deps: &GroupDeps) -> Individual {
    let nonce_seq = derive_nonce_seq(&raw_seq, deps);
    Individual {
        raw_seq,
        nonce_seq,
        profit: U256::ZERO,
        gas: 0,
    }
}

/// Derive a nonce-valid subsequence from a raw permutation of all N orders.
///
/// Step 1 — conflict resolution: walk raw_seq; the first order to claim each
/// nonce slot wins; later orders conflicting on any claimed slot are excluded.
///
/// Step 2 — per-individual DAG: build the dependency DAG for the active set.
///
/// Step 3 — topo sort: repair_topo_sort with the active-set projection of
/// raw_seq as the proposed ordering.  The DAG is discarded after use.
pub fn derive_nonce_seq(raw_seq: &[usize], deps: &GroupDeps) -> Vec<usize> {
    let mut claimed: AHashSet<_> = AHashSet::default();
    let mut active_set: Vec<usize> = Vec::new();

    for &oi in raw_seq {
        let provides = &deps.order_deps[oi].provides;
        if provides.iter().any(|s| claimed.contains(s)) {
            continue;
        }
        for s in provides {
            claimed.insert(s.clone());
        }
        active_set.push(oi);
    }

    let active_hash: AHashSet<usize> = active_set.iter().copied().collect();
    let dag = deps.build_dag(&active_hash);
    let proposed: Vec<usize> = raw_seq
        .iter()
        .copied()
        .filter(|oi| active_hash.contains(oi))
        .collect();
    repair_topo_sort(&proposed, &dag)
}

#[cfg(debug_assertions)]
pub fn validate_individual(ind: &Individual, deps: &GroupDeps) -> bool {
    let mut claimed: AHashSet<_> = AHashSet::default();
    let mut active: AHashSet<usize> = AHashSet::default();
    for &oi in &ind.raw_seq {
        let provides = &deps.order_deps[oi].provides;
        if provides.iter().any(|s| claimed.contains(s)) {
            continue;
        }
        for s in provides {
            claimed.insert(s.clone());
        }
        active.insert(oi);
    }

    let dag = deps.build_dag(&active);
    let seq_set: AHashSet<usize> = ind.nonce_seq.iter().copied().collect();
    let dag_set: AHashSet<usize> = dag.nodes.iter().copied().collect();
    if seq_set != dag_set {
        return false;
    }

    let pos: HashMap<usize, usize> = ind
        .nonce_seq
        .iter()
        .enumerate()
        .map(|(p, &oi)| (oi, p))
        .collect();

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

// Two-point Order Crossover (OX) on raw_seq:
//   1. Draw two independent uniform cut-points over [0, n] and sort them.
//   2. Copy core slice from parent A.
//   3. Fill remainder from parent B preserving B's relative order, skipping
//      elements already in the core.
//
// nonce_seq is left empty; caller (generate_dc_children) calls derive_nonce_seq.
pub fn crossover(
    parent_a: &Individual,
    parent_b: &Individual,
    rng: &mut SmallRng,
) -> Individual {
    let n = parent_a.raw_seq.len();
    if n <= 1 {
        return Individual {
            raw_seq: parent_a.raw_seq.clone(),
            nonce_seq: Vec::new(),
            profit: U256::ZERO,
            gas: 0,
        };
    }

    let p1 = rng.gen_range(0..=n);
    let p2 = rng.gen_range(0..=n);
    let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };

    let core: Vec<usize> = parent_a.raw_seq[lo..hi].to_vec();
    let core_set: AHashSet<usize> = core.iter().copied().collect();

    let remainder: Vec<usize> = parent_b
        .raw_seq
        .iter()
        .copied()
        .filter(|oi| !core_set.contains(oi))
        .collect();

    let split = lo.min(remainder.len());
    let mut raw_seq = Vec::with_capacity(n);
    raw_seq.extend_from_slice(&remainder[..split]);
    raw_seq.extend_from_slice(&core);
    raw_seq.extend_from_slice(&remainder[split..]);

    Individual {
        raw_seq,
        nonce_seq: Vec::new(),
        profit: U256::ZERO,
        gas: 0,
    }
}

/// Repair a proposed sequence into a valid topological sort of `dag`.
///
/// Uses a min-heap variant of Kahn's algorithm: O((N + E) log N).
/// At each step the ready node with the smallest position in `proposed` is
/// selected, preserving the proposed ordering preference.
fn repair_topo_sort(proposed: &[usize], dag: &DependencyDag) -> Vec<usize> {
    let n = dag.len();
    if n == 0 {
        return Vec::new();
    }

    let pos: HashMap<usize, usize> = proposed
        .iter()
        .enumerate()
        .map(|(i, &oi)| (oi, i))
        .collect();

    let mut in_deg = dag.in_degree.clone();
    let mut result = Vec::with_capacity(n);

    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for ni in 0..n {
        if in_deg[ni] == 0 {
            let oi = dag.nodes[ni];
            let p = pos.get(&oi).copied().unwrap_or(usize::MAX);
            heap.push(Reverse((p, oi)));
        }
    }

    while let Some(Reverse((_, oi))) = heap.pop() {
        result.push(oi);
        let ni = dag.node_of[&oi];
        for &succ in &dag.successors[ni] {
            in_deg[succ] -= 1;
            if in_deg[succ] == 0 {
                let succ_oi = dag.nodes[succ];
                let p = pos.get(&succ_oi).copied().unwrap_or(usize::MAX);
                heap.push(Reverse((p, succ_oi)));
            }
        }
    }

    result
}

/// Per-position mutation on raw_seq: iterate each position i; with probability
/// mutation_rate, swap raw_seq[i] with a uniformly chosen position j.
/// nonce_seq is left stale; caller must call derive_nonce_seq afterward.
pub fn mutate(ind: &mut Individual, rng: &mut SmallRng, mutation_rate: f64) {
    let n = ind.raw_seq.len();
    if n < 2 {
        return;
    }
    for i in 0..n {
        if rng.gen::<f64>() < mutation_rate {
            let j = rng.gen_range(0..n);
            ind.raw_seq.swap(i, j);
        }
    }
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

/// Two-component DC distance between two individuals based on their nonce_seqs.
///
/// distance = (1 - w_choice) * ordering_tau + w_choice * choice_mismatch
///
/// choice_mismatch: fraction of contested slots where the two individuals chose
/// different orders (or one included the slot and the other didn't).
///
/// ordering_tau: normalised Kendall-tau on the subsequence of orders common to
/// both nonce_seqs, mapped to representative slot IDs.
pub fn nonce_seq_dc_distance(
    a: &Individual,
    b: &Individual,
    deps: &GroupDeps,
    w_choice: f64,
) -> f64 {
    let contested: Vec<_> = deps
        .slot_providers
        .iter()
        .filter(|(_, providers)| providers.len() > 1)
        .collect();

    let choice_mismatch = if contested.is_empty() {
        0.0
    } else {
        let slot_winner = |nonce_seq: &[usize]| -> HashMap<_, usize> {
            let mut map: HashMap<_, usize> = HashMap::default();
            for &oi in nonce_seq {
                for slot in &deps.order_deps[oi].provides {
                    map.entry(slot.clone()).or_insert(oi);
                }
            }
            map
        };
        let winners_a = slot_winner(&a.nonce_seq);
        let winners_b = slot_winner(&b.nonce_seq);
        let mismatches = contested
            .iter()
            .filter(|(slot, _)| winners_a.get(*slot) != winners_b.get(*slot))
            .count();
        mismatches as f64 / contested.len() as f64
    };

    let set_a: AHashSet<usize> = a.nonce_seq.iter().copied().collect();
    let set_b: AHashSet<usize> = b.nonce_seq.iter().copied().collect();
    let common: AHashSet<usize> = set_a.intersection(&set_b).copied().collect();

    let ordering_tau = if common.is_empty() {
        1.0
    } else {
        let rep_slot = |oi: usize| -> (Address, u64) {
            deps.order_deps[oi]
                .provides
                .iter()
                .map(|s| (s.address, s.nonce))
                .min_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)))
                .unwrap_or((Address::ZERO, u64::MAX))
        };

        let mut all_slots: Vec<(usize, (Address, u64))> =
            common.iter().map(|&oi| (oi, rep_slot(oi))).collect();
        all_slots.sort_by(|a, b| a.1.cmp(&b.1));
        let slot_rank: HashMap<usize, usize> = all_slots
            .iter()
            .enumerate()
            .map(|(rank, &(oi, _))| (oi, rank))
            .collect();

        let sub_a: Vec<usize> = a
            .nonce_seq
            .iter()
            .copied()
            .filter(|o| common.contains(o))
            .collect();
        let sub_b: Vec<usize> = b
            .nonce_seq
            .iter()
            .copied()
            .filter(|o| common.contains(o))
            .collect();

        let mapped_a: Vec<usize> = sub_a.iter().map(|oi| slot_rank[oi]).collect();
        let mapped_b: Vec<usize> = sub_b.iter().map(|oi| slot_rank[oi]).collect();

        kendall_tau_distance(&mapped_a, &mapped_b)
    };

    (1.0 - w_choice) * ordering_tau + w_choice * choice_mismatch
}

/// Normalized Kendall-tau distance between two sequences of the same elements.
pub fn kendall_tau_distance(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    if n <= 1 {
        return 0.0;
    }

    let mut pos_in_b: HashMap<usize, usize> = HashMap::default();
    for (i, &oi) in b.iter().enumerate() {
        pos_in_b.insert(oi, i);
    }

    let mut mapped: Vec<usize> = a
        .iter()
        .map(|&oi| pos_in_b.get(&oi).copied().unwrap_or(0))
        .collect();

    let mut buf = vec![0usize; n];
    let inv = count_inversions(&mut mapped, &mut buf) as f64;
    let max_inv = (n as f64) * ((n as f64) - 1.0) / 2.0;
    if max_inv == 0.0 {
        0.0
    } else {
        inv / max_inv
    }
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

pub struct PendingChild {
    pub island_idx: usize,
    pub pair_idx: usize,
    pub child_slot: usize,
    pub ind: Individual,
}

pub struct ParentPairInfo {
    pub p1_idx: usize,
    pub p2_idx: usize,
}

/// Generate all children for one island without evaluating them.
/// Applies crossover then mutation on raw_seq, then derives nonce_seq once per child.
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
            (crossover(p1, p2, rng), crossover(p2, p1, rng))
        } else {
            (p1.clone(), p2.clone())
        };

        mutate(&mut c1, rng, params.mutation_rate);
        mutate(&mut c2, rng, params.mutation_rate);

        c1.nonce_seq = derive_nonce_seq(&c1.raw_seq, deps);
        c2.nonce_seq = derive_nonce_seq(&c2.raw_seq, deps);

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

/// Apply DC competition using evaluated children.
pub fn apply_dc_competition(
    island: &mut Island,
    pair_infos: &[ParentPairInfo],
    evaluated_children: &mut Vec<Option<Individual>>,
    deps: &GroupDeps,
    w_choice: f64,
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

        let dist_p1c1 = nonce_seq_dc_distance(p1, &c1, deps, w_choice);
        let dist_p2c2 = nonce_seq_dc_distance(p2, &c2, deps, w_choice);
        let dist_p1c2 = nonce_seq_dc_distance(p1, &c2, deps, w_choice);
        let dist_p2c1 = nonce_seq_dc_distance(p2, &c1, deps, w_choice);

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

    fn build_deps(group: &ConflictGroup) -> GroupDeps {
        GroupDeps::from_group(group).unwrap()
    }

    fn assert_valid_permutation(seq: &[usize], n: usize) {
        let mut seen = vec![false; n];
        assert_eq!(seq.len(), n, "Permutation has wrong length");
        for &x in seq {
            assert!(x < n, "Element {} out of range 0..{}", x, n);
            assert!(!seen[x], "Duplicate element {} in permutation", x);
            seen[x] = true;
        }
    }

    fn assert_valid_nonce_seq(nonce_seq: &[usize], deps: &GroupDeps) {
        let active: AHashSet<usize> = nonce_seq.iter().copied().collect();
        let dag = deps.build_dag(&active);
        let pos: HashMap<usize, usize> = nonce_seq.iter().enumerate().map(|(p, &oi)| (oi, p)).collect();
        for (ni, succs) in dag.successors.iter().enumerate() {
            let from = dag.nodes[ni];
            for &si in succs {
                let to = dag.nodes[si];
                assert!(pos[&from] < pos[&to], "{} must come before {} in {:?}", from, to, nonce_seq);
            }
        }
    }

    // individual_from_seq

    #[test]
    fn individual_from_seq_creates_valid_individual() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 50, &mut gen),
        ]);
        let deps = build_deps(&group);
        let ind = individual_from_seq(vec![3, 1, 0, 2], &deps);
        assert_eq!(ind.raw_seq, vec![3, 1, 0, 2]);
        assert_eq!(ind.profit, U256::ZERO);
        assert_eq!(ind.gas, 0);
        assert_valid_nonce_seq(&ind.nonce_seq, &deps);
    }

    #[test]
    fn individual_from_seq_empty() {
        let group = mk_group(vec![]);
        let deps = GroupDeps::from_group(&group).unwrap_or_else(|| {
            let g2 = mk_group(vec![]);
            GroupDeps::from_group(&g2).unwrap()
        });
        let ind = individual_from_seq(vec![], &deps);
        assert!(ind.raw_seq.is_empty());
        assert!(ind.nonce_seq.is_empty());
    }

    // dominates
    #[test]
    fn dominates_higher_profit() {
        let a = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(100), gas: 50 };
        let b = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(50), gas: 50 };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_same_profit_lower_gas() {
        let a = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(100), gas: 30 };
        let b = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(100), gas: 50 };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_neither() {
        let a = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(100), gas: 50 };
        let b = Individual { raw_seq: vec![], nonce_seq: vec![], profit: U256::from(100), gas: 50 };
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    // compete
    #[test]
    fn compete_picks_dominant() {
        let mut rng = SmallRng::seed_from_u64(1);
        let better = Individual { raw_seq: vec![0, 1], nonce_seq: vec![0, 1], profit: U256::from(100), gas: 50 };
        let worse = Individual { raw_seq: vec![1, 0], nonce_seq: vec![1, 0], profit: U256::from(50), gas: 100 };
        for _ in 0..20 {
            let winner = compete(&worse, &better, &mut rng);
            assert_eq!(winner.profit, U256::from(100));
        }
    }

    #[test]
    fn compete_equal_is_random() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = Individual { raw_seq: vec![0, 1], nonce_seq: vec![0, 1], profit: U256::from(100), gas: 50 };
        let b = Individual { raw_seq: vec![1, 0], nonce_seq: vec![1, 0], profit: U256::from(100), gas: 50 };
        let mut a_wins = 0;
        let mut b_wins = 0;
        for _ in 0..200 {
            let winner = compete(&a, &b, &mut rng);
            if winner.raw_seq == a.raw_seq { a_wins += 1; } else { b_wins += 1; }
        }
        assert!(a_wins > 50 && b_wins > 50,
            "Expected roughly even split, got a={} b={}", a_wins, b_wins);
    }

    // derive_nonce_seq

    #[test]
    fn derive_nonce_seq_no_conflicts() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let nonce_seq = derive_nonce_seq(&[2, 0, 1], &deps);
        assert_valid_nonce_seq(&nonce_seq, &deps);
        assert_eq!(nonce_seq.len(), 3);
        let pos: HashMap<usize, usize> = nonce_seq.iter().enumerate().map(|(p, &oi)| (oi, p)).collect();
        assert!(pos[&0] < pos[&1], "A@0 must come before A@1");
    }

    #[test]
    fn derive_nonce_seq_conflict_first_wins() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1 (conflict)
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // idx 2
        ]);
        let deps = build_deps(&group);
        // raw_seq puts idx 0 before idx 1 → idx 0 wins conflict
        let nonce_seq = derive_nonce_seq(&[0, 1, 2], &deps);
        assert!(nonce_seq.contains(&0));
        assert!(!nonce_seq.contains(&1));
        assert!(nonce_seq.contains(&2));
    }

    #[test]
    fn derive_nonce_seq_conflict_second_wins() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1 (conflict)
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // idx 2
        ]);
        let deps = build_deps(&group);
        // raw_seq puts idx 1 before idx 0 → idx 1 wins conflict
        let nonce_seq = derive_nonce_seq(&[1, 0, 2], &deps);
        assert!(nonce_seq.contains(&1));
        assert!(!nonce_seq.contains(&0));
        assert!(nonce_seq.contains(&2));
    }

    #[test]
    fn derive_nonce_seq_excluded_predecessor() {
        let mut gen = IdGen::new();
        // idx 0: A@0 (conflict winner in some raw_seqs)
        // idx 1: A@0 (conflict loser)
        // idx 2: A@1 (requires (A,0) — provided by whoever wins the conflict)
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 300, &mut gen),
        ]);
        let deps = build_deps(&group);
        // idx 0 wins → idx 2 depends on idx 0
        let nonce_seq = derive_nonce_seq(&[0, 1, 2], &deps);
        assert!(nonce_seq.contains(&0));
        assert!(!nonce_seq.contains(&1));
        assert!(nonce_seq.contains(&2));
        assert_valid_nonce_seq(&nonce_seq, &deps);
    }

    #[test]
    fn derive_nonce_seq_fully_conflicted() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 0, 300, &mut gen),
        ]);
        let deps = build_deps(&group);
        let nonce_seq = derive_nonce_seq(&[2, 1, 0], &deps);
        assert_eq!(nonce_seq.len(), 1);
        assert_eq!(nonce_seq[0], 2);
    }

    // Crossover

    #[test]
    fn crossover_independent_orders_valid_permutation() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 1, 2], &deps);
        let pb = individual_from_seq(vec![2, 0, 1], &deps);

        for _ in 0..100 {
            let child = crossover(&pa, &pb, &mut rng);
            assert_valid_permutation(&child.raw_seq, 3);
            assert!(child.nonce_seq.is_empty());
            assert_eq!(child.profit, U256::ZERO);
        }
    }

    #[test]
    fn crossover_nonce_chain_child_derives_valid_nonce_seq() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // 0
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen), // 1
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen), // 2
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // 3
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen), // 4
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 3, 1, 4, 2], &deps);
        let pb = individual_from_seq(vec![3, 0, 4, 1, 2], &deps);

        for _ in 0..200 {
            let mut child = crossover(&pa, &pb, &mut rng);
            assert_valid_permutation(&child.raw_seq, 5);
            child.nonce_seq = derive_nonce_seq(&child.raw_seq, &deps);
            assert_valid_nonce_seq(&child.nonce_seq, &deps);
        }
    }

    #[test]
    fn crossover_single_element() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![mk_single_tx_order(SENDER_A, 0, 100, &mut gen)]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(42);
        let pa = individual_from_seq(vec![0], &deps);
        let pb = individual_from_seq(vec![0], &deps);
        let child = crossover(&pa, &pb, &mut rng);
        assert_eq!(child.raw_seq, vec![0]);
    }

    #[test]
    fn crossover_produces_diverse_children() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let pa = individual_from_seq(vec![0, 1, 2], &deps);
        let pb = individual_from_seq(vec![2, 1, 0], &deps);

        let mut unique_seqs: AHashSet<Vec<usize>> = AHashSet::default();
        for _ in 0..100 {
            let child = crossover(&pa, &pb, &mut rng);
            unique_seqs.insert(child.raw_seq);
        }
        assert!(unique_seqs.len() > 1,
            "Crossover should produce diverse children, got {} unique", unique_seqs.len());
    }

    // Mutation

    #[test]
    fn mutation_produces_valid_permutation() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(123);

        let mut ind = individual_from_seq(vec![0, 1, 2], &deps);
        for _ in 0..200 {
            mutate(&mut ind, &mut rng, 0.3);
            assert_valid_permutation(&ind.raw_seq, 3);
        }
    }

    #[test]
    fn mutation_derived_nonce_seq_is_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(456);

        let mut ind = individual_from_seq(vec![0, 1, 3, 2], &deps);
        for _ in 0..200 {
            mutate(&mut ind, &mut rng, 0.5);
            ind.nonce_seq = derive_nonce_seq(&ind.raw_seq, &deps);
            assert_valid_nonce_seq(&ind.nonce_seq, &deps);
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
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(789);

        let original = vec![0, 1, 2];
        let mut changed = false;
        for _ in 0..50 {
            let mut ind = individual_from_seq(original.clone(), &deps);
            mutate(&mut ind, &mut rng, 0.5);
            if ind.raw_seq != original {
                changed = true;
                break;
            }
        }
        assert!(changed, "Mutation should eventually change the sequence");
    }

    #[test]
    fn mutation_can_produce_zero_changes() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(42);

        let original = vec![0, 1];
        let mut unchanged_seen = false;
        for _ in 0..200 {
            let mut ind = individual_from_seq(original.clone(), &deps);
            mutate(&mut ind, &mut rng, 0.01);
            if ind.raw_seq == original {
                unchanged_seen = true;
                break;
            }
        }
        assert!(unchanged_seen, "Low mutation rate should sometimes produce no change");
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
        let b = vec![0, 2, 1, 3];
        let d = kendall_tau_distance(&a, &b);
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

    // validate_individual
    #[test]
    #[cfg(debug_assertions)]
    fn validate_individual_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);
        let deps = build_deps(&group);
        let ind = individual_from_seq(vec![0, 2, 1], &deps);
        assert!(validate_individual(&ind, &deps));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_individual_wrong_order_invalid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
        ]);
        let deps = build_deps(&group);
        let mut ind = individual_from_seq(vec![0, 1], &deps);
        // Manually corrupt nonce_seq to have wrong order
        ind.nonce_seq = vec![1, 0];
        assert!(!validate_individual(&ind, &deps));
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
        let deps = build_deps(&group);

        let params = GAParams {
            population: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            max_generations: 10,
            time_ms: 1000,
            seed: 42,
            num_islands: 1,
            migration_interval: 5,
            w_choice: 0.3,
            early_stopping_generations: 10,
            temp_tight_low: 0.5,
            temp_tight_high: 2.0,
            temp_broad_low: 3.0,
            temp_broad_high: 8.0,
            tight_fraction: 0.6,
        };

        let pop: Vec<Individual> = (0..10).map(|_| {
            individual_from_seq(vec![0, 1, 2], &deps)
        }).collect();

        let mut island = Island {
            population: pop,
            rng: SmallRng::seed_from_u64(42),
        };

        let (children, pair_infos) = generate_dc_children(&mut island, 0, &params, &deps);

        assert_eq!(children.len(), 10);
        assert_eq!(pair_infos.len(), 5);

        for child in &children {
            assert_valid_permutation(&child.ind.raw_seq, 3);
            assert_valid_nonce_seq(&child.ind.nonce_seq, &deps);
        }
    }

    #[test]
    fn generate_dc_children_odd_population() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let deps = build_deps(&group);

        let params = GAParams {
            population: 7,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            max_generations: 10,
            time_ms: 1000,
            seed: 42,
            num_islands: 1,
            migration_interval: 5,
            w_choice: 0.3,
            early_stopping_generations: 10,
            temp_tight_low: 0.5,
            temp_tight_high: 2.0,
            temp_broad_low: 3.0,
            temp_broad_high: 8.0,
            tight_fraction: 0.6,
        };

        let pop: Vec<Individual> = (0..7).map(|_| {
            individual_from_seq(vec![0, 1], &deps)
        }).collect();

        let mut island = Island {
            population: pop,
            rng: SmallRng::seed_from_u64(42),
        };

        let (children, pair_infos) = generate_dc_children(&mut island, 0, &params, &deps);

        assert_eq!(children.len(), 6);
        assert_eq!(pair_infos.len(), 3);
    }

    // DC competition
    #[test]
    fn dc_competition_replaces_with_better_child() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let deps = build_deps(&group);

        let p1 = Individual { raw_seq: vec![0, 1], nonce_seq: vec![0, 1], profit: U256::from(50), gas: 100 };
        let p2 = Individual { raw_seq: vec![1, 0], nonce_seq: vec![1, 0], profit: U256::from(60), gas: 90 };
        let c1 = Individual { raw_seq: vec![0, 1], nonce_seq: vec![0, 1], profit: U256::from(200), gas: 50 };
        let c2 = Individual { raw_seq: vec![1, 0], nonce_seq: vec![1, 0], profit: U256::from(10), gas: 200 };

        let mut island = Island {
            population: vec![p1, p2],
            rng: SmallRng::seed_from_u64(42),
        };

        let pair_infos = vec![ParentPairInfo { p1_idx: 0, p2_idx: 1 }];
        let mut evaluated = vec![Some(c1), Some(c2)];

        apply_dc_competition(&mut island, &pair_infos, &mut evaluated, &deps, 0.3);

        assert_eq!(island.population.len(), 2);
        assert_eq!(island.population[0].profit, U256::from(200));
        assert_eq!(island.population[1].profit, U256::from(60));
    }

    #[test]
    fn dc_competition_handles_failed_eval() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        let deps = build_deps(&group);

        let p1 = Individual { raw_seq: vec![0, 1], nonce_seq: vec![0, 1], profit: U256::from(50), gas: 100 };
        let p2 = Individual { raw_seq: vec![1, 0], nonce_seq: vec![1, 0], profit: U256::from(60), gas: 90 };

        let mut island = Island {
            population: vec![p1, p2],
            rng: SmallRng::seed_from_u64(42),
        };

        let pair_infos = vec![ParentPairInfo { p1_idx: 0, p2_idx: 1 }];
        let mut evaluated: Vec<Option<Individual>> = vec![None, None];

        apply_dc_competition(&mut island, &pair_infos, &mut evaluated, &deps, 0.3);

        assert_eq!(island.population.len(), 2);
        assert_eq!(island.population[0].profit, U256::from(50));
        assert_eq!(island.population[1].profit, U256::from(60));
    }

    // Stress tests
    #[test]
    fn stress_crossover_mutation_interleaved_chains() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // 0
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),                  // 1
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),                  // 2
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // 3
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),                  // 4
            mk_bundle_order(&[(SENDER_A, 3), (SENDER_B, 2)], 500, &mut gen), // 5
        ]);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(54321);

        let pa = individual_from_seq(vec![0, 3, 1, 4, 2, 5], &deps);
        let pb = individual_from_seq(vec![3, 0, 4, 1, 2, 5], &deps);

        for _ in 0..500 {
            let mut child = crossover(&pa, &pb, &mut rng);
            assert_valid_permutation(&child.raw_seq, 6);
            child.nonce_seq = derive_nonce_seq(&child.raw_seq, &deps);
            assert_valid_nonce_seq(&child.nonce_seq, &deps);

            mutate(&mut child, &mut rng, 0.3);
            child.nonce_seq = derive_nonce_seq(&child.raw_seq, &deps);
            assert_valid_nonce_seq(&child.nonce_seq, &deps);

            let grandchild_raw = crossover(&child, &pa, &mut rng);
            assert_valid_permutation(&grandchild_raw.raw_seq, 6);
        }
    }

    #[test]
    fn stress_many_independent_orders() {
        let mut gen = IdGen::new();
        let orders: Vec<_> = (0..20).map(|i| {
            let sender = Address::from_word(U256::from(i + 100).into());
            mk_single_tx_order(sender, 0, (i + 1) as u64 * 10, &mut gen)
        }).collect();
        let group = mk_group(orders);
        let deps = build_deps(&group);
        let mut rng = SmallRng::seed_from_u64(99999);

        let seq_a: Vec<usize> = (0..20).collect();
        let seq_b: Vec<usize> = (0..20).rev().collect();
        let pa = individual_from_seq(seq_a, &deps);
        let pb = individual_from_seq(seq_b, &deps);

        for _ in 0..200 {
            let mut child = crossover(&pa, &pb, &mut rng);
            assert_valid_permutation(&child.raw_seq, 20);
            child.nonce_seq = derive_nonce_seq(&child.raw_seq, &deps);
            assert_valid_nonce_seq(&child.nonce_seq, &deps);
            mutate(&mut child, &mut rng, 0.15);
            child.nonce_seq = derive_nonce_seq(&child.raw_seq, &deps);
            assert_valid_nonce_seq(&child.nonce_seq, &deps);
        }
    }
}
