use ahash::HashMap;
use alloy_primitives::U256;
use rand::rngs::SmallRng;
use rand::Rng;
use super::nonce_interleavings::*;

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

pub fn repair_to_nonce_valid(preferred: &[usize], layout: &NonceLayout) -> Vec<usize> {
    // Project a single preference list onto a valid interleaving.
    // Internally this uses the same ready-set selection as crossover.
    ppx_build_child_from_parents_greedy(preferred, preferred, layout)
}

#[derive(Clone)]
pub struct Individual {
    pub seq: Vec<usize>,
    pub fitness: Option<U256>,
}

pub fn tournament_select<'p>(
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

/// Count inversions via mergesort; returns inversions count.
pub fn count_inversions(arr: &mut [usize], buf: &mut [usize]) -> u64 {
    let n = arr.len();
    if n <= 1 { return 0; }
    let mid = n / 2;
    let (left, right) = arr.split_at_mut(mid);
    let (buf_left, buf_right) = buf.split_at_mut(mid);

    let inv_left  = count_inversions(left,  buf_left);
    let inv_right = count_inversions(right, buf_right);

    // Merge from buf back into arr
    buf[..mid].copy_from_slice(left);
    buf[mid..].copy_from_slice(right);

    let mut i = 0;
    let mut j = mid;
    let mut k = 0;
    let mut inv = inv_left + inv_right;

    while i < mid && j < n {
        if buf[i] <= buf[j] {
            arr[k] = buf[i]; i += 1;
        } else {
            arr[k] = buf[j]; j += 1;
            inv += (mid - i) as u64;
        }
        k += 1;
    }
    while i < mid { arr[k] = buf[i]; i += 1; k += 1; }
    while j < n   { arr[k] = buf[j]; j += 1; k += 1; }

    inv
}

/// Build per-chain offsets so each (chain, step) maps to a dense slot id in [0, total_steps).
pub fn chain_offsets(layout: &NonceLayout) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(layout.chains.len());
    let mut acc = 0usize;
    for c in 0..layout.chains.len() {
        offsets.push(acc);
        acc += layout.chains[c].steps.len();
    }
    offsets
}

/// Map a concrete sequence of candidate indices into slot ids (chain,step) -> dense id.
/// This makes sequences comparable even if they pick different candidates for the same slot.
fn seq_to_slot_ids(seq: &[usize], layout: &NonceLayout, offsets: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(seq.len());
    for &idx in seq {
        let (c, s) = layout.index_of[&idx]; // every chosen idx must exist
        out.push(offsets[c] + s);
    }
    out
}

/// Normalized Kendall-tau distance in [0,1].
/// We can't compare sequences directly because they may pick different candidates for the same sender/nonce,
/// so we first map the ids in the sequences to slot ids (chain, step).
fn kendall_tau_distance_slots_norm(
    a: &[usize],
    b: &[usize],
    layout: &NonceLayout,
    offsets: &[usize],
) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n <= 1 { return 0.0; }

    let total_steps = layout.total_steps;

    // Convert sequences to slot ids
    let mut a_slots = seq_to_slot_ids(a, layout, offsets);
    let b_slots = seq_to_slot_ids(b, layout, offsets);

    // Build position map for b over all slots
    let mut pos_in_b = vec![0usize; total_steps];
    for (i, &slot) in b_slots.iter().enumerate() {
        pos_in_b[slot] = i;
    }

    // Map a_slots to positions in b
    for v in &mut a_slots {
        *v = pos_in_b[*v];
    }

    // Count inversions
    let mut buf = vec![0usize; n];
    let inv = count_inversions(&mut a_slots, &mut buf) as f64;
    let max_inv = (n as u128 * (n as u128 - 1) / 2) as f64;
    inv / max_inv
}

fn candidate_mismatch_rate(a: &[usize], b: &[usize], layout: &NonceLayout, offsets: &[usize]) -> f64 {
    let n = a.len();
    if n == 0 { return 0.0; }

    // For each sequence, build slot -> chosen candidate id
    let mut pick_a = vec![usize::MAX; n];
    for &idx in a {
        let (c, s) = layout.index_of[&idx];
        pick_a[offsets[c] + s] = idx;
    }
    let mut pick_b = vec![usize::MAX; n];
    for &idx in b {
        let (c, s) = layout.index_of[&idx];
        pick_b[offsets[c] + s] = idx;
    }

    let mismatches = pick_a.iter().zip(pick_b.iter()).filter(|(x, y)| *x != *y).count();
    mismatches as f64 / n as f64
}

/// Combined distance for deterministic crowding
pub fn dc_distance(a: &[usize], b: &[usize], layout: &NonceLayout, offsets: &[usize], w_order: f64, w_id: f64) -> f64 {
    let tau = kendall_tau_distance_slots_norm(a, b, layout, offsets);
    let idm = candidate_mismatch_rate(a, b, layout, offsets);
    w_order * tau + w_id * idm
}

/// Build child interleaving using a ready-set, prioritising by parent ranks.
/// - Each chain contributes exactly one candidate per slot (consumes slot on pick).
/// - Ready set = candidates in current slot for each chain.
/// - Priority = min(rank_in_parent_a, rank_in_parent_b), tie-break by sum then by idx.
/// Build a nonce-valid child by precedence-preserving selection from the
/// ready set (current slot of each chain), guided by ranks in both parents.
pub fn ppx_build_child_from_parents_greedy(
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
pub fn ppx_build_child_from_parents(
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
pub fn mut_adjacent_interchain_swap(seq: &mut [usize], layout: &NonceLayout, rng: &mut SmallRng) -> bool {
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

pub fn mut_bubble_move(seq: &mut [usize], layout: &NonceLayout, rng: &mut SmallRng, max_steps: usize) -> bool {
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

pub fn mutation_inter_sender_swap(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng) {
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

pub fn mutation_same_nonce_flip(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng) {
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

pub fn mut_same_nonce_flip(seq: &mut [usize], layout: &NonceLayout, multi: &[(usize,usize)], rng: &mut SmallRng) -> bool {
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

pub fn mutate(seq: &mut Vec<usize>, layout: &NonceLayout, rng: &mut SmallRng, multi: &[(usize,usize)]) {
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

    #[test]
    fn test_nonce_buckets_and_ppx_child_valid() {
        // Build group with duplicate same-nonce choices
        let group = make_group_with_duplicate_buckets();

        // Buckets
        let layout = NonceLayout::from_group(&group).unwrap();
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

        // Start from a valid interleaving (best-per-slot, simple A0,B0,A1,B1)
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);
        let seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &layout);

        // There are at least two slots with multiple candidates (A:nonce0, B:nonce1).
        let mut rng = SmallRng::seed_from_u64(12345);

        // Try flipping until we observe a change
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
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        let mut seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &layout);

        let mut rng = SmallRng::seed_from_u64(777);
        mutation_inter_sender_swap(&mut seq, &layout, &mut rng);

        // Must remain valid
        assert_nonce_valid(&seq, &layout);
    }

    #[test]
    fn test_seq_to_slot_ids_same_order_different_candidates_maps_equal() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();

        // Build two sequences with the SAME slot order but different candidates where possible:
        // Slots: A0(2 choices), B0(1), A1(1), B1(2)
        let chains_best = build_chains_best(&layout, &group);
        // seq1: A0(alt1), B0, A1, B1(alt1)
        let seq1 = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        // seq2: A0(alt2), B0, A1, B1(alt2)  (flip candidates inside same slots)
        // For A0 choose any other candidate in slot A0; for B1 choose any other candidate
        let a0_bucket = &layout.chains[0].steps[0].candidates;
        let a0_alt = if a0_bucket[0] == chains_best[0][0] { a0_bucket[1] } else { a0_bucket[0] };
        let b1_bucket = &layout.chains[1].steps[1].candidates;
        let b1_alt = if b1_bucket[0] == chains_best[1][1] { b1_bucket[1] } else { b1_bucket[0] };
        let seq2 = vec![a0_alt, chains_best[1][0], chains_best[0][1], b1_alt];

        assert_nonce_valid(&seq1, &layout);
        assert_nonce_valid(&seq2, &layout);

        let offs = chain_offsets(&layout);
        let s1 = seq_to_slot_ids(&seq1, &layout, &offs);
        let s2 = seq_to_slot_ids(&seq2, &layout, &offs);
        assert_eq!(s1, s2, "slot mapping must ignore candidate identity");
    }

    #[test]
    fn test_kendall_tau_slots_zero_when_only_candidates_differ() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Same interleaving, different candidates in two slots
        let a0_bucket = &layout.chains[0].steps[0].candidates;
        let a0_alt = if a0_bucket[0] == chains_best[0][0] { a0_bucket[1] } else { a0_bucket[0] };
        let b1_bucket = &layout.chains[1].steps[1].candidates;
        let b1_alt = if b1_bucket[0] == chains_best[1][1] { b1_bucket[1] } else { b1_bucket[0] };

        let seq1 = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        let seq2 = vec![a0_alt,               chains_best[1][0], chains_best[0][1], b1_alt];
        let offs = chain_offsets(&layout);
        let tau = kendall_tau_distance_slots_norm(&seq1, &seq2, &layout, &offs);
        assert!(
            (tau - 0.0).abs() < 1e-12,
            "Kendall-τ over slots should be zero when slot order is identical (tau={})",
            tau
        );
    }

    #[test]
    fn test_kendall_tau_slots_positive_when_slot_order_differs() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Two different interleavings:
        // seqA: A0, B0, A1, B1
        let seq_a = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        // seqB: B0, A0, B1, A1
        let seq_b = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]];
        let offs = chain_offsets(&layout);
        let tau = kendall_tau_distance_slots_norm(&seq_a, &seq_b, &layout, &offs);
        assert!(tau > 0.0, "Kendall-τ should be >0 when slot order differs, got {}", tau);
        assert!(tau <= 1.0 + 1e-12);
    }

    #[test]
    fn test_candidate_mismatch_rate_counts_only_candidate_changes() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Same slot order, flip candidates in two slots -> mismatch rate should be 2/4 = 0.5
        let a0_bucket = &layout.chains[0].steps[0].candidates;
        let a0_alt = if a0_bucket[0] == chains_best[0][0] { a0_bucket[1] } else { a0_bucket[0] };
        let b1_bucket = &layout.chains[1].steps[1].candidates;
        let b1_alt = if b1_bucket[0] == chains_best[1][1] { b1_bucket[1] } else { b1_bucket[0] };

        let seq1 = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        let seq2 = vec![a0_alt,               chains_best[1][0], chains_best[0][1], b1_alt];

        let offs = super::chain_offsets(&layout);
        let m = candidate_mismatch_rate(&seq1, &seq2, &layout, &offs);
        assert!((m - 0.5).abs() < 1e-12, "expected mismatch rate 0.5, got {}", m);

        // If only order changes but candidates remain the same, mismatch rate = 0
        let seq3 = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]];
        let m2 = candidate_mismatch_rate(&seq1, &seq3, &layout, &offs);
        assert!((m2 - 0.0).abs() < 1e-12, "expected mismatch rate 0.0 when candidates equal, got {}", m2);
    }

    #[test]
    fn test_dc_distance_identity_symmetry_and_weights() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Base sequence
        let base = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];

        // Order-changed (swap A0,B0), same candidates
        let order_diff = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]];

        // Candidate-changed (same order, flip A0 & B1 candidates)
        let a0_bucket = &layout.chains[0].steps[0].candidates;
        let a0_alt = if a0_bucket[0] == chains_best[0][0] { a0_bucket[1] } else { a0_bucket[0] };
        let b1_bucket = &layout.chains[1].steps[1].candidates;
        let b1_alt = if b1_bucket[0] == chains_best[1][1] { b1_bucket[1] } else { b1_bucket[0] };
        let id_diff = vec![a0_alt, chains_best[1][0], chains_best[0][1], b1_alt];

        let offs = super::chain_offsets(&layout);

        // Identity
        let d_id = super::dc_distance(&base, &base, &layout, &offs, 0.8, 0.2);
        assert!((d_id - 0.0).abs() < 1e-12, "dc_distance(x,x) must be 0");

        // Symmetry
        let d_sym1 = super::dc_distance(&base, &order_diff, &layout, &offs, 0.8, 0.2);
        let d_sym2 = super::dc_distance(&order_diff, &base, &layout, &offs, 0.8, 0.2);
        assert!((d_sym1 - d_sym2).abs() < 1e-12, "dc_distance must be symmetric");

        // Weight sensitivity: emphasize order
        let d_order_heavy_order = super::dc_distance(&base, &order_diff, &layout, &offs, 0.9, 0.1);
        let d_order_heavy_id    = super::dc_distance(&base, &id_diff,    &layout, &offs, 0.9, 0.1);
        assert!(d_order_heavy_order > d_order_heavy_id, "with high w_order, order changes should dominate");

        // Weight sensitivity: emphasize candidate identity
        let d_id_heavy_order = super::dc_distance(&base, &order_diff, &layout, &offs, 0.1, 0.9);
        let d_id_heavy_id    = super::dc_distance(&base, &id_diff,    &layout, &offs, 0.1, 0.9);
        assert!(d_id_heavy_id > d_id_heavy_order, "with high w_id, candidate changes should dominate");
    }

    #[test]
    fn test_dc_pairing_prefers_min_total_distance() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Parents
        let p1 = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]]; // A0, B0, A1, B1
        let p2 = vec![chains_best[1][0], chains_best[0][0], chains_best[1][1], chains_best[0][1]]; // B0, A0, B1, A1

        // Children near each parent
        // c1: small perturbation of p1 (flip candidate on A0)
        let a0_bucket = &layout.chains[0].steps[0].candidates;
        let a0_alt = if a0_bucket[0] == chains_best[0][0] { a0_bucket[1] } else { a0_bucket[0] };
        let c1 = vec![a0_alt, chains_best[1][0], chains_best[0][1], chains_best[1][1]];

        // c2: small perturbation of p2 (flip candidate on B1)
        let b1_bucket = &layout.chains[1].steps[1].candidates;
        let b1_alt = if b1_bucket[0] == chains_best[1][1] { b1_bucket[1] } else { b1_bucket[0] };
        let c2 = vec![chains_best[1][0], chains_best[0][0], b1_alt, chains_best[0][1]];

        let w_order = 0.8;
        let w_id = 0.2;

        let offs = chain_offsets(&layout);

        let d11 = dc_distance(&c1, &p1, &layout, &offs, w_order, w_id);
        let d12 = dc_distance(&c1, &p2, &layout, &offs, w_order, w_id);
        let d21 = dc_distance(&c2, &p1, &layout, &offs, w_order, w_id);
        let d22 = dc_distance(&c2, &p2, &layout, &offs, w_order, w_id);

        // Deterministic crowding should choose (c1<->p1, c2<->p2)
        let total_pairing_1 = d11 + d22;
        let total_pairing_2 = d12 + d21;
        assert!(
            total_pairing_1 <= total_pairing_2 + 1e-12,
            "expected pairing (p1,c1) & (p2,c2) to minimize total distance: {} <= {}",
            total_pairing_1, total_pairing_2
        );
    }

    #[test]
    fn test_mut_bubble_move_preserves_validity() {
        let group = make_group_with_duplicate_buckets();
        let layout = NonceLayout::from_group(&group).unwrap();
        let chains_best = build_chains_best(&layout, &group);

        // Start with valid sequence
        let seq = vec![chains_best[0][0], chains_best[1][0], chains_best[0][1], chains_best[1][1]];
        assert_nonce_valid(&seq, &layout);

        let mut rng = SmallRng::seed_from_u64(4242);

        // Try multiple times to actually cause a change (bounded attempts)
        let mut changed_once = false;
        for _ in 0..50 {
            let mut tmp = seq.clone();
            let changed = super::mut_bubble_move(&mut tmp, &layout, &mut rng, 3);
            if changed {
                assert_nonce_valid(&tmp, &layout);
                changed_once = true;
                break;
            }
        }
        assert!(changed_once, "bubble move did not produce any change across attempts");
    }

}
