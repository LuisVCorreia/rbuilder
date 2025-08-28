use serde::Serialize;
use ahash::{HashMap, HashSet};
use num_bigint::BigUint;
use num_traits::One;

use super::{
    ConflictGroup,
    task::{ConflictTask, Algorithm},
    conflict_resolvers::generate_sequences_of_orders_to_try,
    nonce_interleavings::NonceLayout,
};

#[derive(Serialize)]
struct MultinomialStat {
    exact: String, // decimal
    log10: f64,
}

#[derive(Serialize)]
struct AlgoCount {
    algorithm: String,
    planned_sequences: usize,
}

#[derive(Serialize)]
pub struct GroupMetrics {
    pub group_id: usize,
    pub group_size: usize,
    pub num_senders: usize,
    pub sender_chain_lengths: Vec<usize>, // per-sender number of nonce steps (dedup’d)
    multinomial_permutations: MultinomialStat, // interleavings of nonce steps only (no per-step choices)
    pub rbuilder_total_sequences_planned: usize,
    pub rbuilder_unique_input_sequences: usize,
    pub rbuilder_unique_executed_sequences: usize, // after canonicalization, full-length
    per_algo: Vec<AlgoCount>,
}

#[derive(Serialize)]
pub struct CacheStats {
    pub full_hits: usize,
    pub partial_hits: usize,
    pub saved: usize,
    pub requested: usize,
    pub requests: usize,
    pub rate_full_hits_pct: f64,
    pub rate_partial_hits_pct: f64,
    pub efficiency_pct: f64,
}

#[derive(Serialize)]
pub struct BlockMetrics {
    pub block_number: u64,
    pub parent_hash: String,
    pub groups_analyzed: usize,
    pub groups_skipped_size_1: usize,
    pub groups: Vec<GroupMetrics>,
    pub simulation_cache: CacheStats,
}

fn algo_name(a: &Algorithm) -> &'static str {
    match a {
        Algorithm::Greedy => "Greedy",
        Algorithm::ReverseGreedy => "ReverseGreedy",
        Algorithm::Length => "Length",
        Algorithm::AllPermutations => "AllPermutations",
        Algorithm::Random { .. } => "Random",
        Algorithm::Genetic { .. } => "Genetic",
        Algorithm::ExhaustiveStreaming { .. } => "ExhaustiveStreaming",
    }
}

fn binom_big(n: usize, k: usize) -> BigUint {
    let k = k.min(n - k);
    let mut res = BigUint::one();
    for i in 1..=k {
        res *= BigUint::from((n - k + i) as u64);
        res /= BigUint::from(i as u64);
    }
    res
}

fn multinomial_exact_and_log10(n: usize, lens: &[usize]) -> (BigUint, f64) {
    // exact = ∏_j C(remaining, k_j)
    let mut remaining = n;
    let mut exact = BigUint::one();
    for &k in lens {
        if k > 0 {
            exact *= binom_big(remaining, k);
            remaining -= k;
        }
    }
    // log10 for intuition
    let log10_fact = |m: usize| (1..=m).map(|i| (i as f64).log10()).sum::<f64>();
    let log10 = log10_fact(n) - lens.iter().map(|&k| log10_fact(k)).sum::<f64>();
    (exact, log10)
}


/// Return a canonical, valid execution order from a raw sequence:
/// - choose at most one tx per (sender,nonce), namely the one that appears earliest in `seq`
/// - within each sender, keep nonce order
/// - across senders (and items with no sender/nonce), order by the raw position of that chosen tx
///
/// If a slot has no candidate present in `seq`, it's skipped (we don't invent one).
pub fn normalize_sequence_to_executable(layout: &NonceLayout, seq: &[usize]) -> Vec<usize> {
    // 1) position of each index in the raw sequence
    let mut pos_of: HashMap<usize, usize> = HashMap::default();
    for (pos, &idx) in seq.iter().enumerate() {
        pos_of.insert(idx, pos);
    }

    // 2) build a set of all indices that belong to any (sender,nonce) slot
    let mut in_any_slot: HashSet<usize> = HashSet::default();
    for chain in &layout.chains {
        for step in &chain.steps {
            for &idx in &step.candidates {
                in_any_slot.insert(idx);
            }
        }
    }

    // 3) for each sender and each slot (nonce asc), pick the representative that appears earliest in `seq`
    //    if none of that slot's candidates appear in `seq`, we skip that slot entirely
    #[derive(Clone)]
    struct SlotRep { idx: usize, pos: usize, slot_id: usize }

    let mut reps_per_sender: Vec<Vec<SlotRep>> = Vec::with_capacity(layout.chains.len());
    for chain in &layout.chains {
        let mut reps: Vec<SlotRep> = Vec::with_capacity(chain.steps.len());
        for (slot_id, step) in chain.steps.iter().enumerate() {
            let mut best: Option<(usize /*idx*/, usize /*pos*/)> = None;
            for &cand in &step.candidates {
                if let Some(&p) = pos_of.get(&cand) {
                    if best.map_or(true, |(_, bp)| p < bp) {
                        best = Some((cand, p));
                    }
                }
            }
            if let Some((idx, pos)) = best {
                reps.push(SlotRep { idx, pos, slot_id });
            }
            // else: no candidate from this slot appears in `seq` -> skip this slot
        }
        reps_per_sender.push(reps);
    }

    // 4) collect “free” items (those not in any slot) in the raw order
    let mut free_items = Vec::new();
    for (pos, &idx) in seq.iter().enumerate() {
        if !in_any_slot.contains(&idx) {
            free_items.push((pos, idx));
        }
    }

    // 5) merge by increasing raw position:
    //    at each step, consider each sender's next slot rep and the next free item,
    //    pick the one with the smallest position
    let mut ptr_per_sender: Vec<usize> = vec![0; reps_per_sender.len()];
    let mut free_ptr = 0usize;
    let mut out: Vec<usize> = Vec::new();

    loop {
        let mut best = None;

        if free_ptr < free_items.len() {
            let (pos, idx) = free_items[free_ptr];
            let cand = (pos, 1u8, usize::MAX, usize::MAX, idx);
            if best.map_or(true, |b| cand < b) {
                best = Some(cand);
            }
        }

        // next slot rep for each sender (if any)
        for (s, reps) in reps_per_sender.iter().enumerate() {
            let p = ptr_per_sender[s];
            if p < reps.len() {
                let r = &reps[p];
                let cand = (r.pos, 0u8, s, r.slot_id, r.idx);
                if best.map_or(true, |b| cand < b) {
                    best = Some(cand);
                }
            }
        }

        let Some((_, kind, s, _, idx)) = best else { break; };

        // emit & advance the corresponding pointer
        out.push(idx);
        if kind == 1 {
            free_ptr += 1;
        } else {
            ptr_per_sender[s] += 1;
        }
    }

    out
}


pub fn build_group_metrics_for_tasks(group: &ConflictGroup, tasks: &[ConflictTask]) -> GroupMetrics {
    // Build the nonce layout once. If None (bundles / multi-tx), we won’t normalize
    let layout_opt = NonceLayout::from_group(group);

    // Per-sender step counts (dedup’d by nonce) + product of per-step candidate counts
    let (lens, n_steps, choice_prod_big, choice_log10): (Vec<usize>, usize, BigUint, f64) =
        if let Some(ref layout) = layout_opt {
            let lens: Vec<usize> = layout.chain_lengths();
            let n_steps: usize = lens.iter().sum();

            // Multiply number of candidates in each NONCE STEP (i.e., duplicate-nonce choices)
            let mut prod = BigUint::one();
            let mut log10_sum = 0.0;
            for mult in layout.multiplicities() {
                let c = mult.max(1);
                prod *= BigUint::from(c as u64);
                log10_sum += (c as f64).log10();
            }
            (lens, n_steps, prod, log10_sum)
        } else {
            // No nonce structure available (bundles/mixed). Treat as plain permutations
            let n = group.orders.len();
            (vec![n], n, BigUint::one(), 0.0)
        };

    // Multinomial over slots, then scale by the per-slot choices
    let multinomial_permutations = {
        let (slot_exact, slot_log10) = multinomial_exact_and_log10(n_steps, &lens);
        let exact = slot_exact * choice_prod_big;     // include per-slot choices
        let log10 = slot_log10 + choice_log10;        // add logs
        MultinomialStat { exact: exact.to_str_radix(10), log10 }
    };

    let mut per_algo = Vec::<AlgoCount>::new();
    let mut total = 0usize;

    let mut uniq_inputs: HashSet<Vec<usize>> = HashSet::default();
    let mut uniq_exec:   HashSet<Vec<usize>> = HashSet::default();

    for t in tasks {
        let algo = algo_name(&t.algorithm).to_string();
        let seqs = generate_sequences_of_orders_to_try(t);
        total += seqs.len();
        per_algo.push(AlgoCount { algorithm: algo, planned_sequences: seqs.len() });

        for s in seqs {
            uniq_inputs.insert(s.clone());

            if let Some(ref layout) = layout_opt {
                let norm = normalize_sequence_to_executable(layout, &s);
                uniq_exec.insert(norm);
            } else {
                // No nonce view, count the raw sequence.
                uniq_exec.insert(s);
            }
        }
    }

    let mut lens_sorted = lens.clone();
    lens_sorted.sort_unstable_by(|a, b| b.cmp(a));

    GroupMetrics {
        group_id: group.id,
        group_size: group.orders.len(),
        num_senders: lens.len(),
        sender_chain_lengths: lens_sorted,
        multinomial_permutations,
        rbuilder_total_sequences_planned: total,
        rbuilder_unique_input_sequences: uniq_inputs.len(),
        rbuilder_unique_executed_sequences: uniq_exec.len(),
        per_algo,
    }
}
