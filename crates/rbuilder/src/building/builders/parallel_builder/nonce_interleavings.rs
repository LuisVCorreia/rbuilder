//! Nonce-aware interleaving utilities.
//!
//! Terminology:
//! - A SenderChain is all transactions from one sender, ordered by nonce ascending.
//! - A NonceStep is a single nonce position within a SenderChain.
//!   If a sender submitted multiple distinct txs with the same nonce, that step has multiple candidates.
//!
//! We interleave steps from different chains while preserving per-chain order (nonce increases).
//! When duplicate-nonce candidates exist, we may either:
//!   - pick one candidate per step (best-by metric or random), then interleave (enumerate_best),
//!   - or branch across all per-step candidates as we interleave (enumerate_with_choices).

use ahash::{HashMap, HashSet as AHashSet};
use alloy_primitives::{Address, U256};
use rand::Rng;
use std::collections::BTreeMap;

use crate::primitives::SimulatedOrder;
use super::ConflictGroup;

pub const ALL_PERMS_CAP: usize = 120;

/// (sender, nonce) extraction from a single-tx order
/// Returns None for bundles / multi-tx orders
pub fn sender_and_nonce(order: &SimulatedOrder) -> Option<(Address, u64)> {
    let txs = order.order.list_txs();
    if txs.len() != 1 {
        return None;
    }
    let (tx, _) = &txs[0];
    Some((tx.signer(), tx.nonce()))
}

#[derive(Debug, Clone)]
pub struct NonceStep {
    /// The nonce value at this step (asc for each chain)
    pub nonce: u64,
    /// Candidate order indices for this (sender, nonce)
    pub candidates: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SenderChain {
    /// Sender address for this chain
    pub sender: Address,
    /// Ascending sequence of nonce steps
    pub steps: Vec<NonceStep>,
}

#[derive(Debug, Clone)]
pub struct NonceLayout {
    /// All sender chains in this group (order of chains is arbitrary)
    pub chains: Vec<SenderChain>,
    /// Reverse lookup: order index -> (chain_id, step_id)
    pub index_of: HashMap<usize, (usize, usize)>,
    /// Total number of steps across all chains (i.e., length of any valid interleaving)
    pub total_steps: usize,
}

impl NonceLayout {
    /// Build from `ConflictGroup`. Returns None when we cannot derive (sender,nonce) for some order
    pub fn from_group(group: &ConflictGroup) -> Option<Self> {
        // sender -> (nonce -> [order indices])
        let mut by_sender: BTreeMap<Address, BTreeMap<u64, Vec<usize>>> = BTreeMap::new();

        for (idx, o) in group.orders.iter().enumerate() {
            let (sender, nonce) = sender_and_nonce(o)?;
            by_sender.entry(sender).or_default()
                     .entry(nonce).or_default()
                     .push(idx);
        }

        let mut chains = Vec::with_capacity(by_sender.len());
        let mut index_of: HashMap<usize, (usize, usize)> = HashMap::default();

        for (sender, by_nonce) in by_sender {
            let mut steps = Vec::with_capacity(by_nonce.len());
            for (nonce, mut indices) in by_nonce {
                indices.sort_unstable(); // canonical per-bucket order
                steps.push(NonceStep { nonce, candidates: indices });
            }
            chains.push(SenderChain { sender, steps });
        }

        let mut total_steps = 0usize;
        for (ci, ch) in chains.iter().enumerate() {
            for (si, st) in ch.steps.iter().enumerate() {
                total_steps += 1;
                for &idx in &st.candidates {
                    index_of.insert(idx, (ci, si));
                }
            }
        }

        Some(Self { chains, index_of, total_steps })
    }

    #[inline]
    pub fn chain_lengths(&self) -> Vec<usize> {
        self.chains.iter().map(|c| c.steps.len()).collect()
    }

    #[inline]
    pub fn multiplicities(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.total_steps);
        for ch in &self.chains {
            for st in &ch.steps {
                v.push(st.candidates.len());
            }
        }
        v
    }

    /// Return all (chain_id, step_id) where there are duplicate-nonce choices (multiplicity > 1)
    pub fn multi_steps(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (ci, ch) in self.chains.iter().enumerate() {
            for (si, st) in ch.steps.iter().enumerate() {
                if st.candidates.len() > 1 {
                    out.push((ci, si));
                }
            }
        }
        out
    }
}


#[derive(Clone, Copy, Debug)]
pub enum GreedyKey { Profit, MevGasPrice }

#[inline]
pub fn value_for(key: GreedyKey, o: &SimulatedOrder) -> U256 {
    match key {
        GreedyKey::Profit => o.sim_value.coinbase_profit,
        GreedyKey::MevGasPrice => o.sim_value.mev_gas_price,
    }
}


/// Choose one candidate per (sender,nonce) step using the given metric and direction.
/// Primary key = `key` (max by default, min if `reverse`), secondary = the other metric
/// (same direction), then stable tie-break by lowest index.
pub fn build_chains_best_by(
    layout: &NonceLayout,
    group: &ConflictGroup,
    key: GreedyKey,
    reverse: bool,
) -> Vec<Vec<usize>> {
    let secondary = match key {
        GreedyKey::Profit => GreedyKey::MevGasPrice,
        GreedyKey::MevGasPrice => GreedyKey::Profit,
    };

    let better = |a: usize, b: usize| {
        let oa = &group.orders[a];
        let ob = &group.orders[b];

        let pa = value_for(key, oa);
        let pb = value_for(key, ob);
        if pa != pb {
            if reverse { pa < pb } else { pa > pb }
        } else {
            let sa = value_for(secondary, oa);
            let sb = value_for(secondary, ob);
            if sa != sb {
                if reverse { sa < sb } else { sa > sb }
            } else {
                a < b
            }
        }
    };

    let mut out: Vec<Vec<usize>> = Vec::with_capacity(layout.chains.len());
    for ch in &layout.chains {
        let mut picks: Vec<usize> = Vec::with_capacity(ch.steps.len());
        for st in &ch.steps {
            let mut best = st.candidates[0];
            for &cand in st.candidates.iter().skip(1) {
                if better(cand, best) { best = cand; }
            }
            picks.push(best);
        }
        out.push(picks);
    }
    out
}

#[inline]
pub fn build_chains_best(layout: &NonceLayout, group: &ConflictGroup) -> Vec<Vec<usize>> {
    build_chains_best_by(layout, group, GreedyKey::Profit, false)
}

/// HashSet of all indices that survive the best-per-step dedup (useful for Greedy filters).
pub fn allowed_indices_after_nonce_dedup(
    group: &ConflictGroup,
    key: GreedyKey,
    reverse: bool,
) -> Option<AHashSet<usize>> {
    let layout = NonceLayout::from_group(group)?;
    let chains = build_chains_best_by(&layout, group, key, reverse);
    let mut set = AHashSet::default();
    for ch in chains { for i in ch { set.insert(i); } }
    Some(set)
}

/// ln(n!) helper (exact sum of logs; OK for moderate n)
pub fn ln_fact(n: usize) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
}

/// log-safe multinomial compare: interleavings ≤ cap ?
pub fn interleavings_leq_cap(lengths: &[usize], cap: usize) -> bool {
    let n: usize = lengths.iter().sum();
    let ln_n = ln_fact(n);
    let ln_den: f64 = lengths.iter().map(|&l| ln_fact(l)).sum();
    let ln_mult = ln_n - ln_den;
    ln_mult <= (cap as f64).ln() + 1e-12
}

#[derive(Debug, Clone)]
pub struct InterleavingStats {
    pub chain_lengths: Vec<usize>,
    pub n_steps: usize,
    pub ln_multinomial: f64,
    pub ln_with_choice: f64,
    pub multiplicities: Vec<usize>,
}

/// Compute interleavings stats from the group (nonce-aware).
pub fn compute_interleaving_stats(group: &ConflictGroup) -> Option<InterleavingStats> {
    let layout = NonceLayout::from_group(group)?;
    let chain_lengths = layout.chain_lengths();
    let n_steps = chain_lengths.iter().sum();
    let ln_n = ln_fact(n_steps);
    let ln_den: f64 = chain_lengths.iter().map(|&l| ln_fact(l)).sum();
    let ln_multinomial = ln_n - ln_den;

    let multiplicities = layout.multiplicities();
    let ln_with_choice = ln_multinomial
        + multiplicities.iter().map(|&m| (m as f64).ln()).sum::<f64>();

    Some(InterleavingStats {
        chain_lengths,
        n_steps,
        ln_multinomial,
        ln_with_choice,
        multiplicities,
    })
}

pub fn format_approx_from_ln(ln_count: f64) -> String {
    if !ln_count.is_finite() || ln_count <= 0.0 {
        return "≈0".to_string();
    }
    if ln_count < 50.0 {
        let v = ln_count.exp().round();
        return format!("{}", v as u128);
    }
    const LN10: f64 = std::f64::consts::LN_10;
    let log10 = ln_count / LN10;
    let expo = log10.floor();
    let mant = 10f64.powf(log10 - expo);
    format!("{:.3}e{:+.0}", mant, expo)
}

pub fn enumerate_all_interleavings_best(per_chain: &[Vec<usize>], cap: usize) -> Vec<Vec<usize>> {
    let k = per_chain.len();
    if k == 0 { return vec![]; }
    let total: usize = per_chain.iter().map(|c| c.len()).sum();
    let mut cursors = vec![0usize; k];
    let mut cur: Vec<usize> = Vec::with_capacity(total);
    let mut out: Vec<Vec<usize>> = Vec::new();

    fn dfs(
        per_chain: &[Vec<usize>],
        cursors: &mut [usize],
        cur: &mut Vec<usize>,
        total: usize,
        out: &mut Vec<Vec<usize>>,
        cap: usize,
    ) {
        if cur.len() == total {
            out.push(cur.clone());
            return;
        }
        for ci in 0..per_chain.len() {
            if cursors[ci] < per_chain[ci].len() {
                let x = per_chain[ci][cursors[ci]];
                cursors[ci] += 1;
                cur.push(x);
                dfs(per_chain, cursors, cur, total, out, cap);
                cur.pop();
                cursors[ci] -= 1;
                if out.len() >= cap { return; }
            }
        }
    }

    dfs(per_chain, &mut cursors, &mut cur, total, &mut out, cap);
    out
}

/// Build one uniformly random nonce-respecting interleaving from `per_chain`
/// (one chosen candidate per step already).
pub fn sample_one_uniform_interleaving<R: Rng + ?Sized>(
    per_chain: &[Vec<usize>],
    rng: &mut R,
) -> Vec<usize> {
    let k = per_chain.len();
    let total: usize = per_chain.iter().map(|c| c.len()).sum();
    let mut cursors = vec![0usize; k];
    let mut remains: Vec<usize> = per_chain.iter().map(|c| c.len()).collect();
    let mut seq: Vec<usize> = Vec::with_capacity(total);

    for _ in 0..total {
        let total_rem: usize = remains.iter().sum();
        let mut r = rng.gen_range(0..total_rem);
        let mut chosen = 0usize;
        for i in 0..k {
            let w = remains[i];
            if w == 0 { continue; }
            if r < w { chosen = i; break; }
            r -= w;
        }
        let idx = per_chain[chosen][cursors[chosen]];
        cursors[chosen] += 1;
        remains[chosen] -= 1;
        seq.push(idx);
    }
    seq
}

/// Enumerate all interleavings with per-step choices (branches on duplicate-nonce steps).
pub fn enumerate_all_interleavings_with_choices(layout: &NonceLayout, cap: usize) -> Vec<Vec<usize>> {
    struct Frame { next_i: usize, applied_chain: Option<usize> }

    let n_chains = layout.chains.len();
    if n_chains == 0 { return vec![]; }
    let total = layout.total_steps;

    let mut cursors = vec![0usize; n_chains]; // next step per chain
    let mut seq: Vec<usize> = Vec::with_capacity(total);
    let mut stack: Vec<Frame> = vec![Frame { next_i: 0, applied_chain: None }];
    let mut out: Vec<Vec<usize>> = Vec::new();

    let ready_count = |cursors: &[usize], layout: &NonceLayout| -> usize {
        let mut tot = 0usize;
        for (ci, ch) in layout.chains.iter().enumerate() {
            let s = cursors[ci];
            if s < ch.steps.len() { tot += ch.steps[s].candidates.len(); }
        }
        tot
    };

    let nth_ready = |mut n: usize, cursors: &[usize], layout: &NonceLayout| -> (usize, usize) {
        for (ci, ch) in layout.chains.iter().enumerate() {
            let s = cursors[ci];
            if s >= ch.steps.len() { continue; }
            let len = ch.steps[s].candidates.len();
            if n < len { return (ci, n); }
            n -= len;
        }
        unreachable!("nth_ready called with n >= ready_count()");
    };

    loop {
        if stack.is_empty() { break; }
        let mut frame = stack.pop().unwrap();
        let tot_ready = ready_count(&cursors, layout);

        if frame.next_i >= tot_ready {
            if let Some(chain) = frame.applied_chain {
                seq.pop();
                cursors[chain] -= 1;
            }
            continue;
        }

        let (chain, pos_in_bucket) = nth_ready(frame.next_i, &cursors, layout);
        frame.next_i += 1;
        stack.push(frame);

        let step = cursors[chain];
        let cand = layout.chains[chain].steps[step].candidates[pos_in_bucket];
        seq.push(cand);
        cursors[chain] += 1;

        if seq.len() == total {
            out.push(seq.clone());
            seq.pop();
            cursors[chain] -= 1;
            if out.len() >= cap { break; }
        } else {
            stack.push(Frame { next_i: 0, applied_chain: Some(chain) });
        }
    }

    out
}


/// Pick 1 random candidate per step, then interleave uniformly across chains
pub fn random_interleaving_with_random_choices<R: Rng + ?Sized>(
    layout: &NonceLayout,
    rng: &mut R,
) -> Vec<usize> {
    let mut per_chain: Vec<Vec<usize>> = Vec::with_capacity(layout.chains.len());
    for ch in &layout.chains {
        let mut picks = Vec::with_capacity(ch.steps.len());
        for st in &ch.steps {
            let cand = st.candidates[rng.gen_range(0..st.candidates.len())];
            picks.push(cand);
        }
        per_chain.push(picks);
    }
    sample_one_uniform_interleaving(&per_chain, rng)
}



pub fn is_simple_chain(group: &ConflictGroup) -> bool {
    if let Some(layout) = NonceLayout::from_group(group) {
        layout.chains.len() == 1 && group.orders.len() == layout.chains[0].steps.len()
    } else {
        false
    }
}