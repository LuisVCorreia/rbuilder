use ahash::HashMap;
use alloy_primitives::{Address, U256};
use std::collections::BTreeMap;

use crate::primitives::SimulatedOrder;

use super::ConflictGroup;

pub const ALL_PERMS_CAP: usize = 120;

/// (sender, nonce) extraction from a single-tx order.
/// Returns None for bundles / multi-tx orders.
pub fn sender_and_nonce(order: &SimulatedOrder) -> Option<(Address, u64)> {
    let txs = order.order.list_txs();
    if txs.len() != 1 {
        return None;
    }
    let (tx, _) = &txs[0];
    Some((tx.signer(), tx.nonce()))
}

/// For each sender, we keep a chain of "nonce slots", and each slot has
/// the list of candidate order indices (duplicate nonces -> multiple candidates).
#[derive(Debug, Clone)]
pub struct SenderNonceView {
    /// chains_slots[sender_idx][slot_idx] = Vec<order_idx> (candidates at that nonce)
    pub chains_slots: Vec<Vec<Vec<usize>>>,
}

impl SenderNonceView {
    pub fn chain_lengths(&self) -> Vec<usize> {
        self.chains_slots.iter().map(|c| c.len()).collect()
    }
    pub fn total_slots(&self) -> usize {
        self.chains_slots.iter().map(|c| c.len()).sum()
    }
    /// Multiplicity of each slot across all senders (for counting "with choice").
    pub fn slot_multiplicities(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for chain in &self.chains_slots {
            for slot in chain {
                v.push(slot.len());
            }
        }
        v
    }
}

/// Build a full (sender -> nonce -> [indices]) view, with nonces sorted asc per sender.
/// Returns None when we cannot derive (sender, nonce) for any order.
pub fn build_sender_nonce_view(group: &ConflictGroup) -> Option<SenderNonceView> {
    // sender -> (nonce -> Vec<idx>)
    let mut by_sender: HashMap<Address, BTreeMap<u64, Vec<usize>>> = HashMap::default();

    for (idx, o) in group.orders.iter().enumerate() {
        let (sender, nonce) = sender_and_nonce(o)?;
        by_sender.entry(sender).or_default().entry(nonce).or_default().push(idx);
    }

    let mut chains_slots: Vec<Vec<Vec<usize>>> = Vec::with_capacity(by_sender.len());
    for (_sender, by_nonce) in by_sender {
        // BTreeMap keeps keys ordered by nonce asc
        let mut chain: Vec<Vec<usize>> = Vec::with_capacity(by_nonce.len());
        for (_nonce, indices) in by_nonce {
            chain.push(indices);
        }
        chains_slots.push(chain);
    }

    Some(SenderNonceView { chains_slots })
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

/// Choose one candidate per (sender,nonce) slot using the given metric and direction.
/// Primary key = `key` (max by default, min if `reverse`), secondary = the other metric
/// (same direction), then stable tie-break by lowest index.
pub fn build_sender_chains_best_by(
    view: &SenderNonceView,
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

    let mut out: Vec<Vec<usize>> = Vec::with_capacity(view.chains_slots.len());
    for chain in &view.chains_slots {
        let mut best_chain: Vec<usize> = Vec::with_capacity(chain.len());
        for slot in chain {
            let mut best_idx = slot[0];
            for &cand in slot.iter().skip(1) {
                if better(cand, best_idx) {
                    best_idx = cand;
                }
            }
            best_chain.push(best_idx);
        }
        out.push(best_chain);
    }
    out
}

#[inline]
pub fn build_sender_chains_best(
    view: &SenderNonceView,
    group: &ConflictGroup,
) -> Vec<Vec<usize>> {
    build_sender_chains_best_by(view, group, GreedyKey::Profit, false)
}


/// HashSet of all indices that survive the best-per-slot dedup (useful for Greedy filters).
pub fn allowed_indices_after_nonce_dedup(group: &ConflictGroup, key: GreedyKey, reverse: bool) -> Option<ahash::HashSet<usize>> {
    let view = build_sender_nonce_view(group)?;
    let chains = build_sender_chains_best_by(&view, group, key, reverse);
    let mut set = ahash::HashSet::default();
    for ch in chains {
        for i in ch {
            set.insert(i);
        }
    }
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

/// Compact stats for dedupbed interleavings and "with duplicate choices".
#[derive(Debug, Clone)]
pub struct InterleavingStats {
    pub chain_lengths: Vec<usize>,
    pub n_slots: usize,
    pub ln_multinomial: f64,
    pub ln_with_choice: f64,
    pub multiplicities: Vec<usize>,
}

/// Compute interleavings stats from the group (nonce-aware).
pub fn compute_interleaving_stats(group: &ConflictGroup) -> Option<InterleavingStats> {
    let view = build_sender_nonce_view(group)?;
    let chain_lengths = view.chain_lengths();
    let n_slots = chain_lengths.iter().sum();
    let ln_n = ln_fact(n_slots);
    let ln_den: f64 = chain_lengths.iter().map(|&l| ln_fact(l)).sum();
    let ln_multinomial = ln_n - ln_den;

    let multiplicities = view.slot_multiplicities();
    let ln_with_choice = ln_multinomial
        + multiplicities.iter().map(|&m| (m as f64).ln()).sum::<f64>();

    Some(InterleavingStats {
        chain_lengths,
        n_slots,
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

/// Enumerate all interleavings (dedupbed: 1 best candidate per slot).
/// `chains` is the result of `build_sender_chains_best`.
pub fn enumerate_all_interleavings_best(chains: &[Vec<usize>], cap: usize) -> Vec<Vec<usize>> {
    let k = chains.len();
    if k == 0 { return vec![]; }
    let total: usize = chains.iter().map(|c| c.len()).sum();
    let mut cursors = vec![0usize; k];
    let mut cur: Vec<usize> = Vec::with_capacity(total);
    let mut out: Vec<Vec<usize>> = Vec::new();

    fn dfs(
        chains: &[Vec<usize>],
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
        for ci in 0..chains.len() {
            if cursors[ci] < chains[ci].len() {
                let x = chains[ci][cursors[ci]];
                cursors[ci] += 1;
                cur.push(x);
                dfs(chains, cursors, cur, total, out, cap);
                cur.pop();
                cursors[ci] -= 1;

                if out.len() >= cap {
                    return;
                }
            }
        }
    }

    dfs(chains, &mut cursors, &mut cur, total, &mut out, cap);
    out
}

/// Enumerate all interleavings with choices per duplicate nonce slot.
/// Each time we take the next slot from some sender, we branch over all candidates in that slot.
pub fn enumerate_all_interleavings_with_choices(view: &SenderNonceView, cap: usize) -> Vec<Vec<usize>> {
    let k = view.chains_slots.len();
    if k == 0 { return vec![]; }
    let total: usize = view.total_slots();
    let mut pos = vec![0usize; k]; // which slot index we are on for each sender
    let mut cur: Vec<usize> = Vec::with_capacity(total);
    let mut out: Vec<Vec<usize>> = Vec::new();

    fn dfs(
        view: &SenderNonceView,
        pos: &mut [usize],
        cur: &mut Vec<usize>,
        total: usize,
        out: &mut Vec<Vec<usize>>,
        cap: usize,
    ) {
        if cur.len() == total {
            out.push(cur.clone());
            return;
        }
        for ci in 0..view.chains_slots.len() {
            let p = pos[ci];
            if p < view.chains_slots[ci].len() {
                // we can take this sender's next slot; branch on all candidates in that slot
                let candidates = &view.chains_slots[ci][p];
                for &idx in candidates {
                    pos[ci] += 1;
                    cur.push(idx);
                    dfs(view, pos, cur, total, out, cap);
                    cur.pop();
                    pos[ci] -= 1;

                    if out.len() >= cap {
                        return;
                    }
                }
            }
        }
    }

    dfs(view, &mut pos, &mut cur, total, &mut out, cap);
    out
}
