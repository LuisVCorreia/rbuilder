use ahash::{HashMap as AHashMap, HashSet as AHashSet};
use alloy_primitives::{Address, U256};
use crate::primitives::{OrderId, SimulatedOrder};
use std::{cmp::Ordering, sync::Arc};

#[derive(Clone, Debug)]
struct OrderDesc {
    id: OrderId,
    profit: U256,
    blob_gas: u64,
    signer: Option<Address>,
    nonce: Option<u64>,
}

fn build_descriptors(sim_orders: &[Arc<SimulatedOrder>]) -> Vec<OrderDesc> {
    sim_orders.iter().map(|o| {
        let profit = o.sim_value.coinbase_profit;
        let blob_gas = o.sim_value.blob_gas_used;
        let nonces = o.order.nonces();
        let (signer, nonce) = if nonces.len() == 1 {
            (Some(nonces[0].address), Some(nonces[0].nonce))
        } else {
            (None, None)
        };
        OrderDesc { id: o.order.id(), profit, blob_gas, signer, nonce }
    }).collect()
}

/// Choose the “best” blob at a nonce (highest profit, tie -> smaller blob_gas).
fn choose_best_blob<'a>(blobs: &[&'a OrderDesc]) -> &'a OrderDesc {
    let mut best = blobs[0];
    for &d in blobs.iter().skip(1) {
        match d.profit.cmp(&best.profit) {
            Ordering::Greater => best = d,
            Ordering::Equal if d.blob_gas < best.blob_gas => best = d,
            _ => {}
        }
    }
    best
}

#[derive(Clone, Debug)]
struct RawGroup {
    nonce: u64,
    normals_ids: Vec<OrderId>,
    blobs_ids: Vec<OrderId>,
    chosen_blob_id: Option<OrderId>, // if this nonce has blobs
}

/// One condensed representative per nonce (used only for scoring).
#[derive(Clone, Debug)]
struct RepNode<'a> {
    nonce: u64,
    rep:   &'a OrderDesc,
    is_blob: bool,
}

/// Value/gas step contributed by selecting a blob at position i (inclusive) up to next blob (exclusive).
#[derive(Clone, Debug)]
struct Step {
    delta_v: U256,  // sum of representative profits in that segment
    delta_w: u64,  // blob gas of the starting blob
}

/// A signer class: raw groups for reconstruction + a condensed chain for scoring
#[derive(Clone, Debug)]
struct Class<'a> {
    raw_groups: Vec<RawGroup>,  // full duplicates kept (by nonce)
    rep_chain: Vec<RepNode<'a>>,  // one representative per nonce
    blob_idx: Vec<usize>,         // indices into rep_chain that are blobs
    steps: Vec<Step>,             // incremental steps between consecutive blobs
}

#[derive(Clone, Copy)]
enum Strategy { Ratio, Absolute }

fn cmp_ratio(vw_a: (U256, u64), vw_b: (U256, u64)) -> Ordering {
    let (va, wa) = vw_a;
    let (vb, wb) = vw_b;
    let lhs = va.saturating_mul(U256::from(wb));
    let rhs = vb.saturating_mul(U256::from(wa));
    match lhs.cmp(&rhs) {
        Ordering::Equal => match va.cmp(&vb) {
            Ordering::Equal => wb.cmp(&wa), // prefer smaller weight
            o => o,
        },
        o => o,
    }
}

fn cmp_abs(vw_a: (U256, u64), vw_b: (U256, u64)) -> Ordering {
    let (va, wa) = vw_a;
    let (vb, wb) = vw_b;
    match va.cmp(&vb) {
        Ordering::Equal => wb.cmp(&wa), // prefer smaller weight
        o => o,
    }
}

/// Prepare per-signer classes:
///  - Build raw nonce groups (keep all normal duplicates; pick a single chosen blob if any).
///  - Build a condensed chain (one representative per nonce: chosen blob if present, else best normal).
///  - Precompute blob steps on the condensed chain for scoring.
fn prepare_classes<'a>(
    descs: &'a [OrderDesc]
) -> (Vec<Class<'a>>, AHashSet<OrderId>, AHashMap<OrderId, U256>) {
    let mut always_keep: AHashSet<OrderId> = AHashSet::default();
    let mut profit_by_id: AHashMap<OrderId, U256> = AHashMap::default();
    for d in descs {
        profit_by_id.insert(d.id, d.profit);
    }

    // group candidates with clear (signer,nonce)
    let mut by_signer: AHashMap<Address, Vec<&OrderDesc>> = AHashMap::default();
    for d in descs {
        match (d.signer, d.nonce) {
            (Some(a), Some(_)) => by_signer.entry(a).or_default().push(d),
            _ => { always_keep.insert(d.id); }
        }
    }

    let mut classes: Vec<Class> = Vec::new();

    for (_, mut chain_raw) in by_signer {
        chain_raw.sort_by_key(|d| d.nonce.unwrap());

        // Build raw nonce-groups
        let mut raw_groups: Vec<RawGroup> = Vec::new();
        {
            let mut i = 0usize;
            while i < chain_raw.len() {
                let n = chain_raw[i].nonce.unwrap();
                let mut normals: Vec<&OrderDesc> = Vec::new();
                let mut blobs:   Vec<&OrderDesc> = Vec::new();
                while i < chain_raw.len() && chain_raw[i].nonce.unwrap() == n {
                    if chain_raw[i].blob_gas > 0 { blobs.push(chain_raw[i]); }
                    else { normals.push(chain_raw[i]); }
                    i += 1;
                }
                let chosen_blob_id = if blobs.is_empty() {
                    None
                } else {
                    Some(choose_best_blob(&blobs).id)
                };
                raw_groups.push(RawGroup {
                    nonce: n,
                    normals_ids: normals.into_iter().map(|d| d.id).collect(),
                    blobs_ids:   blobs.into_iter().map(|d| d.id).collect(),
                    chosen_blob_id,
                });
            }
        }

        // If no blob nonces, keep entire raw chain and continue
        if !raw_groups.iter().any(|g| !g.blobs_ids.is_empty()) {
            for g in &raw_groups {
                for &id in &g.normals_ids { always_keep.insert(id); }
                // no blobs in these groups by construction
            }
            continue;
        }

        // Build condensed representative chain (one per nonce).
        let mut rep_chain: Vec<RepNode> = Vec::with_capacity(raw_groups.len());
        for g in &raw_groups {
            // find the representative
            let rep = if let Some(cb) = g.chosen_blob_id {
                // find desc by id (we have chain_raw, so find it)
                let rep = chain_raw.iter().find(|d| d.id == cb).expect("chosen blob present");
                RepNode { nonce: g.nonce, rep, is_blob: true }
            } else {
                // choose best normal (must exist by construction)
                // locate the desc
                let best_norm_id = {
                    // for scoring: best by profit
                    // normals_ids is non-empty here (else group is empty which can't happen)
                    let mut best = g.normals_ids[0];
                    for &id in &g.normals_ids {
                        let a = chain_raw.iter().find(|d| d.id == id).unwrap();
                        let b = chain_raw.iter().find(|d| d.id == best).unwrap();
                        if a.profit > b.profit { best = id; }
                    }
                    best
                };
                let rep = chain_raw.iter().find(|d| d.id == best_norm_id).unwrap();
                RepNode { nonce: g.nonce, rep, is_blob: false }
            };
            rep_chain.push(rep);
        }

        let blob_idx: Vec<usize> = rep_chain.iter()
            .enumerate()
            .filter_map(|(i, rn)| rn.is_blob.then_some(i))
            .collect();

        // Precompute steps between consecutive blobs
        let mut steps: Vec<Step> = Vec::with_capacity(blob_idx.len());
        for (j, &bi) in blob_idx.iter().enumerate() {
            let start = bi;
            let end = if j + 1 < blob_idx.len() { blob_idx[j + 1] } else { rep_chain.len() };
            let mut dv = U256::ZERO;
            for rn in &rep_chain[start..end] {
                dv = dv.saturating_add(rn.rep.profit);
            }
            steps.push(Step {
                delta_v: dv,
                delta_w: rep_chain[bi].rep.blob_gas,
            });
        }

        classes.push(Class { raw_groups, rep_chain, blob_idx, steps });
    }

    (classes, always_keep, profit_by_id)
}

/// One greedy pass (ratio or absolute) over precomputed steps.
fn greedy_once<'a>(classes: &'a [Class<'a>], blob_cap: u64, strategy: Strategy) -> (Vec<usize>, AHashSet<OrderId>) {
    let mut chosen_prefix: Vec<usize> = vec![0; classes.len()];
    let mut remaining: i64 = blob_cap as i64;

    loop {
        // Build candidates: next step per class
        let mut candidates: Vec<(usize, U256, u64)> = Vec::new(); // (class_idx, v, w)
        for (ci, cls) in classes.iter().enumerate() {
            if let Some(step) = cls.steps.get(chosen_prefix[ci]) {
                candidates.push((ci, step.delta_v, step.delta_w));
            }
        }
        if candidates.is_empty() { break; }

        match strategy {
            Strategy::Ratio => {
                candidates.sort_unstable_by(|a, b| cmp_ratio((a.1, a.2), (b.1, b.2)).reverse());
            }
            Strategy::Absolute => {
                candidates.sort_unstable_by(|a, b| cmp_abs((a.1, a.2), (b.1, b.2)).reverse());
            }
        }

        // Pick the best that fits
        let mut picked = false;
        for (ci, _, w) in candidates {
            let w_i64 = w as i64;
            if w_i64 <= remaining {
                remaining -= w_i64;
                chosen_prefix[ci] += 1;
                picked = true;
                break;
            }
        }
        if !picked { break; }
    }

    let mut keep: AHashSet<OrderId> = AHashSet::default();

    for (i, cls) in classes.iter().enumerate() {
        let k = chosen_prefix[i]; // number of blobs accepted for this signer
        if k >= cls.blob_idx.len() {
            // selected all blobs -> keep entire chain:
            // for blob nonces: keep only the chosen blob (not all blob-alternatives)
            // for normal-only nonces: keep all normal tx duplicates
            for g in &cls.raw_groups {
                if let Some(chosen_blob) = g.chosen_blob_id {
                    keep.insert(chosen_blob);
                } else {
                    for &id in &g.normals_ids {
                        keep.insert(id);
                    }
                }
            }
        } else {
            // we did not select the blob at blob_idx[k] -> cutoff at that blob's nonce
            let cutoff_nonce = cls.rep_chain[cls.blob_idx[k]].nonce;
            for g in &cls.raw_groups {
                if g.nonce < cutoff_nonce {
                    if let Some(chosen_blob) = g.chosen_blob_id {
                        // earlier blob (must have been selected); keep only the chosen blob here
                        keep.insert(chosen_blob);
                    } else {
                        // normal-only nonce before cutoff -> keep all normal alternatives
                        for &id in &g.normals_ids {
                            keep.insert(id);
                        }
                    }
                } else {
                    break; // groups are in ascending nonce
                }
            }
        }
    }

    (chosen_prefix, keep)
}

fn select_keep_set(descs: &[OrderDesc], blob_cap: u64) -> AHashSet<OrderId> {
    let (classes, always_keep, profit_by_id) = prepare_classes(descs);

    if classes.is_empty() {
        return always_keep;
    }

    let (_kr, keep_ratio) = greedy_once(&classes, blob_cap, Strategy::Ratio);
    let (_ka, keep_abs)   = greedy_once(&classes, blob_cap, Strategy::Absolute);

    let mut with_ratio = keep_ratio;
    let mut with_abs   = keep_abs;
    for k in &always_keep { with_ratio.insert(*k); with_abs.insert(*k); }

    let score = |set: &AHashSet<OrderId>| -> U256 {
        let mut s = U256::ZERO;
        for id in set {
            if let Some(p) = profit_by_id.get(id) {
                s = s.saturating_add(*p);
            }
        }
        s
    };

    if score(&with_abs) >= score(&with_ratio) { with_abs } else { with_ratio }
}

pub fn select_orders_under_blob_cap(
    sim_orders: &[Arc<SimulatedOrder>],
    blob_cap: u64,
) -> Vec<Arc<SimulatedOrder>> {
    let descs = build_descriptors(sim_orders);
    let keep = select_keep_set(&descs, blob_cap);
    sim_orders.iter()
        .filter(|o| keep.contains(&o.order.id()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{address, Address, B256};

    fn dummy_tx_id(n: u8) -> OrderId {
        let mut bytes = [0u8; 32];
        bytes[31] = n;
        OrderId::Tx(B256::new(bytes))
    }

    fn od(id: u8, profit: u128, blob_gas: u64, signer: Option<Address>, nonce: Option<u64>) -> OrderDesc {
        OrderDesc {
            id: dummy_tx_id(id),
            profit: U256::from(profit),
            blob_gas,
            signer,
            nonce,
        }
    }

    #[test]
    fn keeps_all_when_no_blobs() {
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let d = vec![
            od(1, 10, 0, Some(a), Some(1)),
            od(2, 20, 0, Some(a), Some(2)),
        ];
        let keep = select_keep_set(&d, 0);
        assert!(keep.contains(&dummy_tx_id(1)));
        assert!(keep.contains(&dummy_tx_id(2)));
    }

    #[test]
    fn single_signer_two_blobs_various_caps() {
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        // Nonces: 1(N), 2(B,10), 3(N), 4(B,10), 5(N)
        // Profits:  2     5       3       4       7
        let d = vec![
            od(1, 2,  0, Some(a), Some(1)),
            od(2, 5, 10, Some(a), Some(2)), // b1
            od(3, 3,  0, Some(a), Some(3)),
            od(4, 4, 10, Some(a), Some(4)), // b2
            od(5, 7,  0, Some(a), Some(5)),
        ];

        // cap=0: cutoff at nonce 2, keep only id=1
        let keep0 = select_keep_set(&d, 0);
        assert_eq!(keep0.len(), 1);
        assert!(keep0.contains(&dummy_tx_id(1)));

        // cap=10: cutoff at nonce 4, keep ids 1,2,3
        let keep10 = select_keep_set(&d, 10);
        let exp10: AHashSet<OrderId> = [1u8,2,3].into_iter().map(dummy_tx_id).collect();
        assert_eq!(keep10, exp10);

        // cap=20: keep all
        let keep20 = select_keep_set(&d, 20);
        let exp20: AHashSet<OrderId> = [1u8,2,3,4,5].into_iter().map(dummy_tx_id).collect();
        assert_eq!(keep20, exp20);
    }

    #[test]
    fn pre_blob_duplicate_normals_preserved_when_blob_not_selected() {
        // Two normal txs at nonce=1 (duplicates), then a blob at nonce=2.
        // If cap=0 (blob not selected), BOTH pre-blob normals should be kept.
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let d = vec![
            od(1, 2,  0, Some(a), Some(1)), // normal alt A
            od(2, 3,  0, Some(a), Some(1)), // normal alt B (better)
            od(3, 5, 10, Some(a), Some(2)), // blob
        ];
        let keep = select_keep_set(&d, 0);
        let exp: AHashSet<OrderId> = [1u8,2].into_iter().map(dummy_tx_id).collect();
        assert_eq!(keep, exp, "both pre-blob normals must be preserved when blob not selected");
    }

    #[test]
    fn pre_blob_duplicate_normals_preserved_and_blob_chosen_when_selected() {
        // Two normals at nonce=1 (duplicates), then a blob at nonce=2.
        // If cap>=10 (blob selected), keep BOTH normals at nonce=1 PLUS the chosen blob at nonce=2.
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let d = vec![
            od(1, 2,  0, Some(a), Some(1)), // normal alt A
            od(2, 3,  0, Some(a), Some(1)), // normal alt B
            od(3, 5, 10, Some(a), Some(2)), // blob
        ];
        let keep = select_keep_set(&d, 10);
        let exp: AHashSet<OrderId> = [1u8,2,3].into_iter().map(dummy_tx_id).collect();
        assert_eq!(keep, exp, "both pre-blob normals + chosen blob must be kept when blob selected");
    }

    #[test]
    fn multi_signer_dual_greedy_pick() {
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let b = address!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        // Signer A:   1(N,2), 2(B,10,5), 3(N,3)
        // Signer B:   1(B,10,9)
        let d = vec![
            od(11, 2,  0, Some(a), Some(1)),
            od(12, 5, 10, Some(a), Some(2)), // A blob
            od(13, 3,  0, Some(a), Some(3)),
            od(21, 9, 10, Some(b), Some(1)), // B blob
        ];

        // cap=10 -> expect {11,21} (ratio beats absolute here)
        let keep = select_keep_set(&d, 10);
        let expected: AHashSet<OrderId> = [dummy_tx_id(11), dummy_tx_id(21)].into_iter().collect();
        assert_eq!(keep, expected);

        // cap=20 -> both blobs fit -> {11,12,13,21}
        let keep2 = select_keep_set(&d, 20);
        let exp2: AHashSet<OrderId> = [11u8,12,13,21].into_iter().map(dummy_tx_id).collect();
        assert_eq!(keep2, exp2);
    }
}
