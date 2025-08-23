use ahash::{HashMap, HashSet};
use alloy_primitives::{Address, U256};
use crate::primitives::{OrderId, SimulatedOrder};
use std::sync::Arc;

/// Lightweight per-order description feeding the selector.
#[derive(Clone, Debug)]
struct OrderDesc {
    id: OrderId,
    profit: U256,
    blob_gas: u64,
    signer: Option<Address>,
    nonce: Option<u64>,
}

/// Extract per-order descriptors.
/// - If an order has exactly one (address, nonce) we use it as chain key;
/// - Otherwise we mark signer/nonce as None -> always keep (not controlled by blob-cap DP).
fn build_descriptors(sim_orders: &[Arc<SimulatedOrder>]) -> Vec<OrderDesc> {
    println!("=== BUILD DESCRIPTORS ===");
    println!("Total orders to process: {}", sim_orders.len());
    
    let descriptors: Vec<OrderDesc> = sim_orders.iter().map(|o| {
        let profit = o.sim_value.coinbase_profit;
        let blob_gas = o.sim_value.blob_gas_used;
        let nonces = o.order.nonces();
        let (signer, nonce) = if nonces.len() == 1 {
            (Some(nonces[0].address), Some(nonces[0].nonce))
        } else {
            (None, None)
        };
        
        if blob_gas > 0 {
            println!("BLOB TX: id={}, profit={}, blob_gas={}, signer={:?}, nonce={:?}", 
                o.order.id(), profit, blob_gas, signer, nonce);
        }
        
        OrderDesc { id: o.order.id(), profit, blob_gas, signer, nonce }
    }).collect();
    
    let blob_count = descriptors.iter().filter(|d| d.blob_gas > 0).count();
    println!("Total blob transactions: {}", blob_count);
    println!("=========================");
    
    descriptors
}

/// Among multiple txs at the same (signer, nonce), keep exactly one:
/// - max profit
/// - tie-breaker: lower blob_gas
fn condense_chain_by_nonce<'a>(chain: &[&'a OrderDesc]) -> Vec<&'a OrderDesc> {
    let mut best: HashMap<u64, &'a OrderDesc> = HashMap::default();
    for d in chain {
        let n = d.nonce.expect("chain elements have nonce");
        match best.get(&n) {
            None => { best.insert(n, *d); }
            // Some(cur) => {
            //     if d.profit > cur.profit || (d.profit == cur.profit && d.blob_gas < cur.blob_gas) {
            //         best.insert(n, *d);
            //     }
            // }
            Some(cur) => { best.insert(n, *d); }
        }
    }
    let mut v: Vec<&'a OrderDesc> = best.into_values().collect();
    v.sort_by_key(|d| d.nonce.unwrap());
    v
}

/// For each signer with m blobs (in nonce order), define options k=0..m:
///   weight[k] = sum(blob_gas of the first k blobs)             (k=0 => 0)
///   cutoff[k] = nonce of blob_{k+1} (first discarded blob)     (k=m => None)
///   value[k]  = sum profits of orders with nonce < cutoff[k]
/// Choose exactly one option per signer under global blob-gas cap to maximize total value.
/// Returns the set of OrderId to keep (plus all multi-signer orders).
fn select_keep_set(descs: &[OrderDesc], blob_cap: u64) -> HashSet<OrderId> {
    // Keep all multi-signer / ambiguous-nonce orders up-front.
    let mut keep: HashSet<OrderId> = HashSet::default();

    // Group single-signer orders by signer; sort by nonce.
    let mut chains: HashMap<Address, Vec<&OrderDesc>> = HashMap::default();
    for d in descs {
        match (d.signer, d.nonce) {
            (Some(a), Some(_)) => chains.entry(a).or_default().push(d),
            _ => { keep.insert(d.id); }
        }
    }
    for v in chains.values_mut() {
        v.sort_by_key(|d| d.nonce.unwrap());
    }

    println!("=== SIGNER CHAINS ===");
    for (signer, chain) in &chains {
        println!("Signer {:?} has {} orders:", signer, chain.len());
        for d in chain {
            println!("  - Order {}: nonce={}, profit={}, blob_gas={}", 
                d.id, d.nonce.unwrap(), d.profit, d.blob_gas);
        }
    }
    println!("=====================");

    // Build per-signer choice classes only for signers that have >= 1 blob after condensation.
    #[derive(Clone, Copy, Debug)]
    struct ClassOpt {
        weight: u64,
        value: U256,
        cutoff: Option<u64>, // first discarded blob's nonce; None => keep all
    }
    #[derive(Clone, Debug)]
    struct SignerClass {
        signer: Address,
        options: Vec<ClassOpt>,
    }

    let mut classes: Vec<SignerClass> = Vec::new();
    // We also need the (condensed) chain later to realize the chosen cutoff.
    let mut signer_chain: HashMap<Address, Vec<&OrderDesc>> = HashMap::default();

    for (addr, raw_chain) in chains {
        // Deduplicate per nonce
        let chain = condense_chain_by_nonce(&raw_chain);
        signer_chain.insert(addr, chain.clone());

        println!("=== PROCESSING SIGNER {:?} ===", addr);
        println!("Raw chain length: {}, Condensed chain length: {}", raw_chain.len(), chain.len());
        for d in &chain {
            println!("  Condensed order: id={}, nonce={}, profit={}, blob_gas={}", 
                d.id, d.nonce.unwrap(), d.profit, d.blob_gas);
        }

        // Find the blob positions in the condensed chain
        let blob_pos: Vec<usize> = chain.iter()
            .enumerate()
            .filter_map(|(i, d)| if d.blob_gas > 0 { Some(i) } else { None })
            .collect();

        println!("Blob positions in chain: {:?}", blob_pos);

        if blob_pos.is_empty() {
            // No blobs for this signer → everything is kept (no impact on blob cap)
            println!("No blobs for signer {:?}, keeping all {} orders", addr, chain.len());
            for d in chain { keep.insert(d.id); }
            continue;
        }

        // Prefix sum of blob gas over the blob items (in chain order)
        let m = blob_pos.len();
        let mut blob_prefix: Vec<u64> = Vec::with_capacity(m + 1);
        blob_prefix.push(0);
        for &idx in &blob_pos {
            blob_prefix.push(blob_prefix.last().copied().unwrap() + chain[idx].blob_gas);
        }

        // blob nonces (to define cutoffs). blob_nonces[k] = nonce of blob_{k+1}
        let blob_nonces: Vec<u64> = blob_pos.iter().map(|&i| chain[i].nonce.unwrap()).collect();

        // Build options k=0..=m
        let mut options: Vec<ClassOpt> = Vec::with_capacity(m + 1);
        for k in 0..=m {
            let cutoff = if k < m { Some(blob_nonces[k]) } else { None };
            let weight = blob_prefix[k];

            // Sum profits of orders with nonce < cutoff (or all if None)
            let value = match cutoff {
                Some(cn) => {
                    chain.iter()
                         .take_while(|d| d.nonce.unwrap() < cn)
                         .fold(U256::ZERO, |acc, d| acc.saturating_add(d.profit))
                }
                None => {
                    chain.iter()
                         .fold(U256::ZERO, |acc, d| acc.saturating_add(d.profit))
                }
            };

            println!("  Option k={}: weight={}, value={}, cutoff={:?}", k, weight, value, cutoff);
            options.push(ClassOpt { weight, value, cutoff });
        }

        classes.push(SignerClass { signer: addr, options });
        println!("==============================");
    }

    if classes.is_empty() {
        // No blob-bearing signers; we're done (we already inserted keep-all).
        println!("No blob-bearing signers found, keeping all orders");
        return keep;
    }

    println!("=== KNAPSACK OPTIMIZATION ===");
    println!("Blob cap: {}", blob_cap);
    println!("Number of signer classes: {}", classes.len());

    // Multi-choice knapsack over classes (value in U256).
    let cap = blob_cap as usize;
    let mut dp: Vec<U256> = vec![U256::ZERO; cap + 1];
    // parent[i][c] = which option k was chosen for class i at capacity c
    let mut parent: Vec<Vec<Option<usize>>> = vec![vec![None; cap + 1]; classes.len() + 1];

    for (i, class) in classes.iter().enumerate() {
        let mut next = dp.clone();
        let mut row: Vec<Option<usize>> = vec![None; cap + 1];

        for c in 0..=cap {
            for (k, opt) in class.options.iter().enumerate() {
                let w = opt.weight as usize;
                if w <= c {
                    let cand = dp[c - w].saturating_add(opt.value);
                    if cand > next[c] {
                        next[c] = cand;
                        row[c] = Some(k);
                    }
                }
            }
        }
        dp = next;
        parent[i + 1] = row;
    }

    // Reconstruct chosen option per class
    let mut chosen_k: Vec<usize> = vec![0; classes.len()];
    let mut c = cap;
    for i in (1..=classes.len()).rev() {
        let k = parent[i][c].unwrap_or(0);
        chosen_k[i - 1] = k;
        let w = classes[i - 1].options[k].weight as usize;
        if w <= c { c -= w; }
    }

    println!("=== OPTIMAL SOLUTION ===");
    for (i, class) in classes.iter().enumerate() {
        let k = chosen_k[i];
        let opt = &class.options[k];
        println!("Signer {:?}: chose option k={}, weight={}, value={}, cutoff={:?}", 
            class.signer, k, opt.weight, opt.value, opt.cutoff);
    }
    println!("========================");

    // Materialize the keep set from cutoffs
    println!("=== MATERIALIZING KEEP SET ===");
    for (i, class) in classes.iter().enumerate() {
        let k = chosen_k[i];
        let cutoff = class.options[k].cutoff; // first discarded blob nonce; None => keep all
        let chain = signer_chain.get(&class.signer).expect("present");
        match cutoff {
            None => {
                // selected all blobs => keep entire chain
                println!("Signer {:?}: keeping entire chain ({} orders)", class.signer, chain.len());
                for d in chain { keep.insert(d.id); }
            }
            Some(cn) => {
                println!("Signer {:?}: keeping orders with nonce < {}", class.signer, cn);
                for d in chain {
                    if d.nonce.unwrap() < cn {
                        println!("Keeping order {} with nonce {} for signer {}", d.id, d.nonce.unwrap(), class.signer);
                        keep.insert(d.id);
                    } else {
                        println!("Discarding order {} with nonce {} for signer {}", d.id, d.nonce.unwrap(), class.signer);
                    }
                }
            }
        }
    }
    println!("===============================");

    keep
}

/// returns filtered sim_orders according to the optimal blob selection.
pub fn select_orders_under_blob_cap(
    sim_orders: &[Arc<SimulatedOrder>],
    blob_cap: u64,
) -> Vec<Arc<SimulatedOrder>> {
    println!("\n=== BLOB TX SELECTION START ===");
    println!("Input: {} orders, blob cap: {}", sim_orders.len(), blob_cap);
    
    let descs = build_descriptors(sim_orders);
    let keep = select_keep_set(&descs, blob_cap);
    let result = sim_orders.iter()
        .filter(|o| keep.contains(&o.order.id()))
        .cloned()
        .collect::<Vec<_>>();
    
    println!("=== FINAL SELECTION SUMMARY ===");
    println!("Selected {} out of {} orders", result.len(), sim_orders.len());
    let selected_blob_gas: u64 = result.iter()
        .map(|o| o.sim_value.blob_gas_used)
        .sum();
    let total_profit: U256 = result.iter()
        .map(|o| o.sim_value.coinbase_profit)
        .fold(U256::ZERO, |acc, p| acc.saturating_add(p));
    println!("Total blob gas used: {} (cap: {})", selected_blob_gas, blob_cap);
    println!("Total coinbase profit: {}", total_profit);
    
    let selected_blob_count = result.iter().filter(|o| o.sim_value.blob_gas_used > 0).count();
    println!("Selected blob transactions: {}", selected_blob_count);
    
    for order in &result {
        if order.sim_value.blob_gas_used > 0 {
            println!("  SELECTED BLOB: id={}, profit={}, blob_gas={}", 
                order.order.id(), order.sim_value.coinbase_profit, order.sim_value.blob_gas_used);
        }
    }
    println!("==============================\n");
    
    result
}


#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{address, B256};

    fn dummy_tx_id(n: u8) -> OrderId {
        let mut bytes = [0u8; 32];
        bytes[31] = n; // unique last byte
        OrderId::Tx(B256::new(bytes))
    }

    fn od(id: u8, profit: impl Into<U256>, blob_gas: u64, signer: Option<Address>, nonce: Option<u64>) -> OrderDesc {
        OrderDesc {
            id: dummy_tx_id(id),
            profit: profit.into(),
            blob_gas,
            signer,
            nonce,
        }
    }

    #[test]
    fn keeps_all_when_no_blobs() {
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let d = vec![
            od(1, U256::from(10u64), 0, Some(a), Some(1)),
            od(2, U256::from(20u64), 0, Some(a), Some(2)),
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
            od(1, U256::from(2u64),  0, Some(a), Some(1)),
            od(2, U256::from(5u64), 10, Some(a), Some(2)), // b1
            od(3, U256::from(3u64),  0, Some(a), Some(3)),
            od(4, U256::from(4u64), 10, Some(a), Some(4)), // b2
            od(5, U256::from(7u64),  0, Some(a), Some(5)),
        ];

        // cap=0: cutoff at nonce 2, keep only id=1
        let keep0 = select_keep_set(&d, 0);
        assert_eq!(keep0.len(), 1);
        assert!(keep0.contains(&dummy_tx_id(1)));
        assert!(!keep0.contains(&dummy_tx_id(2)));

        // cap=10: cutoff at nonce 4, keep ids 1,2,3
        let keep10 = select_keep_set(&d, 10);
        let s10: HashSet<OrderId> = keep10.iter().copied().collect();
        assert_eq!(s10, [dummy_tx_id(1), dummy_tx_id(2), dummy_tx_id(3)].into_iter().collect());

        // cap=20: keep all
        let keep20 = select_keep_set(&d, 20);
        let s20: HashSet<OrderId> = keep20.iter().copied().collect();
        assert_eq!(
            s20,
            [dummy_tx_id(1), dummy_tx_id(2), dummy_tx_id(3), dummy_tx_id(4), dummy_tx_id(5)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn multi_signer_multi_choice_knapsack() {
        let a = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let b = address!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        // Signer A:   1(N,2), 2(B,10,5), 3(N,3)
        // Signer B:   1(B,10,9)
        let d = vec![
            od(11, U256::from(2u64),  0, Some(a), Some(1)),   // A n1 normal (2)
            od(12, U256::from(5u64), 10, Some(a), Some(2)),   // A n2 blob   (5, 10 gas)
            od(13, U256::from(3u64),  0, Some(a), Some(3)),   // A n3 normal (3)
            od(21, U256::from(9u64), 10, Some(b), Some(1)),   // B n1 blob   (9, 10 gas)
        ];

        // cap=10: optimal is {11,21}
        let keep = select_keep_set(&d, 10);
        let s: HashSet<OrderId> = keep.iter().copied().collect();
        let expected: HashSet<OrderId> = [dummy_tx_id(11), dummy_tx_id(21)].into_iter().collect();
        assert_eq!(s, expected, "cap=10 should keep pre-blob A normal and B blob");

        // cap=20: can take both A(k=1) and B(k=1) => keep {11,12,13,21}
        let keep2 = select_keep_set(&d, 20);
        let s2: HashSet<OrderId> = keep2.iter().copied().collect();
        assert!(s2.contains(&dummy_tx_id(11)));
        assert!(s2.contains(&dummy_tx_id(12)));
        assert!(s2.contains(&dummy_tx_id(13)));
        assert!(s2.contains(&dummy_tx_id(21)));
    }

    #[test]
    fn keeps_multi_signer_orders_safely() {
        // Multi-signer (signer=None) is always kept.
        let d = vec![
            OrderDesc { id: dummy_tx_id(1), profit: U256::from(100u64), blob_gas: 0, signer: None, nonce: None },
            OrderDesc { id: dummy_tx_id(2), profit: U256::from(0u64),   blob_gas: 0, signer: None, nonce: None },
        ];
        let keep = select_keep_set(&d, 0);
        assert!(keep.contains(&dummy_tx_id(1)));
        assert!(keep.contains(&dummy_tx_id(2)));
    }
}