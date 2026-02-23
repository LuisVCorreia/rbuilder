//! Nonce-aware ordering via dependency graphs.
//!
//! This module generalises the old chain-interleaving approach to handle
//! **bundles** (orders containing transactions from multiple senders).
//!
//! # Core idea
//!
//! Every order *provides* one or more nonce slots `(sender, nonce)` — one per
//! transaction it contains.  An order *requires* the predecessor slot for each
//! sender it touches: if it includes sender A at nonce 3, it needs whoever
//! fills `(A, 2)` to run first (unless no such order exists in the group,
//! meaning the on-chain nonce already satisfies the requirement).
//!
//! These provide/require relationships form a DAG over orders.  Valid execution
//! orderings are exactly the **topological sorts** of this DAG.
//!
//! When multiple orders compete for the same slot (duplicate-nonce candidates),
//! they are mutually exclusive — at most one can be active.  We handle this via
//! either greedy dedup (pick the best candidate) or branching enumeration.
//!

use ahash::{HashMap, HashSet as AHashSet};
use alloy_primitives::{Address, U256};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::BTreeMap;

use crate::building::sim::NonceKey;

use super::ConflictGroup;
use rbuilder_primitives::SimulatedOrder;

pub const ALL_PERMS_CAP: usize = 120;

// ─── Greedy metric helpers ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum GreedyKey {
    Profit,
    MevGasPrice,
}

#[inline]
pub fn value_for(key: GreedyKey, o: &SimulatedOrder) -> U256 {
    match key {
        GreedyKey::Profit => o.sim_value.full_profit_info().coinbase_profit(),
        GreedyKey::MevGasPrice => o.sim_value.full_profit_info().mev_gas_price(),
    }
}

/// Return the "better" index according to primary key (+ direction),
/// secondary key (same direction), then lowest index as tie-break.
fn is_better(a: usize, b: usize, group: &ConflictGroup, key: GreedyKey, reverse: bool) -> bool {
    let secondary = match key {
        GreedyKey::Profit => GreedyKey::MevGasPrice,
        GreedyKey::MevGasPrice => GreedyKey::Profit,
    };
    let (oa, ob) = (&group.orders[a], &group.orders[b]);

    let pa = value_for(key, oa);
    let pb = value_for(key, ob);
    if pa != pb {
        return if reverse { pa < pb } else { pa > pb };
    }
    let sa = value_for(secondary, oa);
    let sb = value_for(secondary, ob);
    if sa != sb {
        return if reverse { sa < sb } else { sa > sb };
    }
    a < b
}

// ─── Per-order dependency info ──────────────────────────────────────────

/// Nonce-level dependency information for a single order.
#[derive(Debug, Clone)]
pub struct OrderDeps {
    /// Nonce slots this order fills (one per transaction in the order).
    pub provides: Vec<NonceKey>,
    /// Predecessor slots that must be filled before this order can execute.
    ///
    /// For each sender the order touches, this contains
    /// `(sender, min_nonce_for_sender - 1)` — but only when `min_nonce > 0`.
    /// An edge is created only if another order in the group provides the
    /// required slot.
    pub requires: Vec<NonceKey>,
}

// ─── Group-level dependency info ────────────────────────────────────────

/// Aggregate dependency information for every order in a [`ConflictGroup`].
///
/// This is the entry point: build one with [`GroupDeps::from_group`], then use
/// it to construct a [`DependencyDag`] for enumeration / sampling.
#[derive(Debug, Clone)]
pub struct GroupDeps {
    /// Per-order dependency info, indexed by position in `group.orders`.
    pub order_deps: Vec<OrderDeps>,
    /// Slot → order indices that can fill it.
    pub slot_providers: HashMap<NonceKey, Vec<usize>>,
    /// Total number of orders in the group.
    pub n: usize,
}

impl GroupDeps {
    /// Extract dependency info from a conflict group.
    ///
    /// Returns `None` if any order has zero transactions (shouldn't happen in
    /// practice but guards against malformed data).
    pub fn from_group(group: &ConflictGroup) -> Option<Self> {
        let n = group.orders.len();
        let mut order_deps = Vec::with_capacity(n);
        let mut slot_providers: HashMap<NonceKey, Vec<usize>> = HashMap::default();

        for (idx, order) in group.orders.iter().enumerate() {
            let txs = order.order.list_txs();
            if txs.is_empty() {
                return None;
            }

            // Group transactions by sender to find the min nonce per sender.
            let mut by_sender: BTreeMap<Address, Vec<u64>> = BTreeMap::new();
            for (tx, _) in &txs {
                by_sender.entry(tx.signer()).or_default().push(tx.nonce());
            }

            let mut provides = Vec::new();
            let mut requires = Vec::new();

            for (sender, mut nonces) in by_sender {
                nonces.sort_unstable();
                nonces.dedup();

                // Every (sender, nonce) pair is a slot this order provides.
                for &nonce in &nonces {
                    provides.push(NonceKey { address: sender, nonce });
                    slot_providers
                        .entry(NonceKey { address: sender, nonce })
                        .or_default()
                        .push(idx);
                }

                // The order needs the predecessor of its *lowest* nonce for
                // this sender. Higher nonces from the same sender within the
                // same bundle are satisfied internally.
                let min_nonce = nonces[0];
                if min_nonce > 0 {
                    requires.push(NonceKey { address: sender, nonce: min_nonce - 1 });
                }
            }

            order_deps.push(OrderDeps { provides, requires });
        }

        Some(Self {
            order_deps,
            slot_providers,
            n,
        })
    }

    /// Are there duplicate-nonce conflicts (multiple orders competing for one slot)?
    pub fn has_conflicts(&self) -> bool {
        self.slot_providers.values().any(|p| p.len() > 1)
    }

    /// Return every slot that has more than one candidate provider.
    pub fn conflicting_slots(&self) -> Vec<(NonceKey, Vec<usize>)> {
        self.slot_providers
            .iter()
            .filter(|(_, providers)| providers.len() > 1)
            .map(|(slot, providers)| (slot.clone(), providers.clone()))
            .collect()
    }

    /// Greedy dedup: iterate orders by value and greedily include each order
    /// whose provided slots are all still free.  Returns the set of active
    /// (selected) order indices.
    ///
    /// For single-tx orders this is equivalent to the old per-step best pick.
    /// For bundles it correctly treats the bundle as atomic (all-or-nothing).
    pub fn dedup_best(
        &self,
        group: &ConflictGroup,
        key: GreedyKey,
        reverse: bool,
    ) -> AHashSet<usize> {
        // Sort order indices by metric.
        let mut ranked: Vec<usize> = (0..self.n).collect();
        ranked.sort_by(|&a, &b| {
            if is_better(a, b, group, key, reverse) {
                std::cmp::Ordering::Less
            } else if is_better(b, a, group, key, reverse) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        let mut taken_slots: AHashSet<NonceKey> = AHashSet::default();
        let mut active: AHashSet<usize> = AHashSet::default();

        for idx in ranked {
            let deps = &self.order_deps[idx];
            let all_free = deps.provides.iter().all(|s| !taken_slots.contains(s));
            if all_free {
                for s in &deps.provides {
                    taken_slots.insert(s.clone());
                }
                active.insert(idx);
            }
        }

        active
    }

    /// Build a [`DependencyDag`] from a set of active order indices.
    ///
    /// Only orders in `active` become nodes.  Edges are derived from the
    /// provide / require relationships among active orders.
    pub fn build_dag(&self, active: &AHashSet<usize>) -> DependencyDag {
        // Map active order indices to dense 0..n_active node ids.
        let nodes: Vec<usize> = {
            let mut v: Vec<usize> = active.iter().copied().collect();
            v.sort_unstable();
            v
        };
        let n = nodes.len();

        let mut node_of: HashMap<usize, usize> = HashMap::default();
        for (ni, &oi) in nodes.iter().enumerate() {
            node_of.insert(oi, ni);
        }

        // Map each slot to its (unique among active orders) provider node.
        let mut slot_to_node: HashMap<NonceKey, usize> = HashMap::default();
        for (ni, &oi) in nodes.iter().enumerate() {
            for slot in &self.order_deps[oi].provides {
                slot_to_node.insert(slot.clone(), ni);
            }
        }

        // Derive edges from requires → provider lookup.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];
        // Track edges to avoid duplicates (a bundle requiring two slots from
        // the same predecessor would otherwise add the edge twice).
        let mut edge_set: Vec<AHashSet<usize>> = (0..n).map(|_| AHashSet::default()).collect();

        for (ni, &oi) in nodes.iter().enumerate() {
            for req_slot in &self.order_deps[oi].requires {
                if let Some(&pred_ni) = slot_to_node.get(req_slot) {
                    if pred_ni != ni && edge_set[pred_ni].insert(ni) {
                        successors[pred_ni].push(ni);
                        in_degree[ni] += 1;
                    }
                }
            }
        }

        DependencyDag {
            nodes,
            node_of,
            successors,
            in_degree,
        }
    }

    /// Convenience: dedup with default metric (max profit) then build the DAG.
    pub fn build_dag_best(&self, group: &ConflictGroup) -> DependencyDag {
        let active = self.dedup_best(group, GreedyKey::Profit, false);
        self.build_dag(&active)
    }

    /// Convenience: all orders active (only valid when there are no conflicts).
    pub fn build_dag_all(&self) -> DependencyDag {
        let active: AHashSet<usize> = (0..self.n).collect();
        self.build_dag(&active)
    }
}

// ─── Dependency DAG ─────────────────────────────────────────────────────

/// A concrete dependency DAG over a selected subset of orders.
///
/// Nodes are identified internally as `0..n` and map back to the original
/// order indices via [`DependencyDag::nodes`].
#[derive(Debug, Clone)]
pub struct DependencyDag {
    /// DAG node `i` corresponds to `group.orders[nodes[i]]`.
    pub nodes: Vec<usize>,
    /// Reverse lookup: original order index → DAG node index.
    pub node_of: HashMap<usize, usize>,
    /// Adjacency list: `successors[i]` = nodes that depend on node `i`.
    pub successors: Vec<Vec<usize>>,
    /// Number of predecessors for each node (used as starting state for algorithms).
    pub in_degree: Vec<usize>,
}

impl DependencyDag {
    /// Number of nodes in the DAG.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check whether the number of topological sorts is ≤ `cap`.
    pub fn topo_sorts_leq(&self, cap: usize) -> bool {
        self.count_topo_sorts_up_to(cap) <= cap
    }

    /// Count topological sorts, stopping as soon as `count > cap`.
    ///
    /// Uses an incremental ready set so each step costs O(ready set size)
    /// rather than O(n).  Finding `cap + 1` results takes O(cap × n) steps
    /// in the worst case, so callers should ensure n is reasonable (the
    /// heuristics in [`orderings_leq_cap`] handle this).
    pub fn count_topo_sorts_up_to(&self, cap: usize) -> usize {
        let n = self.nodes.len();
        if n == 0 {
            return 0;
        }

        let mut in_deg = self.in_degree.clone();
        let mut count = 0usize;
        let mut ready: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();

        fn dfs(
            succ: &[Vec<usize>],
            in_deg: &mut [usize],
            ready: &mut Vec<usize>,
            depth: usize,
            n: usize,
            count: &mut usize,
            cap: usize,
        ) {
            if *count > cap {
                return;
            }
            if depth == n {
                *count += 1;
                return;
            }

            let ready_snapshot: Vec<usize> = ready.clone();

            for &i in &ready_snapshot {
                if *count > cap {
                    return;
                }

                // Remove i from ready set
                ready.retain(|&x| x != i);

                // Place i: decrement successors' in-degrees, add newly ready
                let mut newly_ready = Vec::new();
                for &s in &succ[i] {
                    in_deg[s] -= 1;
                    if in_deg[s] == 0 {
                        ready.push(s);
                        newly_ready.push(s);
                    }
                }

                dfs(succ, in_deg, ready, depth + 1, n, count, cap);

                // Undo: restore in-degrees, remove newly ready, re-add i
                for &s in &succ[i] {
                    in_deg[s] += 1;
                }
                for &s in &newly_ready {
                    ready.retain(|&x| x != s);
                }
                ready.push(i);
            }
        }

        dfs(
            &self.successors,
            &mut in_deg,
            &mut ready,
            0,
            n,
            &mut count,
            cap,
        );
        count
    }

    /// Enumerate all topological sorts, up to `cap` results.
    ///
    /// Each result is a `Vec<usize>` of **original order indices** (not DAG
    /// node ids).
    pub fn enumerate_topo_sorts(&self, cap: usize) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }
        let mut in_deg = self.in_degree.clone();
        let mut current: Vec<usize> = Vec::with_capacity(n);
        let mut results: Vec<Vec<usize>> = Vec::new();
        let mut ready: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();

        fn dfs(
            nodes: &[usize],
            succ: &[Vec<usize>],
            in_deg: &mut [usize],
            ready: &mut Vec<usize>,
            current: &mut Vec<usize>,
            n: usize,
            out: &mut Vec<Vec<usize>>,
            cap: usize,
        ) {
            if out.len() >= cap {
                return;
            }
            if current.len() == n {
                out.push(current.iter().map(|&ni| nodes[ni]).collect());
                return;
            }

            let ready_snapshot: Vec<usize> = ready.clone();

            for &i in &ready_snapshot {
                if out.len() >= cap {
                    return;
                }

                ready.retain(|&x| x != i);

                let mut newly_ready = Vec::new();
                for &s in &succ[i] {
                    in_deg[s] -= 1;
                    if in_deg[s] == 0 {
                        ready.push(s);
                        newly_ready.push(s);
                    }
                }
                current.push(i);

                dfs(nodes, succ, in_deg, ready, current, n, out, cap);

                current.pop();
                for &s in &succ[i] {
                    in_deg[s] += 1;
                }
                for &s in &newly_ready {
                    ready.retain(|&x| x != s);
                }
                ready.push(i);
            }
        }

        dfs(
            &self.nodes,
            &self.successors,
            &mut in_deg,
            &mut ready,
            &mut current,
            n,
            &mut results,
            cap,
        );
        results
    }

    /// Sample one random topological sort.
    ///
    /// At each step every ready node (in-degree 0) is equally likely to be
    /// picked.  Note: this is *not* a uniform sample over all topological
    /// sorts — that's #P-hard in general — but it's a reasonable heuristic.
    ///
    /// Returns **original order indices**.
    pub fn sample_random_topo_sort<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_deg = self.in_degree.clone();
        let mut used = vec![false; n];
        let mut seq: Vec<usize> = Vec::with_capacity(n);

        for _ in 0..n {
            // Collect the ready set.
            let ready: Vec<usize> = (0..n)
                .filter(|&i| !used[i] && in_deg[i] == 0)
                .collect();
            debug_assert!(!ready.is_empty(), "cycle in DAG or logic bug");

            let &chosen = ready.choose(rng).unwrap();
            used[chosen] = true;
            seq.push(self.nodes[chosen]);

            for &s in &self.successors[chosen] {
                in_deg[s] -= 1;
            }
        }

        seq
    }

    /// Sample a topo sort biased toward high-value orders.
    ///
    /// `order_values[oi]` is the value of original order index `oi`
    /// (e.g. coinbase profit as a `f64`). Higher values are placed earlier.
    ///
    /// Internally converts values to **ranks** so the softmax temperature
    /// is scale-independent:
    ///   - `temperature = 0.0` → fully greedy (always pick highest-value ready)
    ///   - `temperature = 1.0` → strong greedy bias, occasional swaps
    ///   - `temperature = 5.0` → moderate randomness
    ///   - `temperature = 50.0` → nearly uniform
    ///
    /// Uses softmax: `P(node) ∝ exp(-rank / temperature)`.
    pub fn sample_weighted_topo_sort<R: Rng + ?Sized>(
        &self,
        order_values: &[f64],
        temperature: f64,
        rng: &mut R,
    ) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_deg = self.in_degree.clone();
        let mut used = vec![false; n];
        let mut seq: Vec<usize> = Vec::with_capacity(n);

        for _ in 0..n {
            let ready: Vec<usize> = (0..n)
                .filter(|&i| !used[i] && in_deg[i] == 0)
                .collect();
            debug_assert!(!ready.is_empty());

            let chosen = if ready.len() == 1 || temperature <= 0.0 {
                // Greedy: pick the highest-value ready node.
                *ready.iter().max_by(|&&a, &&b| {
                    let va = order_values.get(self.nodes[a]).copied().unwrap_or(0.0);
                    let vb = order_values.get(self.nodes[b]).copied().unwrap_or(0.0);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                }).unwrap()
            } else {
                // Rank the ready nodes by value (highest value = rank 0).
                let mut ranked: Vec<(usize, f64)> = ready.iter()
                    .map(|&ni| (ni, order_values.get(self.nodes[ni]).copied().unwrap_or(0.0)))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Softmax over negative rank: P(node) ∝ exp(-rank / temperature).
                let weights: Vec<f64> = ranked.iter().enumerate()
                    .map(|(rank, _)| (-(rank as f64) / temperature).exp())
                    .collect();
                let sum: f64 = weights.iter().sum();

                if sum <= 0.0 {
                    *ready.choose(rng).unwrap()
                } else {
                    let r = rng.gen::<f64>() * sum;
                    let mut acc = 0.0;
                    let mut picked = ranked.last().unwrap().0;
                    for (i, &w) in weights.iter().enumerate() {
                        acc += w;
                        if r <= acc {
                            picked = ranked[i].0;
                            break;
                        }
                    }
                    picked
                }
            };

            used[chosen] = true;
            seq.push(self.nodes[chosen]);
            for &s in &self.successors[chosen] {
                in_deg[s] -= 1;
            }
        }

        seq
    }
}

// ─── Enumeration with duplicate-nonce branching ─────────────────────────

/// Enumerate all valid orderings, branching over **both** slot-conflict
/// choices (which candidate fills each duplicate-nonce slot) **and**
/// topological sort orderings.  Capped at `cap` total results.
///
/// Also respects a work budget so it won't hang on large inputs.
///
/// Each result is a `Vec<usize>` of original order indices.
pub fn enumerate_all_with_choices(
    group_deps: &GroupDeps,
    cap: usize,
) -> Vec<Vec<usize>> {
    let conflicts = group_deps.conflicting_slots();

    if conflicts.is_empty() {
        let dag = group_deps.build_dag_all();
        return dag.enumerate_topo_sorts(cap);
    }

    const WORK_BUDGET: usize = 1_000_000;

    let mut results: Vec<Vec<usize>> = Vec::new();
    let mut taken_slots: AHashSet<NonceKey> = AHashSet::default();
    let mut excluded: AHashSet<usize> = AHashSet::default();
    let mut work = 0usize;

    choices_dfs(
        group_deps,
        &conflicts,
        0,
        &mut taken_slots,
        &mut excluded,
        &mut results,
        cap,
        &mut work,
        WORK_BUDGET,
    );

    results
}

fn choices_dfs(
    deps: &GroupDeps,
    conflicts: &[(NonceKey, Vec<usize>)],
    ci: usize,
    taken_slots: &mut AHashSet<NonceKey>,
    excluded: &mut AHashSet<usize>,
    results: &mut Vec<Vec<usize>>,
    cap: usize,
    work: &mut usize,
    budget: usize,
) {
    if results.len() >= cap || *work >= budget {
        return;
    }

    // All conflicts resolved — build the active set and enumerate topo sorts.
    if ci == conflicts.len() {
        let active: AHashSet<usize> = (0..deps.n).filter(|i| !excluded.contains(i)).collect();
        let dag = deps.build_dag(&active);
        let remaining = cap.saturating_sub(results.len());
        let mut sorts = dag.enumerate_topo_sorts(remaining);
        results.append(&mut sorts);
        return;
    }

    *work += 1;

    let (ref slot, ref candidates) = conflicts[ci];

    // A previously chosen bundle may already provide this slot.
    if taken_slots.contains(&slot) {
        choices_dfs(deps, conflicts, ci + 1, taken_slots, excluded, results, cap, work, budget);
        return;
    }

    // Branch: try each candidate for this slot.
    for &chosen in candidates {
        if excluded.contains(&chosen) {
            continue;
        }

        // Include `chosen` — mark all its slots as taken and exclude rivals.
        let mut newly_taken: Vec<NonceKey> = Vec::new();
        let mut newly_excluded: Vec<usize> = Vec::new();

        for s in &deps.order_deps[chosen].provides {
            if taken_slots.insert(s.clone()) {
                newly_taken.push(s.clone());
            }
            // Exclude every *other* provider of slots this order fills.
            if let Some(providers) = deps.slot_providers.get(&s) {
                for &other in providers {
                    if other != chosen && excluded.insert(other) {
                        newly_excluded.push(other);
                    }
                }
            }
        }

        choices_dfs(deps, conflicts, ci + 1, taken_slots, excluded, results, cap, work, budget);

        // Backtrack.
        for s in &newly_taken {
            taken_slots.remove(s);
        }
        for &e in &newly_excluded {
            excluded.remove(&e);
        }

        if results.len() >= cap || *work >= budget {
            return;
        }
    }
}

/// Sample a random valid ordering, including a random slot-conflict
/// resolution when duplicates exist.
pub fn random_ordering_with_random_choices<R: Rng + ?Sized>(
    group_deps: &GroupDeps,
    rng: &mut R,
) -> Vec<usize> {
    // Resolve slot conflicts randomly.
    let mut taken_slots: AHashSet<NonceKey> = AHashSet::default();
    let mut excluded: AHashSet<usize> = AHashSet::default();

    // Shuffle conflict order to avoid bias.
    let mut conflicts = group_deps.conflicting_slots();
    conflicts.shuffle(rng);

    for (ref slot, ref candidates) in &conflicts {
        if taken_slots.contains(slot) {
            continue;
        }
        let eligible: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|c| !excluded.contains(c))
            .collect();
        if eligible.is_empty() {
            continue;
        }
        let &chosen = eligible.choose(rng).unwrap();

        for s in &group_deps.order_deps[chosen].provides {
            taken_slots.insert(s.clone());
            if let Some(providers) = group_deps.slot_providers.get(s) {
                for &other in providers {
                    if other != chosen {
                        excluded.insert(other);
                    }
                }
            }
        }
    }

    let active: AHashSet<usize> = (0..group_deps.n)
        .filter(|i| !excluded.contains(i))
        .collect();
    let dag = group_deps.build_dag(&active);
    dag.sample_random_topo_sort(rng)
}

// ─── Convenience wrappers (backward-compatible signatures) ──────────────

/// Set of order indices surviving the best-per-slot dedup.
pub fn allowed_indices_after_nonce_dedup(
    group: &ConflictGroup,
    key: GreedyKey,
    reverse: bool,
) -> Option<AHashSet<usize>> {
    let deps = GroupDeps::from_group(group)?;
    Some(deps.dedup_best(group, key, reverse))
}

/// Quick check: can we enumerate all orderings within `cap`?
///
/// Uses fast heuristics to reject obviously-large groups, then falls
/// back to work-budgeted DFS enumeration for the rest.
pub fn orderings_leq_cap(group: &ConflictGroup, cap: usize) -> Option<bool> {
    let deps = GroupDeps::from_group(group)?;

    // After dedup, build the DAG to reason about structure.
    let active = deps.dedup_best(group, GreedyKey::Profit, false);
    let dag = deps.build_dag(&active);
    let n = dag.len();

    // Trivial cases.
    if n <= 1 {
        return Some(true);
    }

    // Fast rejection: root count lower bound
    // If there are r roots (in-degree 0), the ordering count is at least
    // r! (roots are mutually independent). Check cheaply with ln
    let n_roots = dag.in_degree.iter().filter(|&&d| d == 0).count();
    if n_roots > 1 {
        let ln_roots_fact = ln_fact(n_roots);
        if ln_roots_fact > (cap as f64).ln() + 1e-12 {
            return Some(false);
        }
    }

    // Fast rejection: conflict branching multiplier
    // Each conflict with k candidates multiplies orderings by k.
    // If that alone exceeds cap, bail out
    if deps.has_conflicts() {
        let conflicts = deps.conflicting_slots();
        let mut ln_choices: f64 = 0.0;
        for (_, candidates) in &conflicts {
            ln_choices += (candidates.len() as f64).ln();
        }
        if ln_choices > (cap as f64).ln() + 1e-12 {
            return Some(false);
        }
        // Few conflicts — enumerate with the work-budgeted DFS
        let count = enumerate_all_with_choices(&deps, cap + 1).len();
        Some(count <= cap)
    } else {
        // No conflicts — work-budgeted topo sort count
        Some(dag.topo_sorts_leq(cap))
    }
}

// ─── Logging / stats helpers ────────────────────────────────────────────

/// ln(n!) computed via sum of logs (exact enough for moderate n).
pub fn ln_fact(n: usize) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
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

/// Detect whether the group is a trivial single chain (one sender, no
/// duplicate nonces or only one step).
pub fn is_simple_chain(group: &ConflictGroup) -> bool {
    let Some(deps) = GroupDeps::from_group(group) else {
        return false;
    };

    // Count distinct senders.
    let mut senders: AHashSet<Address> = AHashSet::default();
    for d in &deps.order_deps {
        for slot in &d.provides {
            senders.insert(slot.address);
        }
    }

    if senders.len() != 1 {
        return false;
    }

    // Single sender: it's a simple chain if there are no slot conflicts, or
    // if everything collapses to a single step after dedup.
    if !deps.has_conflicts() {
        return true;
    }
    let unique_nonces: AHashSet<u64> = deps
        .order_deps
        .iter()
        .flat_map(|d| d.provides.iter().map(|slot| slot.nonce))
        .collect();
    unique_nonces.len() <= 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{address, Address, Signature, TxHash, B256, U256};
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use crate::primitives::{
        Bundle, MempoolTx, Metadata, Order, SimValue, SimulatedOrder,
        TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
    };

    // ─── Test addresses ─────────────────────────────────────────────

    const SENDER_A: Address = address!("0x000000000000000000000000000000000000000a");
    const SENDER_B: Address = address!("0x000000000000000000000000000000000000000b");
    const SENDER_C: Address = address!("0x000000000000000000000000000000000000000c");

    // ─── Helpers ────────────────────────────────────────────────────

    struct IdGen(u64);

    impl IdGen {
        fn new() -> Self {
            Self(0)
        }

        fn next_hash(&mut self) -> TxHash {
            self.0 += 1;
            TxHash::from(U256::from(self.0))
        }
    }

    /// Build a `Recovered<TransactionSigned>` with a given sender and nonce.
    fn mk_tx(sender: Address, nonce: u64, gen: &mut IdGen) -> Recovered<TransactionSigned> {
        let tx_legacy = TxLegacy {
            nonce,
            ..Default::default()
        };
        Recovered::new_unchecked(
            TransactionSigned::new(
                Transaction::Legacy(tx_legacy),
                Signature::test_signature(),
                gen.next_hash(),
            ),
            sender,
        )
    }

    /// Create a single-tx `SimulatedOrder` (mempool tx) with the given profit.
    fn mk_single_tx_order(
        sender: Address,
        nonce: u64,
        profit: u64,
        gen: &mut IdGen,
    ) -> Arc<SimulatedOrder> {
        let rec = mk_tx(sender, nonce, gen);
        let with_blobs = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(rec).unwrap();
        Arc::new(SimulatedOrder {
            order: Order::Tx(MempoolTx {
                tx_with_blobs: with_blobs,
            }).into(),
            used_state_trace: None,
            sim_value: SimValue {
                coinbase_profit: U256::from(profit),
                ..Default::default()
            },
        })
    }

    /// Create a bundle `SimulatedOrder` containing txs from multiple senders.
    /// `tx_specs` is a list of `(sender, nonce)` pairs for each tx in the bundle.
    fn mk_bundle_order(
        tx_specs: &[(Address, u64)],
        profit: u64,
        gen: &mut IdGen,
    ) -> Arc<SimulatedOrder> {
        let txs: Vec<_> = tx_specs
            .iter()
            .map(|&(sender, nonce)| {
                TransactionSignedEcRecoveredWithBlobs::new_no_blobs(mk_tx(sender, nonce, gen))
                    .unwrap()
            })
            .collect();

        let bundle = Bundle {
            block: Some(0),
            min_timestamp: None,
            max_timestamp: None,
            txs,
            reverting_tx_hashes: Vec::new(),
            hash: B256::ZERO,
            uuid: Uuid::new_v4(),
            replacement_data: None,
            signer: None,
            metadata: Metadata::default(),
            dropping_tx_hashes: Vec::new(),
            refund: None,
            version: LAST_BUNDLE_VERSION,
        };

        Arc::new(SimulatedOrder {
            order: Order::Bundle(bundle),
            used_state_trace: None,
            sim_value: SimValue {
                coinbase_profit: U256::from(profit),
                ..Default::default()
            },
        })
    }

    /// Shorthand to build a ConflictGroup from a list of `Arc<SimulatedOrder>`.
    fn mk_group(orders: Vec<Arc<SimulatedOrder>>) -> ConflictGroup {
        ConflictGroup {
            id: 0,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(HashSet::default()),
        }
    }

    /// Assert that a list of orderings contains a specific expected ordering.
    fn assert_contains_ordering(orderings: &[Vec<usize>], expected: &[usize]) {
        assert!(
            orderings.iter().any(|o| o.as_slice() == expected),
            "Expected ordering {:?} not found in:\n{:?}",
            expected,
            orderings
        );
    }

    /// Verify that an ordering respects all DAG edges.
    fn assert_valid_topo_sort(ordering: &[usize], dag: &DependencyDag) {
        let pos: HashMap<usize, usize> = ordering
            .iter()
            .enumerate()
            .map(|(pos, &oi)| (oi, pos))
            .collect();

        for (ni, succ_list) in dag.successors.iter().enumerate() {
            let from_oi = dag.nodes[ni];
            for &si in succ_list {
                let to_oi = dag.nodes[si];
                assert!(
                    pos[&from_oi] < pos[&to_oi],
                    "Ordering {:?} violates edge: order {} must come before order {}",
                    ordering,
                    from_oi,
                    to_oi,
                );
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // GroupDeps construction
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn group_deps_single_tx_orders() {
        let mut gen = IdGen::new();
        // A@0, A@1, B@0
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        assert_eq!(deps.n, 3);

        // Order 0 provides (A,0), requires nothing (nonce 0).
        assert_eq!(deps.order_deps[0].provides, vec![NonceKey { address: SENDER_A, nonce: 0 }]);
        assert!(deps.order_deps[0].requires.is_empty());

        // Order 1 provides (A,1), requires (A,0).
        assert_eq!(deps.order_deps[1].provides, vec![NonceKey { address: SENDER_A, nonce: 1 }]);
        assert_eq!(deps.order_deps[1].requires, vec![NonceKey { address: SENDER_A, nonce: 0 }]);

        // Order 2 provides (B,0), requires nothing.
        assert_eq!(deps.order_deps[2].provides, vec![NonceKey { address: SENDER_B, nonce: 0 }]);
        assert!(deps.order_deps[2].requires.is_empty());

        assert!(!deps.has_conflicts());
    }

    #[test]
    fn group_deps_bundle_multi_sender() {
        let mut gen = IdGen::new();
        // Bundle with txs from A@3 and B@5.
        let group = mk_group(vec![mk_bundle_order(
            &[(SENDER_A, 3), (SENDER_B, 5)],
            100,
            &mut gen,
        )]);

        let deps = GroupDeps::from_group(&group).unwrap();
        assert_eq!(deps.n, 1);

        let d = &deps.order_deps[0];
        assert!(d.provides.contains(&NonceKey { address: SENDER_A, nonce: 3 }));
        assert!(d.provides.contains(&NonceKey { address: SENDER_B, nonce: 5 }));
        // Requires predecessors: (A,2) and (B,4).
        assert!(d.requires.contains(&NonceKey { address: SENDER_A, nonce: 2 }));
        assert!(d.requires.contains(&NonceKey { address: SENDER_B, nonce: 4 }));
    }

    #[test]
    fn group_deps_duplicate_nonce_detected() {
        let mut gen = IdGen::new();
        // Two txs competing for A@0.
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        assert!(deps.has_conflicts());
        let conflicts = deps.conflicting_slots();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, NonceKey { address: SENDER_A, nonce: 0 });
        assert_eq!(conflicts[0].1.len(), 2);
    }

    // ═════════════════════════════════════════════════════════════════
    // DAG: single sender chain
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_single_chain_three_steps() {
        // A@0 → A@1 → A@2: only one valid ordering.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_A, 2, 300, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.len(), 3);
        assert_eq!(dag.count_topo_sorts_up_to(10), 1);

        let sorts = dag.enumerate_topo_sorts(10);
        assert_eq!(sorts.len(), 1);
        assert_eq!(sorts[0], vec![0, 1, 2]);
    }

    // ═════════════════════════════════════════════════════════════════
    // DAG: two independent senders
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_two_independent_senders() {
        // A@0, A@1, B@0
        // Edge: 0→1 only. B@0 is fully independent.
        // Valid orderings: 3 (B can be in position 0, 1, or 2 relative to the A chain).
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.count_topo_sorts_up_to(10), 3);

        let sorts = dag.enumerate_topo_sorts(10);
        assert_eq!(sorts.len(), 3);
        assert_contains_ordering(&sorts, &[2, 0, 1]); // B first
        assert_contains_ordering(&sorts, &[0, 2, 1]); // B in middle
        assert_contains_ordering(&sorts, &[0, 1, 2]); // B last
    }

    #[test]
    fn dag_three_independent_senders() {
        // A@0, B@0, C@0  — no edges, all independent.
        // Valid orderings: 3! = 6
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.count_topo_sorts_up_to(100), 6);
        assert_eq!(dag.enumerate_topo_sorts(100).len(), 6);
    }

    // ═════════════════════════════════════════════════════════════════
    // DAG: bundle creates cross-chain edges
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_bundle_cross_chain_fully_serial() {
        // idx0: A@0 (tx)
        // idx1: Bundle[A@1, B@0]   — requires (A,0) → edge from idx0
        // idx2: B@1 (tx)           — requires (B,0) → edge from idx1
        // DAG: 0 → 1 → 2 (fully serial)
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 150, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.count_topo_sorts_up_to(10), 1);
        let sorts = dag.enumerate_topo_sorts(10);
        assert_eq!(sorts, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn dag_bundle_partial_dependency() {
        // idx0: A@0 (tx)
        // idx1: B@0 (tx)
        // idx2: Bundle[A@1, B@1]  — requires (A,0) and (B,0) → edges from 0 and 1
        // DAG: 0→2, 1→2.  Orders 0 and 1 are independent.
        // Valid orderings: 2  ([0,1,2] and [1,0,2])
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.count_topo_sorts_up_to(10), 2);
        let sorts = dag.enumerate_topo_sorts(10);
        assert_contains_ordering(&sorts, &[0, 1, 2]);
        assert_contains_ordering(&sorts, &[1, 0, 2]);
    }

    #[test]
    fn dag_bundle_with_internal_nonce_sequence() {
        // A bundle with two txs from the same sender: A@2 and A@3.
        // The bundle provides both slots but only requires (A,1).
        // idx0: A@0 (tx), idx1: A@1 (tx), idx2: Bundle[A@2, A@3]
        // DAG: 0→1→2 (fully serial)
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_bundle_order(&[(SENDER_A, 2), (SENDER_A, 3)], 300, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        // Bundle requires (A,1) — the min nonce is 2, so predecessor is (A,1).
        assert_eq!(deps.order_deps[2].requires, vec![NonceKey { address: SENDER_A, nonce: 1 }]);

        let dag = deps.build_dag_all();
        assert_eq!(dag.count_topo_sorts_up_to(10), 1);
    }

    #[test]
    fn dag_diamond_shape() {
        // idx0: A@0 (tx)     — root
        // idx1: Bundle[A@1, B@0]  — depends on idx0 via (A,0)
        // idx2: Bundle[A@2, C@0]  — depends on idx1 via (A,1)
        // idx3: B@1 (tx)     — depends on idx1 via (B,0)
        // idx4: Bundle[B@2, C@1]  — depends on idx3 via (B,1), idx2 via (C,0)
        //
        // DAG: 0→1→2→4, 1→3→4  (diamond from 1, merge at 4)
        // Orderings: 0 must be first. Then 1. Then 2 and 3 can interleave. Then 4.
        // Valid: [0,1,2,3,4] and [0,1,3,2,4] = 2
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // idx 0
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 0)], 200, &mut gen), // idx 1
            mk_bundle_order(&[(SENDER_A, 2), (SENDER_C, 0)], 300, &mut gen), // idx 2
            mk_single_tx_order(SENDER_B, 1, 150, &mut gen),                  // idx 3
            mk_bundle_order(&[(SENDER_B, 2), (SENDER_C, 1)], 250, &mut gen), // idx 4
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        let sorts = dag.enumerate_topo_sorts(100);
        assert_eq!(sorts.len(), 2);
        assert_contains_ordering(&sorts, &[0, 1, 2, 3, 4]);
        assert_contains_ordering(&sorts, &[0, 1, 3, 2, 4]);
    }

    // ═════════════════════════════════════════════════════════════════
    // DAG: nonce 0 orders have no predecessor requirement
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_nonce_zero_has_no_requirements() {
        // A@0 and B@0: neither requires anything.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        assert!(deps.order_deps[0].requires.is_empty());
        assert!(deps.order_deps[1].requires.is_empty());

        let dag = deps.build_dag_all();
        // Both orderings are valid.
        assert_eq!(dag.count_topo_sorts_up_to(10), 2);
    }

    // ═════════════════════════════════════════════════════════════════
    // DAG: requirement points outside group (no edge created)
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_missing_predecessor_outside_group() {
        // A@5 and B@3: both require predecessors not in the group.
        // No edges — both are roots, 2 orderings.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 5, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 3, 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        // Requires (A,4) and (B,2) — but neither is provided.
        assert!(!deps.order_deps[0].requires.is_empty());
        assert!(!deps.order_deps[1].requires.is_empty());

        let dag = deps.build_dag_all();
        // No edges since the required slots aren't in the group.
        assert_eq!(dag.count_topo_sorts_up_to(10), 2);
    }

    // ═════════════════════════════════════════════════════════════════
    // Duplicate-nonce dedup
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dedup_picks_highest_profit() {
        let mut gen = IdGen::new();
        // Two orders for A@0: profit 100 vs 500.
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 500, &mut gen), // idx 1
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let active = deps.dedup_best(&group, GreedyKey::Profit, false);
        assert!(active.contains(&1));
        assert!(!active.contains(&0));
    }

    #[test]
    fn dedup_reverse_picks_lowest() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 500, &mut gen), // idx 1
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let active = deps.dedup_best(&group, GreedyKey::Profit, true);
        assert!(active.contains(&0));
        assert!(!active.contains(&1));
    }

    #[test]
    fn dedup_bundle_atomic_all_or_nothing() {
        // idx0: A@0 (tx, profit 50)
        // idx1: Bundle[A@0, B@0] (profit 300) — takes both slots
        // idx2: B@0 (tx, profit 200)
        //
        // Greedy by profit: idx1 (300) is best, takes (A,0) and (B,0).
        // idx0 and idx2 are excluded.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 50, &mut gen),
            mk_bundle_order(&[(SENDER_A, 0), (SENDER_B, 0)], 300, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let active = deps.dedup_best(&group, GreedyKey::Profit, false);
        assert_eq!(active.len(), 1);
        assert!(active.contains(&1));
    }

    #[test]
    fn dedup_individual_txs_beat_bundle() {
        // idx0: A@0 (tx, profit 400)
        // idx1: B@0 (tx, profit 300)
        // idx2: Bundle[A@0, B@0] (profit 200) — lower profit
        //
        // Greedy: idx0 (400) first → takes (A,0).  idx1 (300) next → takes (B,0).
        // idx2 (200) can't take either slot → excluded.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 400, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 300, &mut gen),
            mk_bundle_order(&[(SENDER_A, 0), (SENDER_B, 0)], 200, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let active = deps.dedup_best(&group, GreedyKey::Profit, false);
        assert_eq!(active.len(), 2);
        assert!(active.contains(&0));
        assert!(active.contains(&1));
        assert!(!active.contains(&2));
    }

    // ═════════════════════════════════════════════════════════════════
    // Enumeration with choices (duplicate-nonce branching)
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn enumerate_with_choices_duplicate_nonce() {
        // Two competing txs for A@0, plus A@1.
        // Choice 1: pick idx0 → chain [0,2]  → 1 ordering
        // Choice 2: pick idx1 → chain [1,2]  → 1 ordering
        // Total: 2
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1
            mk_single_tx_order(SENDER_A, 1, 300, &mut gen), // idx 2
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let all = enumerate_all_with_choices(&deps, 100);
        assert_eq!(all.len(), 2);
        assert_contains_ordering(&all, &[0, 2]);
        assert_contains_ordering(&all, &[1, 2]);
    }

    #[test]
    fn enumerate_with_choices_respects_cap() {
        // Large enough scenario that we can verify the cap is respected.
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        // 3! = 6 orderings, but cap at 3.
        let all = enumerate_all_with_choices(&deps, 3);
        assert_eq!(all.len(), 3);
    }

    // ═════════════════════════════════════════════════════════════════
    // Random sampling
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn sample_random_topo_sort_is_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();
        let mut rng = rand::thread_rng();

        // Sample many times and verify every result is a valid topo sort.
        for _ in 0..50 {
            let ordering = dag.sample_random_topo_sort(&mut rng);
            assert_eq!(ordering.len(), 4);
            assert_valid_topo_sort(&ordering, &dag);
        }
    }

    #[test]
    fn sample_random_with_bundles_is_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // idx 0
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // idx 1
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 300, &mut gen), // idx 2
            mk_single_tx_order(SENDER_C, 0, 200, &mut gen),                  // idx 3
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let ordering = dag.sample_random_topo_sort(&mut rng);
            assert_eq!(ordering.len(), 4);
            assert_valid_topo_sort(&ordering, &dag);

            // idx 2 must come after idx 0 and idx 1.
            let pos_of = |idx: usize| ordering.iter().position(|&x| x == idx).unwrap();
            assert!(pos_of(0) < pos_of(2));
            assert!(pos_of(1) < pos_of(2));
        }
    }

    #[test]
    fn random_ordering_with_choices_is_valid() {
        let mut gen = IdGen::new();
        // Duplicate nonce + bundle scenario.
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen), // idx 0
            mk_single_tx_order(SENDER_A, 0, 200, &mut gen), // idx 1 (conflict with 0)
            mk_single_tx_order(SENDER_A, 1, 300, &mut gen), // idx 2
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen), // idx 3
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let ordering = random_ordering_with_random_choices(&deps, &mut rng);
            // Should have 3 orders (one of the A@0 duplicates excluded).
            assert_eq!(ordering.len(), 3);
            // Must contain exactly one of idx 0 or idx 1.
            let has_0 = ordering.contains(&0);
            let has_1 = ordering.contains(&1);
            assert!(has_0 ^ has_1, "Exactly one of the duplicates should survive");
            // Must always contain idx 2 and idx 3.
            assert!(ordering.contains(&2));
            assert!(ordering.contains(&3));
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // count / cap helpers
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn topo_sorts_leq_works() {
        let mut gen = IdGen::new();
        // 3 independent orders → 6 orderings.
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert!(!dag.topo_sorts_leq(5));  // 6 > 5
        assert!(dag.topo_sorts_leq(6));   // 6 ≤ 6
        assert!(dag.topo_sorts_leq(100)); // 6 ≤ 100
    }

    #[test]
    fn orderings_leq_cap_no_conflicts() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
        ]);
        // Single chain → 1 ordering.
        assert_eq!(orderings_leq_cap(&group, 1), Some(true));
        assert_eq!(orderings_leq_cap(&group, 0), Some(false));
    }

    // ═════════════════════════════════════════════════════════════════
    // is_simple_chain
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn is_simple_chain_single_sender() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
        ]);
        assert!(is_simple_chain(&group));
    }

    #[test]
    fn is_simple_chain_false_for_two_senders() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 200, &mut gen),
        ]);
        assert!(!is_simple_chain(&group));
    }

    // ═════════════════════════════════════════════════════════════════
    // Empty / single-order edge cases
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn dag_single_order() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![mk_single_tx_order(SENDER_A, 0, 100, &mut gen)]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();

        assert_eq!(dag.len(), 1);
        assert_eq!(dag.count_topo_sorts_up_to(10), 1);
        assert_eq!(dag.enumerate_topo_sorts(10), vec![vec![0]]);
    }

    #[test]
    fn dag_empty_group() {
        let group = mk_group(vec![]);
        let deps = GroupDeps::from_group(&group);
        // Empty group: from_group succeeds but the DAG is empty.
        if let Some(deps) = deps {
            let dag = deps.build_dag_all();
            assert!(dag.is_empty());
            assert_eq!(dag.count_topo_sorts_up_to(10), 0);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Regression: all enumerated orderings are valid topo sorts
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn all_enumerated_orderings_are_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),
            mk_single_tx_order(SENDER_A, 1, 200, &mut gen),
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),
            mk_single_tx_order(SENDER_B, 1, 250, &mut gen),
            mk_single_tx_order(SENDER_C, 0, 300, &mut gen),
        ]);
        // Two chains of length 2 + one independent = C(4,2)*C(2,1)... = 30 orderings.

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();
        let sorts = dag.enumerate_topo_sorts(1000);

        assert_eq!(sorts.len(), 30);
        for ordering in &sorts {
            assert_valid_topo_sort(ordering, &dag);
        }
    }

    #[test]
    fn all_enumerated_orderings_with_bundle_are_valid() {
        let mut gen = IdGen::new();
        let group = mk_group(vec![
            mk_single_tx_order(SENDER_A, 0, 100, &mut gen),                  // idx 0
            mk_single_tx_order(SENDER_B, 0, 150, &mut gen),                  // idx 1
            mk_bundle_order(&[(SENDER_A, 1), (SENDER_B, 1)], 300, &mut gen), // idx 2
            mk_single_tx_order(SENDER_A, 2, 400, &mut gen),                  // idx 3
        ]);

        let deps = GroupDeps::from_group(&group).unwrap();
        let dag = deps.build_dag_all();
        let sorts = dag.enumerate_topo_sorts(1000);

        assert!(sorts.len() >= 1);
        for ordering in &sorts {
            assert_valid_topo_sort(ordering, &dag);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // Utility functions
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn ln_fact_basic() {
        assert_eq!(ln_fact(0), 0.0);
        assert_eq!(ln_fact(1), 0.0);
        let ln6 = ln_fact(3); // ln(6)
        assert!((ln6 - 6.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn format_approx_small_and_large() {
        assert_eq!(format_approx_from_ln(0.0), "≈0");
        assert_eq!(format_approx_from_ln(-1.0), "≈0");
        // ln(6) ≈ 1.79
        let s = format_approx_from_ln(6.0_f64.ln());
        assert_eq!(s, "6");
        // Very large
        let s = format_approx_from_ln(100.0);
        assert!(s.contains('e'));
    }
}
