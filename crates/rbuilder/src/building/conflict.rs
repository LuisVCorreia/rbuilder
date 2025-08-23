use ahash::{HashMap, HashSet};
use alloy_primitives::{Address, U256};
use itertools::Itertools;
use reth_provider::{StateProvider};
use std::sync::Arc;

use crate::building::{
    BlockBuildingContext, BlockState, PartialBlockFork, ThreadBlockBuildingContext,
    simulate_order, sim::{SimTree, OrderSimResult, SimulatedResult, NonceKey},
};
use crate::primitives::{Order, OrderId};
use crate::provider::StateProviderFactory;
use crate::utils::NonceCache;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Conflict {
    NoConflict,
    Nonce(Address),
    Fatal,
    DifferentProfit { profit_alone: U256, profit_with_conflict: U256 },
}

#[derive(Clone)]
pub struct SimulationUnit {
    target: Order,
    parents: Vec<Order>,
}

impl SimulationUnit {
    fn id(&self) -> OrderId {
        self.target.id()
    }
    fn full_chain(&self) -> Vec<Order> {
        let mut v = self.parents.clone();
        v.push(self.target.clone());
        v
    }
    fn nonce_index(&self) -> HashMap<Address, HashMap<u64, OrderId>> {
        let mut idx: HashMap<Address, HashMap<u64, OrderId>> = HashMap::default();
        for o in self.full_chain() {
            for n in o.nonces() {
                idx.entry(n.address)
                    .or_default()
                    .insert(n.nonce, o.id());
            }
        }
        idx
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConflictCounts {
    pub total_pairs: usize,
    pub nonce: usize,
    pub fatal: usize,
    pub diff: usize,
    pub none: usize,
}


#[derive(Debug, Clone)]
struct CachedChainState {
    bundle: revm::database::BundleState,
    total_profit: U256,
    per_order_profit: Vec<(OrderId, U256)>,
    cumulative_gas_used: u64,
    cumulative_blob_gas_used: u64,
}

#[derive(Default, Clone)]
struct ChainSimCache {
    inner: HashMap<Vec<OrderId>, Arc<CachedChainState>>,
}

impl ChainSimCache {
    fn get(&self, ordering: &[OrderId]) -> (Option<Arc<CachedChainState>>, usize) {
        let mut key = ordering.to_vec();
        while !key.is_empty() {
            if let Some(cs) = self.inner.get(&key) {
                return (Some(cs.clone()), key.len());
            }
            key.pop();
        }
        (None, 0)
    }
    fn put(&mut self, ordering: &[OrderId], cs: CachedChainState) {
        self.inner.insert(ordering.to_vec(), Arc::new(cs));
    }
}

fn order_ids(chain: &[Order]) -> Vec<OrderId> {
    chain.iter().map(|o| o.id()).collect()
}

fn dedupe_by_id<'a>(a: &'a [Order], b: &'a [Order]) -> Vec<Order> {
    let mut seen: HashSet<OrderId> = HashSet::default();
    let mut out = Vec::with_capacity(a.len() + b.len());
    for o in a.iter().chain(b.iter()) {
        let id = o.id();
        if seen.insert(id) {
            out.push(o.clone());
        }
    }
    out
}

fn discover_units<P: StateProviderFactory + Clone>(
    provider: P,
    ctx: &BlockBuildingContext,
    orders: Vec<Order>,
) -> eyre::Result<Vec<SimulationUnit>> {
    // Reuse sim_tree to compute parents for each target.
    let nonces = {
        let state = provider.history_by_block_hash(ctx.attributes.parent)?;
        NonceCache::new(state.into())
    };
    let mut sim_tree = SimTree::new(nonces);

    sim_tree.push_orders(orders.to_vec())?;

    let mut units = Vec::new();
    let mut state_for_sim =
        Arc::<dyn StateProvider>::from(provider.history_by_block_hash(ctx.attributes.parent)?);
    let mut local_ctx = ThreadBlockBuildingContext::default();

    loop {
        let tasks = sim_tree.pop_simulation_tasks(1000);
        if tasks.is_empty() {
            break;
        }

        let mut results = Vec::new();
        for task in tasks {
            let mut block_state = BlockState::new_arc(state_for_sim.clone());
            let res = simulate_order(
                task.parents.clone(),
                task.order.clone(),
                ctx,
                &mut local_ctx,
                &mut block_state,
            )?;

            let (_, next_provider) = block_state.into_parts();
            state_for_sim = next_provider;

            match res.result {
                OrderSimResult::Success(sim_order, nonces_after) => {
                    // record unit
                    units.push(SimulationUnit {
                        target: sim_order.order.clone(),
                        parents: task.parents.clone(),
                    });

                    results.push(SimulatedResult {
                        id: task.id,
                        simulated_order: sim_order,
                        previous_orders: task.parents.clone(),
                        nonces_after: nonces_after
                            .into_iter()
                            .map(|(address, nonce)| NonceKey { address, nonce })
                            .collect(),
                        simulation_time: std::time::Duration::from_millis(0), // duration not needed here
                    });
                }
                OrderSimResult::Failed(_err) => {
                    // do not produce a unit, but still let sim_tree move on if needed
                }
            }
        }
        sim_tree.submit_simulation_tasks_results(results)?;
    }

    Ok(units)
}

fn simulate_chain_with_cache(
    provider: Arc<dyn StateProvider>,
    ctx: &BlockBuildingContext,
    local_ctx: &mut ThreadBlockBuildingContext,
    cache: &mut ChainSimCache,
    chain: &[Order],
) -> eyre::Result<(HashMap<OrderId, U256>, U256)> {
    let ids = order_ids(chain);
    let (cached, start) = cache.get(&ids);

    let mut state = if let Some(cs) = &cached {
        BlockState::new_arc(provider.clone()).with_bundle_state(cs.bundle.clone())
    } else {
        BlockState::new_arc(provider.clone())
    };

    let mut per_order: Vec<(OrderId, U256)> = cached
        .as_ref()
        .map(|cs| cs.per_order_profit.clone())
        .unwrap_or_default();
    let mut total = cached.as_ref().map(|cs| cs.total_profit).unwrap_or(U256::ZERO);
    let mut cumulative_gas = cached.as_ref().map(|cs| cs.cumulative_gas_used).unwrap_or(0u64);
    let mut cumulative_blob = cached
        .as_ref()
        .map(|cs| cs.cumulative_blob_gas_used)
        .unwrap_or(0u64);

    for i in start..chain.len() {
        let mut fork = PartialBlockFork::new(&mut state, ctx, local_ctx);
        match fork.commit_order(&chain[i], cumulative_gas, 0, cumulative_blob, true)? {
            Ok(ok) => {
                total = total.saturating_add(ok.coinbase_profit);
                per_order.push((chain[i].id(), ok.coinbase_profit));
                cumulative_gas = ok.cumulative_gas_used;
                cumulative_blob = ok.cumulative_blob_gas_used;

                // cache prefix
                let prefix_ids = order_ids(&chain[..=i]);
                let cs = CachedChainState {
                    bundle: state.clone_bundle(),
                    total_profit: total,
                    per_order_profit: per_order.clone(),
                    cumulative_gas_used: cumulative_gas,
                    cumulative_blob_gas_used: cumulative_blob,
                };
                cache.put(&prefix_ids, cs);
            }
            Err(_err) => {
                // return a fatal at caller by bubbling an error
                return Err(eyre::eyre!("fatal in chain at {}", chain[i].id()));
            }
        }
    }

    let map = per_order.into_iter().collect::<HashMap<_, _>>();
    Ok((map, total))
}


pub struct ExhaustiveOutput {
    pub pairwise: HashMap<(OrderId, OrderId), Conflict>,
    pub groups: Vec<HashSet<OrderId>>,
    pub units: Vec<SimulationUnit>,
    pub baseline: HashMap<OrderId, U256>,
    pub counts: ConflictCounts,
}

pub fn find_conflicts_exhaustive<P>(
    provider: P,
    ctx: &BlockBuildingContext,
    orders: Vec<Order>,
) -> eyre::Result<ExhaustiveOutput>
where
    P: StateProviderFactory + Clone,
{
    let units = discover_units(provider.clone(), ctx, orders)?;

    // baseline per unit
    let provider_arc =
        Arc::<dyn StateProvider>::from(provider.history_by_block_hash(ctx.attributes.parent)?);
    let mut local_ctx = ThreadBlockBuildingContext::default();
    let mut cache = ChainSimCache::default();

    let mut baseline: HashMap<OrderId, U256> = HashMap::default();
    for u in &units {
        let chain = u.full_chain();
        let (_per, total) =
            simulate_chain_with_cache(provider_arc.clone(), ctx, &mut local_ctx, &mut cache, &chain)?;
        baseline.insert(u.id(), total);
    }

    // Pairwise
    let mut pairwise: HashMap<(OrderId, OrderId), Conflict> = HashMap::default();
    let nonce_idx: HashMap<OrderId, HashMap<Address, HashMap<u64, OrderId>>> = units
        .iter()
        .map(|u| (u.id(), u.nonce_index()))
        .collect();

    for (u1, u2) in units.iter().cartesian_product(units.iter()) {
        if u1.id() == u2.id() {
            continue;
        }

        let k = (u1.id(), u2.id());

        // direct parent/child
        let u1_parents = u1.parents.iter().map(|o| o.id()).collect::<HashSet<_>>();
        let u2_parents = u2.parents.iter().map(|o| o.id()).collect::<HashSet<_>>();
        if u1_parents.contains(&u2.id()) || u2_parents.contains(&u1.id()) {
            // arbitrary: tag with the child's sender if any, else zero address
            let addr = u2
                .target
                .nonces()
                .get(0)
                .map(|n| n.address)
                .unwrap_or(Address::ZERO);
            pairwise.insert(k, Conflict::Nonce(addr));
            continue;
        }

        // same (addr,nonce) used by different txs
        // same (addr,nonce) used by different txs
        if let (Some(n1), Some(n2)) = (nonce_idx.get(&u1.id()), nonce_idx.get(&u2.id())) {
            let mut addr_hit: Option<Address> = None;

            for (addr, m1) in n1 {
                if let Some(m2) = n2.get(addr) {
                    for (nn, id1) in m1 {
                        if let Some(id2) = m2.get(nn) {
                            if id1 != id2 {
                                addr_hit = Some(*addr);
                                break;
                            }
                        }
                    }
                    if addr_hit.is_some() {
                        break;
                    }
                }
            }

            if let Some(a) = addr_hit {
                pairwise.insert(k, Conflict::Nonce(a));
                continue;
            }
        }


        // profit compare
        let chain1 = u1.full_chain();
        let chain2 = u2.full_chain();
        let merged = dedupe_by_id(&chain1, &chain2);

        let res = simulate_chain_with_cache(
            provider_arc.clone(),
            ctx,
            &mut local_ctx,
            &mut cache,
            &merged,
        );

        match res {
            Ok((per, _total)) => {
                let want_ids = u2.full_chain().into_iter().map(|o| o.id()).collect::<HashSet<_>>();
                let mut sum = U256::ZERO;
                for id in want_ids {
                    if let Some(p) = per.get(&id) {
                        sum = sum.saturating_add(*p);
                    }
                }
                let alone = baseline.get(&u2.id()).copied().unwrap_or(U256::ZERO);
                if sum != alone {
                    pairwise.insert(
                        k,
                        Conflict::DifferentProfit {
                            profit_alone: alone,
                            profit_with_conflict: sum,
                        },
                    );
                } else {
                    pairwise.insert(k, Conflict::NoConflict);
                }
            }
            Err(_e) => {
                println!("Fatal in chain for pair ({}, {})", u1.id(), u2.id());
                pairwise.insert(k, Conflict::Fatal);
            }
        }
    }

    fn dsu_find(parent: &mut HashMap<OrderId, OrderId>, x: OrderId) -> OrderId {
        let p = *parent.get(&x).unwrap_or(&x);
        if p == x {
            parent.entry(x).or_insert(x);
            x
        } else {
            let r = dsu_find(parent, p);
            parent.insert(x, r);
            r
        }
    }

    fn dsu_union(parent: &mut HashMap<OrderId, OrderId>, a: OrderId, b: OrderId) {
        let ra = dsu_find(parent, a);
        let rb = dsu_find(parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
    }


    // Groups
    let mut parent: HashMap<OrderId, OrderId> = HashMap::default();

    for u in &units {
        let ids = u.full_chain().into_iter().map(|o| o.id()).collect::<Vec<_>>();
        for id in &ids {
            dsu_find(&mut parent, *id);
        }
        for w in ids.windows(2) {
            dsu_union(&mut parent, w[0], w[1]);
        }
    }

    for ((a, b), c) in &pairwise {
        if !matches!(c, Conflict::NoConflict) {
            dsu_union(&mut parent, *a, *b);
        }
    }

    let mut groups_map: HashMap<OrderId, HashSet<OrderId>> = HashMap::default();
    for id in parent.keys().copied().collect::<Vec<_>>() {
        let r = dsu_find(&mut parent, id);
        groups_map.entry(r).or_default().insert(id);
    }
    let mut groups = groups_map.into_values().collect::<Vec<_>>();
    groups.sort_by_key(|s| std::cmp::Reverse(s.len()));

    
    
    let mut counts = ConflictCounts::default();
    counts.total_pairs = units.len() * units.len() - units.len();

    for c in pairwise.values() {
        match c {
            Conflict::Nonce(_) => counts.nonce += 1,
            Conflict::Fatal => counts.fatal += 1,
            Conflict::DifferentProfit { .. } => counts.diff += 1,
            Conflict::NoConflict => counts.none += 1,
        }
    }


    Ok(ExhaustiveOutput {
        pairwise,
        groups,
        units,
        baseline,
        counts,
    })
}

// Optional: same grouping helper as before
pub fn conflict_sets_from_pairwise(
    conflicts: &HashMap<(OrderId, OrderId), Conflict>,
) -> Vec<HashSet<OrderId>> {
    let mut set_id = 0i32;
    let mut groups: HashMap<i32, HashSet<OrderId>> = HashMap::default();
    let mut loc: HashMap<OrderId, i32> = HashMap::default();

    for ((a, b), c) in conflicts {
        if matches!(c, Conflict::NoConflict) {
            continue;
        }
        let sa = loc.get(a).copied();
        let sb = loc.get(b).copied();
        match (sa, sb) {
            (Some(x), Some(y)) if x == y => {}
            (Some(x), Some(y)) => {
                let mut g1 = groups.remove(&x).unwrap();
                let g2 = groups.remove(&y).unwrap();
                for k in g2 {
                    loc.insert(k, x);
                    g1.insert(k);
                }
                groups.insert(x, g1);
            }
            (Some(x), None) | (None, Some(x)) => {
                let g = groups.get_mut(&x).unwrap();
                g.insert(*a);
                g.insert(*b);
                loc.insert(*a, x);
                loc.insert(*b, x);
            }
            (None, None) => {
                let mut g = HashSet::default();
                g.insert(*a);
                g.insert(*b);
                loc.insert(*a, set_id);
                loc.insert(*b, set_id);
                groups.insert(set_id, g);
                set_id += 1;
            }
        }
    }
    let mut v = groups.into_values().collect::<Vec<_>>();
    v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    v
}
