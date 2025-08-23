//! Backtest app to build a single block in a similar way as we do in live.
//! It gets the orders from a HistoricalDataStorage, simulates the orders and then runs the building algorithms.
//! It outputs the best algorithm (most profit) so we can check for improvements in our [crate::building::builders::BlockBuildingAlgorithm]s
//! BlockBuildingAlgorithm are defined on the config file but selected on the command line via "--builders"
//! Sample call:
//! backtest-build-block --config /home/happy_programmer/config.toml --builders mgp-ordering --builders mp-ordering 19380913 --show-orders --show-missing


use crate::{
    backtest::{
        execute::{backtest_prepare_orders_from_building_context, BacktestBlockInput},
        OrdersWithTimestamp,
    },
    building::{builders::{BacktestSimulateBlockInput, Block, parallel_builder::{ConflictFinder, ConflictCause}}, BlockBuildingContext, evm_inspector::UsedStateTrace, ExecutionResult, conflict::{find_conflicts_exhaustive, Conflict}},
    live_builder::cli::LiveBuilderConfig,
    primitives::{Order, OrderId, SimulatedOrder},
    provider::StateProviderFactory,
};
use ahash::HashMap;
use alloy_primitives::{utils::format_ether, U256};
use clap::Parser;
use eyre::Context;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Parser, Debug)]
pub struct BuildBlockCfg {
    #[clap(long, help = "Config file path", env = "RBUILDER_CONFIG")]
    pub config: PathBuf,
    #[clap(long, help = "Show all available orders")]
    pub show_orders: bool,
    #[clap(long, help = "Show order data and top of block simulation results")]
    pub show_sim: bool,
    #[clap(long, help = "don't build block")]
    pub no_block_building: bool,
    #[clap(
        long,
        help = "builders to build block with (see config builders)",
        default_value = "mp-ordering"
    )]
    pub builders: Vec<String>,
}

/// Provides all the orders needed to simulate the construction of a block.
/// It also provides the needed context to execute those orders.
pub trait OrdersSource<ConfigType, ProviderType>
where
    ConfigType: LiveBuilderConfig,
    ProviderType: StateProviderFactory + Clone + 'static,
{
    fn config(&self) -> &ConfigType;
    /// Orders available to build blocks with their time of arrival.
    fn available_orders(&self) -> Vec<OrdersWithTimestamp>;
    /// Start of the slot for the block.
    /// Usually all the orders will arrive before block_time_as_unix_ms + 4secs (max get_header time from validator to relays).
    fn block_time_as_unix_ms(&self) -> u64;

    /// ugly: it takes BaseConfig but not all implementations need it.....
    fn create_provider_factory(&self) -> eyre::Result<ProviderType>;

    fn create_block_building_context(&self) -> eyre::Result<BlockBuildingContext>;

    /// Prints any stats specific to the particular OrdersSource implementation (eg: parameters, block simulation)
    fn print_custom_stats(&self, provider: ProviderType) -> eyre::Result<()>;
}

/// Defines the overall structure of the final sims_{block_number}.json file.
#[derive(Serialize)]
struct SimResultFile<'a> {
    simulations: Vec<JsonSimulatedOrder<'a>>,
}

/// Defines the EXACT JSON structure for a single simulated order.
#[derive(Serialize)]
struct JsonSimulatedOrder<'a> {
    order_id: String,
    gas_used: u64,
    coinbase_profit: U256,
    blob_gas_used: u64,
    /// The field will be named "state_trace" in the JSON and omitted if None.
    #[serde(rename = "state_trace", skip_serializing_if = "Option::is_none")]
    used_state_trace: &'a Option<UsedStateTrace>,
}

#[derive(Serialize)]
struct JsonBuilderResult {
    block_summary: JsonBlockSummary,
    ordered_transactions: Vec<JsonOrderedTransaction>,
}

#[derive(Serialize)]
struct JsonBlockSummary {
    gas_used: u64,
    blob_gas_used: u64,
    num_orders: usize,
    raw_coinbase_profit: U256,
}

#[derive(Serialize)]
struct JsonOrderedTransaction {
    order_id: String,
    gas_used: u64,
    coinbase_profit: U256,
}

pub async fn run_backtest_build_block<ConfigType, OrdersSourceType, ProviderType>(
    build_block_cfg: BuildBlockCfg,
    orders_source: OrdersSourceType,
) -> eyre::Result<()>
where
    ConfigType: LiveBuilderConfig,
    ProviderType: StateProviderFactory + Clone + 'static,
    OrdersSourceType: OrdersSource<ConfigType, ProviderType>,
{
    let config = orders_source.config();
    config.base_config().setup_tracing_subscriber()?;

    let available_orders = orders_source.available_orders();
    println!("Available orders: {}", available_orders.len());

    if build_block_cfg.show_orders {
        print_order_and_timestamp(&available_orders, orders_source.block_time_as_unix_ms());
    }

    let provider_factory = orders_source.create_provider_factory()?;

    orders_source.print_custom_stats(provider_factory.clone())?;

    let ctx = orders_source.create_block_building_context()?;

    // let orders: Vec<Order> = available_orders
    //     .iter()
    //     .map(|owt| owt.order.clone())
    //     .collect();

    // use ahash::HashMap as AHashMap;
    // let mut order_by_id: AHashMap<OrderId, Order> = AHashMap::default();
    // for o in &orders {
    //     order_by_id.insert(o.id(), o.clone());
    // }

    // // Helper to turn an Order's nonces into JSON
    // fn serialize_nonces(o: &Order) -> Vec<serde_json::Value> {
    //     o.nonces()
    //         .into_iter()
    //         .map(|n| serde_json::json!({
    //             "address": format!("{:#x}", n.address),
    //             "nonce": n.nonce,
    //         }))
    //         .collect()
    // }


    // use std::fs;
    // use std::path::PathBuf;
    // use serde_json::json;

    // let out = find_conflicts_exhaustive(provider_factory.clone(), &ctx, orders.clone())?;
    // println!(
    //     "pairs={} nonce={} diff={} fatal={} none={}",
    //     out.counts.total_pairs, out.counts.nonce, out.counts.diff, out.counts.fatal, out.counts.none
    // );

    // let pairwise = out.pairwise;
    // let groups = out.groups;

    // if !groups.is_empty() {
    //     println!("Found {} conflict groups:", groups.len());
    //     for group in &groups {
    //         println!("  - Group with {} orders:", group.len());
    //         for order_id in group {
    //             println!("    - {}", order_id);
    //         }
    //     }
    // } else {
    //     println!("No conflict groups found.");
    // }

    // // ---------- write JSON ----------
    // let block_number = ctx.evm_env.block_env.number;

    // let groups_json: Vec<serde_json::Value> = groups
    //     .iter()
    //     .enumerate()
    //     .map(|(i, g)| {
    //         // stable order for IDs
    //         let mut ids: Vec<OrderId> = g.iter().copied().collect();
    //         ids.sort();

    //         // build rich order entries with nonces
    //         let mut orders_entry = Vec::with_capacity(ids.len());
    //         let mut distinct_addrs = ahash::AHashSet::default();
    //         for oid in &ids {
    //             if let Some(ord) = order_by_id.get(oid) {
    //                 let nonces = serialize_nonces(ord);
    //                 // track distinct senders across all nonces in this order
    //                 for n in ord.nonces() {
    //                     distinct_addrs.insert(n.address);
    //                 }
    //                 orders_entry.push(serde_json::json!({
    //                     "order_id": oid.to_string(),
    //                     "nonces": nonces
    //                 }));
    //             } else {
    //                 // order not found (shouldn't happen), still emit id
    //                 orders_entry.push(serde_json::json!({
    //                     "order_id": oid.to_string(),
    //                     "nonces": []
    //                 }));
    //             }
    //         }

    //         serde_json::json!({
    //             "group_id": i,
    //             "order_ids": ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
    //             "orders": orders_entry,
    //             // quick summary to help analyze “how many nonce chains per group”
    //             "distinct_senders": distinct_addrs.len(),
    //         })
    //     })
    //     .collect();


    // let pairwise_json: Vec<serde_json::Value> = pairwise
    //     .into_iter()
    //     .map(|((u1, u2), c)| {
    //         match c {
    //             Conflict::NoConflict => json!({
    //                 "u1": u1.to_string(),
    //                 "u2": u2.to_string(),
    //                 "type": "none"
    //             }),
    //             Conflict::Nonce(addr) => json!({
    //                 "u1": u1.to_string(),
    //                 "u2": u2.to_string(),
    //                 "type": "nonce",
    //                 "address": format!("{:#x}", addr),
    //             }),
    //             Conflict::Fatal => json!({
    //                 "u1": u1.to_string(),
    //                 "u2": u2.to_string(),
    //                 "type": "fatal"
    //             }),
    //             Conflict::DifferentProfit { profit_alone, profit_with_conflict } => json!({
    //                 "u1": u1.to_string(),
    //                 "u2": u2.to_string(),
    //                 "type": "different_profit",
    //                 "profit_alone": profit_alone.to_string(),
    //                 "profit_with_conflict": profit_with_conflict.to_string(),
    //             }),
    //         }
    //     })
    //     .collect();

    // let root = json!({
    //     "header": {
    //         "block_number": block_number,
    //         "counts": {
    //             "total_pairs": out.counts.total_pairs,
    //             "nonce": out.counts.nonce,
    //             "different_profit": out.counts.diff,
    //             "fatal": out.counts.fatal,
    //             "none": out.counts.none,
    //         },
    //         "n_groups": groups_json.len(),
    //         // "n_units": out.units.len() // uncomment if you include units
    //     },
    //     "groups": groups_json,
    //     "pairwise_conflicts": pairwise_json,
    // });

    // let dir = PathBuf::from("conflict_testing/exhaustive_approach");
    // fs::create_dir_all(&dir)?;
    // let path = dir.join(format!("exhaustive_{}.json", block_number));
    // fs::write(&path, serde_json::to_string_pretty(&root)?)?;
    // println!("Saved exhaustive results to {}", path.display());

    use std::{fs, path::PathBuf};
    use serde_json::json;
    use ahash::{HashSet as AHashSet, HashMap as AHashMap};


    let BacktestBlockInput { sim_orders, .. } = backtest_prepare_orders_from_building_context(
        ctx.clone(),
        available_orders.clone(),
        provider_factory.clone(),
        &config.base_config().sbundle_mergeable_signers(),
    )?;

    let mut conflict_finder = ConflictFinder::new();

    let sorted_orders = {
        let mut orders = sim_orders.clone();
        orders.sort_by_key(|o| o.order.id());
        orders
    };

    conflict_finder.add_orders(sorted_orders);
    let groups = conflict_finder.get_order_groups();

    // helper (only needed if you did NOT add #[serde(rename_all="snake_case")] on ConflictCause)
    fn cause_str(c: &ConflictCause) -> &'static str {
        match c {
            ConflictCause::Nonce          => "nonce",
            ConflictCause::StorageRW      => "storage_rw",
            ConflictCause::BalanceRW      => "balance_rw",
            ConflictCause::CodeVsStorage  => "code_vs_storage",
            ConflictCause::CodeVsCode     => "code_vs_code",
        }
    }

    let block_number = ctx.evm_env.block_env.number;

    // ------- groups_json (unchanged from your version) -------
    fn serialize_nonces(o: &Order) -> Vec<serde_json::Value> {
        o.nonces()
            .into_iter()
            .map(|n| json!({
                "address": format!("{:#x}", n.address),
                "nonce":   n.nonce,
            }))
            .collect()
    }

    let mut groups_json: Vec<serde_json::Value> = Vec::with_capacity(groups.len());
    for (gid, g) in groups.iter().enumerate() {
        let mut entries: Vec<(OrderId, &Order)> =
            g.orders.iter().map(|so| (so.order.id(), &so.order)).collect();

        // deterministic order inside group
        entries.sort_by_key(|(oid, _)| *oid);

        let mut order_ids: Vec<String> = Vec::with_capacity(entries.len());
        let mut orders_entry: Vec<serde_json::Value> = Vec::with_capacity(entries.len());
        let mut distinct_addrs: AHashSet<alloy_primitives::Address> = AHashSet::default();

        for (oid, ord) in entries {
            order_ids.push(oid.to_string());
            for n in ord.nonces() {
                distinct_addrs.insert(n.address);
            }
            orders_entry.push(json!({
                "order_id": oid.to_string(),
                "nonces": serialize_nonces(ord),
            }));
        }

        groups_json.push(json!({
            "group_id": gid,
            "order_ids": order_ids,
            "orders": orders_entry,
            "distinct_senders": distinct_addrs.len(),
        }));
    }

    // ------- pairwise_conflicts from ConflictFinder -------
    let pairs_map = conflict_finder.pair_conflicts_map();

    // counts summary by type
    let mut type_counts: AHashMap<&'static str, usize> = AHashMap::default();
    let mut total_pairs = 0usize;

    // Collect and sort pairs deterministically
    let mut pairs_vec: Vec<(&(OrderId, OrderId), &AHashSet<ConflictCause>)> =
        pairs_map.iter().collect();

    // Deterministic ordering by (u1,u2) string
    pairs_vec.sort_by(|a, b| {
        let (a1, a2) = (a.0 .0.to_string(), a.0 .1.to_string());
        let (b1, b2) = (b.0 .0.to_string(), b.0 .1.to_string());
        a1.cmp(&b1).then_with(|| a2.cmp(&b2))
    });

    // Serialize pairwise_conflicts
    let mut pairwise_json = Vec::with_capacity(pairs_vec.len());
    for ((u1, u2), causes) in pairs_vec {
        total_pairs += 1;

        // convert causes to stable, deduped list of strings
        let mut types: Vec<&'static str> = causes.iter().map(cause_str).collect();
        types.sort_unstable();
        types.dedup();

        // bump counts
        for t in &types {
            *type_counts.entry(*t).or_default() += 1;
        }

        pairwise_json.push(json!({
            "u1": u1.to_string(),
            "u2": u2.to_string(),
            "types": types,  // e.g. ["nonce"], ["storage_rw","code_vs_storage"], ...
        }));
    }

    // Optional: sort groups_json deterministically by group_id
    groups_json.sort_by(|a, b| {
        let ag = a.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let bg = b.get("group_id").and_then(|v| v.as_u64()).unwrap_or(0);
        ag.cmp(&bg)
    });

    // Root JSON (mirrors your exhaustive layout)
    let root = json!({
        "header": {
            "block_number": block_number,
            "n_orders": sim_orders.len(),
            "n_groups": groups_json.len(),
            "counts": {
                "total_pairs": total_pairs,
                // only conflicts you actually recorded; “none” not included
                "by_type": type_counts,
            }
        },
        "groups": groups_json,
        "pairwise_conflicts": pairwise_json,
    });

    // Write file
    let outdir = PathBuf::from("conflict_testing/groups_orig");
    fs::create_dir_all(&outdir)?;
    let path = outdir.join(format!("parallel_conflicts_{}.json", block_number));
    fs::write(&path, serde_json::to_string_pretty(&root)?)?;
    println!("Saved parallel groups + pairwise conflicts to {}", path.display());






    Ok(())


    // println!("Generating simulation results JSON...");

    // let json_sims: Vec<JsonSimulatedOrder> = sim_orders
    //     .iter()
    //     .map(|sim_order| JsonSimulatedOrder {
    //         order_id: sim_order.order.id().to_string(),
    //         gas_used: sim_order.sim_value.gas_used,
    //         coinbase_profit: sim_order.sim_value.coinbase_profit,
    //         blob_gas_used: sim_order.sim_value.blob_gas_used,
    //         used_state_trace: &sim_order.used_state_trace,
    //     })
    //     .collect();

    // let sim_result_file = SimResultFile { simulations: json_sims };

    // let output_dir = Path::new("rbuilder_results_1/build_outputs");
    // let filename = format!("sims_{}.json", ctx.block());
    // let file_path = output_dir.join(&filename);

    // fs::create_dir_all(output_dir)
    //     .wrap_err_with(|| format!("Failed to create output directory at {:?}", output_dir))?;

    // let json_string = serde_json::to_string_pretty(&sim_result_file)
    //     .wrap_err("Failed to serialize simulation results to JSON")?;

    // fs::write(&file_path, json_string)
    //     .wrap_err_with(|| format!("Failed to write simulation results to file at {:?}", file_path))?;

    // println!("Successfully saved simulation results to {}", file_path.display());

    // if build_block_cfg.show_sim {
    //     let order_and_timestamp: HashMap<OrderId, u64> = available_orders
    //         .iter()
    //         .map(|order| (order.order.id(), order.timestamp_ms))
    //         .collect();
    //     print_simulated_orders(
    //         &sim_orders,
    //         &order_and_timestamp,
    //         orders_source.block_time_as_unix_ms(),
    //     );
    // }

    // if !build_block_cfg.no_block_building {
    //     let builder_results: Vec<Block> = build_block_cfg
    //     .builders
    //     .iter()
    //     .filter_map(|builder_name: &String| {
    //         println!("\nBuilding block with builder: {}", builder_name);
    //         let input = BacktestSimulateBlockInput {
    //             ctx: ctx.clone(),
    //             builder_name: builder_name.clone(),
    //             sim_orders: &sim_orders,
    //             provider: provider_factory.clone(),
    //         };
    //         let build_res = config.build_backtest_block(builder_name, input);
    //             if let Err(err) = &build_res {
    //                 println!("  - Error building block: {:?}", err);
    //                 return None;
    //             }
                
    //             // On success, unwrap the block and print detailed logs
    //             let block = build_res.ok()?;

    //             println!("  - Built block {} with builder: {:?}", ctx.block(), builder_name);
    //             println!("  - Builder profit: {}", format_ether(block.trace.bid_value));
    //             println!("  - Number of used orders: {}", block.trace.included_orders.len());

    //             println!("  - Used orders:");
    //             for order_result in &block.trace.included_orders {
    //                 println!(
    //                     "    {:>74} gas: {:>8} profit: {}",
    //                     order_result.order.id().to_string(),
    //                     order_result.gas_used,
    //                     format_ether(order_result.coinbase_profit),
    //                 );
    //                 if let Order::Bundle(_) | Order::ShareBundle(_) = &order_result.order {
    //                     for tx in &order_result.txs {
    //                         println!("      ↳ {:?}", tx.hash());
    //                     }

    //                     for (to, value) in &order_result.paid_kickbacks {
    //                         println!("      - kickback to: {:?} value: {}", to, format_ether(*value));
    //                     }
    //                 }
    //             }

    //             Some(block)
    //     })
    //     .collect();

    // // --- 2. Process the collected results to generate the final JSON output ---
    // let mut final_json_output = HashMap::default();
    // for block in &builder_results {
    //     let gas_used: u64 = block
    //         .trace
    //         .included_orders
    //         .iter()
    //         .map(|o: &ExecutionResult| o.gas_used) 
    //         .sum::<u64>();                     

    //     let blob_gas_used: u64 = block
    //         .trace
    //         .included_orders
    //         .iter()
    //         .map(|o: &ExecutionResult| o.inplace_sim.blob_gas_used) 
    //         .sum::<u64>(); 

    //     let block_summary = JsonBlockSummary {
    //         gas_used,
    //         blob_gas_used,
    //         num_orders: block.trace.included_orders.len(),
    //         raw_coinbase_profit: block.trace.coinbase_reward,
    //     };

    //     let ordered_transactions = block
    //         .trace
    //         .included_orders
    //         .iter()
    //         .map(|order_result| JsonOrderedTransaction {
    //             order_id: order_result.order.id().to_string(),
    //             gas_used: order_result.gas_used,
    //             coinbase_profit: order_result.coinbase_profit,
    //         })
    //         .collect();

    //     let builder_json = JsonBuilderResult {
    //         block_summary,
    //         ordered_transactions,
    //     };
    //     let raw_name = block.builder_name.as_str();
    //     let key = if raw_name == "backtest_builder" {
    //         "parallel"
    //     } else {
    //         raw_name
    //     };
    //     final_json_output.insert(key.to_string(), builder_json);
    // }


    //     // --- 3. Write the builder results to a new JSON file ---
    //     let builder_results_filename = format!("built_blocks_{}.json", ctx.block());
    //     let builder_results_file_path = output_dir.join(&builder_results_filename);
    //     let builder_json_string = serde_json::to_string_pretty(&final_json_output)?;
    //     fs::write(&builder_results_file_path, builder_json_string).wrap_err("Failed to write builder results file")?;
    //     println!("\nSuccessfully saved builder results to {}", builder_results_file_path.display());


    //     // --- 4. Find and print the winning builder from the collected results ---
    //     let winning_builder = builder_results.iter().max_by_key(|block| block.trace.bid_value);

    //     if let Some(winner) = winning_builder {
    //         println!(
    //             "\nWinning builder: {} with profit: {}",
    //             winner.builder_name,
    //             format_ether(winner.trace.bid_value)
    //         );
    //     } else {
    //         println!("\nNo successful builders found.");
    //     }
    // }

    // Ok(())
}

/// Convert a timestamp in milliseconds to the slot time relative to the given block timestamp.
fn timestamp_ms_to_slot_time(timestamp_ms: u64, block_timestamp: u64) -> i64 {
    (block_timestamp * 1000) as i64 - (timestamp_ms as i64)
}

/// Print the available orders sorted by timestamp.
fn print_order_and_timestamp(orders_with_ts: &[OrdersWithTimestamp], block_time_as_unix_ms: u64) {
    let mut order_by_ts = orders_with_ts.to_vec();
    order_by_ts.sort_by_key(|owt| owt.timestamp_ms);
    for owt in order_by_ts {
        let id = owt.order.id();
        println!(
            "{:>74} ts: {}",
            id.to_string(),
            timestamp_ms_to_slot_time(owt.timestamp_ms, block_time_as_unix_ms)
        );
        for (tx, optional) in owt.order.list_txs() {
            println!("    {:?} {:?}", tx.hash(), optional);
            println!(
                "        from: {:?} to: {:?} nonce: {}",
                tx.signer(),
                tx.to(),
                tx.nonce()
            )
        }
    }
}

/// Print information about simulated orders.
fn print_simulated_orders(
    sim_orders: &[Arc<SimulatedOrder>],
    order_and_timestamp: &HashMap<OrderId, u64>,
    block_time_as_unix_ms: u64,
) {
    println!("Simulated orders: ({} total)", sim_orders.len());
    let mut sorted_orders = sim_orders.to_owned();
    sorted_orders.sort_by_key(|order| order.sim_value.coinbase_profit);
    sorted_orders.reverse();
    for order in sorted_orders {
        let order_timestamp = order_and_timestamp
            .get(&order.order.id())
            .copied()
            .unwrap_or_default();

        let slot_time_ms = timestamp_ms_to_slot_time(order_timestamp, block_time_as_unix_ms);

        println!(
            "{:>74} slot_time_ms: {:>8}, gas: {:>8} profit: {}",
            order.order.id().to_string(),
            slot_time_ms,
            order.sim_value.gas_used,
            format_ether(order.sim_value.coinbase_profit),
        );
    }
    println!();
}
