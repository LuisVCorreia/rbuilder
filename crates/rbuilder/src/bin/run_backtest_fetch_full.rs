use anyhow::Context;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use reqwest;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::PathBuf;
use tokio::{process::Command, time::{sleep, Duration}};


const BASE_COMMAND: &[&str] = &[
    "./target/debug/backtest-fetch",
    "--config",
    "config-backtest.toml",
    "fetch",
];

const OUTPUT_DIR: &str = "rbuilder_results_high_mev_sample/fetch_outputs";
const MAX_RETRIES: u32 = 20;
const DESIRED_BLOCKS: usize = 200;

const SAFETY_START_OFFSET_SECS: i64 = 3 * 60; // +4 minutes
const SAFETY_END_MARGIN_SECS: i64 = 50;        // -50 seconds

const TARGET_DATE_ENV: &str = "MEV_DAY";
const DEFAULT_TARGET_DATE: &str = "2025-07-04";

const API_CALL_PAUSE_MS: u64 = 200;

#[derive(Deserialize)]
struct EtherscanResponse {
    status: String,
    result: String,
}


fn parse_target_date() -> anyhow::Result<NaiveDate> {
    let s = std::env::var(TARGET_DATE_ENV).unwrap_or_else(|_| DEFAULT_TARGET_DATE.to_string());
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d/%m/%Y") { return Ok(d); }
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") { return Ok(d); }
    anyhow::bail!("Could not parse MEV day {:?}. Use DD/MM/YYYY or YYYY-MM-DD.", s)
}

async fn get_block_number_by_timestamp(ts: i64) -> anyhow::Result<Option<u64>> {
    let api_key = std::env::var("ETHERSCAN_API_KEY")
        .context("[Config Error] ETHERSCAN_API_KEY not found in environment.")?;
    let url = format!(
        "https://api.etherscan.io/v2/api?chainid=1&module=block&action=getblocknobytime&timestamp={}&closest=before&apikey={}",
        ts, api_key
    );

    for attempt in 0..5 {
        match reqwest::get(&url).await {
            Ok(resp) => {
                let resp = resp.error_for_status()?;
                let data: EtherscanResponse = resp.json().await?;
                if data.status == "1" {
                    return Ok(Some(data.result.parse::<u64>()?));
                } else {
                    eprintln!("  [API Error] {}", data.result);
                    return Ok(None);
                }
            }
            Err(e) => {
                eprintln!("Attempt {} failed: {}", attempt + 1, e);
                let backoff = Duration::from_secs(2_u64.pow(attempt));
                sleep(backoff).await;
            }
        }
    }

    eprintln!("  [Failure] All retries failed for timestamp {}", ts);
    Ok(None)
}

fn choose_timestamps_evenly(start_ts: i64, end_ts: i64, n: usize) -> Vec<i64> {
    if n == 0 || end_ts <= start_ts { return vec![]; }
    if n == 1 { return vec![start_ts]; }
    let span = (end_ts - start_ts) as f64;
    (0..n).map(|i| {
        let off = (i as f64) * (span / (n as f64 - 1.0));
        start_ts + off.round() as i64
    }).collect()
}

async fn run_backtest_for_block_capped(initial_block_number: u64, end_block_cap: u64) -> anyhow::Result<()> {
    for i in 0..MAX_RETRIES {
        let current_block = initial_block_number + i as u64;
        if current_block > end_block_cap {
            eprintln!("  [Cap] Reached safe end block cap ({}) while retrying.", end_block_cap);
            break;
        }

        let mut expected_json_path = PathBuf::from(OUTPUT_DIR);
        expected_json_path.push(format!("results_{}.json", current_block));
        if tokio::fs::try_exists(&expected_json_path).await? {
            println!("  -- Skipping block {}, already fetched. --", current_block);
            return Ok(());
        }

        if i > 0 {
            println!("  -> Previous block failed, retrying with next block: {}", current_block);
        }

        let cmd_str = format!("{} {}", BASE_COMMAND.join(" "), current_block);
        println!("  -> Running command: {}", cmd_str);

        let output = Command::new(BASE_COMMAND[0])
            .args(&BASE_COMMAND[1..])
            .arg(current_block.to_string())
            .output()
            .await?;

        if output.status.success() {
            let mut log_path = PathBuf::from(OUTPUT_DIR);
            log_path.push(format!("{}.log", current_block));
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let log_content = format!("--- STDOUT ---\n{}\n\n--- STDERR ---\n{}", stdout, stderr);
            tokio::fs::write(&log_path, log_content).await?;
            println!("  -- Success on block {}, output saved to {:?} --", current_block, log_path);
            return Ok(());
        } else {
            eprintln!(
                "  [Exec Error] Command failed for block {}: {}",
                current_block,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    eprintln!(
        "  [Failure] Could not find a valid block after {} tries, starting at {}.",
        MAX_RETRIES, initial_block_number
    );
    Ok(())
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok(); // load .env if present
    tokio::fs::create_dir_all(OUTPUT_DIR).await?;

    let target_date = parse_target_date()?;
    println!("--- High-MEV sampling for {} (UTC) ---", target_date);

    // Day bounds in UTC, then apply safety margins
    let day_start = NaiveDateTime::new(target_date, NaiveTime::from_hms_opt(0, 0, 0).unwrap());
    let day_end   = NaiveDateTime::new(target_date, NaiveTime::from_hms_opt(23, 59, 59).unwrap());
    let t_start_safe = day_start.and_utc().timestamp() + SAFETY_START_OFFSET_SECS; // +3m
    let t_end_safe   = day_end.and_utc().timestamp() - SAFETY_END_MARGIN_SECS;     // -5s

    if t_end_safe <= t_start_safe {
        anyhow::bail!("Safe time window is empty. Check your offsets.");
    }

    // Determine safe block range using the safe timestamps
    let start_block = get_block_number_by_timestamp(t_start_safe).await?
        .context("Failed to get safe start-of-day block")?;
    sleep(Duration::from_millis(API_CALL_PAUSE_MS)).await;

    let end_block = get_block_number_by_timestamp(t_end_safe).await?
        .context("Failed to get safe end-of-day block")?;
    println!(
        "  Safe block range: [{} .. {}], safe UTC window: [{} .. {}]",
        start_block, end_block, t_start_safe, t_end_safe
    );

    // Evenly spaced timestamps within the safe window
    let timestamps = choose_timestamps_evenly(t_start_safe, t_end_safe, DESIRED_BLOCKS);

    // Resolve each timestamp to a block (closest=before), dedupe and clamp to safe range
    let mut picks: BTreeSet<u64> = BTreeSet::new();
    for ts in timestamps {
        if let Some(b) = get_block_number_by_timestamp(ts).await? {
            if b >= start_block && b <= end_block {
                picks.insert(b);
            }
        }
        sleep(Duration::from_millis(API_CALL_PAUSE_MS)).await;
    }

    let chosen: Vec<u64> = picks.into_iter().collect();
    println!("  Selected {} unique blocks within safe range.", chosen.len());

    for (i, b) in chosen.iter().enumerate() {
        println!("\n[{}/{}] Processing block {}", i + 1, chosen.len(), b);
        if let Err(e) = run_backtest_for_block_capped(*b, end_block).await {
            eprintln!("  [Error] Block {}: {}", b, e);
        }
    }

    println!("\n--- Done ---");
    Ok(())
}
