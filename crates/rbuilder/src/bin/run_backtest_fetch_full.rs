use anyhow::Context;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime};
use serde::Deserialize;
use std::path::PathBuf;
use tokio::process::Command;
use reqwest;

const BASE_COMMAND: &[&str] = &[
    "./target/debug/backtest-fetch",
    "--config",
    "config-backtest.toml",
    "fetch",
];
const OUTPUT_DIR: &str = "rbuilder_results/fetch_outputs";
const MAX_RETRIES: u32 = 20;
const TEST_DAY_OF_MONTH: u32 = 15;

/// Represents the JSON response from the Etherscan API.
#[derive(Deserialize)]
struct EtherscanResponse {
    status: String,
    result: String,
}

/// Queries the Etherscan API to get the block number for a given Unix timestamp.
async fn get_block_number_by_timestamp(timestamp: i64) -> anyhow::Result<Option<u64>> {
    let api_key = std::env::var("ETHERSCAN_API_KEY")
        .context("[Config Error] ETHERSCAN_API_KEY not found in environment.")?;

    let api_url = format!(
        "https://api.etherscan.io/v2/api?chainid=1&module=block&action=getblocknobytime&timestamp={}&closest=before&apikey={}",
        timestamp, api_key
    );

    let response = reqwest::get(&api_url).await?.error_for_status()?;
    let data: EtherscanResponse = response.json().await?;

    if data.status == "1" {
        let block_number = data.result.parse::<u64>()?;
        Ok(Some(block_number))
    } else {
        eprintln!("  [API Error] Message: {}", data.result);
        Ok(None)
    }
}

async fn run_backtest_for_block(initial_block_number: u64) -> anyhow::Result<()> {
    for i in 0..MAX_RETRIES {
        let current_block = initial_block_number + i as u64;

        let mut expected_json_path = PathBuf::from(OUTPUT_DIR);
        expected_json_path.push(format!("results_{}.json", current_block));

        if tokio::fs::try_exists(&expected_json_path).await? {
            println!("  -- Skipping block {}, result file already exists. --", current_block);
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
            let log_content = format!(
                "--- STDOUT ---\n{}\n\n--- STDERR (Logs) ---\n{}",
                stdout, stderr
            );

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
        "  [Failure] Could not find a valid block after trying {} blocks, starting from {}.",
        MAX_RETRIES, initial_block_number
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok(); // Load .env file

    let start_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end_date = NaiveDate::from_ymd_opt(2025, 6, 30).unwrap();

    tokio::fs::create_dir_all(OUTPUT_DIR).await?;

    println!("--- Starting Backtest Fetching Process (Strategy: Hourly blocks on the {}th of each month) ---", TEST_DAY_OF_MONTH);
    println!("--- Successful outputs will be saved to the '{}/' directory. ---", OUTPUT_DIR);

    let mut current_date = start_date;
    while current_date <= end_date {
        let last_day_of_month = NaiveDate::from_ymd_opt(
            current_date.year(),
            current_date.month() + 1,
            1,
        )
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(current_date.year() + 1, 1, 1).unwrap())
        .pred_opt()
        .unwrap()
        .day();

        let day_to_test = TEST_DAY_OF_MONTH.min(last_day_of_month);
        let target_day = NaiveDate::from_ymd_opt(current_date.year(), current_date.month(), day_to_test).unwrap();

        println!("\nProcessing date: {}", target_day.format("%Y-%m-%d"));

        for hour in 0..24 {
            let target_dt = NaiveDateTime::new(
                target_day,
                NaiveTime::from_hms_opt(hour, 4, 0).unwrap(),
            );
            
            println!("  Fetching block for {} UTC...", target_dt.format("%H:%M:%S"));
            let timestamp = target_dt.and_utc().timestamp();

            match get_block_number_by_timestamp(timestamp).await {
                Ok(Some(block_number)) => {
                    run_backtest_for_block(block_number).await?;
                }
                Ok(None) => {
                    eprintln!("  Could not fetch block for timestamp {}. Skipping.", timestamp);
                }
                Err(e) => {
                    eprintln!("  An error occurred: {}", e);
                }
            }
        }
        
        // Move to the next month
        current_date = if current_date.month() == 12 {
            NaiveDate::from_ymd_opt(current_date.year() + 1, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(current_date.year(), current_date.month() + 1, 1).unwrap()
        };
    }

    println!("\n--- Backtest Fetching Process Complete ---");
    Ok(())
}
