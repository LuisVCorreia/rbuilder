use anyhow::Context;
use sqlx::SqlitePool;
use std::path::Path;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};
use futures::stream::{self, StreamExt};

// --- Configuration ---

// The path to the SQLite database.
const DB_PATH: &str = "rbuilder_results_1/main.sqlite";

// The directory where successful build logs and JSON outputs will be saved.
const OUTPUT_DIR: &str = "performance_testing/exhaustive_streaming";

const EXECUTABLE: &str = "./target/debug/backtest-build-block";
const BASE_ARGS: &[&str] = &["--config", "config-backtest.toml"];
const BUILDER_ARGS: &[&str] = &[
    "--builders",
    "parallel",
    "--builders",
    "mp-ordering",
    "--builders",
    "mgp-ordering",
];

/// Reads the list of block numbers to be built from the SQLite database using sqlx.
async fn get_blocks_to_build(db_path: &str) -> anyhow::Result<Vec<u64>> {
    if !tokio::fs::try_exists(db_path).await? {
        anyhow::bail!("Database not found at '{}'", db_path);
    }

    // sqlx requires a connection string with a protocol prefix.
    let connection_string = format!("sqlite://{}", db_path);
    let pool = SqlitePool::connect(&connection_string)
        .await
        .context("Failed to connect to SQLite database")?;

    // A simple struct to map the query result to.
    struct BlockRow {
        block_number: i64,
    }

    // Use the query_as! macro for type-safe query execution.
    let rows = sqlx::query_as!(BlockRow, "SELECT block_number FROM blocks ORDER BY block_number ASC")
        .fetch_all(&pool)
        .await?;

    // We can close the pool now that we're done with it.
    pool.close().await;

    // Convert the results (i64) to the u64 we need.
    let blocks: Vec<u64> = rows.into_iter().map(|row| row.block_number as u64).collect();

    println!("  Found {} blocks to build in the database.", blocks.len());
    Ok(blocks)
}

async fn run_build_for_block(block_number: u64) -> anyhow::Result<()> {
    let log_path = Path::new(OUTPUT_DIR).join(format!("{}.log", block_number));
    if tokio::fs::try_exists(&log_path).await? {
        println!("  -- Skipping block {}, result file already exists. --", block_number);
        return Ok(());
    }

    let block_number_str = block_number.to_string();

    let mut command_args = Vec::new();
    command_args.extend_from_slice(BASE_ARGS);
    command_args.push(&block_number_str);
    command_args.extend_from_slice(BUILDER_ARGS);

    let cmd_str = format!("{} {}", EXECUTABLE, command_args.join(" "));
    println!("  -> Running command: {}", cmd_str);
    println!("--- Subprocess Output ---");

    let mut child = Command::new(EXECUTABLE)
        .args(&command_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context(format!("Failed to spawn command for block {}", block_number))?;

    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut output_lines = Vec::new();

    let mut stdout_line = String::new();
    let mut stderr_line = String::new();

    // Optional: prefix lines so parallel runs don’t interleave confusingly
    let prefix = format!("[{}] ", block_number);

    // Read both until BOTH are done (don’t break when one hits EOF)
    let mut stdout_done = false;
    let mut stderr_done = false;

    while !(stdout_done && stderr_done) {
        tokio::select! {
            res = stdout.read_line(&mut stdout_line), if !stdout_done => {
                match res {
                    Ok(0) => { stdout_done = true; }
                    Ok(_) => {
                        print!("{}{}", prefix, stdout_line);
                        output_lines.push(stdout_line.clone());
                        stdout_line.clear();
                    }
                    Err(e) => {
                        eprintln!("{}[Stdout Error] {}", prefix, e);
                        stdout_done = true;
                    }
                }
            }
            res = stderr.read_line(&mut stderr_line), if !stderr_done => {
                match res {
                    Ok(0) => { stderr_done = true; }
                    Ok(_) => {
                        eprint!("{}{}", prefix, stderr_line);
                        output_lines.push(stderr_line.clone());
                        stderr_line.clear();
                    }
                    Err(e) => {
                        eprintln!("{}[Stderr Error] {}", prefix, e);
                        stderr_done = true;
                    }
                }
            }
        }
    }

    let status = child.wait().await?;
    println!("--- End Subprocess Output ---");

    if status.success() {
        let full_output = output_lines.join("");
        tokio::fs::write(&log_path, full_output).await?;
        println!("  -- Success on block {}, output saved to {:?} --", block_number, log_path);
    } else {
        eprintln!(
            "  [Exec Error] Command failed for block {} with exit code {:?}",
            block_number,
            status.code()
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tokio::fs::create_dir_all(OUTPUT_DIR).await?;

    println!("--- Starting rbuilder Block Building Process ---");
    println!("--- Logs for successful builds will be saved to the '{}/' directory. ---", OUTPUT_DIR);

    // let blocks_to_process = get_blocks_to_build(DB_PATH).await?;

    let blocks_to_process = vec![19872271u64];

    if blocks_to_process.is_empty() {
        println!("\nNo blocks found to process. Exiting.");
        return Ok(());
    }

    let concurrency: usize = std::env::var("RBUILDER_TESTING_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let total_blocks = blocks_to_process.len();
    println!("  Planning to process {} blocks with concurrency = {}", total_blocks, concurrency);

    // Queue them all, run up to `concurrency` at a time
    let results = stream::iter(blocks_to_process.into_iter().enumerate())
        .map(|(i, block_number)| async move {
            println!("\n--- Queued block {} ({}/{}) ---", block_number, i + 1, total_blocks);
            let res = run_build_for_block(block_number).await;
            (block_number, res)
        })
        .buffer_unordered(concurrency) // Control concurrency here
        .collect::<Vec<_>>()
        .await;

    // Summarize
    let mut ok = 0usize;
    let mut fail = 0usize;
    for (block, res) in results {
        if let Err(e) = res {
            eprintln!("  [Critical Error] Failed to process block {}: {}", block, e);
            fail += 1;
        } else {
            ok += 1;
        }
    }

    println!("\n--- Block Building Process Complete ---");
    println!("    Success: {} | Failed: {}", ok, fail);
    Ok(())
}
