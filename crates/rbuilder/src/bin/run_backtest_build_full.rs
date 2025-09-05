use anyhow::Context;
use sqlx::SqlitePool;
use std::path::Path;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

const DB_PATH: &str = "rbuilder_results_1/main.sqlite";
const OUTPUT_DIR: &str = "performance_testing/improvements_all";

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

    let rows = sqlx::query_as!(BlockRow, "SELECT block_number FROM blocks ORDER BY block_number ASC")
        .fetch_all(&pool)
        .await?;

    pool.close().await;

    let blocks: Vec<u64> = rows.into_iter().map(|row| row.block_number as u64).collect();

    println!("  Found {} blocks to build in the database.", blocks.len());
    Ok(blocks)
}


async fn run_build_for_block(block_number: u64) -> anyhow::Result<()> {
    let log_path = Path::new(OUTPUT_DIR).join(format!("{}.log", block_number));
    let json_path = Path::new(OUTPUT_DIR).join(format!("block_{}.json", block_number));
    if tokio::fs::try_exists(&json_path).await? {
        println!("  -- Skipping block {}, result file already exists. --", block_number);
        return Ok(());
    }
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

    loop {
        tokio::select! {
            result = stdout.read_line(&mut stdout_line) => {
                match result {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        print!("{}", stdout_line);
                        output_lines.push(stdout_line.clone());
                        stdout_line.clear();
                    }
                    Err(e) => {
                        eprintln!("[Stdout Error] {}", e);
                        break;
                    }
                }
            }
            result = stderr.read_line(&mut stderr_line) => {
                match result {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        eprint!("{}", stderr_line);
                        output_lines.push(stderr_line.clone());
                        stderr_line.clear();
                    }
                    Err(e) => {
                        eprintln!("[Stderr Error] {}", e);
                        break;
                    }
                }
            }
        }
    }

    let status = child.wait().await?;
    println!("--- End Subprocess Output ---");

    if status.success() {
        // let full_output = output_lines.join("");
        // tokio::fs::write(&log_path, full_output).await?;
        // println!("  -- Success on block {}, output saved to {:?} --", block_number, log_path);
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

    let blocks_to_process = get_blocks_to_build(DB_PATH).await?;

    if blocks_to_process.is_empty() {
        println!("\nNo blocks found to process. Exiting.");
        return Ok(());
    }

    let total_blocks = blocks_to_process.len();
    for (i, block_number) in blocks_to_process.into_iter().enumerate() {
        println!("\n--- Processing block {} ({}/{}) ---", block_number, i + 1, total_blocks);
        if let Err(e) = run_build_for_block(block_number).await {
            eprintln!(
                "  [Critical Error] Failed to process block {}: {}. Continuing...",
                block_number, e
            );
        }
    }

    println!("\n--- Block Building Process Complete ---");
    Ok(())
}
