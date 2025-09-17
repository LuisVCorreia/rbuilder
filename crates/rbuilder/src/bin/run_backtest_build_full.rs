use anyhow::Context;
use sqlx::SqlitePool;
use std::path::Path;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

const DB_PATH: &str = "rbuilder_results_1/main.sqlite";
const OUTPUT_DIR: &str = "high_gas_mev/build_outputs";

const EXECUTABLE: &str = "./target/debug/backtest-build-block";
const BASE_ARGS: &[&str] = &["--config", "config-backtest.toml"];

/// Reads the list of block numbers to be built from the SQLite database.
async fn get_blocks_to_build(db_path: &str) -> anyhow::Result<Vec<u64>> {
    if !tokio::fs::try_exists(db_path).await? {
        eprintln!("  [Info] Database not found at '{}'; skipping.", db_path);
        return Ok(Vec::new());
    }

    let connection_string = format!("sqlite://{}", db_path);
    let pool = SqlitePool::connect(&connection_string)
        .await
        .context("Failed to connect to SQLite database")?;

    let rows: Vec<i64> = sqlx::query_scalar("SELECT block_number FROM blocks ORDER BY block_number ASC")
        .fetch_all(&pool)
        .await
        .context("Failed to query block numbers")?;

    pool.close().await;

    let blocks: Vec<u64> = rows.into_iter().filter(|&n| n >= 0).map(|n| n as u64).collect();
    println!("  Found {} blocks to build in the database.", blocks.len());
    Ok(blocks)
}

const ARCHIVE_URL: &str = "https://datateam-archive.flashbots.dev/";

async fn execute_build_all_builders(block_number: u64) -> anyhow::Result<()> {
    let builders = ["parallel"];

    println!("\n  -> Running builders in one go: {:?}", builders);

    let log_filename = format!("{}_all_builders.log", block_number);
    let log_path = Path::new(OUTPUT_DIR).join(&log_filename);

    if tokio::fs::try_exists(&log_path).await? {
        match tokio::fs::read_to_string(&log_path).await {
            Ok(existing) => {
                if !existing.contains(ARCHIVE_URL) {
                    println!(
                        "     -- Skipping: log exists and no '{}' link found: {} --",
                        ARCHIVE_URL, log_filename
                    );
                    return Ok(());
                } else {
                    println!(
                        "     -- Rebuilding: '{}' link present in existing log --",
                        ARCHIVE_URL
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "     [Warn] Could not read existing log '{}': {}. Rebuilding...",
                    log_filename, e
                );
                // fall through to rebuild
            }
        }
    }

    let block_number_str = block_number.to_string();

    let mut command_args: Vec<String> = Vec::new();
    command_args.extend(BASE_ARGS.iter().map(|s| s.to_string()));
    command_args.push(block_number_str);

    for b in builders {
        command_args.push("--builders".into());
        command_args.push(b.into());
    }

    let cmd_str = format!("{} {}", EXECUTABLE, command_args.join(" "));
    println!("     Running command: {}", cmd_str);
    println!("--- Subprocess Output ---");

    let mut child = Command::new(EXECUTABLE)
        .args(command_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context(format!("Failed to spawn command for block {}", block_number))?;

    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut output_lines = Vec::new();

    let mut stdout_line = String::new();
    let mut stderr_line = String::new();
    let mut stdout_done = false;
    let mut stderr_done = false;

    // Read until both stdout and stderr are closed
    while !(stdout_done && stderr_done) {
        tokio::select! {
            read = stdout.read_line(&mut stdout_line), if !stdout_done => {
                match read {
                    Ok(0) => stdout_done = true,
                    Ok(_) => {
                        print!("{}", stdout_line);
                        output_lines.push(stdout_line.clone());
                        stdout_line.clear();
                    }
                    Err(e) => {
                        eprintln!("[read stdout error] {}", e);
                        stdout_done = true;
                    }
                }
            }
            read = stderr.read_line(&mut stderr_line), if !stderr_done => {
                match read {
                    Ok(0) => stderr_done = true,
                    Ok(_) => {
                        eprint!("{}", stderr_line);
                        output_lines.push(stderr_line.clone());
                        stderr_line.clear();
                    }
                    Err(e) => {
                        eprintln!("[read stderr error] {}", e);
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
        println!("     -- Success, output saved to {:?} --", log_path);
    } else {
        eprintln!(
            "     [Exec Error] Command failed with exit code {:?}",
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
    for pass in 1..=3 {
        println!("\n=== Pass {}/3 ===", pass);
        for (i, &block_number) in blocks_to_process.iter().enumerate() {
            println!(
                "\n--- Processing block {} ({}/{}) ---",
                block_number, i + 1, total_blocks
            );
            if let Err(e) = execute_build_all_builders(block_number).await {
                eprintln!(
                    "  [Critical Error] Failed to process block {}: {}. Continuing...",
                    block_number, e
                );
            }
        }
    }

    println!("\n--- Block Building Process Complete ---");
    Ok(())
}
