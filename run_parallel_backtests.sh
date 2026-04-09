#!/usr/bin/env bash

set -u

START_BLOCK=23549981
END_BLOCK=23550460
BIN="./target/debug/backtest-build-block-testing"
CONFIG="config-backtest-testing.toml"
BUILDER="parallel"

count=0
for block_num in $(seq "$START_BLOCK" "$END_BLOCK"); do
  if [ "$count" -ge 200 ]; then
    break
  fi

  echo "count = $count"

  echo "==> Running block ${block_num}"

  if "$BIN" --config "$CONFIG" "$block_num" --builders "$BUILDER"; then
    echo "Finished ${block_num}"
    count=$((count + 1))
  else
    echo "Skipping ${block_num} (failed or missing in DB)"
  fi

  echo
done