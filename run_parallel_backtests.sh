#!/usr/bin/env bash

set -u

START_BLOCK=23549801
END_BLOCK=23550250
BIN="./target/debug/backtest-build-block"
CONFIG="config-backtest.toml"
BUILDER="parallel"

for block_num in $(seq "$START_BLOCK" "$END_BLOCK"); do
  echo "==> Running block ${block_num}"

  if "$BIN" --config "$CONFIG" "$block_num" --builders "$BUILDER"; then
    echo "Finished ${block_num}"
  else
    echo "Skipping ${block_num} (failed or missing in DB)"
  fi

  echo
done