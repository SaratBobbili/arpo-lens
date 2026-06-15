#!/bin/bash
set -euo pipefail

# Source run directory containing rolling global_step_* checkpoints.
SOURCE_RUN_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3BInstruct"
# Destination root where independent snapshots are stored for side evaluation.
DEST_ARCHIVE_ROOT="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoint_snapshots/echo3BInstruct"
# Poll interval in seconds between scans for a newer global_step_* checkpoint.
POLL_SECONDS=30
# true: copy only global_step_*/actor (usually enough for eval); false: copy full global_step_*.
COPY_ACTOR_ONLY="true"
# --once copies the current latest checkpoint once and exits.
RUN_ONCE="false"

print_usage() {
  echo "Usage:"
  echo "  $0 [--source <run_dir>] [--dest <snapshot_dir>] [--poll <seconds>] [--copy-full] [--once]"
}

while (( "$#" )); do
  case "$1" in
    --source)
      SOURCE_RUN_DIR="$2"
      shift 2
      ;;
    --dest)
      DEST_ARCHIVE_ROOT="$2"
      shift 2
      ;;
    --poll)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --copy-full)
      COPY_ACTOR_ONLY="false"
      shift
      ;;
    --once)
      RUN_ONCE="true"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

mkdir -p "$DEST_ARCHIVE_ROOT"

find_latest_checkpoint_dir() {
  local latest_dir=""
  local latest_step=-1
  local candidate=""
  local step=""

  for candidate in "$SOURCE_RUN_DIR"/global_step_*; do
    [[ -d "$candidate" ]] || continue
    step="${candidate##*_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step > latest_step )); then
      latest_step="$step"
      latest_dir="$candidate"
    fi
  done

  [[ -n "$latest_dir" ]] && echo "$latest_dir"
}

snapshot_checkpoint() {
  local src_step_dir="$1"
  local step_name
  local final_dst
  local tmp_dst

  step_name="$(basename "$src_step_dir")"
  final_dst="$DEST_ARCHIVE_ROOT/$step_name"
  tmp_dst="$DEST_ARCHIVE_ROOT/.tmp_${step_name}_$$"

  if [[ -d "$final_dst" ]]; then
    echo "Snapshot already exists, skipping: $final_dst"
    return 0
  fi

  rm -rf "$tmp_dst"
  mkdir -p "$tmp_dst"

  if [[ "$COPY_ACTOR_ONLY" == "true" ]]; then
    rsync -a "$src_step_dir/actor/" "$tmp_dst/actor/"
  else
    rsync -a "$src_step_dir/" "$tmp_dst/"
  fi

  mv "$tmp_dst" "$final_dst"
  echo "Saved snapshot: $final_dst"
}

echo "Watching source checkpoint directory: $SOURCE_RUN_DIR"
echo "Snapshot destination directory: $DEST_ARCHIVE_ROOT"
echo "Copy mode (actor only): $COPY_ACTOR_ONLY"
echo "Run once: $RUN_ONCE"

while true; do
  latest_checkpoint_dir="$(find_latest_checkpoint_dir || true)"

  if [[ -n "$latest_checkpoint_dir" ]]; then
    snapshot_checkpoint "$latest_checkpoint_dir"
  else
    echo "No global_step_* checkpoint found yet in: $SOURCE_RUN_DIR"
  fi

  if [[ "$RUN_ONCE" == "true" ]]; then
    break
  fi

  sleep "$POLL_SECONDS"
done
