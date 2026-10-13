#!/bin/bash
# Build the V4 runner Apptainer (SIF) image.
# Run from repo root: bash executors/container_images/v4_runner/build_sif.sh
#
# Mirrors build.sh (docker) but produces a .sif for clusters without Docker.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR"
SIF_OUT="${SIF_OUT:-/path/to/scratch/images/dsgym_v4_runner.sif}"

# Stage agent_runner.py and prompts into build context (for %files).
cp "$REPO_ROOT/dsgym/harness/agent_runner.py" "$BUILD_DIR/agent_runner.py"
cp -r "$REPO_ROOT/dsgym/harness/prompts_dir_v4" "$BUILD_DIR/prompts_dir_v4"

mkdir -p "$(dirname "$SIF_OUT")"

# Build (must run from BUILD_DIR so %files relative paths resolve).
cd "$BUILD_DIR"
apptainer build --force --fakeroot "$SIF_OUT" kaggle_v4.def

# Cleanup staged files.
rm -f "$BUILD_DIR/agent_runner.py"
rm -rf "$BUILD_DIR/prompts_dir_v4"

echo "Built $SIF_OUT"
