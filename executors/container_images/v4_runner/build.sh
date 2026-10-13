#!/bin/bash
# Build the V4 runner Docker image.
# Run from repo root: bash executors/container_images/v4_runner/build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR"

# Copy agent_runner.py and prompts into build context
cp "$REPO_ROOT/dsgym/harness/agent_runner.py" "$BUILD_DIR/agent_runner.py"
cp -r "$REPO_ROOT/dsgym/harness/prompts_dir_v4" "$BUILD_DIR/prompts_dir_v4"

# Build
docker build -t dsgym-v4-runner:latest "$BUILD_DIR"

# Cleanup copied files
rm -f "$BUILD_DIR/agent_runner.py"
rm -rf "$BUILD_DIR/prompts_dir_v4"

echo "Built dsgym-v4-runner:latest"
