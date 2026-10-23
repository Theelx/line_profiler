#!/bin/bash
# Mirror this checkout (including .git, excluding the venvs) to the same
# path on another machine, e.g. a quieter one to benchmark on.
#
# Usage: bench/sync.sh <host>
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$BENCH_DIR")
host=$1
rel=${REPO_DIR#"$HOME"/}
rsync -a --delete --exclude 'bench/.venvs' --exclude '__pycache__' \
    "$REPO_DIR/" "$host:$rel/"
