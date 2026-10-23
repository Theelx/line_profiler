#!/bin/bash
# Run bench.py for each built venv of <tag>, pinned to a single core using the
# same core-selection logic as jsonpickle's make benchmark.
#
# Usage: bench/run.sh <tag> [repeats] [scenario ...]
# Benchmarks the 3.10, 3.13 and 3.15 venvs; override with
# BENCH_PYVERS="312 315".
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
NUM_CORES=$(nproc)
# Cores 0, 1 and the last get the most noise; halve for SMT siblings
if [ "$NUM_CORES" -gt 16 ]; then CORE=7; else CORE=$((NUM_CORES / 2 - 1)); fi
# Override with e.g. BENCH_CORE=11 if that core is busy on this machine
CORE=${BENCH_CORE:-$CORE}
for pyver in ${BENCH_PYVERS:-310 313 315}; do
    venv=$BENCH_DIR/.venvs/$tag$pyver
    if [ ! -d "$venv" ]; then echo "== $tag$pyver: not built, skipping"; continue; fi
    echo "== $(basename "$venv") (core $CORE of $NUM_CORES)"
    # Run from / with -I so the repo's uncompiled sources aren't imported
    (cd / && taskset -c "$CORE" "$venv/bin/python" -I "$BENCH_DIR/bench.py" "$@")
done
