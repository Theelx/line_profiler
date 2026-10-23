#!/bin/bash
# Time a full wheel build (cythonize + compile + package) of line_profiler,
# with the pinned Cython from constraints.txt.
#
# Usage: bench/build_time.sh [<git-ref> | worktree] [runs]
# Uses Python 3.10 by default; override with PYTHON=/path/to/python.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=${REPO_DIR:-$(dirname "$BENCH_DIR")}
ref=${1:-worktree}
runs=${2:-3}
PYTHON=${PYTHON:-$HOME/.pyenv/versions/3.10.9/bin/python}
export PIP_CONSTRAINT=$BENCH_DIR/constraints.txt

src=$(mktemp -d); out=$(mktemp -d)
trap 'rm -rf "$src" "$out"' EXIT
if [ "$ref" = worktree ]; then
    (cd "$REPO_DIR" && git ls-files -z --cached --others --exclude-standard \
        | xargs -0 tar -cf - --ignore-failed-read) | tar -xf - -C "$src"
else
    git -C "$REPO_DIR" archive "$ref" | tar -xf - -C "$src"
fi
# Warm pip's cache of the build dependencies so they don't skew the first run
"$PYTHON" -m pip wheel -q --disable-pip-version-check --no-deps -w "$out" "$src" >/dev/null
for i in $(seq "$runs"); do
    rm -rf "$out"/*
    start=$(date +%s.%N)
    "$PYTHON" -m pip wheel -q --disable-pip-version-check --no-deps -w "$out" "$src" >/dev/null
    end=$(date +%s.%N)
    LC_NUMERIC=C printf '%s run %d: %.1f s\n' "$ref" "$i" "$(echo "$end - $start" | bc)"
done
