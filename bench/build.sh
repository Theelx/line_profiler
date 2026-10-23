#!/bin/bash
# Build line_profiler into one venv per Python version, all with the same
# (pinned) Cython from constraints.txt.
#
# Usage: bench/build.sh [--all] <tag> [<git-ref> | worktree]
#   <git-ref>   build that commit (e.g. origin/main), exported via git archive
#   worktree    build the current working tree, uncommitted changes included
#               (the default)
#   --all       also build 3.11, 3.12 and 3.14, which are only needed for running
#               the tests (benchmarks use 3.10, 3.13 and 3.15)
# Venvs land in bench/.venvs/<tag><pyver>, e.g. bench/.venvs/base312.
# Each version is the newest installed patch release (with pyenv) of that
# minor version; missing ones are skipped.  Override the interpreters with
# PYTHONS="/path/to/python3.x ...".
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
# Override REPO_DIR to build/test another checkout (e.g. a worktree)
REPO_DIR=${REPO_DIR:-$(dirname "$BENCH_DIR")}
VERSIONS="3.15 3.13 3.10"
if [ "${1:-}" = --all ]; then
    VERSIONS="3.15 3.14 3.13 3.12 3.11 3.10"
    shift
fi
tag=$1
ref=${2:-worktree}
newest_patch() {
    ls "$HOME/.pyenv/versions" 2>/dev/null | grep -E "^${1//./\\.}\.[0-9]+$" \
        | sort -V | tail -n 1
}
PYTHONS=${PYTHONS:-"$(for v in $VERSIONS; do
    p=$(newest_patch "$v"); echo "$HOME/.pyenv/versions/${p:-$v-missing}/bin/python"
done)"}
export PIP_CONSTRAINT=$BENCH_DIR/constraints.txt

src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT
if [ "$ref" = worktree ]; then
    # Tracked + untracked (non-ignored) files, so new files are included too
    (cd "$REPO_DIR" && git ls-files -z --cached --others --exclude-standard \
        | xargs -0 tar -cf - --ignore-failed-read) | tar -xf - -C "$src"
else
    git -C "$REPO_DIR" archive "$ref" | tar -xf - -C "$src"
fi

for py in $PYTHONS; do
    if [ ! -x "$py" ]; then echo "skipping $py (not installed)"; continue; fi
    s=$("$py" -c 'import sys; print("%d%d" % sys.version_info[:2])')
    venv=$BENCH_DIR/.venvs/$tag$s
    [ -d "$venv" ] || "$py" -m venv "$venv"
    "$venv/bin/pip" install -q --disable-pip-version-check --force-reinstall "$src" >/dev/null
    # The Cython code profiled by the cy_* scenarios of bench.py (built once
    # per venv, since it doesn't depend on line_profiler)
    if ! (cd / && "$venv/bin/python" -c 'import lp_bench_cython' 2>/dev/null); then
        "$venv/bin/pip" install -q --disable-pip-version-check "$BENCH_DIR/cython_bench" >/dev/null
    fi
    so=$(cd / && "$venv/bin/python" -c "import line_profiler._line_profiler as m; print(m.__file__)")
    echo "$tag$s: python $("$venv/bin/python" -c 'import sys; print(sys.version.split()[0])')," \
         "cython $(grep -ao '_cython_3_[0-9]*_[0-9]*' "$so" | sort -u | sed 's/_cython_//; s/_/./g'), ref $ref"
done
