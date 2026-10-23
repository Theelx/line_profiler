#!/bin/bash
# Run the test suite against the installed build in bench/.venvs/<venv>,
# from a scratch copy so the uncompiled sources in the repo aren't imported.
#
# Usage: bench/test.sh <venv> [pytest args ...]   e.g. bench/test.sh new313 -x
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
# Override REPO_DIR to build/test another checkout (e.g. a worktree)
REPO_DIR=${REPO_DIR:-$(dirname "$BENCH_DIR")}
venv=$BENCH_DIR/.venvs/$1; shift
"$venv/bin/pip" install -q --disable-pip-version-check -r "$REPO_DIR/requirements/tests.txt"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cp -r "$REPO_DIR/tests" "$REPO_DIR/pyproject.toml" "$scratch/"
# Some tests also read files elsewhere in the repo (but not the package
# itself, which must come from the venv)
for extra in dev docs; do
    if [ -d "$REPO_DIR/$extra" ]; then cp -r "$REPO_DIR/$extra" "$scratch/"; fi
done
cd "$scratch"
# Tests also spawn python/kernprof from PATH
export PATH="$venv/bin:$PATH"
"$venv/bin/python" -m pytest -p no:cacheprovider tests "$@"
