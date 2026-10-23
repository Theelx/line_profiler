#!/bin/bash
# Cythonize the extension modules with annotation, producing HTML reports
# which highlight the lines interacting with the Python C-API (the yellower,
# the more), for finding further speedups.
#
# Usage: bench/annotate.sh [<git-ref> | worktree]
# The reports land in bench/annotate/<module>.html.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=${REPO_DIR:-$(dirname "$BENCH_DIR")}
ref=${1:-worktree}
PYTHON=${PYTHON:-$HOME/.pyenv/versions/3.12.11/bin/python}
out=$BENCH_DIR/annotate

# A venv with the pinned Cython
venv=$BENCH_DIR/.venvs/cython
if [ ! -x "$venv/bin/cython" ]; then
    "$PYTHON" -m venv "$venv"
    "$venv/bin/pip" install -q --disable-pip-version-check \
        -c "$BENCH_DIR/constraints.txt" Cython
fi

# Work on a scratch copy so that no generated files end up in the repo
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT
if [ "$ref" = worktree ]; then
    (cd "$REPO_DIR" && git ls-files -z --cached --others --exclude-standard \
        | xargs -0 tar -cf - --ignore-failed-read 2>/dev/null) | tar -xf - -C "$src"
else
    git -C "$REPO_DIR" archive "$ref" | tar -xf - -C "$src"
fi

mkdir -p "$out"
cd "$src"
# The directives (language level, etc.) are in the `# cython:` headers of
# the .pyx files; also build as C++ like setup.py does
for pyx in line_profiler/*.pyx; do
    "$venv/bin/cython" --annotate --cplus -I line_profiler "$pyx" \
        -o "${pyx%.pyx}.cpp" 2>&1 | grep -v "Implicit noexcept" || true
    cp "${pyx%.pyx}.html" "$out/"
    echo "$out/$(basename "${pyx%.pyx}").html"
done
