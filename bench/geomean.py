"""Summarize A-B-A-B results as geometric-mean speedups per scenario type.

Usage: python bench/geomean.py A1.txt A2.txt B1.txt B2.txt

Each file is the output of `bench/run.sh <tag>`.  The speedup of each
scenario is (min profiled time of A) / (min profiled time of B); these are
then combined with geometric means per scenario type (sync Python, async,
and Cython), per Python version and over all versions.
"""
import math
import sys

from compare import load

TYPES = {
    'sync': ['tight_loop', 'many_calls', 'calls_unprofiled', 'recursive',
             'generators'],
    'async': ['async_calls', 'async_tasks'],
    'cython': ['cy_tight_loop', 'cy_many_calls', 'cy_recursive'],
}


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def main():
    a1, a2, b1, b2 = map(load, sys.argv[1:5])
    speedups = {key: min(a1[key], a2[key]) / min(b1[key], b2[key])
                for key in a1 if key in b1}
    versions = sorted({version for version, _ in speedups})
    columns = [*versions, 'all']
    print(f"{'type':10}" + ''.join(f'{c:>9}' for c in columns))
    for kind, scenarios in [*TYPES.items(), ('overall', sum(TYPES.values(), []))]:
        row = []
        for column in columns:
            values = [s for (version, name), s in speedups.items()
                      if name in scenarios and column in (version, 'all')]
            row.append(f'{geomean(values):8.2f}x' if values else f'{"-":>9}')
        print(f'{kind:10}' + ''.join(row))


if __name__ == '__main__':
    main()
