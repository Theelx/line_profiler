"""Compare two builds' bench.py results, from A-B-A-B style runs.

Usage: python bench/compare.py A1.txt A2.txt B1.txt B2.txt

Each file is the output of bench/run.sh <tag>.  Prints the minimum
profiled time of each build per Python version and scenario, the change
from A to B, and the run-to-run spread of each build (as a noise gauge).
"""
import re
import sys


def load(path):
    results = {}
    version = None
    for line in open(path):
        m = re.match(r'== \S*?(\d{3}) ', line)
        if m:
            version = m.group(1)
            continue
        m = re.match(r'(\w+)\s+base\s+([\d.]+) ms\s+profiled\s+([\d.]+) ms',
                     line)
        if m:
            results[version, m.group(1)] = float(m.group(3))
    return results


def main():
    a1, a2, b1, b2 = map(load, sys.argv[1:5])
    print(f"{'py':4} {'scenario':17} {'A ms':>9} {'B ms':>9} {'change':>8}"
          '   (spread A / B)')
    for key in a1:
        a, b = min(a1[key], a2[key]), min(b1[key], b2[key])
        spread_a = abs(a1[key] - a2[key]) / a * 100
        spread_b = abs(b1[key] - b2[key]) / b * 100
        print(f'{key[0]:4} {key[1]:17} {a:9.1f} {b:9.1f} '
              f'{(b / a - 1) * 100:+7.1f}%   ({spread_a:.1f}% / {spread_b:.1f}%)')


if __name__ == '__main__':
    main()
