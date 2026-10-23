"""Print per-line hit counts for each bench.py scenario as JSON.

Timings differ from run to run but hit counts are deterministic, so
diffing this output between two builds checks that a hot-path change
didn't alter what gets recorded.

Usage: python -I bench/dump_hits.py > hits.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
from line_profiler import LineProfiler  # noqa: E402


def main():
    result = {}
    for name, (entry, to_profile) in bench.SCENARIOS.items():
        prof = LineProfiler()
        for f in to_profile:
            prof.add_function(f)
        prof.enable()
        try:
            entry()
        finally:
            prof.disable()
        result[name] = {
            f'{func}:{first_lineno}': [[lineno, nhits]
                                       for lineno, nhits, _ in entries]
            for (_, first_lineno, func), entries
            in sorted(prof.get_stats().timings.items())
        }
    json.dump(result, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
