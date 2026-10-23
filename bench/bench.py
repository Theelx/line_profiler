"""Overhead benchmark for line_profiler's hot path.

Usage: python bench.py [repeats] [scenario ...]
Prints min wall time (ms) for each scenario unprofiled and profiled, and the
slowdown factor.

The cy_* scenarios profile Cython code compiled with line tracing enabled
(bench/cython_bench, installed into the venvs by bench/build.sh); they are
skipped where that isn't installed, or where line_profiler can't line-trace
Cython (Python 3.12).

With LP_BENCH_WRAP=1, a no-op "other tool" is installed during the profiled
runs (a sys.monitoring tool on the profiler tool ID, or a sys.settrace()
tracer with the legacy trace system), and the profiler is created with
wrap_trace=True, so that it calls the other tool's callbacks.
"""
import asyncio
import contextlib
import gc
import os
import sys
import time

from line_profiler import LineProfiler

N = 1_000_000


def tight_loop():
    total = 0
    for x in range(N):
        total += x
    return total


def tiny(x):
    return x + 1


def many_calls():
    total = 0
    for x in range(N // 4):
        total = tiny(total)
    return total


def unprofiled_work(n):
    total = 0
    for x in range(n):
        total += x & 3
    return total


def calls_unprofiled():
    # Profiled caller, unprofiled callee: exercises the reject path
    return unprofiled_work(N)


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def recursive():
    return fib(22)


def gen(n):
    for x in range(n):
        yield x


def generators():
    total = 0
    for x in gen(N // 2):
        total += x
    return total


async def add_one(x):
    return x + 1


async def async_calls_main(n):
    total = 0
    for _ in range(n):
        total = await add_one(total)
    return total


def async_calls():
    # Coroutines awaiting coroutines (which don't suspend)
    return asyncio.run(async_calls_main(N // 4))


async def async_task(n):
    total = 0
    for x in range(n):
        total += x
        await asyncio.sleep(0)
    return total


async def async_tasks_main(ntasks, n):
    return sum(await asyncio.gather(*(async_task(n) for _ in range(ntasks))))


def async_tasks():
    # Many tasks suspending to and resuming from the (unprofiled) event loop
    return asyncio.run(async_tasks_main(100, N // 2000))


SCENARIOS = {
    'tight_loop': (tight_loop, [tight_loop]),
    'many_calls': (many_calls, [many_calls, tiny]),
    'calls_unprofiled': (calls_unprofiled, [calls_unprofiled]),
    'recursive': (recursive, [recursive, fib]),
    'generators': (generators, [generators, gen]),
    'async_calls': (async_calls, [async_calls_main, add_one]),
    'async_tasks': (async_tasks, [async_tasks_main, async_task]),
}
# Scenarios which can't be run here, and why
SKIPPED = {}

try:
    import lp_bench_cython
except ImportError:
    _reason = 'lp_bench_cython not installed (see bench/build.sh)'
    for _name in 'cy_tight_loop', 'cy_many_calls', 'cy_recursive':
        SKIPPED[_name] = _reason
else:
    from line_profiler._line_profiler import CANNOT_LINE_TRACE_CYTHON

    def cy_tight_loop():
        return lp_bench_cython.tight_loop(N)

    def cy_many_calls():
        return lp_bench_cython.many_calls(N // 4)

    def cy_recursive():
        return lp_bench_cython.fib(22)

    if CANNOT_LINE_TRACE_CYTHON:
        _reason = ('line_profiler cannot line-trace Cython on Python '
                   + '.'.join(map(str, sys.version_info[:2])))
        for _name in 'cy_tight_loop', 'cy_many_calls', 'cy_recursive':
            SKIPPED[_name] = _reason
    else:
        SCENARIOS['cy_tight_loop'] = (
            cy_tight_loop, [lp_bench_cython.tight_loop])
        SCENARIOS['cy_many_calls'] = (
            cy_many_calls,
            [lp_bench_cython.many_calls, lp_bench_cython.tiny])
        SCENARIOS['cy_recursive'] = (cy_recursive, [lp_bench_cython.fib])


WRAP = os.environ.get('LP_BENCH_WRAP', '') == '1'


@contextlib.contextmanager
def other_tool():
    """
    Install a no-op tracer for the profiler to wrap around.
    """
    from line_profiler._line_profiler import USE_LEGACY_TRACE

    if USE_LEGACY_TRACE:
        def tracer(frame, event, arg):
            return tracer

        sys.settrace(tracer)
        try:
            yield
        finally:
            sys.settrace(None)
        return

    mon = sys.monitoring
    tool_id = mon.PROFILER_ID
    event_ids = [mon.events.LINE, mon.events.PY_RETURN, mon.events.PY_YIELD]

    def noop(*args):
        return None

    mon.use_tool_id(tool_id, 'bench')
    for event_id in event_ids:
        mon.register_callback(tool_id, event_id, noop)
    mon.set_events(tool_id, sum(event_ids))
    try:
        yield
    finally:
        mon.set_events(tool_id, 0)
        for event_id in event_ids:
            mon.register_callback(tool_id, event_id, None)
        mon.free_tool_id(tool_id)


def timeit(func, repeats):
    func()  # warmup
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def main():
    gc.disable()  # reduce noise, mirroring jsonpickle's make benchmark
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    names = sys.argv[2:] or [*SCENARIOS, *SKIPPED]
    for name in names:
        if name in SKIPPED:
            print(f'{name:18s} skipped: {SKIPPED[name]}')
            continue
        entry, to_profile = SCENARIOS[name]
        base = timeit(entry, repeats)
        prof = LineProfiler(wrap_trace=WRAP)
        for f in to_profile:
            prof.add_function(f)
        with other_tool() if WRAP else contextlib.nullcontext():
            prof.enable()
            try:
                profiled = timeit(entry, repeats)
            finally:
                prof.disable()
        print(f'{name:18s} base {base:8.1f} ms  profiled {profiled:9.1f} ms'
              f'  slowdown {profiled / base:6.2f}x')


if __name__ == '__main__':
    main()
