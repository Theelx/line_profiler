# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
"""
Cython counterparts of the tight_loop, many_calls, and recursive scenarios in
bench/bench.py, compiled with line tracing enabled so that line_profiler
can profile them (the same options as in tests/cython_example).
"""


def tight_loop(long n):
    cdef long total = 0
    cdef long x
    for x in range(n):
        total += x
    return total


def tiny(long x):
    return x + 1


def many_calls(long n):
    cdef long total = 0
    cdef long i
    for i in range(n):
        total = tiny(total)
    return total


def fib(long n):
    # (Recursing through the module-level function, i.e. a Python-level call)
    return n if n < 2 else fib(n - 1) + fib(n - 2)
