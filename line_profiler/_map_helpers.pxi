# cython: language_level=3
# cython: legacy_implicit_noexcept=True
# used in _line_profiler.pyx
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref

# If your module already defines PY_LONG_LONG, reuse that.
# long long int is at least 64 bytes assuming c99
ctypedef unsigned long long int uint64
ctypedef long long int int64

cdef extern from "Python_wrapper.h":
    ctypedef long long PY_LONG_LONG

cdef struct LastTime:
    int f_lineno
    PY_LONG_LONG time

cdef struct LineTime:
    long long code
    int lineno
    PY_LONG_LONG total_time
    long nhits

# Types used for mappings from code hash to last/line times.
ctypedef unordered_map[int64, LastTime] LastTimeMap
ctypedef unordered_map[int64, LineTime] LineTimeMap

cdef extern from "timers.c":
    PY_LONG_LONG hpTimer()  # add 'nogil' here only if it truly is nogil

# ---- helpers ----

cdef inline bint last_contains(LastTimeMap* m, int64 key) noexcept nogil:
    return deref(m).count(key) != 0

cdef inline LastTime last_get_value(LastTimeMap* m, int64 key) noexcept:
    # Precondition: last_contains(m, key)
    return deref(m)[key]

cdef inline LastTime* last_find_ptr(LastTimeMap* m, int64 key) noexcept nogil:
    cdef LastTimeMap.iterator it = deref(m).find(key)
    if it == deref(m).end():
        return NULL
    return &deref(it).second

cdef inline void last_set_now(LastTimeMap* m, int64 key, int lineno) noexcept:
    deref(m)[key] = LastTime(lineno, hpTimer())

cdef inline void last_erase_if_present(LastTimeMap* m, int64 key) noexcept nogil:
    cdef LastTimeMap.iterator it = deref(m).find(key)
    if it != deref(m).end():
        deref(m).erase(it)

cdef inline LineTime* line_find_ptr(LineTimeMap* m, int lineno) noexcept nogil:
    cdef LineTimeMap.iterator it = deref(m).find(lineno)
    if it == deref(m).end():
        return NULL
    return &deref(it).second

cdef inline LineTime* line_ensure_entry(LineTimeMap* m, int lineno, long long code_hash) noexcept:
    if not deref(m).count(lineno):
        deref(m)[lineno] = LineTime(code_hash, lineno, 0, 0)
    return &(deref(m)[lineno])
