#include "Python.h"

/* The following timer code comes from Python 2.5.2's _lsprof.c */

#if !defined(HAVE_LONG_LONG)
#error "This module requires long longs!"
#endif

/*** Selection of a high-precision timer ***/

#ifdef MS_WINDOWS

#include <windows.h>

PY_LONG_LONG
hpTimer(void)
{
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return li.QuadPart;
}

double
hpTimerUnit(void)
{
        LARGE_INTEGER li;
        if (QueryPerformanceFrequency(&li))
                return 1.0 / li.QuadPart;
        else
                return 0.000001;  /* unlikely */
}

#elif (defined(PYOS_OS2) && defined(PYCC_GCC))

#include <sys/time.h>

PY_LONG_LONG
hpTimer(void)
{
        struct timeval tv;
        PY_LONG_LONG ret;
        gettimeofday(&tv, (struct timezone *)NULL);
        ret = tv.tv_sec;
        ret = ret * 1000000 + tv.tv_usec;
        return ret;
}

double
hpTimerUnit(void)
{
        return 0.000001;
}

#else

#include <sys/resource.h>
#include <sys/times.h>

/*
 * On x86-64 Linux, time with the CPU's timestamp counter (TSC) where it
 * is reliable, which is much cheaper to read than clock_gettime()
 * (~7ns vs. ~17ns, and there are two reads per line event).
 */
#if defined(__linux__) && defined(__x86_64__) \
        && (defined(__GNUC__) || defined(__clang__))
#   define LINE_PROFILER_HAVE_TSC 1
#   include <cpuid.h>
#   include <stdio.h>
#   include <string.h>
#   include <x86intrin.h>
#endif

#ifdef LINE_PROFILER_HAVE_TSC
static int lp_use_tsc = 0;
#endif
/* Nanoseconds per tick of hpTimer() */
static double lp_ticks_to_ns = 1.0;

static inline PY_LONG_LONG
clock_ns(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (PY_LONG_LONG)ts.tv_sec * 1000000000 + (PY_LONG_LONG)ts.tv_nsec;
}

static inline PY_LONG_LONG
hpTimer(void)
{
#ifdef LINE_PROFILER_HAVE_TSC
        if (lp_use_tsc)
                return (PY_LONG_LONG)__rdtsc();
#endif
        return clock_ns();
}

double
hpTimerUnit(void)
{
        return 0.000000001;
}

#ifdef LINE_PROFILER_HAVE_TSC
/*
 * The CPUID leaf (the value of EAX passed to the CPUID instruction) for
 * "Advanced Power Management Information", and the bit of EDX that it
 * sets if the TSC is invariant, i.e. ticks at a constant rate regardless
 * of frequency scaling and power states (P-, C- and T-states).
 *
 * Note: the meaning of each bit depends on the leaf, e.g. bit 8 of EDX
 * means something else entirely for leaf 7 (structured extended
 * features).
 *
 * References:
 *     Intel 64 and IA-32 Architectures Software Developer's Manual,
 *     Vol. 2A, CPUID, "Initial EAX Value 80000007H" (and Vol. 3B,
 *     section "Invariant TSC");
 *     AMD64 Architecture Programmer's Manual, Vol. 3, appendix E,
 *     "CPUID Fn8000_0007_EDX", bit 8 "TscInvariant";
 *     Linux's X86_FEATURE_CONSTANT_TSC and X86_FEATURE_NONSTOP_TSC,
 *     which are set from the same bit (arch/x86/kernel/cpu/).
 */
#define CPUID_LEAF_POWER_MANAGEMENT 0x80000007u
#define CPUID_EDX_INVARIANT_TSC (1u << 8)

/*
 * Whether the TSC ticks at a constant rate (regardless of frequency
 * scaling and power states), and the kernel trusts it as its clock source
 * (which it doesn't when e.g. the TSCs aren't synchronized between CPUs,
 * as on some VMs).
 */
static int
tsc_is_reliable(void)
{
        unsigned int eax, ebx, ecx, edx;
        char clocksource[64] = {0};
        FILE *f;
        /* (__get_cpuid() fails if the CPU doesn't have the leaf) */
        if (!__get_cpuid(CPUID_LEAF_POWER_MANAGEMENT,
                         &eax, &ebx, &ecx, &edx)
                        || !(edx & CPUID_EDX_INVARIANT_TSC))
                return 0;
        f = fopen("/sys/devices/system/clocksource/clocksource0/"
                  "current_clocksource", "r");
        if (f == NULL)
                return 0;
        if (fgets(clocksource, sizeof(clocksource), f) == NULL)
                clocksource[0] = '\0';
        fclose(f);
        return strncmp(clocksource, "tsc", 3) == 0
                && (clocksource[3] == '\n' || clocksource[3] == '\0');
}
#endif

/*
 * Choose the timer; must be called before hpTimer() is used.
 *
 * Returns: whether the TSC is used
 */
int
hpTimerInit(int allow_tsc)
{
#ifdef LINE_PROFILER_HAVE_TSC
        PY_LONG_LONG ns_start, ns_end;
        unsigned long long ticks_start, ticks_end;
        double ticks_to_ns;
        lp_use_tsc = 0;
        lp_ticks_to_ns = 1.0;
        if (!allow_tsc || !tsc_is_reliable())
                return 0;
        /* Measure the TSC frequency against the monotonic clock, over
         * ~2ms (good to ~1e-5) */
        ns_start = clock_ns();
        ticks_start = __rdtsc();
        do {
                ns_end = clock_ns();
                ticks_end = __rdtsc();
        } while (ns_end - ns_start < 2000000);
        ticks_to_ns = (double)(ns_end - ns_start)
                / (double)(ticks_end - ticks_start);
        /* Sanity check: 100MHz to 20GHz */
        if (!(ticks_to_ns > 0.05 && ticks_to_ns < 10.0))
                return 0;
        lp_ticks_to_ns = ticks_to_ns;
        lp_use_tsc = 1;
        return 1;
#else
        (void)allow_tsc;
        return 0;
#endif
}

/* Multiply hpTimer() differences by this to convert them to
 * hpTimerUnit()s */
double
hpTimerTicksToUnits(void)
{
        return lp_ticks_to_ns;
}

#endif

#if defined(MS_WINDOWS) || (defined(PYOS_OS2) && defined(PYCC_GCC))
int
hpTimerInit(int allow_tsc)
{
        (void)allow_tsc;
        return 0;
}

double
hpTimerTicksToUnits(void)
{
        return 1.0;
}
#endif
