/* SPDX-License-Identifier: MIT */
/* Per-trap cost on the real machine: ns per emulated POPCNT and per
 * emulated PCMPISTRI.  On a CPU that has the instructions this measures
 * the silicon (fractions of a nanosecond) — meaningful only where they
 * trap.  Run on the target: `bench 200000`. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

int main(int argc, char **argv)
{
    long n = argc > 1 ? atol(argv[1]) : 200000;
    volatile uint64_t sink = 0;
    uint64_t v = 0xdeadbeefcafef00dULL;

    double t0 = now_ns();
    for (long i = 0; i < n; i++) { uint64_t r; __asm__ volatile("popcnt %1,%0" : "=r"(r) : "r"(v ^ (uint64_t) i)); sink += r; }
    double t1 = now_ns();

    static const uint8_t a[16] = "needle", b[16] = "hay in a needle";
    for (long i = 0; i < n; i++) {
        uint32_t idx;
        __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\tpcmpistri $0x0c,%%xmm2,%%xmm1\n\tmov %%ecx,%0"
                         : "=&r"(idx) : "r"(a), "r"(b) : "xmm1", "xmm2", "rcx", "cc", "memory");
        sink += idx;
    }
    double t2 = now_ns();

    printf("popcnt    %8.1f ns/op\npcmpistri %8.1f ns/op   (n=%ld, sink=%lu)\n",
           (t1 - t0) / n, (t2 - t1) / n, n, (unsigned long) sink);
    return 0;
}
