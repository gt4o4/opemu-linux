/* SPDX-License-Identifier: MIT */
/* Cycles per direct call of the core's arithmetic, measured with rdtsc on
 * the machine itself — no traps, so it runs on the Penryn the emulator is
 * for and separates the arithmetic from the trap (bench.c measures the
 * whole trap).  The scalar reference (tests/pcmpstr-ref.c) is timed beside
 * the core for PCMPISTRI, so the gain of the SSE2 rewrite is a number, not
 * a claim.  Run: `bench-core [iterations]`. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../core/emu-core.h"
#include "../pcmpstr-ref.h"

static inline uint64_t rdtsc(void)
{ unsigned lo, hi; __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi)); return ((uint64_t) hi << 32) | lo; }

#define TIME(label, stmt) do {                                                \
    uint64_t t0 = rdtsc();                                                    \
    for (long i = 0; i < n; i++) { stmt; }                                    \
    uint64_t t1 = rdtsc();                                                    \
    printf("%-44s %7.1f cycles/op\n", label, (double) (t1 - t0) / (double) n); \
} while (0)

int main(int argc, char **argv)
{
    long n = argc > 1 ? atol(argv[1]) : 200000;
    static const uint8_t needle[16] = "needle", hay[16] = "hay in a needle";
    static const uint8_t lo_hi[16] = { 'a', 'z', '0', '9' };
    uint8_t mask[16], st[16], rk[16], out[16];
    uint32_t idx; uint64_t fl;
    volatile uint64_t sink = 0;
    memset(st, 0x5a, 16); memset(rk, 0xa5, 16);
    sse42emu_core_init();

    TIME("pcmpistri 0x0c EqualOrdered, scalar reference", pcmpstr_ref_istr(needle, hay, 0x0c, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistri 0x0c EqualOrdered, core",             sse42emu_pcmpistr(needle, hay, 0x0c, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistri 0x00 EqualAny, core",                 sse42emu_pcmpistr(needle, hay, 0x00, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistri 0x04 Ranges, core",                   sse42emu_pcmpistr(lo_hi, hay, 0x04, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistri 0x08 EqualEach, core",                sse42emu_pcmpistr(needle, hay, 0x08, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistri 0x0d EqualOrdered words, core",       sse42emu_pcmpistr(needle, hay, 0x0d, &idx, NULL, &fl); sink += idx);
    TIME("pcmpistrm 0x40 expanded mask, core",            sse42emu_pcmpistr(needle, hay, 0x40, NULL, mask, &fl); sink += mask[0]);
    TIME("pcmpestri 0x0c lengths 6/15, core",             sse42emu_pcmpestr(needle, hay, 0x0c, 6, 15, &idx, NULL, &fl); sink += idx);
    TIME("crc32 8 bytes",                                 sink = sse42emu_crc32((uint32_t) sink, 0xdeadbeefcafef00dULL, 8));
    TIME("pclmulqdq",                                     sse42emu_pclmul(out, st, rk, 0x00); sink += out[0]);
    TIME("aesenc",                                        memcpy(out, st, 16); sse42emu_aes(out, rk, SSE42EMU_AESENC); sink += out[0]);
    TIME("aesdec",                                        memcpy(out, st, 16); sse42emu_aes(out, rk, SSE42EMU_AESDEC); sink += out[0]);
    TIME("aeskeygenassist",                               sse42emu_aeskeygen(out, st, 0x1b); sink += out[0]);
    TIME("popcnt",                                        sink += sse42emu_popcnt(sink ^ 0xdeadbeefULL, 8, &fl));
    printf("(sink=%lu)\n", (unsigned long) sink);
    return 0;
}
