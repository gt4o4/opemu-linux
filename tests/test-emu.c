/* SPDX-License-Identifier: MIT */
/* Differential test: emulation vs the real silicon.
 *
 * The build machine (lix-ory) HAS SSE4.2, which is precisely why it can
 * never exercise this library's fault path — and precisely what makes it
 * the right place to check the arithmetic.  Every vector is computed
 * twice, once by the real instruction and once by the emulator, and the
 * results and flags must agree bit for bit.
 *
 * On a CPU without SSE4.2 the differential half is skipped and only the
 * fixed vectors run, so the test is still meaningful (and still exits 0)
 * wherever it lands.
 *
 * Covered here: POPCNT, CRC32, PCMPISTRI, PCMPISTRM.  PCMPISTRM had no
 * differential cover until 2026-09-17 even though it executes in the
 * real workload (twice per `claude --version`) — the index form was
 * swept exhaustively and the mask form was taken on trust.  It isn't
 * now.  AES, PCLMULQDQ, PCMPGTQ and PCMPESTRx are in test-vec.c.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "../preload/sse42emu.h"
#include "imm-tables.h"

static int failures, checks;
static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int have_sse42(void)
{
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1), "c"(0));
    return (c >> 20) & 1;
}

#define FLMASK 0x08D5u   /* CF PF AF ZF SF OF */

/* One function per imm8: the immediate must be a literal, and the table
 * index IS the imm8 value because both come from one expansion. */
typedef void (*istri_fn)(const uint8_t *, const uint8_t *, uint32_t *, uint64_t *);
typedef void (*istrm_fn)(const uint8_t *, const uint8_t *, uint8_t *, uint64_t *);

#define X(IMM)                                                                \
static void istri_##IMM(const uint8_t *a, const uint8_t *b,                   \
                        uint32_t *idx, uint64_t *fl) {                        \
    uint64_t f; uint32_t i;                                                   \
    __asm__ volatile("movdqu (%2), %%xmm1\n\t"                                \
                     "movdqu (%3), %%xmm2\n\t"                                \
                     "pcmpistri $" #IMM ", %%xmm2, %%xmm1\n\t"                \
                     "mov %%ecx, %0\n\t"                                      \
                     "pushfq\n\tpopq %1"                                      \
                     : "=&r"(i), "=&r"(f) : "r"(a), "r"(b)                    \
                     : "xmm1", "xmm2", "rcx", "cc", "memory");                \
    *idx = i; *fl = f; }
IMM_EXPAND_128
#undef X
static istri_fn istri_tab[128] = {
#define X(IMM) istri_##IMM,
IMM_EXPAND_128
#undef X
};

#define X(IMM)                                                                \
static void istrm_##IMM(const uint8_t *a, const uint8_t *b,                   \
                        uint8_t *mask, uint64_t *fl) {                        \
    uint64_t f;                                                               \
    __asm__ volatile("movdqu (%1), %%xmm1\n\t"                                \
                     "movdqu (%2), %%xmm2\n\t"                                \
                     "pcmpistrm $" #IMM ", %%xmm2, %%xmm1\n\t"                \
                     "movdqu %%xmm0, (%3)\n\t"                                \
                     "pushfq\n\tpopq %0"                                      \
                     : "=&r"(f) : "r"(a), "r"(b), "r"(mask)                   \
                     : "xmm0", "xmm1", "xmm2", "cc", "memory");               \
    *fl = f; }
IMM_EXPAND_128
#undef X
static istrm_fn istrm_tab[128] = {
#define X(IMM) istrm_##IMM,
IMM_EXPAND_128
#undef X
};

static void dump(const char *tag, const uint8_t *v)
{
    printf("    %s:", tag);
    for (int i = 0; i < 16; i++) printf(" %02x", v[i]);
    printf("\n");
}

static void check_popcnt(void)
{
    for (int t = 0; t < 4000; t++) {
        uint64_t v = rnd();
        if (t < 4) v = (uint64_t) t;          /* include 0 for the ZF case */
        sse42emu_regs R; memset(&R, 0, sizeof R);
        uint64_t want = 0;
        for (int i = 0; i < 64; i++) if (v & (1ULL << i)) want++;
        R.gpr[REG_RAX] = v;
        /* POPCNT rcx, rax  =  F3 48 0F B8 C8 */
        static const uint8_t code[] = { 0xF3, 0x48, 0x0F, 0xB8, 0xC8 };
        R.rip = (uint64_t) (uintptr_t) code;
        checks++;
        if (sse42emu_step(&R, code, sizeof code) != SSE42EMU_OK) { failures++; continue; }
        if (R.gpr[REG_RCX] != want) {
            printf("  popcnt(%016lx) = %lu, want %lu\n", v, R.gpr[REG_RCX], want); failures++;
        }
        int zf = (R.rflags & 0x40) != 0;
        if (zf != (v == 0)) { printf("  popcnt ZF wrong for %016lx\n", v); failures++; }
        if (R.rip != (uint64_t) (uintptr_t) code + sizeof code) {
            printf("  popcnt did not advance rip by 5\n"); failures++;
        }
    }
}

static void check_crc32(void)
{
    /* CRC-32C of "123456789" is the standard check value. */
    uint32_t c = 0xffffffffu;
    const char *s = "123456789";
    for (int i = 0; i < 9; i++) c = sse42emu_crc32(c, (uint8_t) s[i], 1);
    checks++;
    if ((c ^ 0xffffffffu) != 0xE3069283u) {
        printf("  crc32c(\"123456789\") = %08x, want E3069283\n", c ^ 0xffffffffu); failures++;
    }
}

static void check_istr_differential(void)
{
    if (!have_sse42()) { printf("  (no SSE4.2 here: differential half skipped)\n"); return; }
    for (int t = 0; t < 20000; t++) {
        uint8_t a[16], b[16];
        for (int i = 0; i < 16; i++) {
            /* Bias hard toward short strings and shared bytes: the
             * invalid-element override table only shows up there. */
            a[i] = (uint8_t) (rnd() % ((t % 3) ? 4 : 256));
            b[i] = (uint8_t) (rnd() % ((t % 3) ? 4 : 256));
        }
        for (int imm = 0; imm < 128; imm++) {
            uint32_t ridx = 0, eidx = 0; uint64_t rfl = 0, efl = 0;
            istri_tab[imm](a, b, &ridx, &rfl);
            sse42emu_pcmpistr(a, b, (uint8_t) imm, &eidx, NULL, &efl);
            checks++;
            if (ridx != eidx || (rfl & FLMASK) != (efl & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpistri imm=0x%02x: hw idx=%u fl=%04lx  emu idx=%u fl=%04lx\n",
                           imm, ridx, rfl & FLMASK, eidx, efl & FLMASK);
                    dump("a", a); dump("b", b);
                }
                failures++;
            }

            /* Same vectors through the MASK form.  Only every 4th
             * iteration: it is the same core, and the mask expansion is
             * what is under test, not the aggregation again. */
            if ((t & 3) == 0) {
                uint8_t rm[16], em[16]; uint64_t rf2 = 0, ef2 = 0;
                istrm_tab[imm](a, b, rm, &rf2);
                sse42emu_pcmpistr(a, b, (uint8_t) imm, NULL, em, &ef2);
                checks++;
                if (memcmp(rm, em, 16) || (rf2 & FLMASK) != (ef2 & FLMASK)) {
                    if (failures < 10) {
                        printf("  pcmpistrm imm=0x%02x: fl hw=%04lx emu=%04lx\n",
                               imm, rf2 & FLMASK, ef2 & FLMASK);
                        dump("a ", a); dump("b ", b); dump("hw", rm); dump("em", em);
                    }
                    failures++;
                }
            }
        }
    }
}

int main(void)
{
    printf("sse42emu differential test (host %s SSE4.2)\n",
           have_sse42() ? "HAS" : "lacks");
    check_popcnt();
    check_crc32();
    check_istr_differential();
    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
