/* SPDX-License-Identifier: MIT */
/* Differential test: emulation vs the real silicon.
 *
 * The build machine (lix-ory) HAS SSE4.2, which is precisely why it can
 * never exercise this library's fault path — and precisely what makes it
 * the right place to check the arithmetic.  Every vector is computed
 * twice, once by the real instruction and once by the emulator, and the
 * results and flags must agree bit for bit.
 *
 * The silicon is MANDATORY.  Without SSE4.2 this test exits 2 and runs
 * nothing; until 2026-09-18 it skipped the differential half and still
 * exited 0, so a build scheduled on the wrong machine could certify nothing
 * and pass.  For the same reason the number of checks is asserted against
 * the loop bounds: a path that quietly skips vectors cannot pass either.
 *
 * Covered here: POPCNT, CRC32, PCMPISTRI and PCMPISTRM over ALL 256 control
 * bytes (imm8[7] is reserved; the silicon ignores it and the sweep proves
 * the emulator does too — see imm-tables.h) and the vector shapes of
 * vectors.h.  AES, PCLMULQDQ, PCMPGTQ and PCMPESTRx are in test-vec.c.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "../preload/sse42emu.h"
#include "imm-tables.h"
#include "vectors.h"

#define N_POPCNT 4000
#define N_ISTR   20000                  /* shaped vectors, after the fixed edges */

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
IMM_EXPAND_256
#undef X
static istri_fn istri_tab[256] = {
#define X(IMM) istri_##IMM,
IMM_EXPAND_256
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
IMM_EXPAND_256
#undef X
static istrm_fn istrm_tab[256] = {
#define X(IMM) istrm_##IMM,
IMM_EXPAND_256
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
    for (int t = 0; t < N_POPCNT; t++) {
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

/* The mask form runs on every fixed edge pair and on every 4th shaped
 * vector: it is the same core, and the mask expansion is what is under
 * test there, not the aggregation again. */
static int istrm_here(int t) { return t < VEC_N_EDGE || (t & 3) == 0; }

static void check_istr_differential(void)
{
    for (int t = 0; t < VEC_N_EDGE + N_ISTR; t++) {
        uint8_t a[16], b[16];
        vec_pair(t, a, b);
        for (int imm = 0; imm < 256; imm++) {
            uint32_t ridx = 0, eidx = 0; uint64_t rfl = 0, efl = 0;
            istri_tab[imm](a, b, &ridx, &rfl);
            sse42emu_pcmpistr(a, b, (uint8_t) imm, &eidx, NULL, &efl);
            checks++;
            if (ridx != eidx || (rfl & FLMASK) != (efl & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpistri imm=0x%02x%s%s: hw idx=%u fl=%04lx  emu idx=%u fl=%04lx\n",
                           imm, t < VEC_N_EDGE ? " " : "", t < VEC_N_EDGE ? vec_edges[t].why : "",
                           ridx, rfl & FLMASK, eidx, efl & FLMASK);
                    dump("a", a); dump("b", b);
                }
                failures++;
            }
            if (istrm_here(t)) {
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
    if (!have_sse42()) {
        printf("test-emu: this machine has no SSE4.2 — the differential test cannot run here.\n"
               "          Build where the silicon is (lix-ory); nothing was verified.\n");
        return 2;
    }
    int planned = N_POPCNT + 1;
    for (int t = 0; t < VEC_N_EDGE + N_ISTR; t++) planned += 256 * (1 + istrm_here(t));

    printf("sse42emu differential test (host HAS SSE4.2)\n");
    check_popcnt();
    check_crc32();
    check_istr_differential();
    printf("%d checks, %d failures\n", checks, failures);
    if (checks != planned) { printf("  ran %d checks, planned %d — a path skipped vectors\n", checks, planned); return 1; }
    return failures ? 1 : 0;
}
