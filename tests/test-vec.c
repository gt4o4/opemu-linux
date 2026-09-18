/* SPDX-License-Identifier: MIT */
/* Differential test #2: the instructions added on 2026-09-17, plus the
 * two that were implemented earlier and never checked against silicon.
 *
 * AES-NI, PCLMULQDQ and PCMPESTRx are new.  PCMPGTQ is not — it has been
 * in the emulator since the first commit and executes 8 times in a real
 * `claude --version`, yet nothing ever compared it to the hardware.  It
 * is checked here first, because an untested path that RUNS is worse
 * than an untested path that does not.
 *
 * Two layers, on purpose:
 *   1. the arithmetic, called directly (sse42emu_aes, _pclmul, _pcmpestr)
 *   2. the same instruction through sse42emu_step() on encoded bytes,
 *      which is the only thing that exercises the Zydis operand plumbing
 *      the handler actually depends on
 *
 * Layer 2 is where a correct implementation still ships broken: reading
 * the wrong operand, writing the wrong XMM, or forgetting xmm_written
 * are all invisible to layer 1.
 *
 * The hardware half needs AES-NI/PCLMULQDQ/SSE4.2 — the build machine
 * has them and the target host is the one that does not.  They are
 * MANDATORY: without them this exits 2 and verifies nothing (until
 * 2026-09-18 the halves were skipped and the test still passed), and the
 * number of checks is asserted against the loop bounds so no path can
 * quietly skip vectors.  PCMPESTRx sweeps all 256 control bytes (imm8[7]
 * is reserved and ignored by the silicon — imm-tables.h), the 32-bit and
 * the REX.W form, over the vector shapes of vectors.h.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include "../preload/sse42emu.h"
#include "imm-tables.h"
#include "vectors.h"

static int failures, checks;
static uint64_t rng = 0x13198A2E03707344ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static void fill(uint8_t v[16]) { for (int i = 0; i < 16; i++) v[i] = (uint8_t) rnd(); }

#define FLMASK 0x08D5u   /* CF PF AF ZF SF OF */

static void cpuid1(uint32_t *ecx) {
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1), "c"(0));
    *ecx = c;
}
static int have_sse42(void) { uint32_t c; cpuid1(&c); return (c >> 20) & 1; }
static int have_aes(void)   { uint32_t c; cpuid1(&c); return (c >> 25) & 1; }
static int have_pclmul(void){ uint32_t c; cpuid1(&c); return (c >>  1) & 1; }

static void dump(const char *tag, const uint8_t *v)
{
    printf("    %s:", tag);
    for (int i = 0; i < 16; i++) printf(" %02x", v[i]);
    printf("\n");
}
static int cmp16(const char *what, const uint8_t *hw, const uint8_t *em,
                 const uint8_t *a, const uint8_t *b)
{
    checks++;
    if (!memcmp(hw, em, 16)) return 0;
    if (failures < 10) {
        printf("  %s mismatch\n", what);
        dump("a ", a); dump("b ", b); dump("hw", hw); dump("em", em);
    }
    failures++;
    return 1;
}

/* ---- layer 1: the arithmetic against the real instruction ---------- */

#define AESOP(name, insn)                                                     \
static void hw_##name(uint8_t d[16], const uint8_t s[16]) {                   \
    __asm__ volatile("movdqu (%0), %%xmm1\n\t"                                \
                     "movdqu (%1), %%xmm2\n\t"                                \
                     insn " %%xmm2, %%xmm1\n\t"                               \
                     "movdqu %%xmm1, (%0)"                                    \
                     : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }
AESOP(aesenc,     "aesenc")
AESOP(aesenclast, "aesenclast")
AESOP(aesdec,     "aesdec")
AESOP(aesdeclast, "aesdeclast")
#undef AESOP

static void hw_aesimc(uint8_t d[16], const uint8_t s[16]) {
    __asm__ volatile("movdqu (%1), %%xmm2\n\t"
                     "aesimc %%xmm2, %%xmm1\n\t"
                     "movdqu %%xmm1, (%0)"
                     : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }

/* RCON is a literal; the standard AES key schedule only ever uses these
 * eleven, plus 0 for the 256-bit schedule's even rounds. */
#define RCONS X(0x00) X(0x01) X(0x02) X(0x04) X(0x08) X(0x10) X(0x20) \
              X(0x40) X(0x80) X(0x1b) X(0x36) X(0x8d) X(0xff)
#define X(R)                                                                  \
static void hw_keygen_##R(uint8_t d[16], const uint8_t s[16]) {               \
    __asm__ volatile("movdqu (%1), %%xmm2\n\t"                                \
                     "aeskeygenassist $" #R ", %%xmm2, %%xmm1\n\t"            \
                     "movdqu %%xmm1, (%0)"                                    \
                     : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }
RCONS
#undef X

static void check_aes(void)
{
    static const struct { const char *n; void (*hw)(uint8_t *, const uint8_t *); int op; }
    ops[] = {
        { "aesenc",     hw_aesenc,     SSE42EMU_AESENC     },
        { "aesenclast", hw_aesenclast, SSE42EMU_AESENCLAST },
        { "aesdec",     hw_aesdec,     SSE42EMU_AESDEC     },
        { "aesdeclast", hw_aesdeclast, SSE42EMU_AESDECLAST },
        { "aesimc",     hw_aesimc,     SSE42EMU_AESIMC     },
    };
    for (int t = 0; t < 5000; t++) {
        uint8_t st[16], rk[16];
        fill(st); fill(rk);
        if (t == 0) { memset(st, 0, 16); memset(rk, 0, 16); }
        if (t == 1) { memset(st, 0xff, 16); memset(rk, 0, 16); }
        for (unsigned k = 0; k < sizeof ops / sizeof *ops; k++) {
            uint8_t h[16], e[16];
            memcpy(h, st, 16); memcpy(e, st, 16);
            ops[k].hw(h, rk);
            sse42emu_aes(e, rk, ops[k].op);
            cmp16(ops[k].n, h, e, st, rk);
        }
    }
    /* AESKEYGENASSIST over every RCON the key schedule uses. */
    static const struct { uint8_t r; void (*hw)(uint8_t *, const uint8_t *); } kg[] = {
#define X(R) { (uint8_t) R, hw_keygen_##R },
        RCONS
#undef X
    };
    for (int t = 0; t < 2000; t++) {
        uint8_t src[16];
        fill(src);
        for (unsigned k = 0; k < sizeof kg / sizeof *kg; k++) {
            uint8_t h[16], e[16];
            kg[k].hw(h, src);
            sse42emu_aeskeygen(e, src, kg[k].r);
            cmp16("aeskeygenassist", h, e, src, src);
        }
    }
}

#define PCLMUL_IMMS X(0x00) X(0x01) X(0x10) X(0x11)
#define X(I)                                                                  \
static void hw_pclmul_##I(uint8_t d[16], const uint8_t a[16], const uint8_t b[16]) { \
    __asm__ volatile("movdqu (%1), %%xmm1\n\t"                                \
                     "movdqu (%2), %%xmm2\n\t"                                \
                     "pclmulqdq $" #I ", %%xmm2, %%xmm1\n\t"                  \
                     "movdqu %%xmm1, (%0)"                                    \
                     : : "r"(d), "r"(a), "r"(b) : "xmm1", "xmm2", "memory"); }
PCLMUL_IMMS
#undef X

static void check_pclmul(void)
{
    static const struct { uint8_t i; void (*hw)(uint8_t *, const uint8_t *, const uint8_t *); } v[] = {
#define X(I) { (uint8_t) I, hw_pclmul_##I },
        PCLMUL_IMMS
#undef X
    };
    for (int t = 0; t < 5000; t++) {
        uint8_t a[16], b[16];
        fill(a); fill(b);
        /* The all-ones qword is the worst case for carry-less carry
         * propagation and the one a naive shift-and-xor gets wrong. */
        if (t == 0) { memset(a, 0xff, 16); memset(b, 0xff, 16); }
        for (unsigned k = 0; k < sizeof v / sizeof *v; k++) {
            uint8_t h[16], e[16];
            v[k].hw(h, a, b);
            sse42emu_pclmul(e, a, b, v[k].i);
            cmp16("pclmulqdq", h, e, a, b);
        }
    }
}

static void hw_pcmpgtq(uint8_t d[16], const uint8_t a[16], const uint8_t b[16]) {
    __asm__ volatile("movdqu (%1), %%xmm1\n\t"
                     "movdqu (%2), %%xmm2\n\t"
                     "pcmpgtq %%xmm2, %%xmm1\n\t"
                     "movdqu %%xmm1, (%0)"
                     : : "r"(d), "r"(a), "r"(b) : "xmm1", "xmm2", "memory"); }

/* PCMPGTQ goes through the full decode path: there is no standalone
 * arithmetic entry point for it, which is the point — this is the only
 * check it has ever had. */
static void check_pcmpgtq(void)
{
    /* pcmpgtq %xmm2, %xmm1  =  66 0F 38 37 CA */
    static const uint8_t code[] = { 0x66, 0x0F, 0x38, 0x37, 0xCA };
    for (int t = 0; t < 20000; t++) {
        uint8_t a[16], b[16], h[16];
        fill(a); fill(b);
        /* Signed comparison: force the boundary cases a random draw of
         * 64-bit values essentially never produces. */
        if (t % 5 == 0) { memcpy(b, a, 16); }
        if (t % 7 == 0) { memset(a, 0, 16); memset(b, 0xff, 16); }
        if (t % 11 == 0) { memset(a, 0x80, 8); memset(a + 8, 0x7f, 8);
                           memset(b, 0x7f, 8); memset(b + 8, 0x80, 8); }
        memcpy(h, a, 16);
        hw_pcmpgtq(h, a, b);

        sse42emu_regs R; memset(&R, 0, sizeof R);
        memcpy(R.xmm[1], a, 16);
        memcpy(R.xmm[2], b, 16);
        R.rip = (uint64_t) (uintptr_t) code;
        if (sse42emu_step(&R, code, sizeof code) != SSE42EMU_OK) {
            printf("  pcmpgtq: step refused the instruction\n"); failures++; return;
        }
        if (R.xmm_written != 1) {
            printf("  pcmpgtq: xmm_written = %d, want 1\n", R.xmm_written); failures++; return;
        }
        cmp16("pcmpgtq", h, R.xmm[1], a, b);
    }
}

/* ---- PCMPESTRI / PCMPESTRM ---------------------------------------- */
/* 32-bit form only: GAS emits PCMPESTRI without REX.W, which is also
 * what compilers generate for _mm_cmpestri.  That is the form whose
 * sign-extension the emulator has to get right, and negative lengths
 * below exercise exactly that. */
typedef void (*estri_fn)(const uint8_t *, const uint8_t *, int, int, uint32_t *, uint64_t *);
typedef void (*estrm_fn)(const uint8_t *, const uint8_t *, int, int, uint8_t *, uint64_t *);

#define X(IMM)                                                                \
static void estri_##IMM(const uint8_t *a, const uint8_t *b, int la, int lb,   \
                        uint32_t *idx, uint64_t *fl) {                        \
    uint64_t f; uint32_t i;                                                   \
    __asm__ volatile("movdqu (%4), %%xmm1\n\t"                                \
                     "movdqu (%5), %%xmm2\n\t"                                \
                     "pcmpestri $" #IMM ", %%xmm2, %%xmm1\n\t"                \
                     "mov %%ecx, %0\n\t"                                      \
                     "pushfq\n\tpopq %1"                                      \
                     : "=&r"(i), "=&r"(f)                                     \
                     : "a"(la), "d"(lb), "r"(a), "r"(b)                       \
                     : "xmm1", "xmm2", "rcx", "cc", "memory");                \
    *idx = i; *fl = f; }
IMM_EXPAND_256
#undef X
static estri_fn estri_tab[256] = {
#define X(IMM) estri_##IMM,
IMM_EXPAND_256
#undef X
};

#define X(IMM)                                                                \
static void estrm_##IMM(const uint8_t *a, const uint8_t *b, int la, int lb,   \
                        uint8_t *mask, uint64_t *fl) {                        \
    uint64_t f;                                                               \
    __asm__ volatile("movdqu (%3), %%xmm1\n\t"                                \
                     "movdqu (%4), %%xmm2\n\t"                                \
                     "pcmpestrm $" #IMM ", %%xmm2, %%xmm1\n\t"                \
                     "movdqu %%xmm0, (%5)\n\t"                                \
                     "pushfq\n\tpopq %0"                                      \
                     : "=&r"(f)                                               \
                     : "a"(la), "d"(lb), "r"(a), "r"(b), "r"(mask)            \
                     : "xmm0", "xmm1", "xmm2", "cc", "memory");               \
    *fl = f; }
IMM_EXPAND_256
#undef X
static estrm_fn estrm_tab[256] = {
#define X(IMM) estrm_##IMM,
IMM_EXPAND_256
#undef X
};

#define N_ESTR 600                      /* shaped vectors, after the fixed edges */
static void check_pcmpestr(void)
{
    for (int t = 0; t < VEC_N_EDGE + N_ESTR; t++) {
        uint8_t a[16], b[16];
        vec_pair(t, a, b);
        const int la = vec_len(), lb = vec_len();
        for (int imm = 0; imm < 256; imm++) {
            uint32_t ridx = 0, eidx = 0; uint64_t rfl = 0, efl = 0;
            estri_tab[imm](a, b, la, lb, &ridx, &rfl);
            sse42emu_pcmpestr(a, b, (uint8_t) imm, la, lb, &eidx, NULL, &efl);
            checks++;
            if (ridx != eidx || (rfl & FLMASK) != (efl & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpestri imm=0x%02x la=%d lb=%d: hw idx=%u fl=%04lx  emu idx=%u fl=%04lx\n",
                           imm, la, lb, ridx, rfl & FLMASK, eidx, efl & FLMASK);
                    dump("a", a); dump("b", b);
                }
                failures++;
            }
            uint8_t rm[16], em[16]; uint64_t rf2 = 0, ef2 = 0;
            estrm_tab[imm](a, b, la, lb, rm, &rf2);
            sse42emu_pcmpestr(a, b, (uint8_t) imm, la, lb, NULL, em, &ef2);
            checks++;
            if (memcmp(rm, em, 16) || (rf2 & FLMASK) != (ef2 & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpestrm imm=0x%02x la=%d lb=%d: fl hw=%04lx emu=%04lx\n",
                           imm, la, lb, rf2 & FLMASK, ef2 & FLMASK);
                    dump("a ", a); dump("b ", b); dump("hw", rm); dump("em", em);
                }
                failures++;
            }
        }
    }
}

/* ---- PCMPESTRI / PCMPESTRM with REX.W: lengths from RAX/RDX ------- */
/* GAS has no spelling for this form, so the bytes are given: 66 48 0F 3A
 * 61|60 CA imm = pcmpestri|pcmpestrm $imm, %xmm2, %xmm1 with REX.W.  The
 * emulator side runs the SAME bytes through sse42emu_step, because the
 * width rule lives in the decoder plumbing, not in the int32 core. */
typedef void (*estri64_fn)(const uint8_t *, const uint8_t *, int64_t, int64_t, uint32_t *, uint64_t *);
typedef void (*estrm64_fn)(const uint8_t *, const uint8_t *, int64_t, int64_t, uint8_t *, uint64_t *);

#define X(IMM)                                                                \
static void estri64_##IMM(const uint8_t *a, const uint8_t *b, int64_t la, int64_t lb, \
                          uint32_t *idx, uint64_t *fl) {                      \
    uint64_t f; uint32_t i;                                                   \
    __asm__ volatile("movdqu (%4), %%xmm1\n\t"                                \
                     "movdqu (%5), %%xmm2\n\t"                                \
                     ".byte 0x66,0x48,0x0f,0x3a,0x61,0xca\n\t.byte " #IMM "\n\t" \
                     "mov %%ecx, %0\n\t"                                      \
                     "pushfq\n\tpopq %1"                                      \
                     : "=&r"(i), "=&r"(f)                                     \
                     : "a"(la), "d"(lb), "r"(a), "r"(b)                       \
                     : "xmm1", "xmm2", "rcx", "cc", "memory");                \
    *idx = i; *fl = f; }
IMM_EXPAND_256
#undef X
static estri64_fn estri64_tab[256] = {
#define X(IMM) estri64_##IMM,
IMM_EXPAND_256
#undef X
};

#define X(IMM)                                                                \
static void estrm64_##IMM(const uint8_t *a, const uint8_t *b, int64_t la, int64_t lb, \
                          uint8_t *mask, uint64_t *fl) {                      \
    uint64_t f;                                                               \
    __asm__ volatile("movdqu (%3), %%xmm1\n\t"                                \
                     "movdqu (%4), %%xmm2\n\t"                                \
                     ".byte 0x66,0x48,0x0f,0x3a,0x60,0xca\n\t.byte " #IMM "\n\t" \
                     "movdqu %%xmm0, (%5)\n\t"                                \
                     "pushfq\n\tpopq %0"                                      \
                     : "=&r"(f)                                               \
                     : "a"(la), "d"(lb), "r"(a), "r"(b), "r"(mask)            \
                     : "xmm0", "xmm1", "xmm2", "cc", "memory");               \
    *fl = f; }
IMM_EXPAND_256
#undef X
static estrm64_fn estrm64_tab[256] = {
#define X(IMM) estrm64_##IMM,
IMM_EXPAND_256
#undef X
};

/* Lengths whose upper 32 bits matter: the .w form sees them, the 32-bit
 * form must not. */
static const int64_t lens64[] = { 5, -5, 0x100000003LL, -0x100000003LL, 0xffffffffLL, 0x100000000LL,
                                  INT64_MIN, INT64_MAX, 0, 16, -16, 17 };
#define N_LENS64 ((int) (sizeof lens64 / sizeof *lens64))

/* Run the REX.W encoding through the emulator's decoder. */
static int emu_estr64(const uint8_t a[16], const uint8_t b[16], int64_t la, int64_t lb, int imm, int mform,
                      uint32_t *idx, uint8_t mask[16], uint64_t *fl)
{
    uint8_t code[7] = { 0x66, 0x48, 0x0f, 0x3a, (uint8_t) (mform ? 0x60 : 0x61), 0xca, (uint8_t) imm };
    sse42emu_regs R; memset(&R, 0, sizeof R);
    memcpy(R.xmm[1], a, 16); memcpy(R.xmm[2], b, 16);
    R.gpr[REG_RAX] = (uint64_t) la; R.gpr[REG_RDX] = (uint64_t) lb;
    R.rip = (uint64_t) (uintptr_t) code;
    if (sse42emu_step(&R, code, sizeof code) != SSE42EMU_OK || R.rip != (uint64_t) (uintptr_t) code + 7)
        return 0;
    if (mform && R.xmm_written != 0) return 0;
    *idx = (uint32_t) R.gpr[REG_RCX]; memcpy(mask, R.xmm[0], 16); *fl = R.rflags;
    return 1;
}

#define N_ESTR64 200
static void check_pcmpestr_rexw(void)
{
    for (int t = 0; t < N_ESTR64; t++) {
        uint8_t a[16], b[16];
        vec_shape(a, t); vec_shape(b, t);
        const int64_t la = lens64[rnd() % N_LENS64], lb = lens64[rnd() % N_LENS64];
        for (int imm = 0; imm < 256; imm++) {
            uint32_t ridx = 0, eidx = 0; uint64_t rfl = 0, efl = 0; uint8_t rm[16], em[16];
            estri64_tab[imm](a, b, la, lb, &ridx, &rfl);
            checks++;
            if (!emu_estr64(a, b, la, lb, imm, 0, &eidx, em, &efl) || ridx != eidx || (rfl & FLMASK) != (efl & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpestri.w imm=0x%02x la=%ld lb=%ld: hw idx=%u fl=%04lx  emu idx=%u fl=%04lx\n",
                           imm, la, lb, ridx, rfl & FLMASK, eidx, efl & FLMASK);
                    dump("a", a); dump("b", b);
                }
                failures++;
            }
            estrm64_tab[imm](a, b, la, lb, rm, &rfl);
            checks++;
            if (!emu_estr64(a, b, la, lb, imm, 1, &eidx, em, &efl) || memcmp(rm, em, 16) || (rfl & FLMASK) != (efl & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpestrm.w imm=0x%02x la=%ld lb=%ld: fl hw=%04lx emu=%04lx\n",
                           imm, la, lb, rfl & FLMASK, efl & FLMASK);
                    dump("a ", a); dump("b ", b); dump("hw", rm); dump("em", em);
                }
                failures++;
            }
        }
    }

    /* The .w and the 32-bit form must DIFFER on a length whose upper bits
     * are set, or this whole function proves nothing: "cd" in "abcdef",
     * la = 0x100000002 — 2 for the 32-bit form (a match at 2), saturated
     * to 16 for .w (the needle becomes "cd" + 14 NULs, which is nowhere). */
    {
        uint8_t a[16] = "cd", b[16] = "abcdef";
        uint32_t i32, i64, e32, e64; uint64_t f, ef; uint8_t m[16];
        estri_tab[0x0c](a, b, (int) 0x100000002LL, 6, &i32, &f);
        estri64_tab[0x0c](a, b, 0x100000002LL, 6, &i64, &f);
        checks++;
        if (!(i32 == 2 && i64 == 16)) { printf("  REX.W discriminator: hw 32-bit=%u .w=%u, want 2/16\n", i32, i64); failures++; }
        e32 = 0; e64 = 0;
        { uint8_t code[7] = { 0x66, 0x0f, 0x3a, 0x61, 0xca, 0x0c, 0xc3 };   /* the 32-bit form, 6 bytes */
          sse42emu_regs R; memset(&R, 0, sizeof R);
          memcpy(R.xmm[1], a, 16); memcpy(R.xmm[2], b, 16);
          R.gpr[REG_RAX] = 0x100000002ULL; R.gpr[REG_RDX] = 6; R.rip = (uint64_t) (uintptr_t) code;
          if (sse42emu_step(&R, code, 6) == SSE42EMU_OK) e32 = (uint32_t) R.gpr[REG_RCX]; }
        emu_estr64(a, b, 0x100000002LL, 6, 0x0c, 0, &e64, m, &ef);
        checks++;
        if (!(e32 == 2 && e64 == 16)) { printf("  REX.W discriminator: emu 32-bit=%u .w=%u, want 2/16\n", e32, e64); failures++; }
    }
}

/* ---- layer 2: the encoded instruction through sse42emu_step -------- */
/* Every byte string below is `<insn> %xmm2, %xmm1`: ModRM CA = reg 001
 * (xmm1, the destination) and rm 010 (xmm2, the source).  The HARDWARE
 * side runs those very same bytes out of an executable page, so a
 * mis-encoding cannot pass as agreement — both sides would just be
 * doing the same wrong thing, and the decoded-length check below would
 * still catch it.
 *
 * The buffers are file-scope because the asm addresses them RIP-relative:
 * feeding seven pointers through "r" constraints runs the allocator out
 * of registers once rax/rcx/rdx are also clobbered. */
/* NOT static: the compiler cannot see the references inside the asm
 * string, so a local symbol would be dropped and the link would fail on
 * the asm's own relocation. */
uint8_t  g_a[16], g_b[16], g_x1[16], g_x0[16];
uint64_t g_rcx, g_fl;
uint64_t g_rax, g_rdx;
void   (*g_fn)(void);

static void hw_run(void)
{
    __asm__ volatile("movdqu g_a(%%rip), %%xmm1\n\t"
                     "movdqu g_b(%%rip), %%xmm2\n\t"
                     "mov g_rax(%%rip), %%rax\n\t"
                     "mov g_rdx(%%rip), %%rdx\n\t"
                     "call *g_fn(%%rip)\n\t"
                     "pushfq\n\tpopq g_fl(%%rip)\n\t"
                     "mov %%rcx, g_rcx(%%rip)\n\t"
                     "movdqu %%xmm1, g_x1(%%rip)\n\t"
                     "movdqu %%xmm0, g_x0(%%rip)"
                     : : : "rax", "rcx", "rdx", "xmm0", "xmm1", "xmm2", "cc", "memory");
}

struct site {
    const char *name;
    uint8_t     code[8];
    unsigned    len;
    int         need_aes, need_pclmul, need_sse42;
    int         writes_xmm;      /* which XMM the op must report, -1 for none */
    int         writes_ecx;      /* result index lands in ECX */
    int         sets_flags;
};

static const struct site sites[] = {
    { "aesenc",          { 0x66,0x0F,0x38,0xDC,0xCA },      5, 1,0,0,  1, 0, 0 },
    { "aesenclast",      { 0x66,0x0F,0x38,0xDD,0xCA },      5, 1,0,0,  1, 0, 0 },
    { "aesdec",          { 0x66,0x0F,0x38,0xDE,0xCA },      5, 1,0,0,  1, 0, 0 },
    { "aesdeclast",      { 0x66,0x0F,0x38,0xDF,0xCA },      5, 1,0,0,  1, 0, 0 },
    { "aesimc",          { 0x66,0x0F,0x38,0xDB,0xCA },      5, 1,0,0,  1, 0, 0 },
    { "aeskeygenassist", { 0x66,0x0F,0x3A,0xDF,0xCA,0x1b }, 6, 1,0,0,  1, 0, 0 },
    { "pclmulqdq",       { 0x66,0x0F,0x3A,0x44,0xCA,0x11 }, 6, 0,1,0,  1, 0, 0 },
    { "pcmpgtq",         { 0x66,0x0F,0x38,0x37,0xCA },      5, 0,0,1,  1, 0, 0 },
    { "pcmpistrm",       { 0x66,0x0F,0x3A,0x62,0xCA,0x40 }, 6, 0,0,1,  0, 0, 1 },
    { "pcmpistri",       { 0x66,0x0F,0x3A,0x63,0xCA,0x00 }, 6, 0,0,1, -1, 1, 1 },
    { "pcmpestrm",       { 0x66,0x0F,0x3A,0x60,0xCA,0x40 }, 6, 0,0,1,  0, 0, 1 },
    { "pcmpestri",       { 0x66,0x0F,0x3A,0x61,0xCA,0x00 }, 6, 0,0,1, -1, 1, 1 },
    /* REX.W: lengths from RAX/RDX (the bytes GAS cannot spell), and the
     * reserved imm8[7] set on the index form */
    { "pcmpestri.w",     { 0x66,0x48,0x0F,0x3A,0x61,0xCA,0x0C }, 7, 0,0,1, -1, 1, 1 },
    { "pcmpestrm.w",     { 0x66,0x48,0x0F,0x3A,0x60,0xCA,0x40 }, 7, 0,0,1,  0, 0, 1 },
    { "pcmpistri $0x8c", { 0x66,0x0F,0x3A,0x63,0xCA,0x8C },      6, 0,0,1, -1, 1, 1 },
};
static int dispatch_checks_per_iter(void)
{
    int n = 0;
    for (unsigned k = 0; k < sizeof sites / sizeof *sites; k++)
        n += 1 + (sites[k].writes_xmm >= 0) + sites[k].writes_ecx + sites[k].sets_flags;
    return n;
}

#define N_DISPATCH 3000
static void check_dispatch(void)
{
    uint8_t *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) { printf("  (no RWX page: dispatch test skipped)\n"); return; }

    for (int t = 0; t < N_DISPATCH; t++) {
        fill(g_a); fill(g_b);
        if (t % 3) {                            /* short strings for the STR ops */
            for (int i = 0; i < 16; i++) { g_a[i] &= 3; g_b[i] &= 3; }
        }
        /* Explicit lengths, including the negative form the 32-bit
         * encoding must sign-extend rather than read as 4 billion, and —
         * every 4th round — upper halves set, which only the .w sites may
         * see (RAX carries them to both; the 32-bit sites must ignore them). */
        int64_t la = (t % 5 == 0) ? -5 : (int64_t) (t % 19) - 2;
        int64_t lb = (t % 7 == 0) ? -1 : (int64_t) (t % 23) - 3;
        if ((t & 3) == 0) { la = (int64_t) (uint32_t) la | (1LL << 32); lb = (int64_t) (uint32_t) lb | (3LL << 32); }
        g_rax = (uint64_t) la;
        g_rdx = (uint64_t) lb;

        for (unsigned k = 0; k < sizeof sites / sizeof *sites; k++) {
            const struct site *S = &sites[k];
            uint8_t in_a[16], in_b[16];
            memcpy(in_a, g_a, 16); memcpy(in_b, g_b, 16);

            memcpy(page, S->code, S->len);
            page[S->len] = 0xC3;                 /* ret */
            g_fn = (void (*)(void)) page;
            hw_run();

            sse42emu_regs R; memset(&R, 0, sizeof R);
            memcpy(R.xmm[1], in_a, 16);
            memcpy(R.xmm[2], in_b, 16);
            R.gpr[REG_RAX] = (uint64_t) la;
            R.gpr[REG_RDX] = (uint64_t) lb;
            R.rip = (uint64_t) (uintptr_t) S->code;
            checks++;
            if (sse42emu_step(&R, S->code, S->len) != SSE42EMU_OK) {
                printf("  %s: sse42emu_step refused the encoding\n", S->name);
                failures++; continue;
            }
            if (R.rip != (uint64_t) (uintptr_t) S->code + S->len) {
                printf("  %s: rip advanced by %lu, want %u\n", S->name,
                       (unsigned long) (R.rip - (uint64_t) (uintptr_t) S->code), S->len);
                failures++;
            }
            if (R.xmm_written != S->writes_xmm) {
                printf("  %s: xmm_written = %d, want %d\n", S->name,
                       R.xmm_written, S->writes_xmm);
                failures++;
            }
            if (S->writes_xmm == 1) cmp16(S->name, g_x1, R.xmm[1], in_a, in_b);
            if (S->writes_xmm == 0) cmp16(S->name, g_x0, R.xmm[0], in_a, in_b);
            if (S->writes_ecx) {
                checks++;
                if ((uint32_t) g_rcx != (uint32_t) R.gpr[REG_RCX]) {
                    if (failures < 10)
                        printf("  %s: ecx hw=%u emu=%u (la=%ld lb=%ld)\n", S->name,
                               (uint32_t) g_rcx, (uint32_t) R.gpr[REG_RCX], la, lb);
                    failures++;
                }
            }
            if (S->sets_flags) {
                checks++;
                if ((g_fl & FLMASK) != (R.rflags & FLMASK)) {
                    if (failures < 10)
                        printf("  %s: flags hw=%04lx emu=%04lx\n", S->name,
                               g_fl & FLMASK, R.rflags & FLMASK);
                    failures++;
                }
            }
        }
    }
}

int main(void)
{
    if (!have_sse42() || !have_aes() || !have_pclmul()) {
        printf("test-vec: this machine lacks SSE4.2/AES-NI/PCLMULQDQ (%d/%d/%d) — the differential test cannot\n"
               "          run here.  Build where the silicon is (lix-ory); nothing was verified.\n",
               have_sse42(), have_aes(), have_pclmul());
        return 2;
    }
    const int planned = 20000                               /* pcmpgtq */
                      + 5000 * 5 + 2000 * 13                /* aes ops, aeskeygenassist */
                      + 5000 * 4                            /* pclmulqdq */
                      + (VEC_N_EDGE + N_ESTR) * 256 * 2     /* pcmpestri/m, 32-bit form */
                      + N_ESTR64 * 256 * 2 + 2              /* REX.W form + the discriminator */
                      + N_DISPATCH * dispatch_checks_per_iter();
    sse42emu_prepare();
    printf("sse42emu vector test (host has sse4.2, aes-ni, pclmulqdq)\n");
    check_pcmpgtq();
    check_aes();
    check_pclmul();
    check_pcmpestr();
    check_pcmpestr_rexw();
    check_dispatch();
    printf("%d checks, %d failures\n", checks, failures);
    if (checks != planned) { printf("  ran %d checks, planned %d — a path skipped vectors\n", checks, planned); return 1; }
    return failures ? 1 : 0;
}
