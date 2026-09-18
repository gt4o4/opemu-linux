/* SPDX-License-Identifier: MIT */
/* emu-core — see emu-core.h.  Validated on real SSE4.2/AES/PCLMULQDQ
 * silicon by tests/ (7.0M differential checks over all 256 PCMPxSTRx control
 * bytes, both length forms, four vector shapes), against the scalar core the
 * silicon certified first (tests/test-ref.c), and — that scalar core —
 * independently against GCC's PCMPxSTRx reference model
 * (gcc.target/i386/sse4_2-pcmpstr.h, via mirh/opemu-linux) over 512,000
 * cases with zero mismatches.  Change the arithmetic only under that harness.
 *
 * Reuse over rewriting: the AES round primitives and the carry-less
 * multiply are BearSSL's (Thomas Pornin, MIT — notice below), the AES
 * S-boxes and CRC-32C table come from the kernel's own lib/ when compiled
 * into it (emu-core.h), and only the PCMPxSTRx core is written here, on
 * SSE2, because nothing else implements those instructions.
 *
 * BearSSL portions: Copyright (c) 2016 Thomas Pornin <pornin@bolet.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "emu-core.h"

/* Internal spellings of the flag bits, kept short. */
#define FL_CF EMU_FL_CF
#define FL_PF EMU_FL_PF
#define FL_AF EMU_FL_AF
#define FL_ZF EMU_FL_ZF
#define FL_SF EMU_FL_SF
#define FL_OF EMU_FL_OF

/* ---- POPCNT -------------------------------------------------------- */
uint64_t sse42emu_popcnt(uint64_t src, unsigned bytes, uint64_t *flags)
{
    if (bytes < 8) src &= (1ULL << (8 * bytes)) - 1;
    /* ZF from the SOURCE; every other arithmetic flag cleared. */
    if (flags) *flags = src == 0 ? FL_ZF : 0;
    return emu_hweight64(src);
}

/* ---- CRC32 (Castagnoli, reflected) -------------------------------- */
/* The instruction is the raw byte-serial CRC-32C update, low byte first, no
 * inversion at either end.  In the kernel that is lib/crc's crc32c() (see
 * emu-core.h); here it is a 256-entry table built from the polynomial. */
#ifndef __KERNEL__
static uint32_t crc32c_tab[256];
static int      crc32c_ready;

static void crc32c_init(void)
{
    if (crc32c_ready) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (uint32_t) (-(int32_t) (c & 1u)));
        crc32c_tab[i] = c;
    }
    crc32c_ready = 1;
}

uint32_t emu_crc32c(uint32_t crc, const void *p, unsigned n)
{
    const uint8_t *b = p;
    crc32c_init();
    while (n--) crc = (crc >> 8) ^ crc32c_tab[(crc ^ *b++) & 0xffu];
    return crc;
}
#endif

uint32_t sse42emu_crc32(uint32_t crc, uint64_t data, unsigned bytes)
{
    /* the u64 is little-endian in memory: byte 0 first, as the instruction does */
    return emu_crc32c(crc, &data, bytes);
}

/* ---- PCMPISTRx / PCMPESTRx --------------------------------------------
 * imm8: [1:0] element format (ubyte, uword, sbyte, sword), [3:2] aggregation
 * (EqualAny, Ranges, EqualEach, EqualOrdered), [5:4] polarity, [6] output
 * selection; bit 7 is reserved and ignored, as the silicon ignores it.
 *
 * The comparisons run on the CPU's own SSE2 — Penryn has that — with the
 * needle element broadcast from a GPR (movzx, imul by 0x01010101, movd,
 * pshufd), one pcmpeq against the whole haystack register, and pmovmskb
 * giving the 16-bit row of "b[j] == a[i]" straight into a GPR.  Everything
 * after that is bit arithmetic on those rows: the aggregations, the
 * invalid-element override table (SDM Vol 2, "Summary of Imm8 Control
 * Byte"), polarity, index and mask.  Ranges compare with pcmpgt, which is
 * signed: for the unsigned formats both sides are XORed with 0x80 (0x8000)
 * first, which maps unsigned order onto signed order.
 *
 * Every asm block is self-contained (loads its operands from memory, hands
 * back GPRs or a small array) and uses only xmm12..xmm15, which the kernel
 * saves and restores around the call — see emu-core.h for that contract.
 * The scalar version this replaced lives on as tests/pcmpstr-ref.c, the
 * oracle test-ref compares this one with on any x86-64.
 *
 * Two things the shape of the arithmetic must get right, both from the
 * silicon (tests/vectors.h has the vectors): a needle that runs past the
 * END OF THE REGISTER counts as matched at that position (EqualOrdered of
 * "needle" in "0123456789abcnee" is 13, so a strstr loop re-checks the
 * next chunk), while a needle that runs past the end of the STRING does
 * not (16 for "abcnee\0…").  In the EqualOrdered fold below that is the
 * `| ~full` before the shift: ones come in from above bit 15, zeros from
 * the invalid haystack elements.
 */
#ifdef __SSE2__
/* The compiler may keep values in XMM registers across these blocks: tell
 * it which ones the asm trashes.  Under the kernel's -mno-sse GCC refuses
 * the names (it never allocates them), so the list is empty there. */
# define EMU_XMM_CLOBBERS "xmm12", "xmm13", "xmm14", "xmm15",
#else
# define EMU_XMM_CLOBBERS
#endif

/* bit j set where element j of v is zero */
static inline uint32_t emu_zero_mask(const uint8_t v[16], int words)
{
    uint32_t m;
    if (words)
        __asm__ volatile("movdqu (%1), %%xmm13\n\t"
                         "pxor %%xmm15, %%xmm15\n\t"
                         "pcmpeqw %%xmm15, %%xmm13\n\t"
                         "packsswb %%xmm13, %%xmm13\n\t"
                         "pmovmskb %%xmm13, %0"
                         : "=r"(m) : "r"(v), "m"(*(const uint8_t (*)[16]) v) : EMU_XMM_CLOBBERS "cc");
    else
        __asm__ volatile("movdqu (%1), %%xmm13\n\t"
                         "pxor %%xmm15, %%xmm15\n\t"
                         "pcmpeqb %%xmm15, %%xmm13\n\t"
                         "pmovmskb %%xmm13, %0"
                         : "=r"(m) : "r"(v), "m"(*(const uint8_t (*)[16]) v) : EMU_XMM_CLOBBERS "cc");
    return m & (words ? 0xffu : 0xffffu);
}

/* bit j set where a[j] == b[j] */
static inline uint32_t emu_eq_diag(const uint8_t a[16], const uint8_t b[16], int words)
{
    uint32_t m;
    if (words)
        __asm__ volatile("movdqu (%1), %%xmm14\n\t"
                         "movdqu (%2), %%xmm13\n\t"
                         "pcmpeqw %%xmm13, %%xmm14\n\t"
                         "packsswb %%xmm14, %%xmm14\n\t"
                         "pmovmskb %%xmm14, %0"
                         : "=r"(m) : "r"(a), "r"(b), "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
    else
        __asm__ volatile("movdqu (%1), %%xmm14\n\t"
                         "movdqu (%2), %%xmm13\n\t"
                         "pcmpeqb %%xmm13, %%xmm14\n\t"
                         "pmovmskb %%xmm14, %0"
                         : "=r"(m) : "r"(a), "r"(b), "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
    return m & (words ? 0xffu : 0xffffu);
}

/* Row i: bit j set where b[j] == a[i].  Bytes: 16 rows of 16 bits.  Words:
 * 8 rows; packsswb folds each 16-bit lane to a byte first, so pmovmskb
 * yields the 8-bit row twice (0xMMMM) — callers mask with `full`. */
#define EMU_ROW_B(off, eqoff)                                                 \
    "movzbl " #off "(%[a]), %k[t]\n\t"                                        \
    "imul $0x01010101, %k[t], %k[t]\n\t"                                      \
    "movd %k[t], %%xmm14\n\t"                                                 \
    "pshufd $0, %%xmm14, %%xmm14\n\t"                                         \
    "pcmpeqb %%xmm13, %%xmm14\n\t"                                            \
    "pmovmskb %%xmm14, %k[t]\n\t"                                             \
    "movw %w[t], " #eqoff "(%[eq])\n\t"
#define EMU_ROW_W(off)                                                        \
    "movzwl " #off "(%[a]), %k[t]\n\t"                                        \
    "imul $0x00010001, %k[t], %k[t]\n\t"                                      \
    "movd %k[t], %%xmm14\n\t"                                                 \
    "pshufd $0, %%xmm14, %%xmm14\n\t"                                         \
    "pcmpeqw %%xmm13, %%xmm14\n\t"                                            \
    "packsswb %%xmm14, %%xmm14\n\t"                                           \
    "pmovmskb %%xmm14, %k[t]\n\t"                                             \
    "movw %w[t], " #off "(%[eq])\n\t"

static inline void emu_eq_rows(const uint8_t a[16], const uint8_t b[16], int words, uint16_t eq[16])
{
    uint32_t t;
    if (words)
        __asm__ volatile("movdqu (%[b]), %%xmm13\n\t"
                         EMU_ROW_W(0) EMU_ROW_W(2) EMU_ROW_W(4) EMU_ROW_W(6)
                         EMU_ROW_W(8) EMU_ROW_W(10) EMU_ROW_W(12) EMU_ROW_W(14)
                         : [t] "=&r"(t), "=m"(*(uint16_t (*)[16]) eq)
                         : [a] "r"(a), [b] "r"(b), [eq] "r"(eq),
                           "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
    else
        __asm__ volatile("movdqu (%[b]), %%xmm13\n\t"
                         EMU_ROW_B(0, 0)   EMU_ROW_B(1, 2)   EMU_ROW_B(2, 4)   EMU_ROW_B(3, 6)
                         EMU_ROW_B(4, 8)   EMU_ROW_B(5, 10)  EMU_ROW_B(6, 12)  EMU_ROW_B(7, 14)
                         EMU_ROW_B(8, 16)  EMU_ROW_B(9, 18)  EMU_ROW_B(10, 20) EMU_ROW_B(11, 22)
                         EMU_ROW_B(12, 24) EMU_ROW_B(13, 26) EMU_ROW_B(14, 28) EMU_ROW_B(15, 30)
                         : [t] "=&r"(t), "=m"(*(uint16_t (*)[16]) eq)
                         : [a] "r"(a), [b] "r"(b), [eq] "r"(eq),
                           "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
}

/* Pair k = (a[2k], a[2k+1]) as an inclusive range: bit j of oor[k] set where
 * b[j] is OUTSIDE it, i.e. (lo > b[j]) | (b[j] > hi).  pcmpgt is signed;
 * `bias` (0x80 / 0x8000, or 0 for the signed formats) is XORed into every
 * element on both sides first.  xmm13 = biased b, xmm14 = lo, xmm15 = hi,
 * xmm12 = scratch. */
#define EMU_RANGE_B(lo, hi, out)                                              \
    "movzbl " #lo "(%[a]), %k[t]\n\t"  "xor %k[bias], %k[t]\n\t"              \
    "imul $0x01010101, %k[t], %k[t]\n\t" "movd %k[t], %%xmm14\n\t" "pshufd $0, %%xmm14, %%xmm14\n\t" \
    "movzbl " #hi "(%[a]), %k[t]\n\t"  "xor %k[bias], %k[t]\n\t"              \
    "imul $0x01010101, %k[t], %k[t]\n\t" "movd %k[t], %%xmm15\n\t" "pshufd $0, %%xmm15, %%xmm15\n\t" \
    "pcmpgtb %%xmm13, %%xmm14\n\t"          /* lo > b */                      \
    "movdqa %%xmm13, %%xmm12\n\t"                                             \
    "pcmpgtb %%xmm15, %%xmm12\n\t"          /* b > hi */                      \
    "por %%xmm12, %%xmm14\n\t"                                                \
    "pmovmskb %%xmm14, %k[t]\n\t"                                             \
    "movw %w[t], " #out "(%[oor])\n\t"
#define EMU_RANGE_W(lo, hi, out)                                              \
    "movzwl " #lo "(%[a]), %k[t]\n\t"  "xor %k[bias], %k[t]\n\t"              \
    "imul $0x00010001, %k[t], %k[t]\n\t" "movd %k[t], %%xmm14\n\t" "pshufd $0, %%xmm14, %%xmm14\n\t" \
    "movzwl " #hi "(%[a]), %k[t]\n\t"  "xor %k[bias], %k[t]\n\t"              \
    "imul $0x00010001, %k[t], %k[t]\n\t" "movd %k[t], %%xmm15\n\t" "pshufd $0, %%xmm15, %%xmm15\n\t" \
    "pcmpgtw %%xmm13, %%xmm14\n\t"                                            \
    "movdqa %%xmm13, %%xmm12\n\t"                                             \
    "pcmpgtw %%xmm15, %%xmm12\n\t"                                            \
    "por %%xmm12, %%xmm14\n\t"                                                \
    "packsswb %%xmm14, %%xmm14\n\t"                                           \
    "pmovmskb %%xmm14, %k[t]\n\t"                                             \
    "movw %w[t], " #out "(%[oor])\n\t"

static inline void emu_range_rows(const uint8_t a[16], const uint8_t b[16], int words, int signd, uint16_t oor[8])
{
    uint32_t t;
    const uint32_t bias   = signd ? 0 : (words ? 0x8000u : 0x80u);
    const uint32_t bias32 = signd ? 0 : (words ? 0x80008000u : 0x80808080u);
    if (words)
        __asm__ volatile("movdqu (%[b]), %%xmm13\n\t"
                         "movd %k[bias32], %%xmm12\n\t" "pshufd $0, %%xmm12, %%xmm12\n\t" "pxor %%xmm12, %%xmm13\n\t"
                         EMU_RANGE_W(0, 2, 0) EMU_RANGE_W(4, 6, 2) EMU_RANGE_W(8, 10, 4) EMU_RANGE_W(12, 14, 6)
                         : [t] "=&r"(t), "=m"(*(uint16_t (*)[8]) oor)
                         : [a] "r"(a), [b] "r"(b), [oor] "r"(oor), [bias] "r"(bias), [bias32] "r"(bias32),
                           "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
    else
        __asm__ volatile("movdqu (%[b]), %%xmm13\n\t"
                         "movd %k[bias32], %%xmm12\n\t" "pshufd $0, %%xmm12, %%xmm12\n\t" "pxor %%xmm12, %%xmm13\n\t"
                         EMU_RANGE_B(0, 1, 0)   EMU_RANGE_B(2, 3, 2)   EMU_RANGE_B(4, 5, 4)   EMU_RANGE_B(6, 7, 6)
                         EMU_RANGE_B(8, 9, 8)   EMU_RANGE_B(10, 11, 10) EMU_RANGE_B(12, 13, 12) EMU_RANGE_B(14, 15, 14)
                         : [t] "=&r"(t), "=m"(*(uint16_t (*)[8]) oor)
                         : [a] "r"(a), [b] "r"(b), [oor] "r"(oor), [bias] "r"(bias), [bias32] "r"(bias32),
                           "m"(*(const uint8_t (*)[16]) a), "m"(*(const uint8_t (*)[16]) b)
                         : EMU_XMM_CLOBBERS "cc");
}

static void pcmpstr_core(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                         int explicit_len, int32_t exp_la, int32_t exp_lb,
                         uint32_t *out_index, uint8_t out_mask[16],
                         uint64_t *out_flags)
{
    const int words  = (imm & 1) != 0;          /* 0: bytes, 1: words */
    const int signd  = (imm & 2) != 0;
    const int agg    = (imm >> 2) & 3;
    const int pol    = (imm >> 4) & 3;
    const int outsel = (imm >> 6) & 1;
    const int n      = words ? 8 : 16;
    const uint32_t full = words ? 0xffu : 0xffffu;

    int la, lb;
    if (explicit_len) {
        /* PCMPESTRx: lengths come from EAX/RAX and EDX/RDX, SIGNED, and
         * are saturated by ABSOLUTE value — a negative length means the
         * same as its magnitude.  Done in int64 so INT32_MIN cannot
         * negate into itself. */
        int64_t ta = exp_la < 0 ? -(int64_t) exp_la : (int64_t) exp_la;
        int64_t tb = exp_lb < 0 ? -(int64_t) exp_lb : (int64_t) exp_lb;
        la = ta > n ? n : (int) ta;
        lb = tb > n ? n : (int) tb;
    } else {
        /* PCMPISTRx: implicit — elements before the first zero element */
        la = __builtin_ctz(emu_zero_mask(a, words) | (1u << n));
        lb = __builtin_ctz(emu_zero_mask(b, words) | (1u << n));
    }
    const uint32_t valida = (1u << la) - 1u, validb = (1u << lb) - 1u;

    /* The override table, per aggregation: a comparison with an INVALID
     * element does not use the compared value.  EqualAny/Ranges: false.
     * EqualEach: true when both are invalid, false when one is.
     * EqualOrdered: true when the NEEDLE element is invalid (the needle
     * has ended: a match), false when only the haystack one is. */
    uint32_t res1 = 0;
    switch (agg) {
    case 0: {                                   /* EqualAny: is b[j] any of a[0..la) */
        uint16_t eq[16];
        emu_eq_rows(a, b, words, eq);
        for (int i = 0; i < la; i++) res1 |= eq[i];
        res1 &= validb;
        break;
    }
    case 1: {                                   /* Ranges: pairs (a[2k], a[2k+1]) are inclusive bounds */
        uint16_t oor[8];
        emu_range_rows(a, b, words, signd, oor);
        for (int k = 0; 2 * k + 1 < la; k++) res1 |= (uint32_t) ~oor[k];
        res1 &= validb & full;
        break;
    }
    case 2:                                     /* EqualEach: elementwise */
        res1 = (emu_eq_diag(a, b, words) & valida & validb) | (~(valida | validb) & full);
        break;
    case 3: {                                   /* EqualOrdered: substring search of a within b */
        uint16_t eq[16];
        emu_eq_rows(a, b, words, eq);
        res1 = full;
        for (int k = 0; k < la; k++)
            res1 &= ((eq[k] & validb) | ~full) >> k;   /* ones shift in past the register end */
        res1 &= full;
        break;
    }
    }

    uint32_t res2 = res1;
    if (pol == 1) res2 = (~res1) & full;
    else if (pol == 3) res2 = (res1 ^ validb) & full;

    if (out_index) {
        uint32_t idx = (uint32_t) n;           /* no bit set => n */
        if (res2 & full)
            idx = outsel ? 31u - (uint32_t) __builtin_clz(res2) : (uint32_t) __builtin_ctz(res2);
        *out_index = idx;
    }
    if (out_mask) {
        memset(out_mask, 0, 16);
        if (outsel) {                           /* expanded byte/word mask */
            for (int k = 0; k < n; k++) {
                uint8_t v = (uint8_t) -(int8_t) ((res2 >> k) & 1u);
                if (words) { out_mask[2 * k] = v; out_mask[2 * k + 1] = v; }
                else       { out_mask[k] = v; }
            }
        } else {                                /* zero-extended bit mask */
            out_mask[0] = (uint8_t) (res2 & 0xffu);
            if (n == 16) out_mask[1] = (uint8_t) ((res2 >> 8) & 0xffu);
        }
    }
    if (out_flags) {
        uint64_t f = 0;
        if (res2 & full) f |= FL_CF;
        if (lb < n)      f |= FL_ZF;
        if (la < n)      f |= FL_SF;
        if (res2 & 1u)   f |= FL_OF;
        *out_flags = f;                          /* AF, PF cleared */
    }
}

void sse42emu_pcmpistr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                       uint32_t *out_index, uint8_t out_mask[16],
                       uint64_t *out_flags)
{
    pcmpstr_core(a, b, imm, 0, 0, 0, out_index, out_mask, out_flags);
}

void sse42emu_pcmpestr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                       int32_t la, int32_t lb,
                       uint32_t *out_index, uint8_t out_mask[16],
                       uint64_t *out_flags)
{
    pcmpstr_core(a, b, imm, 1, la, lb, out_index, out_mask, out_flags);
}

/* ---- PCMPGTQ ------------------------------------------------------- */
void sse42emu_pcmpgtq(uint8_t dst[16], const uint8_t src[16])
{
    for (int lane = 0; lane < 2; lane++) {
        int64_t x, y;
        memcpy(&x, dst + 8 * lane, 8);
        memcpy(&y, src + 8 * lane, 8);
        uint64_t m = (x > y) ? ~0ULL : 0ULL;
        memcpy(dst + 8 * lane, &m, 8);
    }
}

/* ---- AES-NI -------------------------------------------------------
 * Penryn has neither AES-NI nor PCLMULQDQ, and agy (antigravity-cli,
 * a Go binary) carries 878 AESENC, 326 AESDEC and 156 PCLMULQDQ — all
 * behind CPUID dispatch, so they never execute there; the instructions
 * belong here anyway, for any binary that reaches them without a gate.
 *
 * The round primitives are BearSSL's aes_small (src/symcipher/aes_small_enc.c
 * and aes_small_dec.c), verbatim but for the names: a 16-entry state in the
 * column-major byte order the instructions use (byte i is row i%4, column
 * i/4), SubBytes/ShiftRows/MixColumns and their inverses.  The S-boxes are
 * the kernel's own exported crypto_aes_sbox/crypto_aes_inv_sbox when this
 * file is compiled into it (CONFIG_CRYPTO_LIB_AES, selected by
 * X86_UD_EMULATE) and BearSSL's tables in userspace; the two are the same
 * 512 bytes by definition, and the differential test checks every byte of
 * every round against AES-NI silicon. */
#ifdef __KERNEL__
# include <crypto/aes.h>
# define aes_sbox     crypto_aes_sbox
# define aes_inv_sbox crypto_aes_inv_sbox
#else
static const uint8_t aes_sbox[256] = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B,
    0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
    0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26,
    0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
    0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
    0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED,
    0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F,
    0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
    0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC,
    0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14,
    0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
    0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
    0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F,
    0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
    0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8, 0x98, 0x11,
    0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F,
    0xB0, 0x54, 0xBB, 0x16
};
static const uint8_t aes_inv_sbox[256] = {
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E,
    0x81, 0xF3, 0xD7, 0xFB, 0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87,
    0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB, 0x54, 0x7B, 0x94, 0x32,
    0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49,
    0x6D, 0x8B, 0xD1, 0x25, 0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92, 0x6C, 0x70, 0x48, 0x50,
    0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05,
    0xB8, 0xB3, 0x45, 0x06, 0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02,
    0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B, 0x3A, 0x91, 0x11, 0x41,
    0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8,
    0x1C, 0x75, 0xDF, 0x6E, 0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89,
    0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B, 0xFC, 0x56, 0x3E, 0x4B,
    0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59,
    0x27, 0x80, 0xEC, 0x5F, 0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D,
    0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF, 0xA0, 0xE0, 0x3B, 0x4D,
    0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63,
    0x55, 0x21, 0x0C, 0x7D
};
#endif

static void aes_sub_bytes(unsigned *state)
{
    for (int i = 0; i < 16; i++) state[i] = aes_sbox[state[i]];
}

static void aes_inv_sub_bytes(unsigned *state)
{
    for (int i = 0; i < 16; i++) state[i] = aes_inv_sbox[state[i]];
}

static void aes_shift_rows(unsigned *state)
{
    unsigned tmp;
    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;
    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;
    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;
    tmp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = tmp;
}

static void aes_inv_shift_rows(unsigned *state)
{
    unsigned tmp;
    tmp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = tmp;
    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;
    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;
    tmp = state[3];
    state[3] = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = tmp;
}

static void aes_mix_columns(unsigned *state)
{
    for (int i = 0; i < 16; i += 4) {
        unsigned s0, s1, s2, s3;
        unsigned t0, t1, t2, t3;
        s0 = state[i + 0];
        s1 = state[i + 1];
        s2 = state[i + 2];
        s3 = state[i + 3];
        t0 = (s0 << 1) ^ s1 ^ (s1 << 1) ^ s2 ^ s3;
        t1 = s0 ^ (s1 << 1) ^ s2 ^ (s2 << 1) ^ s3;
        t2 = s0 ^ s1 ^ (s2 << 1) ^ s3 ^ (s3 << 1);
        t3 = s0 ^ (s0 << 1) ^ s1 ^ s2 ^ (s3 << 1);
        state[i + 0] = t0 ^ ((unsigned) (-(int) (t0 >> 8)) & 0x11B);
        state[i + 1] = t1 ^ ((unsigned) (-(int) (t1 >> 8)) & 0x11B);
        state[i + 2] = t2 ^ ((unsigned) (-(int) (t2 >> 8)) & 0x11B);
        state[i + 3] = t3 ^ ((unsigned) (-(int) (t3 >> 8)) & 0x11B);
    }
}

static inline unsigned gf256red(unsigned x)
{
    unsigned y = x >> 8;
    return (x ^ y ^ (y << 1) ^ (y << 3) ^ (y << 4)) & 0xFF;
}

static void aes_inv_mix_columns(unsigned *state)
{
    for (int i = 0; i < 16; i += 4) {
        unsigned s0, s1, s2, s3;
        unsigned t0, t1, t2, t3;
        s0 = state[i + 0];
        s1 = state[i + 1];
        s2 = state[i + 2];
        s3 = state[i + 3];
        t0 = (s0 << 1) ^ (s0 << 2) ^ (s0 << 3)
            ^ s1 ^ (s1 << 1) ^ (s1 << 3)
            ^ s2 ^ (s2 << 2) ^ (s2 << 3)
            ^ s3 ^ (s3 << 3);
        t1 = s0 ^ (s0 << 3)
            ^ (s1 << 1) ^ (s1 << 2) ^ (s1 << 3)
            ^ s2 ^ (s2 << 1) ^ (s2 << 3)
            ^ s3 ^ (s3 << 2) ^ (s3 << 3);
        t2 = s0 ^ (s0 << 2) ^ (s0 << 3)
            ^ s1 ^ (s1 << 3)
            ^ (s2 << 1) ^ (s2 << 2) ^ (s2 << 3)
            ^ s3 ^ (s3 << 1) ^ (s3 << 3);
        t3 = s0 ^ (s0 << 1) ^ (s0 << 3)
            ^ s1 ^ (s1 << 2) ^ (s1 << 3)
            ^ s2 ^ (s2 << 3)
            ^ (s3 << 1) ^ (s3 << 2) ^ (s3 << 3);
        state[i + 0] = gf256red(t0);
        state[i + 1] = gf256red(t1);
        state[i + 2] = gf256red(t2);
        state[i + 3] = gf256red(t3);
    }
}

void sse42emu_core_init(void)
{
#ifndef __KERNEL__
    crc32c_init();
#endif
}

/* AESENC = ShiftRows, SubBytes, MixColumns, then XOR the round key (the
 * source operand); the LAST forms skip MixColumns; the DEC forms use the
 * inverses; AESIMC is InvMixColumns of the source alone. */
void sse42emu_aes(uint8_t dst[16], const uint8_t src[16], int op)
{
    unsigned st[16];
    int i;

    for (i = 0; i < 16; i++) st[i] = (op == SSE42EMU_AESIMC ? src : dst)[i];
    switch (op) {
    case SSE42EMU_AESENC:     aes_shift_rows(st); aes_sub_bytes(st); aes_mix_columns(st); break;
    case SSE42EMU_AESENCLAST: aes_shift_rows(st); aes_sub_bytes(st); break;
    case SSE42EMU_AESDEC:     aes_inv_shift_rows(st); aes_inv_sub_bytes(st); aes_inv_mix_columns(st); break;
    case SSE42EMU_AESDECLAST: aes_inv_shift_rows(st); aes_inv_sub_bytes(st); break;
    default:                  aes_inv_mix_columns(st);
                              for (i = 0; i < 16; i++) dst[i] = (uint8_t) st[i];
                              return;
    }
    for (i = 0; i < 16; i++) dst[i] = (uint8_t) (st[i] ^ src[i]);
}

void sse42emu_aeskeygen(uint8_t dst[16], const uint8_t src[16], uint8_t rcon)
{
    uint32_t x1, x3;
    memcpy(&x1, src + 4, 4);
    memcpy(&x3, src + 12, 4);
    uint32_t s1 = 0, s3 = 0;
    for (int i = 0; i < 4; i++) {
        s1 |= (uint32_t) aes_sbox[(x1 >> (8 * i)) & 0xff] << (8 * i);
        s3 |= (uint32_t) aes_sbox[(x3 >> (8 * i)) & 0xff] << (8 * i);
    }
    uint32_t r1 = (s1 >> 8) | (s1 << 24);      /* RotWord */
    uint32_t r3 = (s3 >> 8) | (s3 << 24);
    uint32_t d0 = s1, d1 = r1 ^ rcon, d2 = s3, d3 = r3 ^ rcon;
    memcpy(dst + 0, &d0, 4); memcpy(dst + 4, &d1, 4);
    memcpy(dst + 8, &d2, 4); memcpy(dst + 12, &d3, 4);
}

/* ---- PCLMULQDQ: carry-less product of two selected qwords ----------
 * bmul64() and rev64() are BearSSL's (src/hash/ghash_ctmul64.c): the
 * operands are split into four interleaved bit-lanes so that plain integer
 * multiplies never carry across lanes, sixteen products are XORed together
 * lane by lane, and the result is exact for the low 64 bits of the carry-less
 * product.  The high half is the same product on the bit-reversed operands,
 * reversed back and shifted by one (bit 127 of a 64x64 product is always 0).
 * Constant time, ~32 multiplies, no loop over 64 bits. */
static inline uint64_t bmul64(uint64_t x, uint64_t y)
{
    uint64_t x0, x1, x2, x3;
    uint64_t y0, y1, y2, y3;
    uint64_t z0, z1, z2, z3;
    x0 = x & (uint64_t) 0x1111111111111111;
    x1 = x & (uint64_t) 0x2222222222222222;
    x2 = x & (uint64_t) 0x4444444444444444;
    x3 = x & (uint64_t) 0x8888888888888888;
    y0 = y & (uint64_t) 0x1111111111111111;
    y1 = y & (uint64_t) 0x2222222222222222;
    y2 = y & (uint64_t) 0x4444444444444444;
    y3 = y & (uint64_t) 0x8888888888888888;
    z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
    z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
    z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
    z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
    z0 &= (uint64_t) 0x1111111111111111;
    z1 &= (uint64_t) 0x2222222222222222;
    z2 &= (uint64_t) 0x4444444444444444;
    z3 &= (uint64_t) 0x8888888888888888;
    return z0 | z1 | z2 | z3;
}

static uint64_t rev64(uint64_t x)
{
#define RMS(m, s)   do { \
        x = ((x & (uint64_t) (m)) << (s)) \
            | ((x >> (s)) & (uint64_t) (m)); \
    } while (0)
    RMS(0x5555555555555555,  1);
    RMS(0x3333333333333333,  2);
    RMS(0x0F0F0F0F0F0F0F0F,  4);
    RMS(0x00FF00FF00FF00FF,  8);
    RMS(0x0000FFFF0000FFFF, 16);
    return (x << 32) | (x >> 32);
#undef RMS
}

void sse42emu_pclmul(uint8_t dst[16], const uint8_t a[16], const uint8_t b[16], uint8_t imm)
{
    uint64_t x, y, lo, hi;
    memcpy(&x, a + ((imm & 0x01) ? 8 : 0), 8);
    memcpy(&y, b + ((imm & 0x10) ? 8 : 0), 8);
    lo = bmul64(x, y);
    hi = rev64(bmul64(rev64(x), rev64(y))) >> 1;
    memcpy(dst, &lo, 8);
    memcpy(dst + 8, &hi, 8);
}
