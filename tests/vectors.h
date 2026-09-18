/* SPDX-License-Identifier: MIT */
/* Operand vectors for the PCMPxSTRx sweeps, shared by test-emu (against
 * silicon) and test-ref (against the scalar reference), so both tests
 * mean the same thing by "a vector".
 *
 * Uniformly random bytes almost never produce the cases the instruction's
 * corner rules exist for — the invalid-element override table needs short
 * strings, the signed formats need bytes and words on either side of
 * 0x80/0x8000 — so vectors come in four shapes, and a few fixed pairs go
 * first: each is a case with a known trap and a known right answer.
 */
#ifndef TEST_VECTORS_H
#define TEST_VECTORS_H
#include <stdint.h>
#include <string.h>

static uint64_t vec_rng = 0x243F6A8885A308D3ULL;
static inline uint64_t vec_rnd(void)
{ vec_rng ^= vec_rng << 13; vec_rng ^= vec_rng >> 7; vec_rng ^= vec_rng << 17; return vec_rng; }

/*   0  uniformly random bytes
 *   1  the alphabet {0,1,2,3}: a zero every fourth element, so short
 *      strings, and many equal elements, so many matches
 *   2  word-shaped: a low byte from a set straddling the signed-byte
 *      boundary, a high byte that is 0 often enough to end a word string
 *      early or 0x80/0x7f/0xff to straddle the signed-word boundary
 *   3  the signed-byte boundary set {0, 1, 0x7f, 0x80, 0x81, 0xff}      */
static inline void vec_shape(uint8_t v[16], int t)
{
    static const uint8_t lo[] = { 0x00, 0x01, 0x02, 0x7f, 0x80, 0xff };
    static const uint8_t hi[] = { 0x00, 0x00, 0xff, 0x80, 0x7f };
    static const uint8_t sb[] = { 0x00, 0x01, 0x7f, 0x80, 0x81, 0xff };
    switch (t & 3) {
    case 0:  for (int i = 0; i < 16; i++) v[i] = (uint8_t) vec_rnd(); break;
    case 1:  for (int i = 0; i < 16; i++) v[i] = (uint8_t) (vec_rnd() & 3); break;
    case 2:  for (int i = 0; i < 16; i += 2) { v[i] = lo[vec_rnd() % 6]; v[i + 1] = hi[vec_rnd() % 5]; } break;
    default: for (int i = 0; i < 16; i++) v[i] = sb[vec_rnd() % 6]; break;
    }
}

/* Fixed pairs.  A 16-character literal fills the array with no NUL (legal
 * C, and `nonstring` tells GCC 15 it is meant): that is the "no zero
 * anywhere" case, la = lb = 16. */
static const struct {
    const char *why;
    uint8_t a[16] __attribute__((nonstring)), b[16] __attribute__((nonstring));
} vec_edges[] = {
    { "empty needle, every aggregation",       { 0 },                 "abcdefghijklmno"  },
    { "no zero anywhere: la = lb = 16",        "abcdefghijklmnop",    "ponmlkjihgfedcba" },
    { "zero only at element 15",               "abcdefghijklmno",     "abcdefghijklmnp"  },
    { "needle overruns the REGISTER end: 13",  "needle",              "0123456789abcnee" },
    { "needle overruns the STRING end: 16",    "needle",              "abcnee"           },
    { "empty haystack",                        "needle",              { 0 }              },
    { "both empty",                            { 0 },                 { 0 }              },
    { "identical strings",                     "abcdefgh",            "abcdefgh"         },
    { "ranges: pairs straddling 0x80",         { 0x7e, 0x82, 0x00 },  { 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x00 } },
    { "words: zero word at 1, signed straddle",{ 0x41, 0x80, 0x00, 0x00, 0x41, 0x80 }, { 0x41, 0x80, 0x41, 0x7f, 0x00, 0x00 } },
};
#define VEC_N_EDGE ((int) (sizeof vec_edges / sizeof *vec_edges))

/* Vector t of a run: the fixed pairs first, then shaped random ones. */
static inline void vec_pair(int t, uint8_t a[16], uint8_t b[16])
{
    if (t < VEC_N_EDGE) { memcpy(a, vec_edges[t].a, 16); memcpy(b, vec_edges[t].b, 16); return; }
    vec_shape(a, t); vec_shape(b, t);
}

/* Explicit lengths worth sweeping: inside the vector, at it, past it, zero,
 * negative (the SDM defines that as its magnitude), and the two int32 ends
 * (INT32_MIN cannot be negated in int32 — the core does it in int64). */
static const int32_t vec_lens[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 40, -1, -5, -16, -17,
                                    0x7fffffff, (int32_t) 0x80000000 };
#define VEC_N_LENS ((int) (sizeof vec_lens / sizeof *vec_lens))
static inline int32_t vec_len(void) { return vec_lens[vec_rnd() % (uint64_t) VEC_N_LENS]; }

#endif
