/* SPDX-License-Identifier: MIT */
/* See pcmpstr-ref.h.  This is core/emu-core.c's pcmpstr_core of commit
 * 376aafe, verbatim but for the name — do not "improve" it: its value is
 * that it is the version the silicon agreed with. */
#include "pcmpstr-ref.h"
#include "../core/emu-core.h"

#define FL_CF EMU_FL_CF
#define FL_ZF EMU_FL_ZF
#define FL_SF EMU_FL_SF
#define FL_OF EMU_FL_OF

static void pcmpstr_ref(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
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

    int32_t av[16], bv[16];
    for (int i = 0; i < n; i++) {
        if (words) {
            uint16_t ua = (uint16_t) (a[2 * i] | (a[2 * i + 1] << 8));
            uint16_t ub = (uint16_t) (b[2 * i] | (b[2 * i + 1] << 8));
            av[i] = signd ? (int32_t) (int16_t) ua : (int32_t) ua;
            bv[i] = signd ? (int32_t) (int16_t) ub : (int32_t) ub;
        } else {
            av[i] = signd ? (int32_t) (int8_t) a[i] : (int32_t) a[i];
            bv[i] = signd ? (int32_t) (int8_t) b[i] : (int32_t) b[i];
        }
    }

    int la = n, lb = n;
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
        for (int i = 0; i < n; i++) if (av[i] == 0) { la = i; break; }
        for (int i = 0; i < n; i++) if (bv[i] == 0) { lb = i; break; }
    }

    /* bres[i][j]: comparison of a[i] against b[j], with overrides */
    int bres[16][16];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int ai_valid = i < la, bj_valid = j < lb;
            if (ai_valid && bj_valid) {
                bres[i][j] = (av[i] == bv[j]);
            } else if (!ai_valid && !bj_valid) {
                bres[i][j] = (agg == 2 || agg == 3); /* EqualEach/Ordered: true */
            } else if (!ai_valid && bj_valid) {
                bres[i][j] = (agg == 3);             /* EqualOrdered: true */
            } else {
                bres[i][j] = 0;                      /* valid a, invalid b */
            }
        }
    }

    uint32_t res1 = 0;
    switch (agg) {
    case 0: /* EqualAny: is b[j] any of a[0..la) */
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int i = 0; i < n; i++)
                if (i < la && j < lb && av[i] == bv[j]) acc = 1;
            if (acc) res1 |= 1u << j;
        }
        break;
    case 1: /* Ranges: pairs (a[2k], a[2k+1]) are inclusive bounds */
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int i = 0; i + 1 < n; i += 2)
                if (i + 1 < la && j < lb && bv[j] >= av[i] && bv[j] <= av[i + 1]) acc = 1;
            if (acc) res1 |= 1u << j;
        }
        break;
    case 2: /* EqualEach: elementwise */
        for (int j = 0; j < n; j++)
            if (bres[j][j]) res1 |= 1u << j;
        break;
    case 3: /* EqualOrdered: substring search of a within b */
        for (int j = 0; j < n; j++) {
            int acc = 1;
            for (int k = 0; k + j < n && k < n; k++) {
                if (k >= la) break;          /* needle exhausted: match */
                if (!bres[k][j + k]) { acc = 0; break; }
            }
            if (acc) res1 |= 1u << j;
        }
        break;
    }

    const uint32_t full = (n == 16) ? 0xffffu : 0xffu;
    uint32_t res2 = res1;
    if (pol == 1) res2 = (~res1) & full;
    else if (pol == 3) {
        uint32_t validb = (lb >= n) ? full : ((1u << lb) - 1u);
        res2 = (res1 ^ validb) & full;
    }

    if (out_index) {
        uint32_t idx = (uint32_t) n;           /* no bit set => n */
        if (res2 & full) {
            if (outsel) { for (int k = n - 1; k >= 0; k--) if (res2 & (1u << k)) { idx = (uint32_t) k; break; } }
            else        { for (int k = 0; k < n; k++)      if (res2 & (1u << k)) { idx = (uint32_t) k; break; } }
        }
        *out_index = idx;
    }
    if (out_mask) {
        memset(out_mask, 0, 16);
        if (outsel) {                           /* expanded byte/word mask */
            for (int k = 0; k < n; k++) {
                uint8_t v = (res2 & (1u << k)) ? 0xffu : 0x00u;
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

void pcmpstr_ref_istr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                      uint32_t *out_index, uint8_t out_mask[16], uint64_t *out_flags)
{
    pcmpstr_ref(a, b, imm, 0, 0, 0, out_index, out_mask, out_flags);
}

void pcmpstr_ref_estr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                      int32_t la, int32_t lb,
                      uint32_t *out_index, uint8_t out_mask[16], uint64_t *out_flags)
{
    pcmpstr_ref(a, b, imm, 1, la, lb, out_index, out_mask, out_flags);
}
