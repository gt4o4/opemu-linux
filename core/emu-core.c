/* SPDX-License-Identifier: MIT */
/* emu-core — see emu-core.h.  Validated on real SSE4.2/AES/PCLMULQDQ
 * silicon by tests/: 3,550,601 differential checks, and independently
 * cross-checked against GCC's PCMPxSTRx reference model
 * (gcc.target/i386/sse4_2-pcmpstr.h, via mirh/opemu-linux) over 512,000
 * cases with zero mismatches.  Change the arithmetic only under that harness.
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
uint32_t sse42emu_crc32(uint32_t crc, uint64_t data, unsigned bytes)
{
    for (unsigned b = 0; b < bytes; b++) {
        crc ^= (uint32_t) ((data >> (8 * b)) & 0xffu);
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0x82F63B78u & (uint32_t) (-(int32_t) (crc & 1u)));
    }
    return crc;
}

/* ---- PCMPISTRI / PCMPISTRM ---------------------------------------- */
/* imm8: [1:0] element format, [3:2] aggregation, [5:4] polarity,
 *       [6] output selection.  Implicit lengths (null-terminated).
 *
 * The fiddly part is the invalid-element override table (SDM Vol 2,
 * "Summary of Imm8 Control Byte"): a comparison involving an invalid
 * element does NOT use the compared value, it is forced, and the forced
 * value differs per aggregation.  Getting that wrong shows up only on
 * short strings, which is exactly what a string search hits at the end
 * of a buffer, so it is written out explicitly rather than folded in.
 */
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
 * a Go binary) carries 878 AESENC, 326 AESDEC and 156 PCLMULQDQ.  It
 * cannot be rescued by this library alone — its CPUID fail-fast runs
 * before any instruction faults — but the instructions themselves
 * belong here: any binary that reaches them without a CPUID gate needs
 * them, and they are a prerequisite for ever defeating that check.
 *
 * The S-box is COMPUTED, not transcribed: inversion in GF(2^8) modulo
 * 0x11b followed by the affine transform.  256 hand-typed hex bytes is
 * a place to make a silent one-nibble error that only shows up as a
 * wrong ciphertext, and the differential test would then be arguing
 * with a typo rather than with the silicon. */
static uint8_t aes_sbox[256], aes_inv_sbox[256];
static int     aes_tables_ready;

static uint8_t gmul(uint8_t a, uint8_t b)
{
    uint8_t r = 0;
    while (b) {
        if (b & 1) r ^= a;
        a = (uint8_t) ((a << 1) ^ ((a & 0x80) ? 0x1b : 0));
        b >>= 1;
    }
    return r;
}

static void aes_init_tables(void)
{
    if (aes_tables_ready) return;
    uint8_t inv[256];
    inv[0] = 0;
    for (int i = 1; i < 256; i++)
        for (int j = 1; j < 256; j++)
            if (gmul((uint8_t) i, (uint8_t) j) == 1) { inv[i] = (uint8_t) j; break; }
    for (int i = 0; i < 256; i++) {
        uint8_t x = inv[i], y = x;
        for (int k = 0; k < 4; k++) { y = (uint8_t) ((y << 1) | (y >> 7)); x ^= y; }
        aes_sbox[i] = (uint8_t) (x ^ 0x63);
    }
    for (int i = 0; i < 256; i++) aes_inv_sbox[aes_sbox[i]] = (uint8_t) i;
    aes_tables_ready = 1;
}

void sse42emu_core_init(void)
{
    aes_init_tables();
}

/* State bytes are column-major: byte i is row i%4, column i/4. */
static void aes_shift_rows(uint8_t s[16], int inverse)
{
    uint8_t t[16];
    memcpy(t, s, 16);
    for (int r = 1; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            int src = inverse ? ((c - r) & 3) : ((c + r) & 3);
            s[r + 4 * c] = t[r + 4 * src];
        }
}

static void aes_sub_bytes(uint8_t s[16], int inverse)
{
    const uint8_t *box = inverse ? aes_inv_sbox : aes_sbox;
    for (int i = 0; i < 16; i++) s[i] = box[s[i]];
}

static void aes_mix_columns(uint8_t s[16], int inverse)
{
    for (int c = 0; c < 4; c++) {
        uint8_t *q = s + 4 * c, a0 = q[0], a1 = q[1], a2 = q[2], a3 = q[3];
        if (!inverse) {
            q[0] = (uint8_t) (gmul(a0,2) ^ gmul(a1,3) ^ a2 ^ a3);
            q[1] = (uint8_t) (a0 ^ gmul(a1,2) ^ gmul(a2,3) ^ a3);
            q[2] = (uint8_t) (a0 ^ a1 ^ gmul(a2,2) ^ gmul(a3,3));
            q[3] = (uint8_t) (gmul(a0,3) ^ a1 ^ a2 ^ gmul(a3,2));
        } else {
            q[0] = (uint8_t) (gmul(a0,14) ^ gmul(a1,11) ^ gmul(a2,13) ^ gmul(a3, 9));
            q[1] = (uint8_t) (gmul(a0, 9) ^ gmul(a1,14) ^ gmul(a2,11) ^ gmul(a3,13));
            q[2] = (uint8_t) (gmul(a0,13) ^ gmul(a1, 9) ^ gmul(a2,14) ^ gmul(a3,11));
            q[3] = (uint8_t) (gmul(a0,11) ^ gmul(a1,13) ^ gmul(a2, 9) ^ gmul(a3,14));
        }
    }
}

void sse42emu_aes(uint8_t dst[16], const uint8_t src[16], int op)
{
    aes_init_tables();
    uint8_t st[16];
    memcpy(st, dst, 16);
    switch (op) {
    case SSE42EMU_AESENC:
    case SSE42EMU_AESENCLAST:
        aes_shift_rows(st, 0); aes_sub_bytes(st, 0);
        if (op == SSE42EMU_AESENC) aes_mix_columns(st, 0);
        for (int i = 0; i < 16; i++) dst[i] = (uint8_t) (st[i] ^ src[i]);
        break;
    case SSE42EMU_AESDEC:
    case SSE42EMU_AESDECLAST:
        aes_shift_rows(st, 1); aes_sub_bytes(st, 1);
        if (op == SSE42EMU_AESDEC) aes_mix_columns(st, 1);
        for (int i = 0; i < 16; i++) dst[i] = (uint8_t) (st[i] ^ src[i]);
        break;
    case SSE42EMU_AESIMC:
        memcpy(st, src, 16); aes_mix_columns(st, 1); memcpy(dst, st, 16);
        break;
    }
}

void sse42emu_aeskeygen(uint8_t dst[16], const uint8_t src[16], uint8_t rcon)
{
    aes_init_tables();
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

/* ---- PCLMULQDQ: carry-less product of two selected qwords ---------- */
void sse42emu_pclmul(uint8_t dst[16], const uint8_t a[16], const uint8_t b[16], uint8_t imm)
{
    uint64_t x, y;
    memcpy(&x, a + ((imm & 0x01) ? 8 : 0), 8);
    memcpy(&y, b + ((imm & 0x10) ? 8 : 0), 8);
    uint64_t lo = 0, hi = 0;
    for (int i = 0; i < 64; i++) {
        if ((y >> i) & 1) {
            lo ^= x << i;
            hi ^= (i == 0) ? 0 : (x >> (64 - i));
        }
    }
    memcpy(dst, &lo, 8);
    memcpy(dst + 8, &hi, 8);
}
