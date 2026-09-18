/* SPDX-License-Identifier: MIT */
/* opemu_match() against Zydis: does the kernel's fast-path matcher accept
 * exactly the encodings it should, and read every field the way a full
 * decoder does?
 *
 * The matcher is a second decoder in the kernel's #UD path, and a second
 * decoder is a place to be silently wrong: a field read one bit off gives
 * a correct-looking emulation of the wrong register, and nothing downstream
 * can tell.  So every encoding here goes through both, and the two must
 * agree on (1) whether the matcher takes it at all — a false negative is a
 * silent performance regression, a false positive a wrong result — and (2)
 * when it does, on length, opcode, both registers (with the AH..BH versus
 * SPL..DIL rule), operand size, memory-vs-register, base/index/scale/disp,
 * the immediate, REX.W, and the numeric effective address computed by
 * Zydis's own formula from a random register file.  A truncated buffer
 * must be refused at every length below the form's and accepted at it,
 * with the bytes placed against a PROT_NONE page so an over-read faults.
 *
 * Inputs: the canned vectors the kernel checks itself against at boot,
 * exhaustive ModRM x SIB x REX sweeps for a GPR form, a byte-source form
 * and an XMM form, and millions of grammar-shaped random strings.
 *
 * The one deliberate disagreement: F2 and F3 together.  Hardware and Zydis
 * take the last one; the matcher (like the kernel's general path) refuses
 * the pair, so prefix_region_ok() treats it as "must not accept".
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <Zydis/Zydis.h>
#include "../kernel/arch/x86/kernel/opemu-match-vectors.h"

static long checks, failures, accepted;
static uint64_t rng = 0x452821E638D01377ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static ZydisDecoder dec;

static void fail(const char *why, const uint8_t *b, int n)
{
    failures++;
    if (failures > 20) return;
    printf("  FAIL %s:", why);
    for (int i = 0; i < n; i++) printf(" %02x", b[i]);
    printf("\n");
}

/* ---- the oracle for "should the matcher take this?" ------------------ */
static int our_mnemonic(ZydisMnemonic m, int *op)
{
    switch (m) {
    case ZYDIS_MNEMONIC_POPCNT:          *op = OPEMU_OP_POPCNT; return 1;
    case ZYDIS_MNEMONIC_CRC32:           *op = OPEMU_OP_CRC32; return 1;
    case ZYDIS_MNEMONIC_PCMPGTQ:         *op = OPEMU_OP_PCMPGTQ; return 1;
    case ZYDIS_MNEMONIC_PCMPESTRM:       *op = OPEMU_OP_PCMPESTRM; return 1;
    case ZYDIS_MNEMONIC_PCMPESTRI:       *op = OPEMU_OP_PCMPESTRI; return 1;
    case ZYDIS_MNEMONIC_PCMPISTRM:       *op = OPEMU_OP_PCMPISTRM; return 1;
    case ZYDIS_MNEMONIC_PCMPISTRI:       *op = OPEMU_OP_PCMPISTRI; return 1;
    case ZYDIS_MNEMONIC_PCLMULQDQ:       *op = OPEMU_OP_PCLMULQDQ; return 1;
    case ZYDIS_MNEMONIC_AESIMC:          *op = OPEMU_OP_AESIMC; return 1;
    case ZYDIS_MNEMONIC_AESENC:          *op = OPEMU_OP_AESENC; return 1;
    case ZYDIS_MNEMONIC_AESENCLAST:      *op = OPEMU_OP_AESENCLAST; return 1;
    case ZYDIS_MNEMONIC_AESDEC:          *op = OPEMU_OP_AESDEC; return 1;
    case ZYDIS_MNEMONIC_AESDECLAST:      *op = OPEMU_OP_AESDECLAST; return 1;
    case ZYDIS_MNEMONIC_AESKEYGENASSIST: *op = OPEMU_OP_AESKEYGENASSIST; return 1;
    default: return 0;
    }
}

/* The grammar's prefix rules, stated independently of the matcher: only
 * 66/F2/F3, at most four, not both F2 and F3, no 66 after an F2/F3 (the
 * kernel's decoder cannot decode that order, so the emulator refuses it
 * although the silicon and Zydis take it), an optional REX immediately
 * before 0F, nothing else in front of the opcode. */
static int prefix_region_ok(const uint8_t *b, int n)
{
    int i = 0, np = 0, f2 = 0, f3 = 0;
    while (i < n && (b[i] == 0x66 || b[i] == 0xf2 || b[i] == 0xf3)) {
        if (b[i] == 0x66 && (f2 || f3)) return 0;
        if (b[i] == 0xf2) f2 = 1;
        if (b[i] == 0xf3) f3 = 1;
        i++; np++;
    }
    if (np > 4 || (f2 && f3) || i >= n) return 0;
    if ((b[i] & 0xf0) == 0x40) { i++; if (i >= n) return 0; }
    return b[i] == 0x0f;
}

/* Zydis register -> (encoding index 0..15, high-byte flag), or -1 */
static int gpr_index(ZydisRegister r, int *high)
{
    *high = 0;
    if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_BL) return r - ZYDIS_REGISTER_AL;
    if (r >= ZYDIS_REGISTER_AH && r <= ZYDIS_REGISTER_BH) { *high = 1; return 4 + (r - ZYDIS_REGISTER_AH); }
    if (r >= ZYDIS_REGISTER_SPL && r <= ZYDIS_REGISTER_DIL) return 4 + (r - ZYDIS_REGISTER_SPL);
    if (r >= ZYDIS_REGISTER_R8B && r <= ZYDIS_REGISTER_R15B) return 8 + (r - ZYDIS_REGISTER_R8B);
    if (r >= ZYDIS_REGISTER_AX && r <= ZYDIS_REGISTER_R15W) return r - ZYDIS_REGISTER_AX;
    if (r >= ZYDIS_REGISTER_EAX && r <= ZYDIS_REGISTER_R15D) return r - ZYDIS_REGISTER_EAX;
    if (r >= ZYDIS_REGISTER_RAX && r <= ZYDIS_REGISTER_R15) return r - ZYDIS_REGISTER_RAX;
    return -1;
}
static int xmm_index(ZydisRegister r)
{
    if (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15) return r - ZYDIS_REGISTER_XMM0;
    return -1;
}
/* The enum order the mapping above relies on (Zydis 4.x). */
_Static_assert(ZYDIS_REGISTER_BL - ZYDIS_REGISTER_AL == 3, "GPR8 order");
_Static_assert(ZYDIS_REGISTER_BH - ZYDIS_REGISTER_AH == 3, "GPR8 high order");
_Static_assert(ZYDIS_REGISTER_DIL - ZYDIS_REGISTER_SPL == 3, "GPR8 REX order");
_Static_assert(ZYDIS_REGISTER_R15B - ZYDIS_REGISTER_R8B == 7, "GPR8 r8..r15 order");
_Static_assert(ZYDIS_REGISTER_R15W - ZYDIS_REGISTER_AX == 15, "GPR16 order");
_Static_assert(ZYDIS_REGISTER_R15D - ZYDIS_REGISTER_EAX == 15, "GPR32 order");
_Static_assert(ZYDIS_REGISTER_R15 - ZYDIS_REGISTER_RAX == 15, "GPR64 order");
_Static_assert(ZYDIS_REGISTER_XMM15 - ZYDIS_REGISTER_XMM0 == 15, "XMM order");

static const ZydisRegister gpr64[16] = {
    ZYDIS_REGISTER_RAX, ZYDIS_REGISTER_RCX, ZYDIS_REGISTER_RDX, ZYDIS_REGISTER_RBX,
    ZYDIS_REGISTER_RSP, ZYDIS_REGISTER_RBP, ZYDIS_REGISTER_RSI, ZYDIS_REGISTER_RDI,
    ZYDIS_REGISTER_R8,  ZYDIS_REGISTER_R9,  ZYDIS_REGISTER_R10, ZYDIS_REGISTER_R11,
    ZYDIS_REGISTER_R12, ZYDIS_REGISTER_R13, ZYDIS_REGISTER_R14, ZYDIS_REGISTER_R15,
};

/* ---- one encoding through both -------------------------------------- */
static void compare(const uint8_t *b, int n)
{
    ZydisDecodedInstruction in;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    struct opemu_insn d;
    int want_op = 0;
    const int zok = ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, b, (ZyanUSize) n, &in, ops));
    const int expect = zok && our_mnemonic(in.mnemonic, &want_op) && prefix_region_ok(b, n);
    const int got = opemu_match(b, n, &d);

    checks++;
    if (!expect) {
        if (got) fail("matcher accepted what it must not", b, n);
        return;
    }
    if (!got) { fail("matcher refused an encoding that is ours", b, n); return; }
    accepted++;

    if (d.len != in.length) { fail("length", b, n); return; }
    if (d.op != want_op) { fail("opcode", b, n); return; }
    if (d.rexw != (in.raw.rex.W ? 1 : 0)) { fail("REX.W", b, n); return; }
    if (d.has_rex != ((in.attributes & ZYDIS_ATTRIB_HAS_REX) ? 1 : 0)) { fail("has_rex", b, n); return; }

    /* operand 0: the destination register */
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) { fail("zydis op0 not a register?", b, n); return; }
    {
        int high, r = d.op == OPEMU_OP_POPCNT || d.op == OPEMU_OP_CRC32
                        ? gpr_index(ops[0].reg.value, &high) : xmm_index(ops[0].reg.value);
        if (r != d.reg) { fail("dest register", b, n); return; }
    }
    /* operand 1: the source, register or memory */
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        int high = 0, r = d.op == OPEMU_OP_POPCNT || d.op == OPEMU_OP_CRC32
                            ? gpr_index(ops[1].reg.value, &high) : xmm_index(ops[1].reg.value);
        if (d.is_mem) { fail("is_mem set for a register source", b, n); return; }
        if (r != d.rm) { fail("source register", b, n); return; }
        /* the byte-register rule: AH..BH only without REX, SPL..DIL only with */
        if (d.src_bytes == 1 && r >= 4 && r <= 7 && high == d.has_rex) { fail("AH..BH / SPL..DIL rule", b, n); return; }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!d.is_mem) { fail("is_mem clear for a memory source", b, n); return; }
        int base = ops[1].mem.base == ZYDIS_REGISTER_NONE ? OPEMU_REG_NONE :
                   ops[1].mem.base == ZYDIS_REGISTER_RIP  ? OPEMU_REG_RIP : -3;
        if (base == -3) { int h; base = gpr_index(ops[1].mem.base, &h); }
        int index = ops[1].mem.index == ZYDIS_REGISTER_NONE ? OPEMU_REG_NONE : -3;
        if (index == -3) { int h; index = gpr_index(ops[1].mem.index, &h); }
        if (base != d.base) { fail("base", b, n); return; }
        if (index != d.index) { fail("index", b, n); return; }
        if (index != OPEMU_REG_NONE) {
            int sc = ops[1].mem.scale == 1 ? 0 : ops[1].mem.scale == 2 ? 1 : ops[1].mem.scale == 4 ? 2 : 3;
            if (sc != d.scale) { fail("scale", b, n); return; }
        }
        int64_t disp = ops[1].mem.disp.has_displacement ? ops[1].mem.disp.value : 0;
        if (disp != (int64_t) d.disp) { fail("displacement", b, n); return; }

        /* the numeric address, from a random register file, by Zydis's own formula */
        ZydisRegisterContext ctx; memset(&ctx, 0, sizeof ctx);
        uint64_t gpr[16], rip = rnd() & 0x00007fffffffffffULL, ea = 0;
        for (int i = 0; i < 16; i++) { gpr[i] = rnd(); ctx.values[gpr64[i]] = gpr[i]; }
        ctx.values[ZYDIS_REGISTER_RIP] = rip;
        if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddressEx(&in, &ops[1], rip, &ctx, &ea))) { fail("zydis could not compute the EA?", b, n); return; }
        if (opemu_match_ea_gpr(&d, rip, gpr) != ea) { fail("effective address", b, n); return; }
    } else { fail("zydis op1 neither register nor memory?", b, n); return; }
    if ((int) d.src_bytes * 8 != (int) ops[1].size) { fail("source size", b, n); return; }

    /* the immediate, when the form has one */
    if (d.has_imm) {
        if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) { fail("zydis op2 not an immediate?", b, n); return; }
        if ((uint8_t) ops[2].imm.value.u != d.imm) { fail("imm8", b, n); return; }
    } else if (in.operand_count_visible > 2 && ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        fail("has_imm clear on a form with an immediate", b, n); return;
    }
}

/* Truncation: refused at every length below L, taken at L — with the bytes
 * ending exactly at a PROT_NONE page so a read past n faults instead of
 * passing unnoticed. */
static uint8_t *edge;           /* last byte of an RW page, next page PROT_NONE */
static void truncation(const uint8_t *b, int L)
{
    for (int k = 1; k <= L; k++) {
        struct opemu_insn d;
        uint8_t *p = edge + 1 - k;
        memcpy(p, b, (size_t) k);
        int got = opemu_match(p, k, &d);
        checks++;
        if (k < L ? got != 0 : got != L) fail("truncation contract", b, L);
    }
}

int main(void)
{
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    uint8_t *pages = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED || mprotect(pages + 4096, 4096, PROT_NONE) != 0) { printf("mmap failed\n"); return 1; }
    edge = pages + 4095;

    /* 1. the canned vectors: the matcher's own expectations, then Zydis */
    for (int i = 0; i < OPEMU_MATCH_NR_VECTORS; i++) {
        const struct opemu_match_vector *v = &opemu_match_vectors[i];
        struct opemu_insn d;
        int got = opemu_match(v->bytes, v->n, &d);
        checks++;
        if (got != v->want_len || (got && d.op != v->want_op)) fail(v->what, v->bytes, v->n);
        compare(v->bytes, v->n);
        if (v->want_len) truncation(v->bytes, v->want_len);
    }
    const long canned = checks;

    /* 2. exhaustive ModRM x SIB x REX for three forms, disp bytes fixed */
    static const struct { uint8_t pre[4]; int npre; const char *name; } forms[] = {
        { { 0xF3, 0x0F, 0xB8 },       3, "popcnt" },
        { { 0xF2, 0x0F, 0x38, 0xF0 }, 4, "crc32 r/m8" },
        { { 0x66, 0x0F, 0x3A, 0x63 }, 4, "pcmpistri" },
    };
    static const uint8_t disp[4] = { 0x80, 0x00, 0x00, 0x80 };
    for (unsigned f = 0; f < sizeof forms / sizeof *forms; f++) {
        for (int rex = -1; rex < 16; rex++) {
            for (int modrm = 0; modrm < 256; modrm++) {
                const int mod = modrm >> 6, rm = modrm & 7;
                const int nsib = (mod != 3 && rm == 4) ? 256 : 1;
                for (int sib = 0; sib < nsib; sib++) {
                    uint8_t b[16]; int n = 0;
                    b[n++] = forms[f].pre[0];
                    if (rex >= 0) b[n++] = (uint8_t) (0x40 | rex);
                    for (int k = 1; k < forms[f].npre; k++) b[n++] = forms[f].pre[k];
                    b[n++] = (uint8_t) modrm;
                    if (nsib == 256) b[n++] = (uint8_t) sib;
                    for (int k = 0; k < 4; k++) b[n++] = disp[k];
                    b[n++] = 0x5A;                       /* imm8 for the 3A form; junk otherwise */
                    b[n++] = 0xC3;
                    compare(b, n);
                }
            }
        }
    }
    const long exhaustive = checks - canned;

    /* 3. grammar-shaped random strings */
    static const uint8_t pool_pre[] = { 0x66, 0xF2, 0xF3, 0x66, 0xF2, 0xF3, 0x67, 0x64, 0x65, 0x2E, 0xF0, 0x26, 0x36, 0x3E };
    static const uint8_t pool_op3[] = { 0xF0, 0xF1, 0x37, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0x60, 0x61, 0x62, 0x63, 0x44 };
    for (long t = 0; t < 5000000; t++) {
        uint8_t b[16]; int n = 0;
        int np = (int) (rnd() % 6);                       /* 0..5 prefixes, sometimes too many */
        for (int k = 0; k < np && n < 6; k++) b[n++] = pool_pre[rnd() % sizeof pool_pre];
        if (rnd() % 3 == 0) b[n++] = (uint8_t) (0x40 | (rnd() & 0xf));
        if (rnd() % 64 == 0) b[n++] = (uint8_t) rnd();     /* an arbitrary byte in front: usually not 0F */
        else b[n++] = 0x0F;
        switch (rnd() % 8) {
        case 0: b[n++] = 0xB8; break;
        case 1: case 2: case 3: b[n++] = 0x38; break;
        case 4: case 5: case 6: b[n++] = 0x3A; break;
        default: b[n++] = (uint8_t) rnd(); break;
        }
        if (rnd() % 8) b[n++] = pool_op3[rnd() % sizeof pool_op3];
        else b[n++] = (uint8_t) rnd();
        while (n < 15) b[n++] = (uint8_t) rnd();          /* ModRM, SIB, disp, imm: random */
        compare(b, n);
        if (t % 97 == 0) {                                /* the truncation contract on a sample */
            struct opemu_insn d;
            int L = opemu_match(b, n, &d);
            if (L) truncation(b, L);
        }
    }

    printf("%ld checks (%ld canned, %ld exhaustive, %ld random; %ld encodings accepted by both), %ld failures\n",
           checks, canned, exhaustive, checks - canned - exhaustive, accepted, failures);
    /* a sweep in which almost nothing was accepted would prove nothing */
    if (accepted < 200000) { printf("  only %ld accepted encodings — the generator drifted off the grammar\n", accepted); return 1; }
    return failures ? 1 : 0;
}
