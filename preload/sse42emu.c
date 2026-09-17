/* SPDX-License-Identifier: MIT */
/* sse42emu — emulate the SSE4.2 instructions Penryn lacks, from SIGILL.
 *
 * iMacGBC is a Core 2 Duo E8435: SSE4.1 and SSSE3, no SSE4.2.  Prebuilt
 * vendor binaries (upstream claude-code's bun build, notably) execute
 * POPCNT and the PCMPxSTRx string ops unconditionally and die with SIGILL.
 *
 * Scope is set by MEASUREMENT, not by guesswork.  `sde64 -mix` over a
 * `claude --version` (10,324,657 instructions) executed exactly:
 *
 *     PCMPISTRI   862      POPCNT      387
 *     PCMPGTQ       8      PCMPISTRM     2
 *     CRC32         0      PCLMULQDQ     0      PCMPESTRx  0
 *
 * i.e. 1259 instructions, 0.0122% of the stream — everything else the
 * image contains (47k AVX, 2.2k AVX2, AES, BMI) sits behind CPUID
 * dispatch and never runs.  At roughly 3 us per signal that is ~4 ms,
 * which is why this trap-and-emulate core is worth having on its own
 * before any instruction patching: the faults are rare enough that the
 * patcher is an optimisation, not a prerequisite.
 *
 * CRC32 is implemented anyway (it is ~20 lines and a different workload
 * would hit it), and so, since 2026-09-17, is the rest of the set a
 * Penryn lacks either side of SSE4.2:
 *
 *   PCMPESTRI/PCMPESTRM  the explicit-length string forms, sharing one
 *                        core with the implicit ones
 *   PCLMULQDQ            carry-less multiply
 *   AESENC/AESENCLAST/AESDEC/AESDECLAST/AESIMC/AESKEYGENASSIST
 *
 * The AES set is here for agy (antigravity-cli): 878 AESENC, 326 AESDEC,
 * 156 PCLMULQDQ statically.  Note what that does NOT buy — agy checks
 * CPUID and calls go/sigill-fail-fast before executing any of them, and
 * `cpuid_fault` is absent on this CPU, so no handler can ever fire for
 * it.  These are here because they are the instructions this hardware
 * lacks, and because a binary that reaches them WITHOUT a CPUID gate is
 * then covered; agy itself still needs the QEMU route.
 *
 * Decoding is Zydis (static, so the library keeps no runtime
 * dependency — see the runpath trap in common/pkgs/bind-rewrite.nix).
 * The arithmetic is NOT here: it is ../core/emu-core.c, the file this
 * library shares with the kernel-side emulator (kernel/) and the tests.
 * This file is the signal-frame plumbing around it.
 *
 * State is plain .bss on purpose: never const, never RELRO-sealed, the
 * rule docs/SECCOMP.md pays for in its own handler.
 *
 * INDEPENDENTLY CROSS-CHECKED, 2026-09-17.  opemu-linux (mirh/opemu-linux,
 * the XNU-AMD "Opcode Emulator" ported to a Linux kernel module) carries a
 * PCMPxSTRx core that is GCC's own testsuite reference model
 * (gcc.target/i386/sse4_2-pcmpstr.h).  Extracted and run against this
 * implementation and against the silicon, all 128 immediates, 512,000
 * cases: three-way agreement, zero mismatches.  So the SDM reading here
 * matches GCC's, not just Intel's hardware.
 *
 * That code is NOT vendored and must not be: opemu is GPLv2 and the GCC
 * model is GPL'd too, while this library is MIT.  The cross-check was a
 * one-off; the result is the thing worth keeping.
 */
#define _GNU_SOURCE
#include <signal.h>
#include <ucontext.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <Zydis/Zydis.h>

#include "sse42emu.h"   /* pulls ../core/emu-core.h */

/* ---- mutable state, .bss ---- */
static ZydisDecoder   g_decoder;
static int            g_ready;
static struct sigaction g_prev_sigill;
static struct sigaction g_app_sigill;   /* what the program thinks it installed */
static int            g_app_has_sigill;
static int            g_trace;
static int            g_no_hook;

/* The real one, reached past our own interposer. */
typedef int (*sigaction_fn)(int, const struct sigaction *, struct sigaction *);
static int real_sigaction(int sig, const struct sigaction *act, struct sigaction *old)
{
    static sigaction_fn real;
    if (!real) real = (sigaction_fn) dlsym(RTLD_NEXT, "__libc_sigaction");
    if (!real) real = (sigaction_fn) dlsym(RTLD_NEXT, "sigaction");
    return real ? real(sig, act, old) : -1;
}
unsigned long sse42emu_faults;      /* exported: fault counter */
unsigned long sse42emu_unhandled;


/* ---- GPR access -------------------------------------------------- */
/* Map a Zydis 64-bit register enum to its glibc gregs[] index. */
static int greg_index(ZydisRegister r64)
{
    switch (r64) {
    case ZYDIS_REGISTER_RAX: return REG_RAX;
    case ZYDIS_REGISTER_RBX: return REG_RBX;
    case ZYDIS_REGISTER_RCX: return REG_RCX;
    case ZYDIS_REGISTER_RDX: return REG_RDX;
    case ZYDIS_REGISTER_RSI: return REG_RSI;
    case ZYDIS_REGISTER_RDI: return REG_RDI;
    case ZYDIS_REGISTER_RBP: return REG_RBP;
    case ZYDIS_REGISTER_RSP: return REG_RSP;
    case ZYDIS_REGISTER_R8:  return REG_R8;
    case ZYDIS_REGISTER_R9:  return REG_R9;
    case ZYDIS_REGISTER_R10: return REG_R10;
    case ZYDIS_REGISTER_R11: return REG_R11;
    case ZYDIS_REGISTER_R12: return REG_R12;
    case ZYDIS_REGISTER_R13: return REG_R13;
    case ZYDIS_REGISTER_R14: return REG_R14;
    case ZYDIS_REGISTER_R15: return REG_R15;
    default: return -1;
    }
}

uint64_t sse42emu_gpr_read(const sse42emu_regs *R, ZydisRegister reg)
{
    ZydisRegister r64 = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg);
    int i = greg_index(r64);
    if (i < 0) return 0;
    uint64_t v = (uint64_t) R->gpr[i];
    switch (ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, reg)) {
    case 8:  /* no high-byte regs reach us: these opcodes cannot encode AH..BH */
             return v & 0xffu;
    case 16: return v & 0xffffu;
    case 32: return v & 0xffffffffu;
    default: return v;
    }
}

void sse42emu_gpr_write(sse42emu_regs *R, ZydisRegister reg, uint64_t val)
{
    ZydisRegister r64 = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg);
    int i = greg_index(r64);
    if (i < 0) return;
    switch (ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, reg)) {
    case 16: R->gpr[i] = (R->gpr[i] & ~0xffffULL) | (val & 0xffffULL); break;
    /* 32-bit writes zero-extend to 64 — the whole reason POPCNT r32 is safe */
    case 32: R->gpr[i] = val & 0xffffffffULL; break;
    default: R->gpr[i] = val; break;
    }
}

/* ---- operand access ------------------------------------------------ */
static int xmm_index(ZydisRegister r)
{
    if (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15)
        return (int) (r - ZYDIS_REGISTER_XMM0);
    return -1;
}

/* Effective address of a memory operand, from the emulated register file. */
static void *mem_addr(const sse42emu_regs *R, const ZydisDecodedInstruction *in,
                      const ZydisDecodedOperand *op)
{
    uint64_t addr = 0;
    if (op->mem.base == ZYDIS_REGISTER_RIP) {
        addr = R->rip + in->length + (uint64_t) op->mem.disp.value;
    } else {
        if (op->mem.base != ZYDIS_REGISTER_NONE)
            addr += sse42emu_gpr_read(R, op->mem.base);
        if (op->mem.index != ZYDIS_REGISTER_NONE)
            addr += sse42emu_gpr_read(R, op->mem.index) * op->mem.scale;
        if (op->mem.disp.has_displacement)
            addr += (uint64_t) op->mem.disp.value;
    }
    return (void *) (uintptr_t) addr;
}

/* Read a source operand as an integer (register or memory). */
static uint64_t src_int(const sse42emu_regs *R, const ZydisDecodedInstruction *in,
                        const ZydisDecodedOperand *op, unsigned *bytes_out)
{
    unsigned bytes = op->size / 8;
    if (bytes_out) *bytes_out = bytes;
    if (op->type == ZYDIS_OPERAND_TYPE_REGISTER)
        return sse42emu_gpr_read(R, op->reg.value);
    uint64_t v = 0;
    memcpy(&v, mem_addr(R, in, op), bytes);
    return v;
}

/* Read a 128-bit source operand (xmm register or m128). */
static void src_xmm(const sse42emu_regs *R, const ZydisDecodedInstruction *in,
                    const ZydisDecodedOperand *op, uint8_t out[16])
{
    if (op->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        int i = xmm_index(op->reg.value);
        memcpy(out, R->xmm[i < 0 ? 0 : i], 16);
    } else {
        memcpy(out, mem_addr(R, in, op), 16);
    }
}

/* ---- one instruction ----------------------------------------------- */
sse42emu_status sse42emu_step(sse42emu_regs *R, const uint8_t *code, size_t avail)
{
    R->xmm_written = -1;        /* no XMM touched unless an op says so */
    ZydisDecodedInstruction in;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&g_decoder, code, avail, &in, ops)))
        return SSE42EMU_DECODE_FAIL;

    switch (in.mnemonic) {
    case ZYDIS_MNEMONIC_POPCNT: {
        unsigned bytes = 0;
        uint64_t fl = 0;
        uint64_t v = src_int(R, &in, &ops[1], &bytes);
        sse42emu_gpr_write(R, ops[0].reg.value, sse42emu_popcnt(v, bytes, &fl));
        R->rflags = (R->rflags & ~(uint64_t) EMU_FL_ALL) | fl;
        break;
    }
    case ZYDIS_MNEMONIC_CRC32: {
        unsigned bytes = 0;
        uint64_t v = src_int(R, &in, &ops[1], &bytes);
        uint32_t crc = (uint32_t) sse42emu_gpr_read(R, ops[0].reg.value);
        sse42emu_gpr_write(R, ops[0].reg.value, sse42emu_crc32(crc, v, bytes));
        break;                                   /* CRC32 affects no flags */
    }
    case ZYDIS_MNEMONIC_PCMPGTQ: {
        uint8_t a[16], b[16];
        int d = xmm_index(ops[0].reg.value);
        if (d < 0) return SSE42EMU_UNHANDLED;
        (void) a;
        src_xmm(R, &in, &ops[1], b);
        sse42emu_pcmpgtq(R->xmm[d], b);
        R->xmm_written = d;
        break;
    }
    case ZYDIS_MNEMONIC_PCMPISTRI:
    case ZYDIS_MNEMONIC_PCMPISTRM:
    case ZYDIS_MNEMONIC_PCMPESTRI:
    case ZYDIS_MNEMONIC_PCMPESTRM: {
        uint8_t a[16], b[16], mask[16];
        int s = xmm_index(ops[0].reg.value);
        if (s < 0) return SSE42EMU_UNHANDLED;
        memcpy(a, R->xmm[s], 16);
        src_xmm(R, &in, &ops[1], b);
        uint8_t imm = (uint8_t) ops[2].imm.value.u;
        uint32_t idx = 0; uint64_t fl = 0;
        const int explicit_len = (in.mnemonic == ZYDIS_MNEMONIC_PCMPESTRI ||
                                  in.mnemonic == ZYDIS_MNEMONIC_PCMPESTRM);
        if (explicit_len) {
            /* Lengths in (R|E)AX and (R|E)DX.  REX.W picks the width, and
             * the value is SIGNED: a 32-bit form must sign-extend before
             * the core takes its absolute value, or 0xffffffff reads as
             * 4294967295 instead of -1. */
            const int w64 = (in.attributes & ZYDIS_ATTRIB_HAS_REX) && in.raw.rex.W;
            uint64_t ra = sse42emu_gpr_read(R, ZYDIS_REGISTER_RAX);
            uint64_t rd = sse42emu_gpr_read(R, ZYDIS_REGISTER_RDX);
            int64_t la = w64 ? (int64_t) ra : (int64_t) (int32_t) (uint32_t) ra;
            int64_t lb = w64 ? (int64_t) rd : (int64_t) (int32_t) (uint32_t) rd;
            /* Clamp into int32 before the core; magnitudes above 16 are
             * all equivalent, so saturating here loses nothing. */
            if (la >  64) la =  64;
            if (la < -64) la = -64;
            if (lb >  64) lb =  64;
            if (lb < -64) lb = -64;
            sse42emu_pcmpestr(a, b, imm, (int32_t) la, (int32_t) lb, &idx, mask, &fl);
        } else {
            sse42emu_pcmpistr(a, b, imm, &idx, mask, &fl);
        }
        if (in.mnemonic == ZYDIS_MNEMONIC_PCMPISTRI ||
            in.mnemonic == ZYDIS_MNEMONIC_PCMPESTRI)
            sse42emu_gpr_write(R, ZYDIS_REGISTER_ECX, idx);
        else {
            memcpy(R->xmm[0], mask, 16);
            R->xmm_written = 0;
        }
        R->rflags = (R->rflags & ~(uint64_t) EMU_FL_ALL) | fl;
        break;
    }
    case ZYDIS_MNEMONIC_AESENC:
    case ZYDIS_MNEMONIC_AESENCLAST:
    case ZYDIS_MNEMONIC_AESDEC:
    case ZYDIS_MNEMONIC_AESDECLAST:
    case ZYDIS_MNEMONIC_AESIMC: {
        uint8_t src[16];
        int d = xmm_index(ops[0].reg.value);
        if (d < 0) return SSE42EMU_UNHANDLED;
        src_xmm(R, &in, &ops[1], src);
        int op = in.mnemonic == ZYDIS_MNEMONIC_AESENC     ? SSE42EMU_AESENC
               : in.mnemonic == ZYDIS_MNEMONIC_AESENCLAST ? SSE42EMU_AESENCLAST
               : in.mnemonic == ZYDIS_MNEMONIC_AESDEC     ? SSE42EMU_AESDEC
               : in.mnemonic == ZYDIS_MNEMONIC_AESDECLAST ? SSE42EMU_AESDECLAST
                                                          : SSE42EMU_AESIMC;
        sse42emu_aes(R->xmm[d], src, op);
        R->xmm_written = d;
        break;                                   /* AES* affect no flags */
    }
    case ZYDIS_MNEMONIC_AESKEYGENASSIST: {
        uint8_t src[16];
        int d = xmm_index(ops[0].reg.value);
        if (d < 0) return SSE42EMU_UNHANDLED;
        src_xmm(R, &in, &ops[1], src);
        sse42emu_aeskeygen(R->xmm[d], src, (uint8_t) ops[2].imm.value.u);
        R->xmm_written = d;
        break;
    }
    case ZYDIS_MNEMONIC_PCLMULQDQ: {
        uint8_t a[16], b[16];
        int d = xmm_index(ops[0].reg.value);
        if (d < 0) return SSE42EMU_UNHANDLED;
        memcpy(a, R->xmm[d], 16);
        src_xmm(R, &in, &ops[1], b);
        sse42emu_pclmul(R->xmm[d], a, b, (uint8_t) ops[2].imm.value.u);
        R->xmm_written = d;
        break;
    }
    default:
        return SSE42EMU_UNHANDLED;
    }

    R->rip += in.length;
    return SSE42EMU_OK;
}

/* ---- SIGILL handler ------------------------------------------------ */
/* ucontext <-> sse42emu_regs.  XMM lives in the FXSAVE area the kernel
 * saves in the signal frame and restores on sigreturn, so writing it
 * here really does reach the resumed thread. */
static void ctx_load(sse42emu_regs *R, const ucontext_t *uc)
{
    for (int i = 0; i < NGREG; i++) R->gpr[i] = (uint64_t) uc->uc_mcontext.gregs[i];
    R->rip    = (uint64_t) uc->uc_mcontext.gregs[REG_RIP];
    R->rflags = (uint64_t) uc->uc_mcontext.gregs[REG_EFL];
    const struct _libc_fpstate *fp = uc->uc_mcontext.fpregs;
    if (fp) for (int i = 0; i < 16; i++) memcpy(R->xmm[i], &fp->_xmm[i], 16);
    else    memset(R->xmm, 0, sizeof R->xmm);
}

/* Write back ONLY what the emulated instruction actually changed.
 *
 * The first cut copied all 16 XMM registers back on every fault, which
 * is wrong even though it looks like a no-op: the signal frame's legacy
 * FXSAVE area is only authoritative for the components XSTATE_BV marks
 * as saved, so blind round-tripping can hand the thread back state it
 * never had.  367 of the 376 faults in a claude startup are POPCNT,
 * which touches no XMM at all — there is no reason to go near it. */
static void ctx_store(const sse42emu_regs *R, ucontext_t *uc)
{
    static const int gpr_idx[] = {
        REG_RAX, REG_RBX, REG_RCX, REG_RDX, REG_RSI, REG_RDI, REG_RBP, REG_RSP,
        REG_R8, REG_R9, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
    };
    for (unsigned i = 0; i < sizeof gpr_idx / sizeof gpr_idx[0]; i++)
        uc->uc_mcontext.gregs[gpr_idx[i]] = (greg_t) R->gpr[gpr_idx[i]];
    uc->uc_mcontext.gregs[REG_RIP] = (greg_t) R->rip;
    uc->uc_mcontext.gregs[REG_EFL] = (greg_t) R->rflags;
    struct _libc_fpstate *fp = uc->uc_mcontext.fpregs;
    if (fp && R->xmm_written >= 0 && R->xmm_written < 16)
        memcpy(&fp->_xmm[R->xmm_written], R->xmm[R->xmm_written], 16);
}

/* async-signal-safe site log: no printf, no locks, one write(2). */
static void trace_site(char tag, uint64_t rip)
{
    char b[64]; int n = 0;
    const uint8_t *c = (const uint8_t *) (uintptr_t) rip;
    static const char hex[] = "0123456789abcdef";
    b[n++] = tag; b[n++] = ' ';
    for (int k = 60; k >= 0; k -= 4) b[n++] = hex[(rip >> k) & 0xf];
    b[n++] = ' ';
    for (int k = 0; k < 8; k++) { b[n++] = hex[c[k] >> 4]; b[n++] = hex[c[k] & 0xf]; }
    b[n++] = '\n';
    (void) !write(2, b, (size_t) n);
}

static void on_sigill(int sig, siginfo_t *si, void *vuc)
{
    ucontext_t *uc = (ucontext_t *) vuc;
    sse42emu_regs R;
    ctx_load(&R, uc);

    /* 15 is the x86 maximum instruction length; reading short is safe
     * because the faulting instruction is by definition mapped. */
    uint64_t rip0 = R.rip;
    if (sse42emu_step(&R, (const uint8_t *) (uintptr_t) R.rip, 15) == SSE42EMU_OK) {
        ctx_store(&R, uc);
        sse42emu_faults++;
        if (g_trace) trace_site('E', rip0);
        return;
    }

    /* Not ours.  Hand it to whatever the program registered (its own
     * crash reporter, usually), or fall back to the default action.
     * Chaining rather than swallowing: a genuine SIGILL must still kill.
     *
     * This is traced too, and that is the point: when a program dies under
     * the emulator the only question worth answering is WHICH instruction
     * we refused, and a trace that logs successes alone cannot answer it. */
    sse42emu_unhandled++;
    if (g_trace) trace_site('U', rip0);
    if (g_app_has_sigill) {
        if (g_app_sigill.sa_flags & SA_SIGINFO) {
            if (g_app_sigill.sa_sigaction) { g_app_sigill.sa_sigaction(sig, si, vuc); return; }
        } else if (g_app_sigill.sa_handler != SIG_DFL && g_app_sigill.sa_handler != SIG_IGN) {
            g_app_sigill.sa_handler(sig); return;
        }
    }
    real_sigaction(SIGILL, &g_prev_sigill, NULL);
}

/* ---- keep our handler installed -----------------------------------
 * bun (like Go, and for the same reason) installs its own SIGILL
 * handler for its crash reporter, which displaced ours: the first
 * emulated fault worked, then bun's reporter took the next one and the
 * process died with exit 132 anyway.  docs/SECCOMP.md pays for this
 * lesson in its own handler and prescribes the fix — interpose
 * __libc_sigaction, NOT sigaction, because signal(), sigset() and
 * sigvec() all reach the former without touching the latter.
 *
 * We record what the program asked for, keep ours in the kernel, and
 * report its own handler back to it so the bookkeeping it does with the
 * `old` argument stays consistent. */
__attribute__((visibility("default")))
int __libc_sigaction(int sig, const struct sigaction *act, struct sigaction *old)
{
    if (sig != SIGILL || !g_ready || g_no_hook)
        return real_sigaction(sig, act, old);
    if (old) *old = g_app_has_sigill ? g_app_sigill : g_prev_sigill;
    if (act) { g_app_sigill = *act; g_app_has_sigill = 1; }
    return 0;                       /* ours stays installed */
}

__attribute__((visibility("default")))
int sigaction(int sig, const struct sigaction *act, struct sigaction *old)
{
    return __libc_sigaction(sig, act, old);
}

/* Does this CPU already have SSE4.2?  If so we install nothing at all —
 * the library must be provably inert on every other fleet host. */
static int host_has_sse42(void)
{
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
    return (ecx >> 20) & 1;          /* CPUID.01H:ECX.SSE4_2[bit 20] */
}

/* Decoder + AES tables.  Idempotent, and deliberately SEPARATE from the
 * handler install: the unit tests drive sse42emu_step() directly on a
 * builder that HAS SSE4.2, where sse42emu_init() returns early and would
 * otherwise leave g_decoder zeroed.  A zeroed ZydisDecoder decodes as
 * LONG_64 with a 16-bit stack width purely because both enums start at
 * zero — it worked by accident, and the tests now say what they mean.
 * Building the AES S-box here also keeps that 64K-iteration loop out of
 * the signal handler. */
void sse42emu_prepare(void)
{
    static int done;
    if (done) return;
    ZydisDecoderInit(&g_decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    sse42emu_core_init();
    done = 1;
}

__attribute__((constructor))
static void sse42emu_init(void)
{
    if (g_ready || host_has_sse42()) return;
    g_trace = getenv("SSE42EMU_TRACE") != NULL;
    g_no_hook = getenv("SSE42EMU_NO_HOOK") != NULL;
    sse42emu_prepare();

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_sigill;
    /* SA_ONSTACK is not optional.  A Go runtime (agy) faults on goroutine
     * stacks as small as 2 KB, and this handler's frame — sse42emu_regs
     * plus a ZydisDecodedInstruction and ten operands — is well over 1 KB.
     * Without it the handler overruns the goroutine stack and corrupts
     * whatever is below: the observed symptom was the Go GC dying with
     * "unexpected return pc for runtime.(*sweepLocked).sweep called from
     * 0x3aa717af3aa717af", a return address overwritten with data.
     * Runtimes that care (Go, and the JVM) register a sigaltstack per
     * thread; where none is registered the flag is simply ignored, so it
     * is always safe to ask for. */
    sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_NODEFER | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    real_sigaction(SIGILL, &sa, &g_prev_sigill);
    g_ready = 1;
}

__attribute__((destructor))
static void sse42emu_report(void)
{
    if (!g_ready || !getenv("SSE42EMU_REPORT")) return;
    char buf[128];
    int n = snprintf(buf, sizeof buf, "sse42emu: emulated %lu, unhandled %lu\n",
                     sse42emu_faults, sse42emu_unhandled);
    if (n > 0) (void) !write(2, buf, (size_t) n);
}

