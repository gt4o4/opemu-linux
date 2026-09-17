/* SPDX-License-Identifier: MIT */
/* sse42emu — public surface.
 *
 * The emulator is written against a PLAIN register struct, not against
 * ucontext_t, so the unit test can drive it on any CPU — including the
 * build machine, which has SSE4.2 and can therefore never reach the
 * fault path.  That matters here: this host runs `nix.settings.max-jobs
 * = 0`, so every installCheck executes on lix-ory.  See default.nix.
 */
#ifndef SSE42EMU_H
#define SSE42EMU_H

#include <stdint.h>
#include <sys/ucontext.h>
#include <Zydis/Zydis.h>
#include "../core/emu-core.h"

typedef struct {
    uint64_t gpr[NGREG];     /* indexed by glibc REG_* */
    uint8_t  xmm[16][16];    /* xmm0..15, little-endian bytes */
    uint64_t rflags;
    uint64_t rip;
    int      xmm_written;    /* index of the one XMM the op wrote, or -1 */
} sse42emu_regs;

typedef enum {
    SSE42EMU_OK = 0,         /* emulated; rip advanced */
    SSE42EMU_UNHANDLED,      /* not one of ours — caller must re-raise */
    SSE42EMU_DECODE_FAIL,
} sse42emu_status;

/* Initialise the decoder and the AES tables.  The constructor calls this;
 * a caller that drives sse42emu_step() directly must call it first. */
void sse42emu_prepare(void);

/* Decode one instruction at `code` and emulate it into *R. */
sse42emu_status sse42emu_step(sse42emu_regs *R, const uint8_t *code, size_t avail);

/* Register-file access, exposed for the unit tests.  The arithmetic
 * itself is in ../core/emu-core.h. */
uint64_t sse42emu_gpr_read(const sse42emu_regs *R, ZydisRegister reg);
void     sse42emu_gpr_write(sse42emu_regs *R, ZydisRegister reg, uint64_t val);

extern unsigned long sse42emu_faults;      /* instructions emulated */
extern unsigned long sse42emu_unhandled;   /* SIGILLs we passed on */

#endif
