/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _ASM_X86_OPEMU_H
#define _ASM_X86_OPEMU_H

#include <linux/types.h>

struct pt_regs;

#ifdef CONFIG_X86_UD_EMULATE
/*
 * Emulate a user-mode #UD raised by an SSE4.2, POPCNT, AES-NI or PCLMULQDQ
 * instruction the CPU lacks.  Returns true when the instruction was
 * emulated (or converted into the SIGSEGV a #PF on its operand would have
 * raised) and the trap is finished; false to let the normal SIGILL path run.
 */
bool opemu_fixup_ud(struct pt_regs *regs);
#else
static inline bool opemu_fixup_ud(struct pt_regs *regs) { return false; }
#endif

#endif /* _ASM_X86_OPEMU_H */
