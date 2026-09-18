// SPDX-License-Identifier: GPL-2.0
/*
 * opemu — emulate, from the #UD trap, the instructions a Penryn-class CPU
 * lacks: SSE4.2 (POPCNT, CRC32, PCMPGTQ, PCMPxSTRx), AES-NI and PCLMULQDQ.
 *
 * Modelled on umip.c: this is the kernel's existing way of executing a
 * user instruction on the CPU's behalf.  It sits in exc_invalid_op() where
 * handle_invalid_op() would otherwise raise SIGILL.
 *
 * Why in the kernel rather than a SIGILL handler in the process: a signal
 * handler cannot run in a thread that has the signal blocked — the kernel
 * forces a blocked synchronous signal to SIG_DFL and the process dies with
 * the handler installed and never called — and it costs a signal frame plus
 * an rt_sigreturn per instruction.  Here nothing is delivered to userspace
 * at all.
 *
 * Two paths, one executor:
 *
 *   fast  — interrupts stay off, exactly as the trap left them.  The
 *           instruction bytes are read with two unsafe_get_user() under
 *           pagefault_disable(), matched against the fixed grammar of our
 *           encodings (opemu-match.h — no general decoder, no GDT read), a
 *           register or resident memory operand is fetched the same way,
 *           and the instruction is executed on the task's live registers.
 *           ~25 ns of software per trap on the E8435, against ~146 ns for
 *           the general path.  Anything it cannot resolve without sleeping
 *           — a segment override or 67 prefix, a page that is not present,
 *           a form outside the grammar — is handed over, never guessed.
 *   slow  — the general path, interrupts on: insn_fetch_from_user(),
 *           insn_decode_from_regs(), classify(), insn_get_addr_ref(),
 *           copy_from_user() (which may page in, or fail and become the
 *           SIGSEGV the real instruction's #PF would have been),
 *           fpregs_lock_and_load().  This is where "not ours" is decided:
 *           the matcher only ever says "resolved" or "look again", so the
 *           fast path can emulate exactly what the slow path would, or
 *           hand over — nothing else.
 *
 * Invariants:
 *   - user memory is only ever READ — the instruction bytes (16) and one
 *     source operand (<= 16) — through unsafe_get_user() under
 *     pagefault_disable() on the fast path or copy_from_user() on the slow;
 *   - the only state written is the faulting task's pt_regs GPRs, its six
 *     arithmetic flags (CF PF AF ZF SF OF) and its XMM registers;
 *   - kernel-mode #UD (WARN/BUG ud2) never reaches this code;
 *   - a #UD this code cannot own is left exactly as it was (SIGILL), and a
 *     fault on the operand becomes the SIGSEGV the real instruction's #PF
 *     would have been — never a SIGILL, and never decided on the fast path
 *     (a page that is merely not present takes the slow path and is paged
 *     in there);
 *   - the fast path runs with interrupts disabled throughout and touches no
 *     lock: with no preemption and no softirq possible, the task's FPU
 *     registers are live and stay live between xmm_read() and xmm_write()
 *     (TIF_NEED_FPU_LOAD is checked and sends the trap to the slow path,
 *     which takes fpregs_lock_and_load()).
 *
 * The matcher is a second decoder, and a second decoder is a place to be
 * silently wrong, so it is held to two oracles: tests/test-match.c compares
 * it with Zydis over millions of encodings at build time, and opemu_init()
 * compares it with insn_decode_from_regs() + insn_get_addr_ref() over the
 * canned vectors at boot and refuses to enable the emulator on any
 * disagreement ("selfcheck_ok 0" in the stats file).
 *
 * The arithmetic lives in opemu-core.c, a byte-identical copy of the core
 * shared with the userspace LD_PRELOAD library and its differential tests,
 * which compare it against real silicon.  This file is only the plumbing.
 *
 * Single-step fidelity: like UMIP emulation, an emulated instruction does
 * not raise #DB with TF set; a debugger's stepi lands one instruction late.
 */
#define pr_fmt(fmt) "opemu: " fmt

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/jump_label.h>
#include <linux/percpu.h>
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include <linux/sched/signal.h>
#include <linux/thread_info.h>
#include <linux/uaccess.h>
#include <linux/ratelimit.h>
#include <linux/lockdep.h>
#include <asm/ptrace.h>
#include <asm/insn.h>
#include <asm/insn-eval.h>
#include <asm/fpu/api.h>
#include <asm/cpufeature.h>
#include <asm/trapnr.h>
#include <asm/trap_pf.h>
#include <asm/processor-flags.h>
#include <asm/segment.h>
#include <asm/opemu.h>

#include "opemu-core.h"
#include "opemu-match.h"
#define OPEMU_VECTORS_ATTR __initconst
#include "opemu-match-vectors.h"

/* ---- knobs ----------------------------------------------------------- */
static DEFINE_STATIC_KEY_FALSE(opemu_enabled);

static bool enable_param = true;      /* opemu.enable= on the command line */
static bool initialized;              /* late_initcall ran */
static bool selfcheck_failed;         /* the matcher disagreed with the kernel's decoder at boot */
static bool emulate_popcnt, emulate_sse42, emulate_aes, emulate_pclmul, emulate_any;
static bool trace;                    /* opemu.trace=1: log refused sites */
module_param(trace, bool, 0644);
MODULE_PARM_DESC(trace, "log each user #UD this emulator refuses (ratelimited)");

static void apply_enable(void)
{
	if (enable_param && emulate_any && !selfcheck_failed)
		static_branch_enable(&opemu_enabled);
	else
		static_branch_disable(&opemu_enabled);
}

static int enable_set(const char *val, const struct kernel_param *kp)
{
	int ret = kstrtobool(val, &enable_param);

	if (ret)
		return ret;
	if (initialized)
		apply_enable();
	return 0;
}

static int enable_get(char *buf, const struct kernel_param *kp)
{
	return sprintf(buf, "%d\n", static_key_enabled(&opemu_enabled) ? 1 : 0);
}

static const struct kernel_param_ops enable_ops = {
	.set = enable_set,
	.get = enable_get,
};
module_param_cb(enable, &enable_ops, NULL, 0644);
MODULE_PARM_DESC(enable, "emulate for user space (default 1; forced 0 when the CPU has every instruction, or when the boot self-check failed)");

/* ---- accounting ------------------------------------------------------ */
enum opemu_family {
	F_POPCNT, F_CRC32, F_PCMPGTQ, F_PCMPISTRI, F_PCMPISTRM, F_PCMPESTRI,
	F_PCMPESTRM, F_PCLMULQDQ, F_AES, F_AESKEYGEN,
	F_UNHANDLED,		/* a user #UD that was not ours */
	F_COMPAT32,		/* 32-bit process: refused */
	F_SEGV,			/* operand not readable: SIGSEGV delivered */
	F_NR
};

static const char *const family_name[F_NR] = {
	"popcnt", "crc32", "pcmpgtq", "pcmpistri", "pcmpistrm", "pcmpestri",
	"pcmpestrm", "pclmulqdq", "aes", "aeskeygen", "unhandled", "compat32",
	"segv",
};

/* Which path served a trap, and why the fast one gave up when it did. */
enum opemu_path {
	P_FAST,			/* matched, resolved and executed with interrupts off */
	P_FALLBACK,		/* went through the general decoder (whatever the outcome) */
	P_FETCH_FALLBACK,	/* ...because the 16 instruction bytes were not all resident */
	P_OPERAND_FALLBACK,	/* ...because the memory operand was not resident */
	P_NR
};

static const char *const path_name[P_NR] = {
	"path_fast", "path_fallback", "path_fetch_fallback", "path_operand_fallback",
};

static DEFINE_PER_CPU(unsigned long, opemu_count[F_NR]);
static DEFINE_PER_CPU(unsigned long, opemu_path_count[P_NR]);

/* The last SITES_RING emulated sites, per CPU: no shared line, no atomics. */
#define SITES_RING 64
struct opemu_sites {
	unsigned int pos;
	struct { unsigned long ip; u8 family; } e[SITES_RING];
};
static DEFINE_PER_CPU(struct opemu_sites, opemu_sites);

/* Interrupts are off wherever this is called — the fast path never enables
 * them, and the slow path counts after disabling them again — so the
 * per-CPU pointer cannot migrate between the read and the stores. */
static __always_inline void count(enum opemu_family f, unsigned long ip)
{
	lockdep_assert_irqs_disabled();
	this_cpu_inc(opemu_count[f]);
	if (f < F_UNHANDLED) {
		struct opemu_sites *s = this_cpu_ptr(&opemu_sites);
		unsigned int i = s->pos++ & (SITES_RING - 1);

		s->e[i].ip = ip;
		s->e[i].family = f;
	}
}

static int opemu_stats_show(struct seq_file *m, void *v)
{
	int f, p, cpu;

	for (f = 0; f < F_NR; f++) {
		unsigned long sum = 0;

		for_each_possible_cpu(cpu)
			sum += per_cpu(opemu_count[f], cpu);
		seq_printf(m, "%s %lu\n", family_name[f], sum);
	}
	for (p = 0; p < P_NR; p++) {
		unsigned long sum = 0;

		for_each_possible_cpu(cpu)
			sum += per_cpu(opemu_path_count[p], cpu);
		seq_printf(m, "%s %lu\n", path_name[p], sum);
	}
	seq_printf(m, "selfcheck_ok %d\n", selfcheck_failed ? 0 : 1);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(opemu_stats);

static int opemu_sites_show(struct seq_file *m, void *v)
{
	int cpu;

	for_each_possible_cpu(cpu) {
		struct opemu_sites *s = per_cpu_ptr(&opemu_sites, cpu);
		unsigned int pos = READ_ONCE(s->pos), i;

		for (i = 1; i <= SITES_RING; i++) {
			unsigned int k = (pos + i) & (SITES_RING - 1);

			if (s->e[k].ip)
				seq_printf(m, "%d %lx %s\n", cpu, s->e[k].ip, family_name[s->e[k].family]);
		}
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(opemu_sites);

/* ---- the live XMM registers ------------------------------------------ */
/*
 * On both paths the hardware registers hold the current task's user state
 * when these run (fast: interrupts off, TIF_NEED_FPU_LOAD clear; slow:
 * under fpregs_lock_and_load()), so the operand is read from and the result
 * written to the real register — no FXSAVE/XRSTOR per trap.  The kernel is
 * compiled -mno-sse; naming %xmmN in inline asm is deliberate and the
 * compiler never allocates them itself.
 */
#define XMM_CASE_RD(n) case n: asm volatile("movdqu %%xmm" #n ", %0" : "=m" (*(u8 (*)[16])buf)); break
#define XMM_CASE_WR(n) case n: asm volatile("movdqu %0, %%xmm" #n : : "m" (*(const u8 (*)[16])buf)); break

static void xmm_read(int n, u8 buf[16])
{
	switch (n) {
	XMM_CASE_RD(0);  XMM_CASE_RD(1);  XMM_CASE_RD(2);  XMM_CASE_RD(3);
	XMM_CASE_RD(4);  XMM_CASE_RD(5);  XMM_CASE_RD(6);  XMM_CASE_RD(7);
	XMM_CASE_RD(8);  XMM_CASE_RD(9);  XMM_CASE_RD(10); XMM_CASE_RD(11);
	XMM_CASE_RD(12); XMM_CASE_RD(13); XMM_CASE_RD(14); XMM_CASE_RD(15);
	}
}

static void xmm_write(int n, const u8 buf[16])
{
	switch (n) {
	XMM_CASE_WR(0);  XMM_CASE_WR(1);  XMM_CASE_WR(2);  XMM_CASE_WR(3);
	XMM_CASE_WR(4);  XMM_CASE_WR(5);  XMM_CASE_WR(6);  XMM_CASE_WR(7);
	XMM_CASE_WR(8);  XMM_CASE_WR(9);  XMM_CASE_WR(10); XMM_CASE_WR(11);
	XMM_CASE_WR(12); XMM_CASE_WR(13); XMM_CASE_WR(14); XMM_CASE_WR(15);
	}
}

/* ---- the executor, shared by both paths ------------------------------ */

/* Is the feature this op belongs to actually missing on this CPU? */
static __always_inline bool ours(enum opemu_op op)
{
	switch (op) {
	case OPEMU_OP_POPCNT:		return emulate_popcnt;
	case OPEMU_OP_CRC32: case OPEMU_OP_PCMPGTQ: case OPEMU_OP_PCMPESTRM: case OPEMU_OP_PCMPESTRI:
	case OPEMU_OP_PCMPISTRM: case OPEMU_OP_PCMPISTRI:
					return emulate_sse42;
	case OPEMU_OP_PCLMULQDQ:	return emulate_pclmul;
	case OPEMU_OP_AESIMC: case OPEMU_OP_AESENC: case OPEMU_OP_AESENCLAST: case OPEMU_OP_AESDEC:
	case OPEMU_OP_AESDECLAST: case OPEMU_OP_AESKEYGENASSIST:
					return emulate_aes;
	default:			return false;
	}
}

/* The one source operand, already fetched by whichever path resolved it. */
struct opemu_src {
	bool from_reg;		/* XMM ops: b is the XMM register `reg`, else the 16 bytes in mem[] */
	int reg;
	u64 val;		/* POPCNT/CRC32: the (masked) integer source */
	u8 mem[16];
};

/* GPR r (0..15, encoding order) as a pointer into pt_regs. */
static __always_inline unsigned long *gpr_ptr(struct pt_regs *regs, unsigned int r)
{
	int off = pt_regs_offset(regs, (int)(r & 15));

	return (unsigned long *)((char *)regs + off);
}

static __always_inline void set_flags(struct pt_regs *regs, u64 fl)
{
	regs->flags = (regs->flags & ~(unsigned long)EMU_FL_ALL) | fl;
}

/*
 * Execute one matched-and-resolved instruction on the task's state.  The
 * caller has made the XMM registers live (see above) and read the source
 * operand; nothing here can fault or sleep.  Returns the family for the
 * counters.
 */
static __always_inline enum opemu_family opemu_execute(struct pt_regs *regs, const struct opemu_insn *d,
						       const struct opemu_src *src)
{
	switch (d->op) {
	case OPEMU_OP_POPCNT: {
		u64 fl, cnt = sse42emu_popcnt(src->val, d->src_bytes, &fl);

		insn_assign_reg(gpr_ptr(regs, d->reg), cnt, d->src_bytes);
		set_flags(regs, fl);
		return F_POPCNT;
	}
	case OPEMU_OP_CRC32: {
		/* dest is r32, or r64 with REX.W; the result is zero-extended either way */
		unsigned long *dst = gpr_ptr(regs, d->reg);

		insn_assign_reg(dst, sse42emu_crc32((u32)*dst, src->val, d->src_bytes), d->rexw ? 8 : 4);
		return F_CRC32;
	}
	default: {
		/* the XMM family: dest xmm from ModRM.reg, source xmm or m128 */
		u8 a[16], b[16], mask[16], dd[16];

		xmm_read(d->reg, a);
		if (src->from_reg)
			xmm_read(src->reg, b);
		else
			memcpy(b, src->mem, 16);

		switch (d->op) {
		case OPEMU_OP_PCMPGTQ:
			sse42emu_pcmpgtq(a, b);
			xmm_write(d->reg, a);
			return F_PCMPGTQ;
		case OPEMU_OP_PCMPISTRI: case OPEMU_OP_PCMPISTRM:
		case OPEMU_OP_PCMPESTRI: case OPEMU_OP_PCMPESTRM: {
			u32 idx = 0;
			u64 fl = 0;
			enum opemu_family fam;

			if (d->op == OPEMU_OP_PCMPESTRI || d->op == OPEMU_OP_PCMPESTRM) {
				/* lengths in (R|E)AX / (R|E)DX, SIGNED; REX.W picks the width */
				s64 la = d->rexw ? (s64)regs->ax : (s64)(s32)regs->ax;
				s64 lb = d->rexw ? (s64)regs->dx : (s64)(s32)regs->dx;

				la = clamp_t(s64, la, -64, 64);
				lb = clamp_t(s64, lb, -64, 64);
				sse42emu_pcmpestr(a, b, d->imm, (s32)la, (s32)lb, &idx, mask, &fl);
				fam = d->op == OPEMU_OP_PCMPESTRI ? F_PCMPESTRI : F_PCMPESTRM;
			} else {
				sse42emu_pcmpistr(a, b, d->imm, &idx, mask, &fl);
				fam = d->op == OPEMU_OP_PCMPISTRI ? F_PCMPISTRI : F_PCMPISTRM;
			}
			if (d->op == OPEMU_OP_PCMPISTRI || d->op == OPEMU_OP_PCMPESTRI)
				regs->cx = idx;			/* ECX, zero-extended */
			else
				xmm_write(0, mask);
			set_flags(regs, fl);
			return fam;
		}
		case OPEMU_OP_PCLMULQDQ:
			sse42emu_pclmul(dd, a, b, d->imm);
			xmm_write(d->reg, dd);
			return F_PCLMULQDQ;
		case OPEMU_OP_AESKEYGENASSIST:
			sse42emu_aeskeygen(dd, b, d->imm);
			xmm_write(d->reg, dd);
			return F_AESKEYGEN;
		default: {	/* AESIMC / AESENC / AESENCLAST / AESDEC / AESDECLAST */
			int core_op = d->op == OPEMU_OP_AESIMC ? SSE42EMU_AESIMC :
				      d->op == OPEMU_OP_AESENC ? SSE42EMU_AESENC :
				      d->op == OPEMU_OP_AESENCLAST ? SSE42EMU_AESENCLAST :
				      d->op == OPEMU_OP_AESDEC ? SSE42EMU_AESDEC : SSE42EMU_AESDECLAST;

			sse42emu_aes(a, b, core_op);
			xmm_write(d->reg, a);
			return F_AES;
		}
		}
	}
	}
}

/* ---- the fast path: interrupts off, nothing sleeps ------------------- */

/*
 * The 16 bytes at ip, or -EFAULT.  Two qword reads under pagefault_disable():
 * a byte that is not resident takes the exception-table fixup instead of the
 * fault handler (faulthandler_disabled() -> bad_area_nosemaphore -> fixup),
 * so nothing here can sleep and interrupts stay off.  masked_user_access_begin
 * clamps a pointer at USER_PTR_MAX without the LFENCE of access_ok(); RIP is
 * canonical and below it, and the guard page above keeps ip+8 in user space.
 * Reading 16 where the instruction is shorter is harmless: only `len` bytes
 * are consumed, and a fault on bytes the instruction does not have costs the
 * slow path, never a wrong result or a signal.
 */
static __always_inline int opemu_fetch16(unsigned long ip, u8 buf[16])
{
	const u64 __user *p;
	u64 w0, w1;

	pagefault_disable();
	p = masked_user_access_begin((const u64 __user *)ip);
	unsafe_get_user(w0, p, efault);
	unsafe_get_user(w1, p + 1, efault);
	user_access_end();
	pagefault_enable();
	memcpy(buf, &w0, 8);
	memcpy(buf + 8, &w1, 8);
	return 0;
efault:
	user_access_end();
	pagefault_enable();
	return -EFAULT;
}

/*
 * A source operand of exactly `size` bytes (1/2/4/8/16) at ea, the same way.
 * Exact width matters: an 8-byte read of a 1-byte operand at the end of a
 * page would fall back for nothing.  A fault here is never a SIGSEGV — the
 * page may simply not be present yet — it sends the trap to the slow path,
 * whose copy_from_user() pages in or reports the first unreadable byte.
 */
static __always_inline int opemu_read_user_fast(unsigned long ea, unsigned int size, u64 *lo, u64 *hi)
{
	const void __user *p;
	u64 a = 0, b = 0;

	pagefault_disable();
	p = masked_user_access_begin((const void __user *)ea);
	switch (size) {
	case 1:  { u8 v;  unsafe_get_user(v, (const u8 __user *)p, efault);  a = v; break; }
	case 2:  { u16 v; unsafe_get_user(v, (const u16 __user *)p, efault); a = v; break; }
	case 4:  { u32 v; unsafe_get_user(v, (const u32 __user *)p, efault); a = v; break; }
	case 8:  unsafe_get_user(a, (const u64 __user *)p, efault); break;
	default: unsafe_get_user(a, (const u64 __user *)p, efault);
		 unsafe_get_user(b, (const u64 __user *)p + 1, efault); break;
	}
	user_access_end();
	pagefault_enable();
	*lo = a;
	*hi = b;
	return 0;
efault:
	user_access_end();
	pagefault_enable();
	return -EFAULT;
}

/* Effective address of d's memory operand from the task's registers. */
static __always_inline unsigned long opemu_ea(struct pt_regs *regs, const struct opemu_insn *d)
{
	return opemu_match_ea(d, regs->ip + d->len,
			      d->base >= 0 ? *gpr_ptr(regs, d->base) : 0,
			      d->index >= 0 ? *gpr_ptr(regs, d->index) : 0);
}

/*
 * The register form of an integer source.  A byte source without REX names
 * AH/CH/DH/BH for rm 4..7 — the high byte of rAX..rBX; with any REX the same
 * bits name SPL/BPL/SIL/DIL.
 */
static __always_inline u64 opemu_gpr_source(struct pt_regs *regs, const struct opemu_insn *d)
{
	u64 v;

	if (d->src_bytes == 1 && !d->has_rex && d->rm >= 4)
		return (*gpr_ptr(regs, d->rm - 4) >> 8) & 0xff;
	v = *gpr_ptr(regs, d->rm);
	if (d->src_bytes < 8)
		v &= (1ULL << (8 * d->src_bytes)) - 1;
	return v;
}

enum opemu_fast { FAST_DONE, FAST_FALLBACK };

static __always_inline enum opemu_fast emulate_fast(struct pt_regs *regs, enum opemu_family *fam)
{
	u8 buf[16];
	struct opemu_insn d;
	struct opemu_src src;
	u64 lo, hi;

	if (opemu_fetch16(regs->ip, buf)) {
		this_cpu_inc(opemu_path_count[P_FETCH_FALLBACK]);
		return FAST_FALLBACK;
	}
	if (!opemu_match(buf, 16, &d) || !ours(d.op))
		return FAST_FALLBACK;

	if (d.src_bytes <= 8) {				/* POPCNT, CRC32 */
		if (!d.is_mem) {
			src.val = opemu_gpr_source(regs, &d);
		} else {
			if (opemu_read_user_fast(opemu_ea(regs, &d), d.src_bytes, &lo, &hi)) {
				this_cpu_inc(opemu_path_count[P_OPERAND_FALLBACK]);
				return FAST_FALLBACK;
			}
			src.val = lo;
		}
	} else {					/* the XMM family */
		/*
		 * Straight from user mode the task's FPU state is in the
		 * registers (switch_fpu_return() saw to it on the last exit);
		 * if it is not, let the slow path's fpregs_lock_and_load() do it.
		 */
		if (test_thread_flag(TIF_NEED_FPU_LOAD))
			return FAST_FALLBACK;
		if (d.is_mem) {
			if (opemu_read_user_fast(opemu_ea(regs, &d), 16, &lo, &hi)) {
				this_cpu_inc(opemu_path_count[P_OPERAND_FALLBACK]);
				return FAST_FALLBACK;
			}
			memcpy(src.mem, &lo, 8);
			memcpy(src.mem + 8, &hi, 8);
			src.from_reg = false;
		} else {
			src.from_reg = true;
			src.reg = d.rm;
		}
	}

	*fam = opemu_execute(regs, &d, &src);
	regs->ip += d.len;
	return FAST_DONE;
}

/* ---- the slow path: the general decoder, interrupts on ---------------- */

struct prefixes { bool p66, pf2, pf3, lock; };

static void scan_prefixes(const struct insn *insn, struct prefixes *p)
{
	int i;

	memset(p, 0, sizeof(*p));
	/* bytes[0..2] are the distinct legacy prefixes seen, [3] the last one */
	for (i = 0; i < 4; i++) {
		switch (insn->prefixes.bytes[i]) {
		case 0x66: p->p66 = true; break;
		case 0xf2: p->pf2 = true; break;
		case 0xf3: p->pf3 = true; break;
		case 0xf0: p->lock = true; break;
		}
	}
}

/*
 * Which of ours is this?  Strict on purpose: a mandatory prefix that is
 * missing or doubled, LOCK, or any VEX/EVEX/REX2 form is not the
 * instruction we validated, and the CPU's own #UD is the right answer.
 * Note 0F 38 F0/F1 without F2 is MOVBE, not CRC32.
 */
static enum opemu_op classify(const struct insn *insn, const struct prefixes *p)
{
	const u8 *op = insn->opcode.bytes;

	if (insn->vex_prefix.nbytes || insn->rex_prefix.nbytes == 2 || p->lock)
		return OPEMU_OP_NONE;
	if (insn->opcode.nbytes < 2 || op[0] != 0x0f)
		return OPEMU_OP_NONE;

	if (insn->opcode.nbytes == 2) {
		if (op[1] == 0xb8 && p->pf3 && !p->pf2)
			return OPEMU_OP_POPCNT;
		return OPEMU_OP_NONE;
	}

	if (op[1] == 0x38 && (op[2] == 0xf0 || op[2] == 0xf1))
		return (p->pf2 && !p->pf3) ? OPEMU_OP_CRC32 : OPEMU_OP_NONE;

	if (!p->p66 || p->pf2 || p->pf3)
		return OPEMU_OP_NONE;

	if (op[1] == 0x38) {
		switch (op[2]) {
		case 0x37: return OPEMU_OP_PCMPGTQ;
		case 0xdb: return OPEMU_OP_AESIMC;
		case 0xdc: return OPEMU_OP_AESENC;
		case 0xdd: return OPEMU_OP_AESENCLAST;
		case 0xde: return OPEMU_OP_AESDEC;
		case 0xdf: return OPEMU_OP_AESDECLAST;
		}
	} else if (op[1] == 0x3a) {
		switch (op[2]) {
		case 0x60: return OPEMU_OP_PCMPESTRM;
		case 0x61: return OPEMU_OP_PCMPESTRI;
		case 0x62: return OPEMU_OP_PCMPISTRM;
		case 0x63: return OPEMU_OP_PCMPISTRI;
		case 0x44: return OPEMU_OP_PCLMULQDQ;
		case 0xdf: return OPEMU_OP_AESKEYGENASSIST;
		}
	}
	return OPEMU_OP_NONE;
}

/* ---- the operand fault: what the real instruction's #PF would do ------ */
static void opemu_segv(struct pt_regs *regs, unsigned long addr)
{
	struct task_struct *tsk = current;

	tsk->thread.cr2		= addr;
	tsk->thread.error_code	= X86_PF_USER;
	tsk->thread.trap_nr	= X86_TRAP_PF;
	force_sig_fault(SIGSEGV, SEGV_MAPERR, (void __user *)addr);
}

/*
 * Read a source operand of `size` bytes: a GPR (via its pt_regs offset), or
 * memory.  Returns 0, -EFAULT with *fault set to the first unreadable byte,
 * or -EINVAL for an addressing form insn-eval cannot resolve.
 */
static int read_int_source(struct insn *insn, struct pt_regs *regs, unsigned int size,
			   u64 *val, unsigned long *fault)
{
	u8 modrm = insn->modrm.value;

	if (X86_MODRM_MOD(modrm) == 3) {
		int rm = X86_MODRM_RM(modrm);
		int off;

		/* the AH/CH/DH/BH form: insn-eval only knows the 64-bit numbering */
		if (size == 1 && !insn->rex_prefix.nbytes && rm >= 4) {
			off = pt_regs_offset(regs, rm - 4);
			if (off < 0)
				return -EINVAL;
			*val = (regs_get_register(regs, off) >> 8) & 0xff;
			return 0;
		}
		off = insn_get_modrm_rm_off(insn, regs);
		if (off < 0)
			return -EINVAL;
		*val = regs_get_register(regs, off);
		if (size < 8)
			*val &= (1ULL << (8 * size)) - 1;
		return 0;
	} else {
		void __user *addr = insn_get_addr_ref(insn, regs);
		u64 v = 0;
		unsigned long left;

		if (addr == (void __user *)-1L)
			return -EINVAL;
		left = copy_from_user(&v, addr, size);
		if (left) {
			*fault = (unsigned long)addr + (size - left);
			return -EFAULT;
		}
		*val = v;
		return 0;
	}
}

/* A 128-bit source: XMM register, or m128. */
static int read_xmm_source(struct insn *insn, struct pt_regs *regs, struct opemu_src *src,
			   unsigned long *fault)
{
	u8 modrm = insn->modrm.value;

	if (X86_MODRM_MOD(modrm) == 3) {
		src->from_reg = true;
		src->reg = X86_MODRM_RM(modrm) | (X86_REX_B(insn->rex_prefix.value) ? 8 : 0);
		return 0;
	} else {
		void __user *addr = insn_get_addr_ref(insn, regs);
		unsigned long left;

		if (addr == (void __user *)-1L)
			return -EINVAL;
		left = copy_from_user(src->mem, addr, 16);
		if (left) {
			*fault = (unsigned long)addr + (16 - left);
			return -EFAULT;
		}
		src->from_reg = false;
		return 0;
	}
}

enum opemu_slow { SLOW_DONE, SLOW_SEGV, SLOW_UNHANDLED };

/* Interrupts are ON here; may sleep in the user accesses. */
static enum opemu_slow emulate_slow(struct pt_regs *regs, enum opemu_family *fam)
{
	unsigned char buf[MAX_INSN_SIZE];
	struct insn insn;
	struct prefixes p;
	struct opemu_insn d;
	struct opemu_src src;
	enum opemu_op op;
	int nr_copied, ret;
	unsigned long fault = 0;
	u8 rex, modrm;

	nr_copied = insn_fetch_from_user(regs, buf);
	if (nr_copied <= 0)
		return SLOW_UNHANDLED;	/* text vanished under us: let SIGILL happen */

	/*
	 * Cheap pre-classifier: skip the legacy prefixes and bail unless the
	 * opcode map is one of ours, so a ud2 or a genuinely bad instruction
	 * never pays for a full decode.
	 */
	{
		int i = 0;

		while (i < nr_copied && (buf[i] == 0x66 || buf[i] == 0xf2 || buf[i] == 0xf3 ||
					 buf[i] == 0xf0 || buf[i] == 0x67 || buf[i] == 0x2e ||
					 buf[i] == 0x36 || buf[i] == 0x3e || buf[i] == 0x26 ||
					 buf[i] == 0x64 || buf[i] == 0x65))
			i++;
		if (i < nr_copied && (buf[i] & 0xf0) == 0x40)	/* REX */
			i++;
		if (i + 1 >= nr_copied || buf[i] != 0x0f ||
		    (buf[i + 1] != 0xb8 && buf[i + 1] != 0x38 && buf[i + 1] != 0x3a))
			goto unhandled;
	}

	if (!insn_decode_from_regs(&insn, regs, buf, nr_copied))
		return SLOW_UNHANDLED;

	scan_prefixes(&insn, &p);
	op = classify(&insn, &p);
	if (op == OPEMU_OP_NONE || !ours(op))
		goto unhandled;

	rex = insn.rex_prefix.value;
	modrm = insn.modrm.value;
	memset(&d, 0, sizeof(d));
	d.op = op;
	d.len = insn.length;
	d.reg = X86_MODRM_REG(modrm) | (X86_REX_R(rex) ? 8 : 0);
	d.rm = X86_MODRM_RM(modrm) | (X86_REX_B(rex) ? 8 : 0);
	d.is_mem = X86_MODRM_MOD(modrm) != 3;
	d.rexw = X86_REX_W(rex) ? 1 : 0;
	d.has_rex = insn.rex_prefix.nbytes != 0;

	switch (op) {
	case OPEMU_OP_POPCNT:
		d.src_bytes = insn.opnd_bytes;			/* 2, 4 or 8 */
		break;
	case OPEMU_OP_CRC32:
		/* F0: r/m8 source; F1: r/m16/32/64 by 66/REX.W */
		d.src_bytes = insn.opcode.bytes[2] == 0xf0 ? 1 : insn.opnd_bytes;
		break;
	default:
		d.src_bytes = 16;
		if (insn.opcode.bytes[1] == 0x3a) {
			if (insn.immediate.nbytes != 1)
				goto unhandled;
			d.has_imm = 1;
			d.imm = (u8)insn.immediate.value;
		}
		break;
	}

	if (d.src_bytes <= 8) {
		ret = read_int_source(&insn, regs, d.src_bytes, &src.val, &fault);
		if (ret)
			goto fault_or_refuse;
		if (!insn_get_modrm_reg_ptr(&insn, regs))
			goto unhandled;
		*fam = opemu_execute(regs, &d, &src);
	} else {
		ret = read_xmm_source(&insn, regs, &src, &fault);
		if (ret)
			goto fault_or_refuse;
		/* From here to fpregs_unlock(): no sleeping, no user access. */
		fpregs_lock_and_load();
		*fam = opemu_execute(regs, &d, &src);
		fpregs_unlock();
	}
	regs->ip += insn.length;
	return SLOW_DONE;

fault_or_refuse:
	if (ret == -EFAULT) {
		opemu_segv(regs, fault);
		return SLOW_SEGV;	/* the SIGSEGV is the outcome; ip stays */
	}
unhandled:
	if (trace)
		pr_info_ratelimited("unhandled user #UD at %lx: %*ph\n",
				    regs->ip, nr_copied > 0 ? nr_copied : 0, buf);
	return SLOW_UNHANDLED;
}

/* ---- the trap ------------------------------------------------------------ */
bool opemu_fixup_ud(struct pt_regs *regs)
{
	enum opemu_family fam;
	enum opemu_slow out;

	if (!static_branch_unlikely(&opemu_enabled))
		return false;
	if (!user_64bit_mode(regs)) {
		count(F_COMPAT32, regs->ip);
		return false;
	}

	if (emulate_fast(regs, &fam) == FAST_DONE) {
		this_cpu_inc(opemu_path_count[P_FAST]);
		count(fam, regs->ip);
		return true;
	}

	/* A user process that ran with IF clear (iopl 3 + cli) gets no help. */
	if (!(regs->flags & X86_EFLAGS_IF))
		return false;

	local_irq_enable();
	out = emulate_slow(regs, &fam);
	local_irq_disable();

	this_cpu_inc(opemu_path_count[P_FALLBACK]);
	switch (out) {
	case SLOW_DONE:
		count(fam, regs->ip);
		return true;
	case SLOW_SEGV:
		count(F_SEGV, regs->ip);
		return true;
	default:
		count(F_UNHANDLED, regs->ip);
		return false;
	}
}

/* ---- boot self-check: the matcher against the kernel's own decoder ---- */
/*
 * Every canned vector, both ways: what opemu_match() says must be what
 * insn_decode_from_regs() + classify() + insn_get_modrm_*_off() +
 * insn_get_addr_ref() say — length, opcode, operand size, REX.W, both
 * registers, the effective address (RIP-relative through regs->ip +
 * length), the immediate — and a vector marked "fallback" must be
 * refused.  On any disagreement the emulator stays off: a matcher that
 * reads a field one bit wrong would emulate the wrong register silently.
 *
 * The decoder runs on a fake pt_regs in task context: a GDT selector for
 * CS (no LDT mutex), distinct GPR values so a wrong register is
 * attributable, and no FS/GS-based vector asks for an address (those
 * bases come from MSRs and the vectors only assert refusal).
 */
static int __init opemu_selfcheck(void)
{
	struct pt_regs regs;
	unsigned char buf[MAX_INSN_SIZE];
	unsigned long *gpr[16];
	int i, bad = 0;

	memset(&regs, 0, sizeof(regs));
	regs.cs = __USER_CS;
	regs.ss = __USER_DS;
	regs.flags = X86_EFLAGS_IF;
	regs.ip = 0x400000;
	for (i = 0; i < 16; i++) {
		gpr[i] = gpr_ptr(&regs, i);
		*gpr[i] = 0x1000UL * (i + 1);
	}

	for (i = 0; i < OPEMU_MATCH_NR_VECTORS; i++) {
		const struct opemu_match_vector *v = &opemu_match_vectors[i];
		struct opemu_insn d;
		struct insn insn;
		struct prefixes p;
		const char *why = NULL;
		int got;

		memset(buf, 0xcc, sizeof(buf));
		memcpy(buf, v->bytes, v->n);
		got = opemu_match(v->bytes, v->n, &d);

		if (!v->want_len) {
			if (got)
				why = "matched a fallback vector";
		} else if (got != v->want_len) {
			why = "length differs from the vector's";
		} else if (!insn_decode_from_regs(&insn, &regs, buf, MAX_INSN_SIZE)) {
			why = "insn_decode_from_regs failed";
		} else {
			scan_prefixes(&insn, &p);
			if (insn.length != d.len)
				why = "length";
			else if (classify(&insn, &p) != d.op || d.op != v->want_op)
				why = "opcode";
			else if ((X86_REX_W(insn.rex_prefix.value) ? 1 : 0) != d.rexw)
				why = "REX.W";
			else if ((insn.rex_prefix.nbytes != 0) != (d.has_rex != 0))
				why = "REX presence";
			else if (d.op == OPEMU_OP_POPCNT && insn.opnd_bytes != d.src_bytes)
				why = "operand size";
			else if (d.op == OPEMU_OP_CRC32 &&
				 (insn.opcode.bytes[2] == 0xf0 ? 1 : insn.opnd_bytes) != d.src_bytes)
				why = "crc32 source size";
			else if (insn_get_modrm_reg_off(&insn, &regs) != pt_regs_offset(&regs, d.reg))
				why = "destination register";
			else if (!d.is_mem && insn_get_modrm_rm_off(&insn, &regs) != pt_regs_offset(&regs, d.rm))
				why = "source register";
			else if (d.is_mem && (unsigned long)insn_get_addr_ref(&insn, &regs) != opemu_ea(&regs, &d))
				why = "effective address";
			else if (d.has_imm && (insn.immediate.nbytes != 1 || (u8)insn.immediate.value != d.imm))
				why = "immediate";
			else if (!d.has_imm && insn.immediate.nbytes != 0)
				why = "an immediate the matcher did not see";
		}
		if (why) {
			pr_err("self-check: %s (%s): %*ph\n", why, v->what, v->n, v->bytes);
			bad++;
		}
	}
	return bad;
}

/* ---- init ---------------------------------------------------------------- */
static int __init opemu_init(void)
{
	struct dentry *dir;
	int bad;

	emulate_popcnt = !boot_cpu_has(X86_FEATURE_POPCNT);
	emulate_sse42  = !boot_cpu_has(X86_FEATURE_XMM4_2);
	emulate_aes    = !boot_cpu_has(X86_FEATURE_AES);
	emulate_pclmul = !boot_cpu_has(X86_FEATURE_PCLMULQDQ);
	emulate_any = emulate_popcnt || emulate_sse42 || emulate_aes || emulate_pclmul;

	sse42emu_core_init();		/* the AES tables, never inside a trap */

	bad = opemu_selfcheck();
	if (bad) {
		selfcheck_failed = true;
		pr_err("self-check: %d of %d vectors disagree with the kernel's decoder — emulation stays OFF\n",
		       bad, OPEMU_MATCH_NR_VECTORS);
	} else {
		pr_info("self-check: %d vectors agree with the kernel's decoder\n", OPEMU_MATCH_NR_VECTORS);
	}

	dir = debugfs_create_dir("opemu", NULL);
	debugfs_create_file("stats", 0444, dir, NULL, &opemu_stats_fops);
	debugfs_create_file("sites", 0444, dir, NULL, &opemu_sites_fops);

	initialized = true;
	apply_enable();

	if (!emulate_any)
		pr_info("CPU has SSE4.2, POPCNT, AES-NI and PCLMULQDQ: nothing to emulate\n");
	else
		pr_info("emulating for user space:%s%s%s%s%s\n",
			emulate_popcnt ? " popcnt" : "",
			emulate_sse42  ? " sse4.2" : "",
			emulate_aes    ? " aes-ni" : "",
			emulate_pclmul ? " pclmulqdq" : "",
			static_key_enabled(&opemu_enabled) ? "" : " (disabled)");
	return 0;
}
late_initcall(opemu_init);
