// SPDX-License-Identifier: GPL-2.0
/*
 * opemu — emulate, from the #UD trap, the instructions a Penryn-class CPU
 * lacks: SSE4.2 (POPCNT, CRC32, PCMPGTQ, PCMPxSTRx), AES-NI and PCLMULQDQ.
 *
 * Modelled on umip.c: this is the kernel's existing way of executing a
 * user instruction on the CPU's behalf — fetch it with insn_fetch_from_user(),
 * decode it with insn_decode_from_regs(), resolve operands with the insn-eval
 * helpers, do the work, advance regs->ip.  It sits in exc_invalid_op() where
 * handle_invalid_op() would otherwise raise SIGILL, and runs with interrupts
 * enabled so a memory operand can be paged in like any other user access.
 *
 * Why in the kernel rather than a SIGILL handler in the process: a signal
 * handler cannot run in a thread that has the signal blocked — the kernel
 * forces a blocked synchronous signal to SIG_DFL and the process dies with
 * the handler installed and never called — and it costs a signal frame plus
 * an rt_sigreturn per instruction.  Here nothing is delivered to userspace
 * at all.
 *
 * Invariants:
 *   - user memory is only ever READ, through copy_from_user(), and only the
 *     instruction bytes (<= 15) and one source operand (<= 16);
 *   - the only state written is the faulting task's pt_regs GPRs, its six
 *     arithmetic flags (CF PF AF ZF SF OF) and its XMM registers;
 *   - kernel-mode #UD (WARN/BUG ud2) never reaches this code;
 *   - a #UD this code cannot own is left exactly as it was (SIGILL), and a
 *     fault on the operand becomes the SIGSEGV the real instruction's #PF
 *     would have been — never a SIGILL.
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
#include <linux/uaccess.h>
#include <linux/ratelimit.h>
#include <linux/atomic.h>
#include <asm/ptrace.h>
#include <asm/insn.h>
#include <asm/insn-eval.h>
#include <asm/fpu/api.h>
#include <asm/cpufeature.h>
#include <asm/trapnr.h>
#include <asm/trap_pf.h>
#include <asm/processor-flags.h>
#include <asm/opemu.h>

#include "opemu-core.h"

/* ---- knobs ----------------------------------------------------------- */
static DEFINE_STATIC_KEY_FALSE(opemu_enabled);

static bool enable_param = true;      /* opemu.enable= on the command line */
static bool initialized;              /* late_initcall ran */
static bool emulate_popcnt, emulate_sse42, emulate_aes, emulate_pclmul, emulate_any;
static bool trace;                    /* opemu.trace=1: log refused sites */
module_param(trace, bool, 0644);
MODULE_PARM_DESC(trace, "log each user #UD this emulator refuses (ratelimited)");

static void apply_enable(void)
{
	if (enable_param && emulate_any)
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
MODULE_PARM_DESC(enable, "emulate for user space (default 1; forced 0 when the CPU has every instruction)");

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

static DEFINE_PER_CPU(unsigned long, opemu_count[F_NR]);

#define SITES_RING 64
static struct { unsigned long ip; u8 family; } sites[SITES_RING];
static atomic_t sites_pos;

static inline void count(enum opemu_family f, unsigned long ip)
{
	this_cpu_inc(opemu_count[f]);
	if (f < F_UNHANDLED) {
		unsigned int i = (unsigned int)atomic_inc_return(&sites_pos) & (SITES_RING - 1);

		sites[i].ip = ip;
		sites[i].family = f;
	}
}

static int opemu_stats_show(struct seq_file *m, void *v)
{
	int f, cpu;

	for (f = 0; f < F_NR; f++) {
		unsigned long sum = 0;

		for_each_possible_cpu(cpu)
			sum += per_cpu(opemu_count[f], cpu);
		seq_printf(m, "%s %lu\n", family_name[f], sum);
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(opemu_stats);

static int opemu_sites_show(struct seq_file *m, void *v)
{
	unsigned int pos = (unsigned int)atomic_read(&sites_pos), i;

	for (i = 1; i <= SITES_RING; i++) {
		unsigned int k = (pos + i) & (SITES_RING - 1);

		if (sites[k].ip)
			seq_printf(m, "%lx %s\n", sites[k].ip, family_name[sites[k].family]);
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(opemu_sites);

/* ---- the live XMM registers ------------------------------------------ */
/*
 * Under fpregs_lock_and_load() the hardware registers hold the current
 * task's user state, so the operand is read from and the result written to
 * the real register — no FXSAVE/XRSTOR per trap.  The kernel is compiled
 * -mno-sse; naming %xmmN in inline asm is deliberate and the compiler never
 * allocates them itself.
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

/* ---- classification -------------------------------------------------- */
enum opemu_op {
	OP_NONE, OP_POPCNT, OP_CRC32, OP_PCMPGTQ, OP_PCMPESTRM, OP_PCMPESTRI,
	OP_PCMPISTRM, OP_PCMPISTRI, OP_PCLMULQDQ, OP_AESIMC, OP_AESENC,
	OP_AESENCLAST, OP_AESDEC, OP_AESDECLAST, OP_AESKEYGENASSIST,
};

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
		return OP_NONE;
	if (insn->opcode.nbytes < 2 || op[0] != 0x0f)
		return OP_NONE;

	if (insn->opcode.nbytes == 2) {
		if (op[1] == 0xb8 && p->pf3 && !p->pf2)
			return OP_POPCNT;
		return OP_NONE;
	}

	if (op[1] == 0x38 && (op[2] == 0xf0 || op[2] == 0xf1))
		return (p->pf2 && !p->pf3) ? OP_CRC32 : OP_NONE;

	if (!p->p66 || p->pf2 || p->pf3)
		return OP_NONE;

	if (op[1] == 0x38) {
		switch (op[2]) {
		case 0x37: return OP_PCMPGTQ;
		case 0xdb: return OP_AESIMC;
		case 0xdc: return OP_AESENC;
		case 0xdd: return OP_AESENCLAST;
		case 0xde: return OP_AESDEC;
		case 0xdf: return OP_AESDECLAST;
		}
	} else if (op[1] == 0x3a) {
		switch (op[2]) {
		case 0x60: return OP_PCMPESTRM;
		case 0x61: return OP_PCMPESTRI;
		case 0x62: return OP_PCMPISTRM;
		case 0x63: return OP_PCMPISTRI;
		case 0x44: return OP_PCLMULQDQ;
		case 0xdf: return OP_AESKEYGENASSIST;
		}
	}
	return OP_NONE;
}

/* Is the feature this op belongs to actually missing on this CPU? */
static bool ours(enum opemu_op op)
{
	switch (op) {
	case OP_POPCNT:		return emulate_popcnt;
	case OP_CRC32: case OP_PCMPGTQ: case OP_PCMPESTRM: case OP_PCMPESTRI:
	case OP_PCMPISTRM: case OP_PCMPISTRI:
				return emulate_sse42;
	case OP_PCLMULQDQ:	return emulate_pclmul;
	case OP_AESIMC: case OP_AESENC: case OP_AESENCLAST: case OP_AESDEC:
	case OP_AESDECLAST: case OP_AESKEYGENASSIST:
				return emulate_aes;
	default:		return false;
	}
}

/* ---- the operand fault: what the real instruction's #PF would do ------ */
static void opemu_segv(struct pt_regs *regs, unsigned long addr)
{
	struct task_struct *tsk = current;

	tsk->thread.cr2		= addr;
	tsk->thread.error_code	= X86_PF_USER;
	tsk->thread.trap_nr	= X86_TRAP_PF;
	force_sig_fault(SIGSEGV, SEGV_MAPERR, (void __user *)addr);
	count(F_SEGV, regs->ip);
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

		/*
		 * Byte source without a REX prefix: rm 4..7 name AH/CH/DH/BH,
		 * the high byte of rAX/rCX/rDX/rBX.  insn-eval only knows the
		 * 64-bit register numbering, so this one form is by hand; with
		 * any REX the same rm bits name SPL/BPL/SIL/DIL, the low byte.
		 */
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
static int read_xmm_source(struct insn *insn, struct pt_regs *regs, bool *from_reg, int *reg,
			   u8 mem[16], unsigned long *fault)
{
	u8 modrm = insn->modrm.value;

	if (X86_MODRM_MOD(modrm) == 3) {
		*from_reg = true;
		*reg = X86_MODRM_RM(modrm) | (X86_REX_B(insn->rex_prefix.value) ? 8 : 0);
		return 0;
	} else {
		void __user *addr = insn_get_addr_ref(insn, regs);
		unsigned long left;

		if (addr == (void __user *)-1L)
			return -EINVAL;
		left = copy_from_user(mem, addr, 16);
		if (left) {
			*fault = (unsigned long)addr + (16 - left);
			return -EFAULT;
		}
		*from_reg = false;
		return 0;
	}
}

static inline void set_flags(struct pt_regs *regs, u64 fl)
{
	regs->flags = (regs->flags & ~(unsigned long)EMU_FL_ALL) | fl;
}

/* ---- the emulation proper ---------------------------------------------- */
static bool emulate(struct pt_regs *regs)
{
	unsigned char buf[MAX_INSN_SIZE];
	struct insn insn;
	struct prefixes p;
	enum opemu_op op;
	int nr_copied, ret;
	unsigned long fault = 0;
	u8 rex;

	nr_copied = insn_fetch_from_user(regs, buf);
	if (nr_copied <= 0)
		return false;	/* text vanished under us: let SIGILL happen */

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
		return false;

	scan_prefixes(&insn, &p);
	op = classify(&insn, &p);
	if (op == OP_NONE || !ours(op))
		goto unhandled;

	rex = insn.rex_prefix.value;

	switch (op) {
	case OP_POPCNT: {
		unsigned int bytes = insn.opnd_bytes;	/* 2, 4 or 8 */
		unsigned long *dst;
		u64 src, fl, cnt;

		ret = read_int_source(&insn, regs, bytes, &src, &fault);
		if (ret)
			goto fault_or_refuse;
		dst = insn_get_modrm_reg_ptr(&insn, regs);
		if (!dst)
			goto unhandled;
		cnt = sse42emu_popcnt(src, bytes, &fl);
		insn_assign_reg(dst, cnt, bytes);
		set_flags(regs, fl);
		count(F_POPCNT, regs->ip);
		break;
	}
	case OP_CRC32: {
		/* F0: r/m8 source; F1: r/m16/32/64 by 66/REX.W.  Dest is r32,
		 * or r64 with REX.W, and the result is zero-extended either way. */
		unsigned int sbytes = insn.opcode.bytes[2] == 0xf0 ? 1 : insn.opnd_bytes;
		unsigned int dbytes = X86_REX_W(rex) ? 8 : 4;
		unsigned long *dst;
		u64 src;

		ret = read_int_source(&insn, regs, sbytes, &src, &fault);
		if (ret)
			goto fault_or_refuse;
		dst = insn_get_modrm_reg_ptr(&insn, regs);
		if (!dst)
			goto unhandled;
		insn_assign_reg(dst, sse42emu_crc32((u32)*dst, src, sbytes), dbytes);
		count(F_CRC32, regs->ip);
		break;
	}
	default: {
		/* The XMM family: dest xmm from ModRM.reg, source xmm or m128. */
		int dreg = X86_MODRM_REG(insn.modrm.value) | (X86_REX_R(rex) ? 8 : 0);
		int sreg = 0;
		bool from_reg;
		u8 a[16], b[16], mask[16];
		u8 imm = 0;
		enum opemu_family fam;

		if ((op >= OP_PCMPESTRM && op <= OP_PCMPISTRI) || op == OP_PCLMULQDQ ||
		    op == OP_AESKEYGENASSIST) {
			if (insn.immediate.nbytes != 1)
				goto unhandled;
			imm = (u8)insn.immediate.value;
		}
		ret = read_xmm_source(&insn, regs, &from_reg, &sreg, b, &fault);
		if (ret)
			goto fault_or_refuse;

		/* From here to fpregs_unlock(): no sleeping, no user access. */
		fpregs_lock_and_load();
		xmm_read(dreg, a);
		if (from_reg)
			xmm_read(sreg, b);

		switch (op) {
		case OP_PCMPGTQ:
			sse42emu_pcmpgtq(a, b);
			xmm_write(dreg, a);
			fam = F_PCMPGTQ;
			break;
		case OP_PCMPISTRI: case OP_PCMPISTRM:
		case OP_PCMPESTRI: case OP_PCMPESTRM: {
			u32 idx = 0;
			u64 fl = 0;

			if (op == OP_PCMPESTRI || op == OP_PCMPESTRM) {
				/* lengths in (R|E)AX / (R|E)DX, SIGNED; REX.W picks the width */
				s64 la = X86_REX_W(rex) ? (s64)regs->ax : (s64)(s32)regs->ax;
				s64 lb = X86_REX_W(rex) ? (s64)regs->dx : (s64)(s32)regs->dx;

				la = clamp_t(s64, la, -64, 64);
				lb = clamp_t(s64, lb, -64, 64);
				sse42emu_pcmpestr(a, b, imm, (s32)la, (s32)lb, &idx, mask, &fl);
				fam = op == OP_PCMPESTRI ? F_PCMPESTRI : F_PCMPESTRM;
			} else {
				sse42emu_pcmpistr(a, b, imm, &idx, mask, &fl);
				fam = op == OP_PCMPISTRI ? F_PCMPISTRI : F_PCMPISTRM;
			}
			if (op == OP_PCMPISTRI || op == OP_PCMPESTRI)
				regs->cx = idx;			/* ECX, zero-extended */
			else
				xmm_write(0, mask);
			set_flags(regs, fl);
			break;
		}
		case OP_PCLMULQDQ: {
			u8 d[16];

			sse42emu_pclmul(d, a, b, imm);
			xmm_write(dreg, d);
			fam = F_PCLMULQDQ;
			break;
		}
		case OP_AESKEYGENASSIST: {
			u8 d[16];

			sse42emu_aeskeygen(d, b, imm);
			xmm_write(dreg, d);
			fam = F_AESKEYGEN;
			break;
		}
		default: {	/* AESIMC / AESENC / AESENCLAST / AESDEC / AESDECLAST */
			int core_op = op == OP_AESIMC ? SSE42EMU_AESIMC :
				      op == OP_AESENC ? SSE42EMU_AESENC :
				      op == OP_AESENCLAST ? SSE42EMU_AESENCLAST :
				      op == OP_AESDEC ? SSE42EMU_AESDEC : SSE42EMU_AESDECLAST;

			sse42emu_aes(a, b, core_op);
			xmm_write(dreg, a);
			fam = F_AES;
			break;
		}
		}
		fpregs_unlock();
		count(fam, regs->ip);
		break;
	}
	}

	regs->ip += insn.length;
	return true;

fault_or_refuse:
	if (ret == -EFAULT) {
		opemu_segv(regs, fault);
		return true;		/* the SIGSEGV is the outcome; ip stays */
	}
unhandled:
	count(F_UNHANDLED, regs->ip);
	if (trace)
		pr_info_ratelimited("unhandled user #UD at %lx: %*ph\n",
				    regs->ip, nr_copied > 0 ? nr_copied : 0, buf);
	return false;
}

bool opemu_fixup_ud(struct pt_regs *regs)
{
	bool handled;

	if (!static_branch_unlikely(&opemu_enabled))
		return false;
	if (!user_64bit_mode(regs)) {
		count(F_COMPAT32, regs->ip);
		return false;
	}
	/* A user process that ran with IF clear (iopl 3 + cli) gets no help. */
	if (!(regs->flags & X86_EFLAGS_IF))
		return false;

	local_irq_enable();
	handled = emulate(regs);
	local_irq_disable();
	return handled;
}

/* ---- init ---------------------------------------------------------------- */
static int __init opemu_init(void)
{
	struct dentry *dir;

	emulate_popcnt = !boot_cpu_has(X86_FEATURE_POPCNT);
	emulate_sse42  = !boot_cpu_has(X86_FEATURE_XMM4_2);
	emulate_aes    = !boot_cpu_has(X86_FEATURE_AES);
	emulate_pclmul = !boot_cpu_has(X86_FEATURE_PCLMULQDQ);
	emulate_any = emulate_popcnt || emulate_sse42 || emulate_aes || emulate_pclmul;

	sse42emu_core_init();		/* the AES tables, never inside a trap */

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
			static_key_enabled(&opemu_enabled) ? "" : " (disabled by opemu.enable=0)");
	return 0;
}
late_initcall(opemu_init);
