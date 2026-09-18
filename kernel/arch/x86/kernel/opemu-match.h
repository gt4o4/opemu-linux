/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/*
 * opemu-match.h — the fixed grammar of the instructions opemu emulates, as a
 * matcher on raw bytes.  Header-only and freestanding: arch/x86/kernel/opemu.c
 * includes it for the kernel's fast path, and tests/test-match.c includes the
 * same file to compare it with Zydis over millions of encodings.
 *
 *   [66|F2|F3]{0,4} [REX] 0F ( B8 | 38 xx | 3A xx ) ModRM [SIB] [disp8|disp32] [imm8]
 *
 * It answers only "one of ours, in a form I can resolve, and here is every
 * field" or 0 = "let the general decoder look".  It never says "not an
 * instruction": refusing (SIGILL) stays with classify() in opemu.c, on the
 * fallback path behind insn_decode_from_regs(), so a #UD the matcher does not
 * recognise costs the slow path and nothing else.  Handed to that fallback
 * on purpose: segment overrides (64/65 select the FS/GS base; 26/2E/36/3E are
 * ignored in 64-bit mode but kept out to keep the grammar small), the 67
 * address-size override, LOCK, anything VEX/EVEX/REX2 (C4 C5 62 D5 are not
 * prefixes below and fall out by themselves), more than four prefix bytes,
 * F2 and F3 together (hardware takes the last; the general path refuses the
 * pair, and the differential test knows this one exception), a REX that is
 * not immediately followed by 0F, and a buffer too short for the form.
 */
#ifndef OPEMU_MATCH_H
#define OPEMU_MATCH_H

#ifdef __KERNEL__
# include <linux/types.h>
# define OPEMU_INLINE static __always_inline
#else
# include <stdint.h>
# define OPEMU_INLINE static inline
#endif

enum opemu_op {
	OPEMU_OP_NONE,
	OPEMU_OP_POPCNT, OPEMU_OP_CRC32, OPEMU_OP_PCMPGTQ,
	OPEMU_OP_PCMPESTRM, OPEMU_OP_PCMPESTRI, OPEMU_OP_PCMPISTRM, OPEMU_OP_PCMPISTRI,
	OPEMU_OP_PCLMULQDQ,
	OPEMU_OP_AESIMC, OPEMU_OP_AESENC, OPEMU_OP_AESENCLAST, OPEMU_OP_AESDEC,
	OPEMU_OP_AESDECLAST, OPEMU_OP_AESKEYGENASSIST,
};

#define OPEMU_REG_NONE	(-1)
#define OPEMU_REG_RIP	(-2)	/* as base only: EA = address of the NEXT instruction + disp */

struct opemu_insn {
	uint8_t op;		/* enum opemu_op; never OPEMU_OP_NONE on a match */
	uint8_t len;		/* the whole instruction, imm8 included */
	uint8_t reg;		/* ModRM.reg | REX.R<<3: the DESTINATION (a GPR for POPCNT/CRC32, an XMM for the rest) */
	uint8_t rm;		/* ModRM.rm | REX.B<<3: the register source; meaningful only when !is_mem */
	uint8_t is_mem;		/* ModRM.mod != 3: the source is memory (base/index/scale/disp) */
	uint8_t src_bytes;	/* POPCNT 2/4/8; CRC32 1 (F0) or 2/4/8 (F1); the XMM ops 16 */
	uint8_t rexw;		/* REX.W: CRC32 writes r64, PCMPESTRx read 64-bit lengths */
	uint8_t has_rex;	/* ANY REX byte, 0x40 included: rm 4..7 of a byte source then name
				 * SPL/BPL/SIL/DIL instead of AH/CH/DH/BH */
	uint8_t has_imm;	/* the 0F 3A forms carry an imm8 */
	uint8_t imm;
	int8_t	base;		/* 0..15, OPEMU_REG_NONE, or OPEMU_REG_RIP */
	int8_t	index;		/* 0..15 or OPEMU_REG_NONE */
	uint8_t scale;		/* the index shift, 0..3 */
	int32_t disp;		/* sign-extended; 0 when the form has none */
};

/*
 * Match the bytes at b[0..n).  Returns the instruction length (> 0) with *d
 * filled in, or 0 when the bytes are not one of ours in a form this grammar
 * resolves — including "too short": every field the form needs must be
 * inside the n bytes given, nothing past them is read.
 */
OPEMU_INLINE int opemu_match(const uint8_t *b, int n, struct opemu_insn *d)
{
	int i = 0, np = 0, dispsz = 0;
	uint8_t p66 = 0, pf2 = 0, pf3 = 0, rex = 0, has_rex = 0, op2, op3, m, mod, rm;

	/* legacy prefixes: only the three that select among our opcodes */
	for (;;) {
		if (i >= n)
			return 0;
		if (b[i] == 0x66)
			p66 = 1;
		else if (b[i] == 0xf2)
			pf2 = 1;
		else if (b[i] == 0xf3)
			pf3 = 1;
		else
			break;
		if (++np > 4)
			return 0;
		i++;
	}
	if (pf2 && pf3)
		return 0;

	if ((b[i] & 0xf0) == 0x40) {		/* REX: only right before the opcode */
		rex = b[i];
		has_rex = 1;
		if (++i >= n)
			return 0;
	}
	if (b[i] != 0x0f || i + 1 >= n)
		return 0;
	op2 = b[i + 1];
	i += 2;

	d->has_imm = 0;
	if (op2 == 0xb8) {
		if (!pf3 || pf2)
			return 0;
		d->op = OPEMU_OP_POPCNT;
		d->src_bytes = (rex & 8) ? 8 : p66 ? 2 : 4;
	} else if (op2 == 0x38 || op2 == 0x3a) {
		if (i >= n)
			return 0;
		op3 = b[i++];
		if (op2 == 0x38 && (op3 == 0xf0 || op3 == 0xf1)) {
			if (!pf2 || pf3)		/* without F2 this is MOVBE */
				return 0;
			d->op = OPEMU_OP_CRC32;
			d->src_bytes = op3 == 0xf0 ? 1 : (rex & 8) ? 8 : p66 ? 2 : 4;
		} else {
			if (!p66 || pf2 || pf3)
				return 0;
			d->src_bytes = 16;
			if (op2 == 0x38) {
				switch (op3) {
				case 0x37: d->op = OPEMU_OP_PCMPGTQ; break;
				case 0xdb: d->op = OPEMU_OP_AESIMC; break;
				case 0xdc: d->op = OPEMU_OP_AESENC; break;
				case 0xdd: d->op = OPEMU_OP_AESENCLAST; break;
				case 0xde: d->op = OPEMU_OP_AESDEC; break;
				case 0xdf: d->op = OPEMU_OP_AESDECLAST; break;
				default: return 0;
				}
			} else {
				switch (op3) {
				case 0x60: d->op = OPEMU_OP_PCMPESTRM; break;
				case 0x61: d->op = OPEMU_OP_PCMPESTRI; break;
				case 0x62: d->op = OPEMU_OP_PCMPISTRM; break;
				case 0x63: d->op = OPEMU_OP_PCMPISTRI; break;
				case 0x44: d->op = OPEMU_OP_PCLMULQDQ; break;
				case 0xdf: d->op = OPEMU_OP_AESKEYGENASSIST; break;
				default: return 0;
				}
				d->has_imm = 1;
			}
		}
	} else {
		return 0;
	}

	/* ModRM */
	if (i >= n)
		return 0;
	m = b[i++];
	mod = m >> 6;
	rm = m & 7;
	d->reg = ((m >> 3) & 7) | ((rex & 4) << 1);
	d->rexw = (rex >> 3) & 1;
	d->has_rex = has_rex;
	d->rm = 0;
	d->base = OPEMU_REG_NONE;
	d->index = OPEMU_REG_NONE;
	d->scale = 0;
	d->disp = 0;

	if (mod == 3) {
		d->is_mem = 0;
		d->rm = rm | ((rex & 1) << 3);
	} else {
		d->is_mem = 1;
		dispsz = mod == 1 ? 1 : mod == 2 ? 4 : 0;
		if (rm == 4) {				/* SIB */
			uint8_t sib;
			int idx;

			if (i >= n)
				return 0;
			sib = b[i++];
			d->scale = sib >> 6;
			idx = ((sib >> 3) & 7) | ((rex & 2) << 2);
			d->index = idx == 4 ? OPEMU_REG_NONE : (int8_t)idx;	/* 4 without REX.X: none; 12 = r12 */
			if ((sib & 7) == 5 && mod == 0) {	/* no base, disp32 */
				dispsz = 4;
			} else {
				d->base = (int8_t)((sib & 7) | ((rex & 1) << 3));
			}
		} else if (rm == 5 && mod == 0) {		/* RIP-relative, whatever REX.B says */
			d->base = OPEMU_REG_RIP;
			dispsz = 4;
		} else {
			d->base = (int8_t)(rm | ((rex & 1) << 3));
		}
		if (dispsz == 1) {
			if (i >= n)
				return 0;
			d->disp = (int8_t)b[i++];
		} else if (dispsz == 4) {
			if (i + 4 > n)
				return 0;
			d->disp = (int32_t)((uint32_t)b[i] | (uint32_t)b[i + 1] << 8 |
					    (uint32_t)b[i + 2] << 16 | (uint32_t)b[i + 3] << 24);
			i += 4;
		}
	}

	d->imm = 0;
	if (d->has_imm) {
		if (i >= n)
			return 0;
		d->imm = b[i++];
	}
	d->len = (uint8_t)i;
	return i;
}

/*
 * Effective address of a memory form.  The caller supplies the values of
 * d->base and d->index from ITS register file (pt_regs in the kernel, an
 * array in the tests; either value is ignored when the field is
 * OPEMU_REG_NONE or OPEMU_REG_RIP), so this header holds no register file
 * and the kernel builds no 16-entry array per trap.  next_ip is the address
 * of the instruction that follows: rip + d->len.
 */
OPEMU_INLINE uint64_t opemu_match_ea(const struct opemu_insn *d, uint64_t next_ip,
				     uint64_t base_val, uint64_t index_val)
{
	uint64_t ea = (uint64_t)(int64_t)d->disp;

	if (d->base == OPEMU_REG_RIP)
		ea += next_ip;
	else if (d->base != OPEMU_REG_NONE)
		ea += base_val;
	if (d->index != OPEMU_REG_NONE)
		ea += index_val << d->scale;
	return ea;
}

/* The array flavour, for tests: gpr[16] in ENCODING order (rax rcx rdx rbx rsp rbp rsi rdi r8..r15). */
OPEMU_INLINE uint64_t opemu_match_ea_gpr(const struct opemu_insn *d, uint64_t rip, const uint64_t gpr[16])
{
	return opemu_match_ea(d, rip + d->len,
			      d->base >= 0 ? gpr[d->base] : 0,
			      d->index >= 0 ? gpr[d->index] : 0);
}

#endif /* OPEMU_MATCH_H */
