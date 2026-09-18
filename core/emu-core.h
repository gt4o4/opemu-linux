/* SPDX-License-Identifier: MIT */
/* emu-core — the arithmetic of the instructions Penryn lacks, shared
 * verbatim between three users:
 *
 *   preload/   the userspace LD_PRELOAD SIGILL emulator (Zydis-decoded)
 *   tests/     3.5M-check differential tests against real silicon
 *   kernel/    arch/x86/kernel/opemu-core.c, compiled INTO the kernel and
 *              driven from the #UD trap by opemu.c
 *
 * The point of the split is that the kernel compiles ONLY code that the
 * tests have compared against hardware.  Nothing here knows how it was
 * invoked: no decoder, no register file, no signal frame.  Inputs are plain
 * bytes and integers; flag results come back as RFLAGS bit values for the
 * six arithmetic flags, and the caller merges them with EMU_FL_ALL.
 *
 * Freestanding but for three seams, all below: uint*_t/memcpy/memset (the
 * kernel's linux/types.h + linux/string.h), popcount (hweight64() in the
 * kernel — never __builtin_popcountll, which pulls libgcc's __popcountdi2
 * that the kernel does not link), and CRC-32C (the kernel's own lib/crc
 * table; a table built here in userspace).  The AES S-boxes are the
 * kernel's crypto_aes_sbox[] there and BearSSL's here (emu-core.c).
 *
 * The PCMPxSTRx core runs on SSE2 and CLOBBERS xmm12..xmm15.  A signal
 * handler need not care (rt_sigreturn restores the frame's registers); a
 * caller holding a task's LIVE registers — the kernel — saves those four
 * before sse42emu_pcmpistr/pcmpestr and restores them right after, before
 * writing xmm0 (the M forms).  xmm0..xmm11, MXCSR and x87 are never touched.
 */
#ifndef EMU_CORE_H
#define EMU_CORE_H

#ifdef __KERNEL__
# include <linux/types.h>
# include <linux/string.h>
# include <linux/bitops.h>
# include <linux/crc32.h>
static inline unsigned emu_hweight64(uint64_t v) { return (unsigned) hweight64(v); }
/* The raw CRC-32C update the crc32 instruction performs (no inversion).
 * lib/crc's crc32c() uses the crc32q instruction only behind a static key
 * that follows X86_FEATURE_XMM4_2 — the same feature whose absence puts us
 * here — so on this CPU it is the 256-entry table. */
static inline uint32_t emu_crc32c(uint32_t crc, const void *p, unsigned n) { return crc32c(crc, p, n); }
#else
# include <stdint.h>
# include <string.h>
static inline unsigned emu_hweight64(uint64_t v) { return (unsigned) __builtin_popcountll(v); }
uint32_t emu_crc32c(uint32_t crc, const void *p, unsigned n);   /* emu-core.c: a table from the polynomial */
#endif

/* RFLAGS bit positions of the six arithmetic flags. */
#define EMU_FL_CF  0x0001u
#define EMU_FL_PF  0x0004u
#define EMU_FL_AF  0x0010u
#define EMU_FL_ZF  0x0040u
#define EMU_FL_SF  0x0080u
#define EMU_FL_OF  0x0800u
#define EMU_FL_ALL (EMU_FL_CF | EMU_FL_PF | EMU_FL_AF | EMU_FL_ZF | EMU_FL_SF | EMU_FL_OF)

/* Build the userspace CRC-32C table (a no-op in the kernel, whose tables
 * are the tree's own).  Idempotent; also done lazily on first use. */
void sse42emu_core_init(void);

/* POPCNT r, r/m of `bytes` (2/4/8) — returns the count; *flags gets ZF if
 * the (masked) source is zero, else 0 (the other five flags are cleared
 * by the instruction; the caller does rflags = (rflags & ~EMU_FL_ALL) | f). */
uint64_t sse42emu_popcnt(uint64_t src, unsigned bytes, uint64_t *flags);

/* CRC32 (Castagnoli, reflected) of the low `bytes` of data.  No flags. */
uint32_t sse42emu_crc32(uint32_t crc, uint64_t data, unsigned bytes);

/* PCMPGTQ: dst = (dst > src) per signed 64-bit lane.  No flags. */
void     sse42emu_pcmpgtq(uint8_t dst[16], const uint8_t src[16]);

/* PCMPISTRI/PCMPISTRM (implicit lengths) and PCMPESTRI/PCMPESTRM (explicit,
 * SIGNED lengths, saturated by absolute value).  a = operand 0 (DEST xmm),
 * b = operand 1 (SRC xmm/m128).  Either output may be NULL.  Both CLOBBER
 * xmm12..xmm15 — see the file comment. */
void     sse42emu_pcmpistr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                           uint32_t *out_index, uint8_t out_mask[16],
                           uint64_t *out_flags);
void     sse42emu_pcmpestr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                           int32_t la, int32_t lb, uint32_t *out_index,
                           uint8_t out_mask[16], uint64_t *out_flags);

enum { SSE42EMU_AESENC, SSE42EMU_AESENCLAST, SSE42EMU_AESDEC,
       SSE42EMU_AESDECLAST, SSE42EMU_AESIMC };
void     sse42emu_aes(uint8_t dst[16], const uint8_t src[16], int op);
void     sse42emu_aeskeygen(uint8_t dst[16], const uint8_t src[16], uint8_t rcon);

/* PCLMULQDQ: carry-less product of a's and b's qwords selected by imm. */
void     sse42emu_pclmul(uint8_t dst[16], const uint8_t a[16], const uint8_t b[16], uint8_t imm);

#endif
