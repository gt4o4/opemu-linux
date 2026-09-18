/* SPDX-License-Identifier: MIT */
/* The scalar PCMPxSTRx core the emulator shipped with (2026-09-17/18), kept
 * as a test ORACLE.  3,550,601 differential checks against real SSE4.2
 * silicon and 512,000 against GCC's reference model found no mismatch in
 * it, and unlike the silicon it runs on the machine the emulator is for —
 * a Penryn has no PCMPxSTRx to compare against, so this is what lets the
 * SIMD core in core/emu-core.c be iterated on there.  Never linked into the
 * preload or the kernel; test-ref.c is its only user. */
#ifndef PCMPSTR_REF_H
#define PCMPSTR_REF_H
#include <stdint.h>
void pcmpstr_ref_istr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                      uint32_t *out_index, uint8_t out_mask[16], uint64_t *out_flags);
void pcmpstr_ref_estr(const uint8_t a[16], const uint8_t b[16], uint8_t imm,
                      int32_t la, int32_t lb,
                      uint32_t *out_index, uint8_t out_mask[16], uint64_t *out_flags);
#endif
