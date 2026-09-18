/* SPDX-License-Identifier: MIT */
/* The PCMPxSTRx core in core/emu-core.c against the scalar reference in
 * pcmpstr-ref.c: all 256 control bytes, implicit and explicit lengths,
 * index, mask and flags, over the shared vector shapes.
 *
 * Needs no SSE4.2, which is the point: test-emu and test-vec compare the
 * core with the silicon on the build machine, and this compares it with
 * the version that silicon certified — on any x86-64, including the
 * Penryn the emulator exists for.  A disagreement here belongs to the
 * core, not to the reference.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../core/emu-core.h"
#include "pcmpstr-ref.h"
#include "vectors.h"

#define N_VEC 10000
#define FLMASK 0x08D5u

static int failures, checks;

static void dump(const char *tag, const uint8_t *v)
{
    printf("    %s:", tag);
    for (int i = 0; i < 16; i++) printf(" %02x", v[i]);
    printf("\n");
}

int main(void)
{
    const int total = VEC_N_EDGE + N_VEC;
    const int planned = total * 256 * 2;
    for (int t = 0; t < total; t++) {
        uint8_t a[16], b[16];
        vec_pair(t, a, b);
        const int32_t la = vec_len(), lb = vec_len();
        for (int imm = 0; imm < 256; imm++) {
            uint32_t ri = 0, ei = 0; uint64_t rf = 0, ef = 0; uint8_t rm[16], em[16];

            pcmpstr_ref_istr(a, b, (uint8_t) imm, &ri, rm, &rf);
            sse42emu_pcmpistr(a, b, (uint8_t) imm, &ei, em, &ef);
            checks++;
            if (ri != ei || memcmp(rm, em, 16) || (rf & FLMASK) != (ef & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpistr imm=0x%02x%s: ref idx=%u fl=%04lx  core idx=%u fl=%04lx\n", imm,
                           t < VEC_N_EDGE ? vec_edges[t].why : "", ri, rf & FLMASK, ei, ef & FLMASK);
                    dump("a  ", a); dump("b  ", b); dump("ref", rm); dump("cor", em);
                }
                failures++;
            }

            pcmpstr_ref_estr(a, b, (uint8_t) imm, la, lb, &ri, rm, &rf);
            sse42emu_pcmpestr(a, b, (uint8_t) imm, la, lb, &ei, em, &ef);
            checks++;
            if (ri != ei || memcmp(rm, em, 16) || (rf & FLMASK) != (ef & FLMASK)) {
                if (failures < 10) {
                    printf("  pcmpestr imm=0x%02x la=%d lb=%d: ref idx=%u fl=%04lx  core idx=%u fl=%04lx\n",
                           imm, la, lb, ri, rf & FLMASK, ei, ef & FLMASK);
                    dump("a  ", a); dump("b  ", b); dump("ref", rm); dump("cor", em);
                }
                failures++;
            }
        }
    }
    printf("%d checks, %d failures\n", checks, failures);
    if (checks != planned) { printf("  ran %d checks, planned %d\n", checks, planned); return 1; }
    return failures ? 1 : 0;
}
