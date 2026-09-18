/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../preload/sse42emu.h"
static int fails, n;
static uint64_t rng=0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
#define FL 0x08D5u
/* run one encoded instruction through hw and through the emulator */
static void both(const uint8_t *code, unsigned len, uint64_t rax_in, uint64_t rdx_in,
                 const char *what)
{
    uint64_t hw_rax, hw_fl;
    /* hw: place the bytes in an executable page */
    static uint8_t *page;
    if (!page) { extern void *mmap(void*,size_t,int,int,int,long);
                 page = mmap(0,4096,7,0x22,-1,0); }
    memcpy(page, code, len);
    page[len] = 0xC3;                       /* ret */
    /* The stub is `<insn> ; ret` and touches nothing but rax/rdx/flags.
     * Early-clobber outputs: a compiler is free to hand an output the
     * register of a dead input, or (gcc 13 did) a register the asm text
     * itself uses — an older version of this saved and restored %rbx
     * around the call and gcc put hw_fl IN %rbx, so the restore
     * overwrote the flags and every check failed. */
    __asm__ volatile("mov %2,%%rax\n\tmov %3,%%rdx\n\t"
                     "call *%4\n\tmov %%rax,%0\n\tpushfq\n\tpop %1"
                     : "=&r"(hw_rax), "=&r"(hw_fl)
                     : "r"(rax_in), "r"(rdx_in), "r"(page)
                     : "rax","rcx","rdx","cc","memory");
    sse42emu_regs R; memset(&R,0,sizeof R);
    R.gpr[REG_RAX]=rax_in; R.gpr[REG_RDX]=rdx_in; R.rip=(uint64_t)(uintptr_t)code;
    n++;
    if (sse42emu_step(&R, code, 15) != SSE42EMU_OK) { printf("  %s: emu refused\n",what); fails++; return; }
    if (R.gpr[REG_RAX]!=hw_rax || (R.rflags&FL)!=(hw_fl&FL)) {
        if (fails<8) printf("  %-22s in=%016lx/%016lx  hw=%016lx/%04lx emu=%016lx/%04lx\n",
               what, rax_in, rdx_in, hw_rax, hw_fl&FL, R.gpr[REG_RAX], R.rflags&FL);
        fails++;
    }
}
int main(void){
  uint32_t a,b,c,d; __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1),"c"(0));
  if(!((c>>20)&1)){printf("test-gpr: no SSE4.2 here — the differential test cannot run; build on lix-ory\n");return 2;}
  for(int t=0;t<3000;t++){
    uint64_t x = (t<3)?(uint64_t)t:rnd();
    /* popcnt rax, rax : F3 48 0F B8 C0 */
    both((const uint8_t[]){0xF3,0x48,0x0F,0xB8,0xC0},5,x,0,"popcnt rax,rax");
    /* popcnt eax, eax : F3 0F B8 C0 */
    both((const uint8_t[]){0xF3,0x0F,0xB8,0xC0},4,x,0,"popcnt eax,eax");
    /* popcnt eax, edx : F3 0F B8 C2 */
    both((const uint8_t[]){0xF3,0x0F,0xB8,0xC2},4,0,x,"popcnt eax,edx");
    /* popcnt rax, rdx : F3 48 0F B8 C2 */
    both((const uint8_t[]){0xF3,0x48,0x0F,0xB8,0xC2},5,0,x,"popcnt rax,rdx");
    /* popcnt ax, dx  : 66 F3 0F B8 C2 */
    both((const uint8_t[]){0x66,0xF3,0x0F,0xB8,0xC2},5,0,x,"popcnt ax,dx");
    /* crc32 eax, edx : F2 0F 38 F1 C2 */
    both((const uint8_t[]){0xF2,0x0F,0x38,0xF1,0xC2},5,0x1234,x,"crc32 eax,edx");
  }
  printf("%d checks, %d failures\n", n, fails);
  if(n!=3000*6){printf("  ran %d checks, planned %d\n",n,3000*6);return 1;}
  return fails?1:0;
}
