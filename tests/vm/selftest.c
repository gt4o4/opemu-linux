/* SPDX-License-Identifier: MIT */
/* opemu VM self-test — the /init of a one-file initramfs booted under
 * `qemu-system-x86_64 -accel tcg -cpu Penryn,+xsave` with the exact kernel
 * that will run on the target machine.
 *
 * The builder that runs this HAS every instruction being emulated, so no
 * native test on it can reach the kernel's #UD path; QEMU's CPU model is
 * what makes the traps real.  Everything here is asserted against
 * INDEPENDENT known answers (hand-counted bits, the CRC-32C check value,
 * the FIPS-197 C.1 AES-128 vector, string positions you can see), never
 * against the emulator's own arithmetic.
 *
 * Two boots use this binary:
 *   - a CPU model LACKING the features: the emulator must be enabled by
 *     the kernel, the negative control (enable=0 -> a POPCNT child dies of
 *     SIGILL) must hold, every family must pass, and the per-family
 *     counters must equal EXACTLY what was executed — including the
 *     refusal counters, so the accounting of the paths that say "not
 *     mine" is tested too;
 *   - a CPU model HAVING them (Westmere): the emulator must have stayed
 *     disabled, everything passes natively, every counter stays zero.
 *
 * Output is TAP-ish on the serial console and ends with the marker the
 * derivation greps for: `OPEMU-SELFTEST: PASS` or `OPEMU-SELFTEST: FAIL`.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/reboot.h>
#include <sys/sysmacros.h>

/* ---- families, in the order the kernel's stats file prints them ---- */
enum { F_POPCNT, F_CRC32, F_PCMPGTQ, F_PCMPISTRI, F_PCMPISTRM, F_PCMPESTRI,
       F_PCMPESTRM, F_PCLMULQDQ, F_AES, F_AESKEYGEN, F_UNHANDLED, F_COMPAT32,
       F_SEGV, F_NR };
static const char *fam_name[F_NR] = {
    "popcnt", "crc32", "pcmpgtq", "pcmpistri", "pcmpistrm", "pcmpestri",
    "pcmpestrm", "pclmulqdq", "aes", "aeskeygen", "unhandled", "compat32", "segv" };
static unsigned long expect[F_NR];          /* what we executed, per family */

static int failures, checks;

static void say(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    fflush(stdout);
}
static void check(int ok, const char *name, const char *fmt, ...)
{
    checks++;
    if (ok) { say("ok %d - %s\n", checks, name); return; }
    failures++;
    va_list ap; va_start(ap, fmt);
    char b[256]; vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    say("not ok %d - %s: %s\n", checks, name, b);
}

/* ---- CPU facts ------------------------------------------------------ */
static void cpuid1(uint32_t *ecx)
{
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1), "c"(0));
    *ecx = c;
}
static int cpu_has_all(void)
{
    uint32_t c; cpuid1(&c);
    return ((c >> 20) & 1) && ((c >> 23) & 1) && ((c >> 25) & 1) && ((c >> 1) & 1);
}

/* ---- the kernel knobs ----------------------------------------------- */
#define ENABLE_PATH "/sys/module/opemu/parameters/enable"
#define STATS_PATH  "/sys/kernel/debug/opemu/stats"

static int read_enable(void)
{
    char b[8] = {0};
    int fd = open(ENABLE_PATH, O_RDONLY);
    if (fd < 0) return -1;
    (void) !read(fd, b, sizeof b - 1); close(fd);
    return b[0] == 'Y' || b[0] == '1' || b[0] == 'y';
}
static int write_enable(int on)
{
    int fd = open(ENABLE_PATH, O_WRONLY);
    if (fd < 0) return -1;
    int r = write(fd, on ? "1\n" : "0\n", 2) == 2 ? 0 : -1; close(fd);
    return r;
}
static int read_stats(unsigned long got[F_NR])
{
    memset(got, 0, sizeof(unsigned long) * F_NR);
    FILE *f = fopen(STATS_PATH, "r");
    if (!f) return -1;
    char name[32]; unsigned long v; int seen = 0;
    while (fscanf(f, "%31s %lu", name, &v) == 2)
        for (int i = 0; i < F_NR; i++) if (!strcmp(name, fam_name[i])) { got[i] = v; seen++; }
    fclose(f);
    return seen;
}

/* ---- the instructions, as real encodings ---------------------------- */
#define FLMASK 0x8D5u
static uint64_t g_ripdata = 0xdeadbeefcafef00dULL;   /* RIP-relative operand */

static uint64_t popcnt64(uint64_t v, uint64_t *fl)
{ uint64_t r, f; __asm__ volatile("popcnt %2,%0\n\tpushfq\n\tpopq %1" : "=r"(r), "=r"(f) : "r"(v) : "cc"); expect[F_POPCNT]++; if (fl) *fl = f & FLMASK; return r; }
static uint32_t popcnt32(uint32_t v)
{ uint32_t r; __asm__ volatile("popcntl %1,%0" : "=r"(r) : "r"(v) : "cc"); expect[F_POPCNT]++; return r; }
static uint16_t popcnt16(uint16_t v)
{ uint16_t r; __asm__ volatile("popcntw %1,%0" : "=r"(r) : "r"(v) : "cc"); expect[F_POPCNT]++; return r; }
static uint64_t popcnt_mem(const uint64_t *p)
{ uint64_t r; __asm__ volatile("popcntq (%1),%0" : "=r"(r) : "r"(p) : "cc", "memory"); expect[F_POPCNT]++; return r; }
static uint64_t popcnt_sib(const uint64_t *base, uint64_t idx)      /* [base+idx*8+0x1000] */
{ uint64_t r; __asm__ volatile("popcntq 0x1000(%1,%2,8),%0" : "=r"(r) : "r"((uintptr_t) base - 0x1000), "r"(idx) : "cc", "memory"); expect[F_POPCNT]++; return r; }
static uint64_t popcnt_rip(void)
{ uint64_t r; __asm__ volatile("popcntq g_ripdata(%%rip),%0" : "=r"(r) : : "cc", "memory"); expect[F_POPCNT]++; return r; }
static uint64_t popcnt_addr32(uint32_t addr)                        /* 67-prefixed */
{ uint64_t r; __asm__ volatile("popcntq (%k1),%0" : "=r"(r) : "r"((uint64_t) addr) : "cc", "memory"); expect[F_POPCNT]++; return r; }   /* %k1 = 32-bit base -> GAS emits the 67 prefix */

static uint32_t crc32b(uint32_t c, uint8_t v)  { __asm__ volatile("crc32b %1,%0" : "+r"(c) : "r"(v)); expect[F_CRC32]++; return c; }
static uint32_t crc32w(uint32_t c, uint16_t v) { __asm__ volatile("crc32w %1,%0" : "+r"(c) : "r"(v)); expect[F_CRC32]++; return c; }
static uint32_t crc32l(uint32_t c, uint32_t v) { __asm__ volatile("crc32l %1,%0" : "+r"(c) : "r"(v)); expect[F_CRC32]++; return c; }
static uint64_t crc32q(uint64_t c, uint64_t v) { __asm__ volatile("crc32q %1,%0" : "+r"(c) : "r"(v)); expect[F_CRC32]++; return c; }
static uint32_t crc32b_ah(uint32_t c, uint32_t eax_val)             /* the high-byte form Zydis hid */
{ __asm__ volatile("crc32b %%ah,%%ecx" : "+c"(c) : "a"(eax_val)); expect[F_CRC32]++; return c; }   /* dest pinned to ECX: AH cannot meet a REX register */
static uint32_t crc32b_mem(uint32_t c, const uint8_t *p)
{ __asm__ volatile("crc32b (%1),%0" : "+r"(c) : "r"(p) : "memory"); expect[F_CRC32]++; return c; }

/* textbook reflected CRC-32C, the reference for the multi-byte forms */
static uint32_t crc32c_ref(uint32_t crc, const void *buf, size_t n)
{
    const uint8_t *p = buf;
    while (n--) { crc ^= *p++; for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0x82F63B78u & (uint32_t) -(int32_t) (crc & 1u)); }
    return crc;
}

#define XMM_OP2(name, insn)                                                   \
static void name(uint8_t d[16], const uint8_t s[16]) {                        \
    __asm__ volatile("movdqu (%0),%%xmm1\n\tmovdqu (%1),%%xmm2\n\t"           \
                     insn " %%xmm2,%%xmm1\n\tmovdqu %%xmm1,(%0)"              \
                     : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }
XMM_OP2(op_pcmpgtq,    "pcmpgtq")
XMM_OP2(op_aesenc,     "aesenc")
XMM_OP2(op_aesenclast, "aesenclast")
XMM_OP2(op_aesdec,     "aesdec")
XMM_OP2(op_aesdeclast, "aesdeclast")
static void op_aesimc(uint8_t d[16], const uint8_t s[16])
{ __asm__ volatile("movdqu (%1),%%xmm2\n\taesimc %%xmm2,%%xmm1\n\tmovdqu %%xmm1,(%0)" : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }

#define KEYGEN(R)                                                             \
static void keygen_##R(uint8_t d[16], const uint8_t s[16]) {                  \
    __asm__ volatile("movdqu (%1),%%xmm2\n\taeskeygenassist $" #R ",%%xmm2,%%xmm1\n\t" \
                     "movdqu %%xmm1,(%0)" : : "r"(d), "r"(s) : "xmm1", "xmm2", "memory"); }
KEYGEN(0x01) KEYGEN(0x02) KEYGEN(0x04) KEYGEN(0x08) KEYGEN(0x10)
KEYGEN(0x20) KEYGEN(0x40) KEYGEN(0x80) KEYGEN(0x1b) KEYGEN(0x36)
typedef void (*keygen_fn)(uint8_t *, const uint8_t *);
static const keygen_fn keygen[10] = { keygen_0x01, keygen_0x02, keygen_0x04, keygen_0x08, keygen_0x10,
                                      keygen_0x20, keygen_0x40, keygen_0x80, keygen_0x1b, keygen_0x36 };

/* One AES-128 round-key step, the Intel white-paper way: the assist word,
 * broadcast, then the running xor chain — all SSE2 except the assist. */
static void expand_round(const uint8_t prev[16], uint8_t out[16], int i)
{
    uint8_t t[16];
    keygen[i](t, prev); expect[F_AESKEYGEN]++;
    __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\t"
                     "pshufd $0xff,%%xmm2,%%xmm2\n\t"
                     "movdqa %%xmm1,%%xmm3\n\tpslldq $4,%%xmm3\n\tpxor %%xmm3,%%xmm1\n\t"
                     "movdqa %%xmm1,%%xmm3\n\tpslldq $4,%%xmm3\n\tpxor %%xmm3,%%xmm1\n\t"
                     "movdqa %%xmm1,%%xmm3\n\tpslldq $4,%%xmm3\n\tpxor %%xmm3,%%xmm1\n\t"
                     "pxor %%xmm2,%%xmm1\n\tmovdqu %%xmm1,(%0)"
                     : : "r"(out), "r"(prev), "r"(t) : "xmm1", "xmm2", "xmm3", "memory");
}

static void pclmul_00(uint8_t d[16], const uint8_t a[16], const uint8_t b[16])
{ __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\tpclmulqdq $0x00,%%xmm2,%%xmm1\n\tmovdqu %%xmm1,(%0)" : : "r"(d), "r"(a), "r"(b) : "xmm1", "xmm2", "memory"); expect[F_PCLMULQDQ]++; }
static void pclmul_11(uint8_t d[16], const uint8_t a[16], const uint8_t b[16])
{ __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\tpclmulqdq $0x11,%%xmm2,%%xmm1\n\tmovdqu %%xmm1,(%0)" : : "r"(d), "r"(a), "r"(b) : "xmm1", "xmm2", "memory"); expect[F_PCLMULQDQ]++; }

#define ISTRI(name, imm, regs)                                                \
static uint32_t name(const uint8_t a[16], const uint8_t b[16], uint64_t *fl) {\
    uint32_t i; uint64_t f;                                                   \
    __asm__ volatile("movdqu (%2),%%xmm" #regs "\n\tmovdqu (%3),%%xmm2\n\t"   \
                     "pcmpistri $" #imm ",%%xmm2,%%xmm" #regs "\n\t"          \
                     "mov %%ecx,%0\n\tpushfq\n\tpopq %1"                      \
                     : "=&r"(i), "=&r"(f) : "r"(a), "r"(b)                    \
                     : "xmm" #regs, "xmm2", "rcx", "cc", "memory");           \
    expect[F_PCMPISTRI]++; if (fl) *fl = f & FLMASK; return i; }
ISTRI(istri_any,     0x00, 1)     /* EqualAny, unsigned bytes, index of first match */
ISTRI(istri_ordered, 0x0c, 1)     /* EqualOrdered: substring search */
ISTRI(istri_ordered_hi, 0x0c, 9)  /* same, through xmm9 (REX.R on the reg field) */
static uint32_t istri_mem(const uint8_t a[16], const uint8_t *b_mem)   /* m128 source */
{ uint32_t i; __asm__ volatile("movdqu (%1),%%xmm1\n\tpcmpistri $0x0c,(%2),%%xmm1\n\tmov %%ecx,%0" : "=&r"(i) : "r"(a), "r"(b_mem) : "xmm1", "rcx", "cc", "memory"); expect[F_PCMPISTRI]++; return i; }
static void istrm_any_mask(const uint8_t a[16], const uint8_t b[16], uint8_t out[16])
{ __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\tpcmpistrm $0x40,%%xmm2,%%xmm1\n\tmovdqu %%xmm0,(%0)" : : "r"(out), "r"(a), "r"(b) : "xmm0", "xmm1", "xmm2", "cc", "memory"); expect[F_PCMPISTRM]++; }
static uint32_t estri_ordered(const uint8_t a[16], const uint8_t b[16], int la, int lb, uint64_t *fl)
{ uint32_t i; uint64_t f;
  __asm__ volatile("movdqu (%4),%%xmm1\n\tmovdqu (%5),%%xmm2\n\tpcmpestri $0x0c,%%xmm2,%%xmm1\n\tmov %%ecx,%0\n\tpushfq\n\tpopq %1"
                   : "=&r"(i), "=&r"(f) : "a"(la), "d"(lb), "r"(a), "r"(b) : "xmm1", "xmm2", "rcx", "cc", "memory");
  expect[F_PCMPESTRI]++; if (fl) *fl = f & FLMASK; return i; }
static uint32_t estri_ordered_rexw(const uint8_t a[16], const uint8_t b[16], int64_t la, int64_t lb)
{ uint32_t i;   /* 66 48 0F 3A 61 CA 0C = pcmpestri $0x0c,%xmm2,%xmm1 with REX.W: lengths from RAX/RDX */
  __asm__ volatile("movdqu (%3),%%xmm1\n\tmovdqu (%4),%%xmm2\n\t.byte 0x66,0x48,0x0f,0x3a,0x61,0xca,0x0c\n\tmov %%ecx,%0"
                   : "=&r"(i) : "a"(la), "d"(lb), "r"(a), "r"(b) : "xmm1", "xmm2", "rcx", "cc", "memory");
  expect[F_PCMPESTRI]++; return i; }
static void estrm_any_mask(const uint8_t a[16], const uint8_t b[16], int la, int lb, uint8_t out[16])
{ __asm__ volatile("movdqu (%1),%%xmm1\n\tmovdqu (%2),%%xmm2\n\tpcmpestrm $0x40,%%xmm2,%%xmm1\n\tmovdqu %%xmm0,(%0)"
                   : : "r"(out), "r"(a), "r"(b), "a"(la), "d"(lb) : "xmm0", "xmm1", "xmm2", "cc", "memory"); expect[F_PCMPESTRM]++; }

/* ---- children that must die a particular way ------------------------ */
static int run_child(void (*fn)(void *), void *arg)
{
    pid_t p = fork();
    if (p == 0) { fn(arg); _exit(99); }
    int st = 0; waitpid(p, &st, 0);
    return st;
}
static void child_popcnt(void *a) { (void) a; volatile uint64_t r; __asm__ volatile("popcnt %1,%0" : "=r"(r) : "r"(g_ripdata)); }
static void child_ud2(void *a) { (void) a; __asm__ volatile("ud2"); }
static void child_exec_bytes(void *a)
{
    const uint8_t *code = a;
    uint8_t *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t n = 0; while (code[n] != 0xC3) n++;        /* copy through the ret */
    memcpy(page, code, n + 1);
    ((void (*)(void)) page)();
}
static void *g_protnone;
static void segv_handler(int sig, siginfo_t *si, void *uc)
{ (void) sig; (void) uc; _exit(si->si_addr == g_protnone ? 42 : 43); }
static void child_protnone(void *a)
{
    int handled = *(int *) a;
    if (handled) {
        struct sigaction sa; memset(&sa, 0, sizeof sa);
        sa.sa_sigaction = segv_handler; sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, NULL);
    }
    volatile uint64_t r; __asm__ volatile("popcntq (%1),%0" : "=r"(r) : "r"(g_protnone) : "memory");
    _exit(44);                                           /* it returned: wrong */
}
static void *thread_blocked_popcnt(void *arg)
{
    sigset_t all; sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, NULL);          /* what bun does */
    uint64_t r; __asm__ volatile("popcnt %1,%0" : "=r"(r) : "r"(0xdeadbeefcafef00dULL) : "cc");
    expect[F_POPCNT]++;
    *(uint64_t *) arg = r;
    return NULL;
}

/* ---- the run --------------------------------------------------------- */
static void run_families(int lacking)
{
    uint64_t fl = 0;
    check(popcnt64(0xdeadbeefcafef00dULL, &fl) == 42 && fl == 0, "popcnt r64 = 42, flags clear", "fl=%lx", fl);
    check(popcnt64(0, &fl) == 0 && fl == 0x40, "popcnt 0 sets ZF only", "fl=%lx", fl);
    check(popcnt32(0x12345678u) == 13, "popcnt r32 = 13", "");
    check(popcnt16(0xffffu) == 16 && popcnt16(0x0101u) == 2, "popcnt r16 (66 F3 0F B8)", "");
    uint64_t arr[4] = { 0xdeadbeefcafef00dULL, 0xffULL, 0x0ULL, 0x8000000000000001ULL };
    check(popcnt_mem(arr) == 42, "popcnt [base]", "");
    check(popcnt_sib(arr, 1) == 8 && popcnt_sib(arr, 3) == 2, "popcnt [base+idx*8+disp32]", "");
    check(popcnt_rip() == 42, "popcnt RIP-relative", "");
    if ((uintptr_t) &g_ripdata < 0x100000000ULL)
        check(popcnt_addr32((uint32_t) (uintptr_t) &g_ripdata) == 42, "popcnt addr32 (67 prefix)", "");
    else say("# skip addr32: data above 4G\n");

    uint32_t c = 0xffffffffu;
    for (const char *s = "123456789"; *s; s++) c = crc32b(c, (uint8_t) *s);
    check((c ^ 0xffffffffu) == 0xE3069283u, "crc32b over \"123456789\" = E3069283", "got %08x", c ^ 0xffffffffu);
    check(crc32b_ah(0x1234u, 0x0000AB00u) == crc32c_ref(0x1234u, "\xAB", 1), "crc32b %%ah (high-byte register)", "");
    check(crc32w(7u, 0xBEEFu) == crc32c_ref(7u, "\xEF\xBE", 2), "crc32w", "");
    check(crc32l(7u, 0xCAFEBABEu) == crc32c_ref(7u, "\xBE\xBA\xFE\xCA", 4), "crc32l", "");
    { uint64_t q = 0xf00d;
      check(crc32q(0x0000000100000007ULL, q) == crc32c_ref(7u, &q, 8), "crc32q (upper 32 of dest ignored, result zero-extended)", ""); }
    { uint8_t one = 0x5A; check(crc32b_mem(1u, &one) == crc32c_ref(1u, &one, 1), "crc32b [mem]", ""); }

    { uint8_t a[16], b[16]; int64_t x[2] = { 5, -1 }, y[2] = { 3, 2 };
      memcpy(a, x, 16); memcpy(b, y, 16); op_pcmpgtq(a, b); expect[F_PCMPGTQ]++;
      uint64_t lo, hi; memcpy(&lo, a, 8); memcpy(&hi, a + 8, 8);
      check(lo == ~0ULL && hi == 0, "pcmpgtq {5,-1} > {3,2} = {~0,0}", "lo=%lx hi=%lx", lo, hi); }

    { /* FIPS-197 C.1 */
      static const uint8_t key[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
      static const uint8_t pt[16]  = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
      static const uint8_t ct[16]  = {0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a};
      uint8_t rk[11][16], dk[11][16], st[16];
      memcpy(rk[0], key, 16);
      for (int i = 1; i <= 10; i++) expand_round(rk[i - 1], rk[i], i - 1);
      memcpy(st, pt, 16);
      for (int i = 0; i < 16; i++) st[i] ^= rk[0][i];
      for (int r = 1; r <= 9; r++) { op_aesenc(st, rk[r]); expect[F_AES]++; }
      op_aesenclast(st, rk[10]); expect[F_AES]++;
      check(!memcmp(st, ct, 16), "AES-128 encrypt FIPS-197 C.1 (10 keygenassist, 9 aesenc, 1 aesenclast)", "");
      memcpy(dk[0], rk[10], 16); memcpy(dk[10], rk[0], 16);
      for (int r = 1; r <= 9; r++) { op_aesimc(dk[r], rk[10 - r]); expect[F_AES]++; }
      for (int i = 0; i < 16; i++) st[i] ^= dk[0][i];
      for (int r = 1; r <= 9; r++) { op_aesdec(st, dk[r]); expect[F_AES]++; }
      op_aesdeclast(st, dk[10]); expect[F_AES]++;
      check(!memcmp(st, pt, 16), "AES-128 decrypt back to plaintext (9 aesimc, 9 aesdec, 1 aesdeclast)", ""); }

    { uint8_t a[16] = {0}, b[16] = {0}, d[16]; a[0] = 3; b[0] = 7; a[8] = 3; b[8] = 7;
      pclmul_00(d, a, b); check(d[0] == 9 && d[1] == 0 && d[8] == 0, "pclmulqdq $0: 11b x 111b = 1001b", "d0=%02x", d[0]);
      pclmul_11(d, a, b); check(d[0] == 9, "pclmulqdq $0x11 (high qwords)", "d0=%02x", d[0]); }

    { uint8_t needle[16] = "abc", hay[16] = "xxbyy";
      uint32_t i = istri_any(needle, hay, &fl);
      check(i == 2 && fl == (0x1 | 0x40 | 0x80), "pcmpistri EqualAny 'abc' in \"xxbyy\" -> 2, CF|ZF|SF", "i=%u fl=%lx", i, fl);
      uint8_t sub[16] = "cd", str[16] = "abcdef";
      check(istri_ordered(sub, str, &fl) == 2 && (fl & 1), "pcmpistri EqualOrdered \"cd\" in \"abcdef\" -> 2", "");
      check(istri_ordered_hi(sub, str, &fl) == 2, "pcmpistri through xmm9 (REX.R)", "");
      check(istri_mem(sub, str) == 2, "pcmpistri with m128 source", "");
      uint8_t nothing[16] = "zz";
      check(istri_ordered(nothing, str, &fl) == 16 && !(fl & 1), "pcmpistri no match -> 16, CF clear", "fl=%lx", fl);
      uint8_t m[16], set[16] = "a", ban[16] = "banana";
      istrm_any_mask(set, ban, m);
      check(m[0] == 0 && m[1] == 0xff && m[2] == 0 && m[3] == 0xff && m[5] == 0xff && m[6] == 0, "pcmpistrm $0x40 mask of 'a' in \"banana\"", "");
      check(estri_ordered(sub, str, 2, 6, &fl) == 2, "pcmpestri lengths 2/6 -> 2", "");
      check(estri_ordered(sub, str, -2, -6, &fl) == 2, "pcmpestri NEGATIVE lengths -2/-6 -> 2 (absolute value)", "");
      check(estri_ordered(sub, str, 2, 3, &fl) == 16, "pcmpestri lb=3 excludes the match -> 16", "");
      check(estri_ordered_rexw(sub, str, 2, 6) == 2, "pcmpestri REX.W (RAX/RDX lengths)", "");
      estrm_any_mask(set, ban, 1, 6, m);
      check(m[1] == 0xff && m[5] == 0xff && m[6] == 0, "pcmpestrm $0x40 lengths 1/6", ""); }

    /* mechanism: the thread that blocks every signal, then faults */
    { uint64_t r = 0; pthread_t t;
      pthread_create(&t, NULL, thread_blocked_popcnt, &r); pthread_join(t, NULL);
      check(r == 42, "POPCNT in a thread with ALL signals blocked (blocker 1)", "r=%lu", r); }

    /* mechanism: an operand on a page that has never been touched */
    { uint64_t *fresh = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      check(popcnt_mem(fresh) == 0, "popcnt on a never-touched anonymous page (pages in, reads 0)", "");
      uint8_t z8[8] = {0};
      check(crc32q(0x77ULL, 0) == crc32c_ref(0x77u, z8, 8) && crc32b_mem(0x77u, (uint8_t *) fresh + 4090) == crc32c_ref(0x77u, z8, 1),
            "crc32 on the fresh page, tail byte", ""); }

    /* children: the ways it must still die */
    if (lacking) {
        int st;
        int handled = 1;
        st = run_child(child_protnone, &handled);
        check(WIFEXITED(st) && WEXITSTATUS(st) == 42, "PROT_NONE operand -> SIGSEGV with si_addr = the page", "status=%x", st);
        handled = 0;
        st = run_child(child_protnone, &handled);
        check(WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV, "PROT_NONE operand, no handler -> killed by SIGSEGV", "status=%x", st);
        expect[F_SEGV] += 2;
        st = run_child(child_ud2, NULL);
        check(WIFSIGNALED(st) && WTERMSIG(st) == SIGILL, "ud2 still dies of SIGILL (pass-through)", "status=%x", st);
        expect[F_UNHANDLED]++;
        static const uint8_t lock_popcnt[] = { 0xF0, 0xF3, 0x0F, 0xB8, 0xC0, 0xC3 };          /* lock popcnt %eax,%eax */
        static const uint8_t vex_istri[]   = { 0xC4, 0xE3, 0x79, 0x63, 0xC1, 0x00, 0xC3 };    /* vpcmpistri $0,%xmm1,%xmm0 */
        static const uint8_t f2_istri[]    = { 0xF2, 0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x00, 0xC3 }; /* F2 on the 66 group */
        static const uint8_t bare_b8[]     = { 0x0F, 0xB8, 0xC0, 0xC3 };                      /* popcnt without F3 */
        static const uint8_t movbe[]       = { 0x0F, 0x38, 0xF0, 0x00, 0xC3 };                /* movbe (%rax),%eax: F0 without F2 */
        const uint8_t *bad[] = { lock_popcnt, vex_istri, f2_istri, bare_b8, movbe };
        const char *badname[] = { "LOCK popcnt", "VEX vpcmpistri", "F2-prefixed pcmpistri", "0F B8 without F3", "MOVBE (0F 38 F0 without F2)" };
        for (int k = 0; k < 5; k++) {
            st = run_child(child_exec_bytes, (void *) bad[k]);
            check(WIFSIGNALED(st) && WTERMSIG(st) == SIGILL, badname[k], "must die of SIGILL, status=%x", st);
            expect[F_UNHANDLED]++;
        }
    }
}

int main(void)
{
    /* PID 1 in a one-file initramfs: make ourselves a console and the
     * mounts the checks need. */
    mkdir("/dev", 0755); mkdir("/sys", 0755); mkdir("/proc", 0755);
    mknod("/dev/console", S_IFCHR | 0600, makedev(5, 1));
    int fd = open("/dev/console", O_RDWR);
    if (fd >= 0) { dup2(fd, 0); dup2(fd, 1); dup2(fd, 2); if (fd > 2) close(fd); }
    setvbuf(stdout, NULL, _IONBF, 0);
    mount("sysfs", "/sys", "sysfs", 0, "");
    mount("proc", "/proc", "proc", 0, "");
    mount("debugfs", "/sys/kernel/debug", "debugfs", 0, "");

    const int lacking = !cpu_has_all();
    uint32_t ecx; cpuid1(&ecx);
    say("# opemu selftest: cpuid.1:ecx=%08x -> CPU %s the emulated features\n", ecx, lacking ? "LACKS" : "HAS ALL");

    int en = read_enable();
    check(en >= 0, "kernel has CONFIG_X86_UD_EMULATE (" ENABLE_PATH " exists)", "missing");
    if (en < 0) goto done;
    check(en == lacking, lacking ? "emulator enabled at boot (CPU lacks features)" : "emulator stayed disabled (CPU has everything)", "enable=%d", en);

    if (lacking) {
        /* negative control: with the emulator off this same POPCNT must kill */
        check(write_enable(0) == 0 && read_enable() == 0, "enable=0 accepted", "");
        int st = run_child(child_popcnt, NULL);
        check(WIFSIGNALED(st) && WTERMSIG(st) == SIGILL, "NEGATIVE CONTROL: popcnt child dies of SIGILL with enable=0", "status=%x (a 0 exit means the test is vacuous)", st);
        check(write_enable(1) == 0 && read_enable() == 1, "enable=1 restored", "");
    }

    g_protnone = mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_families(lacking);

    unsigned long got[F_NR];
    int seen = read_stats(got);
    check(seen == F_NR, "stats file lists every counter", "seen %d of %d", seen, F_NR);
    for (int i = 0; i < F_NR; i++) {
        unsigned long want = lacking ? expect[i] : 0;
        check(got[i] == want, fam_name[i], "counter %lu, expected exactly %lu", got[i], want);
    }
    say("# stats:"); for (int i = 0; i < F_NR; i++) say(" %s=%lu", fam_name[i], got[i]); say("\n");

done:
    say("# %d checks, %d failures\n", checks, failures);
    say("OPEMU-SELFTEST: %s\n", failures ? "FAIL" : "PASS");
    sync();
    reboot(RB_POWER_OFF);
    sleep(5);
    return failures ? 1 : 0;
}
