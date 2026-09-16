#include "tm_bench.h"

/* 周期计数。
 *
 * 为什么要在**板上**测而不是在 PC 上估：
 * 我在 x86 上量到 286 µs/窗口，但那个数对 Cortex-M4F @64MHz 没有任何参考价值——
 * 指令集不同、频率差 50 倍、cache 行为完全不一样。按 MAC 数乘一个"每 MAC 几周期"
 * 去估，误差轻松到 2~3 倍，而"推理要多久"直接决定占空比和功耗，是个要拿来做
 * 产品决策的数。估出来的数不能用。
 *
 * 所以让固件自己数：板子一上电就打印真实的周期数和微秒数，不用连调试器、
 * 不用示波器、不用我猜。
 *
 * DWT（Data Watchpoint and Trace）是 Cortex-M3/M4/M7 里的一个硬件周期计数器，
 * 免费、零开销、精度就是一个 CPU 周期。要先解锁：M7 和部分 M4 上
 * DWT->LAR 需要写 0xC5ACCE55 才允许改 CYCCNT，不写的话计数器一直是 0——
 * **而"一直是 0"会表现成"推理快得不可思议"**，比报错难发现得多，
 * 所以下面专门检查了计数器到底有没有在走。
 */

#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
#define TM_BENCH_DWT 1
#endif

#if TM_BENCH_DWT

#define TM_DEMCR   (*(volatile uint32_t *)0xE000EDFCu)
#define TM_DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define TM_DWT_CYC  (*(volatile uint32_t *)0xE0001004u)
#define TM_DWT_LAR  (*(volatile uint32_t *)0xE0001FB0u)

#define TM_DEMCR_TRCENA   (1u << 24)
#define TM_DWT_CTRL_CYCEN (1u << 0)

int tm_bench_init(void)
{
    TM_DEMCR |= TM_DEMCR_TRCENA;
    TM_DWT_LAR = 0xC5ACCE55u;   /* 解锁；在不需要解锁的核上写它是无害的 */
    TM_DWT_CYC = 0;
    TM_DWT_CTRL |= TM_DWT_CTRL_CYCEN;

    /* **确认它真的在走。** 有些芯片没接调试时钟、或者被安全配置挡住，
     * CYCCNT 会一直是 0，而那会表现成"推理耗时 0 周期"——
     * 一个好得离谱的数比一个错误提示更容易被当真。 */
    uint32_t a = TM_DWT_CYC;
    for (volatile int i = 0; i < 16; i++) { }
    return (TM_DWT_CYC != a) ? 0 : -1;
}

uint32_t tm_bench_now(void) { return TM_DWT_CYC; }

#else   /* PC 上编译（测试用）：没有 DWT，用一个单调递增的替身 */

#include <time.h>

int tm_bench_init(void) { return 0; }

uint32_t tm_bench_now(void)
{
    /* 用标准 C 的 clock()，不用 clock_gettime——后者要 _POSIX_C_SOURCE，
     * 而这份代码也要能在 -std=c99 的裸机编译器下过。
     *
     * 这条分支**只是替身**：它让 tm_bench 的算术（回绕、溢出、累加）能在 PC 上
     * 测，但它给出的不是 CPU 周期数，所以不要拿它报"推理多少周期"。
     * 真正的数只有板上那条 DWT 分支给得出来。 */
    return (uint32_t)clock();
}

#endif

uint32_t tm_bench_elapsed(uint32_t t0, uint32_t t1)
{
    /* 32 位计数器在 64MHz 下约 67 秒回绕一次。无符号减法天然处理回绕，
     * 只要两次采样间隔不超过一个周期——单次推理几十毫秒，差得远。
     * 写成 (t1 - t0) 而不是判大小，正是为了让回绕自动正确。 */
    return t1 - t0;
}

uint32_t tm_bench_us(uint32_t cycles, uint32_t cpu_hz)
{
    if (cpu_hz == 0) {
        return 0;
    }
    /* 先乘后除会溢出（64MHz 下 67 秒就满 32 位，乘 1e6 立刻炸），
     * 所以走 64 位中间量。整数除法截断，对微秒量级够用。 */
    return (uint32_t)(((uint64_t)cycles * 1000000ull) / cpu_hz);
}

void tm_bench_accum(tm_bench_t *b, uint32_t cycles)
{
    if (b->n == 0 || cycles < b->min) {
        b->min = cycles;
    }
    if (cycles > b->max) {
        b->max = cycles;
    }
    b->total += cycles;
    b->n++;
}

uint32_t tm_bench_mean(const tm_bench_t *b)
{
    return b->n ? (uint32_t)(b->total / b->n) : 0;
}

void tm_bench_reset(tm_bench_t *b)
{
    b->total = 0;
    b->n = 0;
    b->min = 0;
    b->max = 0;
}
