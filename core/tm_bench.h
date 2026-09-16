/* 板上测速：用 Cortex-M 的 DWT 周期计数器量一次推理到底多少个 CPU 周期。
 *
 * 这个文件存在的理由：**"推理要多久"这个数不能估**。它直接决定占空比和功耗，
 * 是拿来做产品决策的。x86 上量到的 286 µs/窗口对 64MHz 的 M4F 没有参考价值，
 * 按 MAC 数乘"每 MAC 几周期"去推，误差轻松 2~3 倍。
 *
 * 所以让固件自己数，板子一上电就把真实数字打出来。
 */

#ifndef TM_BENCH_H
#define TM_BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t total;
    uint32_t n;
    uint32_t min;
    uint32_t max;
} tm_bench_t;

/* 起周期计数器。返回 0 成功，-1 表示计数器没动——
 * 那时候所有耗时都会是 0，**必须当成失败而不是"快得惊人"**。 */
int tm_bench_init(void);

/* 当前周期数。回绕是正常的，用 tm_bench_elapsed 算差值。 */
uint32_t tm_bench_now(void);

/* t1 - t0，自动处理 32 位回绕（64MHz 下约 67 秒一圈）。 */
uint32_t tm_bench_elapsed(uint32_t t0, uint32_t t1);

/* 周期数 → 微秒。走 64 位中间量，先乘后除不会溢出。 */
uint32_t tm_bench_us(uint32_t cycles, uint32_t cpu_hz);

void tm_bench_reset(tm_bench_t *b);
void tm_bench_accum(tm_bench_t *b, uint32_t cycles);
uint32_t tm_bench_mean(const tm_bench_t *b);

#ifdef __cplusplus
}
#endif

#endif /* TM_BENCH_H */
