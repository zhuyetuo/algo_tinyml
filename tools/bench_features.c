/* 量一下 193 维特征里各部分各占多少算力。
 *
 * 为什么要量而不是估：结论会直接决定"要不要砍特征、砍哪些"。凭感觉砍的话，
 * 很可能砍掉一堆共享同一个循环的时域统计量（省不了多少），而留着真正贵的 FFT。
 *
 * x86 的绝对时间跟 M4F 没有可比性，但**各部分之间的比例**是可以参考的——
 * 它们都是同一类浮点运算，没有哪一部分特别吃某种 x86 专有指令。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tm_features.h"
#include "tm_feat_cfg.h"

/* 从 tm_features.c 里借出来的内部函数原型，用 -DTM_BENCH 编译时会被导出 */
void tm_bench_time_stats(const tm_feat_cfg_t *c, const float *x, int n, float *out);
void tm_bench_freq_stats(const tm_feat_cfg_t *c, const float *x, int n, float *out);

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    const int N = 20000;
    static float x[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float out[TM_FEAT_DIM];
    float t11[16], f8[16];

    for (int i = 0; i < TM_FEAT_N_CH * TM_FEAT_N_T; i++) {
        x[i] = (float)((i * 37 % 211) - 105) * 0.1f;
    }

    double t0 = now();
    for (int i = 0; i < N; i++) tm_features(&tm_feat_cfg, x, out);
    double whole = (now() - t0) / N;

    t0 = now();
    for (int i = 0; i < N; i++) tm_bench_time_stats(&tm_feat_cfg, x, TM_FEAT_N_T, t11);
    double one_time = (now() - t0) / N;

    t0 = now();
    for (int i = 0; i < N; i++) tm_bench_freq_stats(&tm_feat_cfg, x, TM_FEAT_N_T, f8);
    double one_freq = (now() - t0) / N;

    /* 整套里各跑了多少次：时域 n_ch + acc模长 + gyro模长 + jerk模长；
     * 频域 min(6,n_ch) + 两个模长 */
    const int n_time = TM_FEAT_N_CH + 3;
    const int n_freq = (TM_FEAT_N_CH < 6 ? TM_FEAT_N_CH : 6) + 2;

    printf("一个窗口（%d 点 × %d 通道，%d 维特征）\n\n",
           TM_FEAT_N_T, TM_FEAT_N_CH, TM_FEAT_DIM);
    printf("  整套特征提取        %8.2f us\n", whole * 1e6);
    printf("  单次时域 11 维      %8.3f us  × %2d 次 = %7.2f us (%4.1f%%)\n",
           one_time * 1e6, n_time, one_time * n_time * 1e6,
           100.0 * one_time * n_time / whole);
    printf("  单次频域 8 维(含FFT)%8.3f us  × %2d 次 = %7.2f us (%4.1f%%)\n",
           one_freq * 1e6, n_freq, one_freq * n_freq * 1e6,
           100.0 * one_freq * n_freq / whole);
    printf("  其余（模长/jerk/相关系数/SMA）        %7.2f us (%4.1f%%)\n",
           (whole - one_time * n_time - one_freq * n_freq) * 1e6,
           100.0 * (whole - one_time * n_time - one_freq * n_freq) / whole);
    printf("\n  一次频域 ≈ %.1f 次时域\n", one_freq / one_time);
    return 0;
}
