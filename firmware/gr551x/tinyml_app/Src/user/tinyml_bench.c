/* 上板测速：拿 golden vector 当输入，跑真实的推理链，数周期。
 *
 * **为什么一定要板子自己测**：x86 上量到 286 µs/窗口，那个数对 64MHz 的 M4F
 * 没有任何参考价值——指令集不同、频率差 50 倍、cache 行为完全不一样。
 * 按 MAC 数乘"每 MAC 几周期"去估，误差轻松 2~3 倍，而"推理要多久"直接决定
 * 占空比和功耗，是拿来做产品决策的数。估出来的不能用。
 *
 * 所以板子一上电就把真实数字打出来：不用连调试器、不用示波器、不用谁去猜。
 *
 * 输入用 golden vector 而不是随机数：那是**真实数据量化出来的**窗口，
 * 走的分支跟实跑一样。随机数在决策树那条会走出完全不同的路径深度，
 * 测出来的耗时是假的。
 */

#include "tinyml_bench.h"

#include "tm_bench.h"

#if defined(TM_HAS_CNN)
#include "tm_runtime.h"
#include "tm_model.h"
#include "tm_golden.h"
#endif

#if defined(TM_HAS_RF)
#include "tm_forest.h"
#include "tm_features.h"
#include "tm_forest_model.h"
#include "tm_feat_cfg.h"
#if defined(TM_HAS_PIPELINE_GOLDEN)
#include "tm_pipeline_golden.h"
#endif
#endif

#ifndef TM_BENCH_REPEAT
/* 重复几轮取均值。第一轮会把 flash 预取和 cache 热起来，所以**第一轮单独报**，
 * 后面几轮取均值——真实场景是每秒一次推理，中间芯片睡着，cache 是冷的，
 * 所以"第一轮"那个数才是产品要关心的，均值只是用来看抖动。 */
#define TM_BENCH_REPEAT 8
#endif

int tm_bench_run(tm_bench_report_t *rep)
{
    rep->ok = 0;
    rep->cnn_first = rep->cnn_mean = 0;
    rep->rf_first = rep->rf_mean = 0;
    rep->feat_first = rep->feat_mean = 0;
    rep->cpu_hz = TM_BENCH_CPU_HZ;

    if (tm_bench_init() != 0) {
        /* 计数器没动。所有耗时都会是 0，而"推理 0 微秒"比一条报错更容易被当真 */
        rep->ok = -1;
        return -1;
    }

#if defined(TM_HAS_CNN)
    {
        static int8_t arena[TM_ARENA_BYTES];
        int8_t out[TM_N_CLASSES];
        tm_bench_t b;
        tm_bench_reset(&b);
        for (int r = 0; r < TM_BENCH_REPEAT; r++) {
            const int8_t *in = tm_golden_in +
                (size_t)(r % TM_GOLDEN_N) * TM_N_CH * TM_N_T;
            uint32_t t0 = tm_bench_now();
            (void)tm_invoke(&tm_model, in, out, arena, (int)sizeof(arena));
            uint32_t d = tm_bench_elapsed(t0, tm_bench_now());
            if (r == 0) {
                rep->cnn_first = d;     /* cache 冷的那一次，产品实际就是这个 */
            }
            tm_bench_accum(&b, d);
        }
        rep->cnn_mean = tm_bench_mean(&b);
    }
#endif

#if defined(TM_HAS_RF) && defined(TM_HAS_PIPELINE_GOLDEN)
    {
        static float feat[TM_FEAT_DIM];
        static float proba[TM_F_N_CLASSES];
        tm_bench_t bf, br;
        tm_bench_reset(&bf);
        tm_bench_reset(&br);
        for (int r = 0; r < TM_BENCH_REPEAT; r++) {
            const float *x = tm_pipeline_in +
                (size_t)(r % TM_P_GOLDEN_N) * TM_P_N_CH * TM_P_N_T;

            /* 特征提取和森林分开计时。RF 这条路上特征提取往往比模型本身还贵，
             * 合在一起报的话，"该优化哪一半"这个问题就没法回答了。 */
            uint32_t t0 = tm_bench_now();
            (void)tm_features(&tm_feat_cfg, x, feat);
            uint32_t t1 = tm_bench_now();
            (void)tm_forest_predict(&tm_forest, feat, proba);
            uint32_t t2 = tm_bench_now();

            uint32_t df = tm_bench_elapsed(t0, t1);
            uint32_t dr = tm_bench_elapsed(t1, t2);
            if (r == 0) {
                rep->feat_first = df;
                rep->rf_first = dr;
            }
            tm_bench_accum(&bf, df);
            tm_bench_accum(&br, dr);
        }
        rep->feat_mean = tm_bench_mean(&bf);
        rep->rf_mean = tm_bench_mean(&br);
    }
#endif

    return 0;
}

uint32_t tm_bench_report_us(uint32_t cycles, const tm_bench_report_t *rep)
{
    return tm_bench_us(cycles, rep->cpu_hz);
}

uint32_t tm_bench_duty_permille(uint32_t cycles, const tm_bench_report_t *rep,
                                uint32_t window_ms)
{
    /* 占空比（千分之几）：一次推理的耗时占"多久推一次"的比例。
     * 这个数比"多少微秒"更能直接回答功耗问题——它就是 CPU 醒着的时间占比。
     * 用千分比不用百分比，是因为预期值在 1% 以下，百分比会全是 0。 */
    if (rep->cpu_hz == 0 || window_ms == 0) {
        return 0;
    }
    uint64_t window_cycles = (uint64_t)rep->cpu_hz * window_ms / 1000ull;
    if (window_cycles == 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)cycles * 1000ull) / window_cycles);
}
