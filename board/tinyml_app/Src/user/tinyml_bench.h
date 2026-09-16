/* 上板测速的结果。板子一上电打印，不用连调试器。 */

#ifndef TINYML_BENCH_H
#define TINYML_BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TM_BENCH_CPU_HZ
/* GR5513 主频。**改了主频这里要跟着改**，否则微秒数是错的而周期数是对的——
 * 那种不一致最难发现，因为两个数都"看起来合理"。 */
#define TM_BENCH_CPU_HZ 64000000u
#endif

typedef struct {
    int ok;                 /* 0 正常；-1 = 周期计数器没走，下面的数全不可信 */
    uint32_t cpu_hz;
    /* first = cache 冷的第一次（产品实际场景：每秒醒一次，cache 是冷的）
     * mean  = 连续几轮的均值，只用来看抖动 */
    uint32_t cnn_first,  cnn_mean;
    uint32_t feat_first, feat_mean;   /* RF 那条的 193 维特征提取 */
    uint32_t rf_first,   rf_mean;     /* RF 那条的森林遍历 */
} tm_bench_report_t;

/* 跑一轮测速。返回 0 成功，-1 表示周期计数器不可用。 */
int tm_bench_run(tm_bench_report_t *rep);

/* 周期数 → 微秒。 */
uint32_t tm_bench_report_us(uint32_t cycles, const tm_bench_report_t *rep);

/* 占空比，千分之几：一次推理占"多久推一次"的比例。
 * 这个数直接回答功耗问题——它就是 CPU 醒着的时间占比。 */
uint32_t tm_bench_duty_permille(uint32_t cycles, const tm_bench_report_t *rep,
                                uint32_t window_ms);

#ifdef __cplusplus
}
#endif

#endif /* TINYML_BENCH_H */
