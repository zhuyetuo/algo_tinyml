/* 手工特征的端侧实现（8 通道 193 维 / 3 轴 5 通道 79 维），喂给 tm_forest。
 *
 * 跟 python/tinyml/features.py **逐位一致**（tests/test_features_c.py 现场编译对答案）。
 * 跟 imu_train 的 scipy 版**不是**逐位一致，也做不到（scipy 是 float64、FFT 算法不同）。
 * 差多少要在有 scipy 的机器上量：python/verify_against_scipy.py。
 *
 * 窗函数和 FFT 旋转因子是**从 Python 导出的常量表**，不在 C 里现算：cosf 在不同
 * libm 上末位可能不同，而那是两边对不上的一个来源，堵掉它比事后查便宜得多。
 *
 * 编译务必带 -ffp-contract=off。
 */

#ifndef TM_FEATURES_H
#define TM_FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 编译期上限。写死是为了不用 malloc——端上动态分配失败得很晚、很难查。 */
#ifndef TM_FEAT_MAX_T
#define TM_FEAT_MAX_T 64
#endif
#ifndef TM_FEAT_MAX_NPERSEG
#define TM_FEAT_MAX_NPERSEG 64
#endif

typedef struct {
    int16_t n_t;        /* 窗口点数 */
    int16_t n_ch;       /* 通道数：5（acc + pitch/roll，3 轴）、6（acc+gyr）或 8（+pitch/roll） */
    int16_t nperseg;    /* Welch 的段长，必须是 2 的幂且 <= n_t */
    float fs;           /* 采样率 */
    const float *win;      /* [nperseg] 周期 Hann */
    const float *cos_t;    /* [nperseg/2] */
    const float *sin_t;    /* [nperseg/2] */
    const int16_t *bitrev; /* [nperseg] 位反序下标 */
} tm_feat_cfg_t;

/* x: float [n_ch][n_t]，**通道在前**（跟 tm_window 的输出一致）。
 * out: float [tm_feat_dim(cfg)]。
 * 返回 0 成功；-1 表示 n_t / nperseg 超出编译期上限。 */
int tm_features(const tm_feat_cfg_t *cfg, const float *x, float *out);

/* 特征维度：5 通道 79，6 通道 171，8 通道 193。 */
int tm_feat_dim(const tm_feat_cfg_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* TM_FEATURES_H */
