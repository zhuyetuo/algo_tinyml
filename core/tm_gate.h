/* 动作门：未佩戴 / 静止 不交给 int8 CNN，按窗口内 |a| 的抖动判。
 *
 * 为什么：int8 量化之后（量程按整段数据定，|a| 一格约 0.002g），静止（抖动 ~0.005g）和
 * 未佩戴（~0.0005g）只差一两个 LSB，CNN 根本分不开——imu_train train_v2 的交叉验证里
 * CNN 的静止 F1 只有 0.32。而「静止」本来就是按 |a| 抖动定义的（train_v2 still_motion_g），
 * 用同一个量判反而最准。
 *
 *   std(|a|) <  notworn_g            → 未佩戴
 *   std(|a|) <  still_g（still >= 0） → 静止
 *   否则                               → CNN 在其余类别里选（这两类不参与 argmax）
 *
 * **跟 imu_train 的 src/ml/train_v2.apply_motion_gate、src/infer_csv_scratch._apply_motion_gate
 * 同一套规则**，参数来自模型的 dl_cnn_best.json（motion_gate），导出时写进 tm_gate_cfg.h。
 * std 用两遍法、double 累加，跟 numpy 的 np.std 同一个算法；正好压在阈值上的窗口两边可能不同，
 * 那种窗口在阈值附近、判哪边都说得过去。
 */
#ifndef TM_GATE_H
#define TM_GATE_H

#include <math.h>

/* x: 一个通道的 n_t 个点（channel-first 窗口里 |a| 那一行）。
 * 返回被门判定的类别下标；-1 = 动起来了，交给 CNN。 */
static inline int tm_motion_gate(const float *x, int n_t, double notworn_g, double still_g,
                                 int notworn_cls, int still_cls)
{
    double m = 0.0, v = 0.0;
    for (int i = 0; i < n_t; i++) {
        m += (double)x[i];
    }
    m /= (double)n_t;
    for (int i = 0; i < n_t; i++) {
        const double d = (double)x[i] - m;
        v += d * d;
    }
    const double sd = sqrt(v / (double)n_t);
    if (notworn_cls >= 0 && sd < notworn_g) {
        return notworn_cls;
    }
    if (still_cls >= 0 && still_g > 0.0 && sd < still_g) {
        return still_cls;
    }
    return -1;
}

#endif /* TM_GATE_H */
