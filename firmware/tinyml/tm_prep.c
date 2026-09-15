#include "tm_prep.h"

#include <math.h>

/* 归一化 + 输入量化，合成一个算子。
 *
 * 为什么一定要有这一步：训练时对输入做了逐通道 z-score（(x - mean) / std），
 * 端上不做同一件事的话，输入分布跟训练时对不上——**效果明显下降但不报错**。
 * 这是那种查起来最费劲的 bug：模型对、推理对、数据也对，就是准确率低一截。
 *
 * 为什么不把 mean/std 折进第一层卷积的权重（数学上完全可行）：
 * 折进去之后端上的输入是原始量纲，而 8 个通道的量纲差两个数量级
 * （加速度 ±40，角速度 ±2000），共用一个 int8 输入 scale 会把加速度那几路
 * 压成几个格子。先归一化再量化，所有通道落到同一个范围，int8 的分辨率才用得满。
 *
 * 为什么用 double 而不是 float：要跟 Python 参考实现逐位一致，而那边是 float64。
 * 代价是 M4F 没有双精度 FPU，这段要走软件浮点——但它只有 n_ch × n_t 次
 * （8 × 16 = 128 次），跟后面几十万次乘加的卷积比可以忽略。
 * 真到了要省这 128 次的时候，再换成定点并用 golden vector 证明结果没变。
 */

void tm_prep(const tm_prep_t *p, const float *x, int8_t *out)
{
    for (int c = 0; c < p->n_ch; c++) {
        const double mean = p->ch_mean[c];
        const double inv_std = 1.0 / p->ch_std[c];
        for (int t = 0; t < p->n_t; t++) {
            const double v = ((double)x[(size_t)c * p->n_t + t] - mean) * inv_std;
            /* 两步走，跟 Python 一样：先归一化，再除 in_scale。
             * 合成一个 (x*a + b) 会改变舍入，两边就对不上了——
             * 而那种差异只有几个 LSB，看起来完全像"正常的数值误差"。 */
            const double q = v / p->in_scale;
            /* round 是四舍五入远离零，跟 quantize_input_ref 的
             * sign(v) * floor(|v| + 0.5) 一致。**不能用 rint/nearbyint**：
             * 那两个默认是银行家舍入，跟 numpy 的 np.round 一样会在 .5 上取偶。 */
            double r = round(q) + (double)p->in_zp;
            if (r < -128.0) r = -128.0;
            if (r > 127.0) r = 127.0;
            out[(size_t)c * p->n_t + t] = (int8_t)r;
        }
    }
}
