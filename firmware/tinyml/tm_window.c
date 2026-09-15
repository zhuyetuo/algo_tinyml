#include "tm_window.h"

#include <math.h>
#include <stddef.h>

int8_t tm_quantize(float v, float scale, int8_t zp)
{
    /* roundf 是"四舍五入、.5 远离零"，跟 numpy 的 np.round（银行家舍入，.5 取偶）
     * **不一样**。这里故意用 roundf 并在 Python 侧照着做 —— 统一成哪一种不重要，
     * 两边一样才重要。传感器数据落在正好 .5 的概率不高，但"不高"不是"没有"，
     * 而这种偶发一位偏差在板上根本查不出来。 */
    float q = roundf(v / scale) + (float)zp;
    if (q < -128.0f) return -128;
    if (q > 127.0f) return 127;
    return (int8_t)q;
}

void tm_window_init(tm_window_t *w, int8_t *buf, int n_ch, int n_t, int hop,
                    float in_scale, int8_t in_zp)
{
    w->buf = buf;
    w->n_ch = (int16_t)n_ch;
    w->n_t = (int16_t)n_t;
    w->hop = (int16_t)(hop > 0 ? hop : n_t);
    w->head = 0;
    w->filled = 0;
    w->since = 0;
    w->in_scale = in_scale;
    w->in_zp = in_zp;
}

int tm_window_push(tm_window_t *w, const float *sample, int8_t *out)
{
    for (int c = 0; c < w->n_ch; c++) {
        w->buf[(size_t)c * w->n_t + w->head] = tm_quantize(sample[c], w->in_scale, w->in_zp);
    }
    w->head = (int16_t)((w->head + 1) % w->n_t);
    if (w->filled < w->n_t) {
        w->filled++;
    }
    w->since++;

    /* 攒满之前不出窗。用半截窗口（后面补零）去推理，等于喂给模型一段训练时
     * 从没见过的信号——它会给出一个看起来正常、实际毫无依据的类别。 */
    if (w->filled < w->n_t || w->since < w->hop) {
        return 0;
    }
    w->since = 0;

    /* 环形展平成时间从旧到新。head 指向下一个要写的位置，也就是最旧的那一列。 */
    for (int c = 0; c < w->n_ch; c++) {
        const int8_t *src = w->buf + (size_t)c * w->n_t;
        int8_t *dst = out + (size_t)c * w->n_t;
        for (int t = 0; t < w->n_t; t++) {
            dst[t] = src[(w->head + t) % w->n_t];
        }
    }
    return 1;
}
