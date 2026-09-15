/* 采样 → 窗口的那一层。把 IMU 连续吐出来的样本攒成固定长度的窗口，攒够一个就
 * 回调一次推理。
 *
 * 为什么要单独一层、还要能测：这层的 bug 很隐蔽。窗口没对齐、hop 算错、量化用错
 * scale——模型再对也白搭，而表现出来只是"准确率比训练时低"，跟推理代码看起来
 * 毫无关系。所以它跟 tm_runtime 一样，有 Python 参考实现逐位对答案。
 *
 * 环形缓冲按 [C][T] 存（通道在前），跟 tm_runtime 的输入布局一致，出窗口时不用转置。
 */

#ifndef TM_WINDOW_H
#define TM_WINDOW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int8_t *buf;        /* 调用方给：n_ch * n_t 字节 */
    int16_t n_ch;
    int16_t n_t;
    int16_t hop;        /* 每前进多少个样本出一个窗口。hop < n_t 就是重叠窗 */
    int16_t head;       /* 下一个样本写到哪一列 */
    int16_t filled;     /* 已经攒了多少个样本，封顶 n_t */
    int16_t since;      /* 上次出窗之后又攒了多少个 */
    float in_scale;     /* 实数 → int8 的量化参数，来自 tm_model.h */
    int8_t in_zp;
} tm_window_t;

void tm_window_init(tm_window_t *w, int8_t *buf, int n_ch, int n_t, int hop,
                    float in_scale, int8_t in_zp);

/* 喂一个样本（n_ch 个 float，单位跟训练时一致：加速度 m/s²、角速度 °/s）。
 * 返回 1 表示这一拍凑齐了一个窗口，已经写进 out（int8 [n_ch][n_t]，时间从旧到新）；
 * 返回 0 表示还没到。
 *
 * out 由调用方给，不复用内部缓冲：内部是环形的，直接把它交出去的话，下一个样本
 * 进来就会改写正在被推理读的数据。多这一次 memcpy 换掉一整类竞态。 */
int tm_window_push(tm_window_t *w, const float *sample, int8_t *out);

/* 量化一个实数到 int8，跟 Python 侧 QNet.quantize_input 一致（四舍五入 + 饱和）。 */
int8_t tm_quantize(float v, float scale, int8_t zp);

#ifdef __cplusplus
}
#endif

#endif /* TM_WINDOW_H */
