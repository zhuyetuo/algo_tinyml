/* 输入预处理：逐通道 z-score + int8 量化。
 *
 * 这一层跟 tm_runtime 分开，是因为它是**唯一跟训练脚本的超参直接绑死**的地方：
 * ch_mean / ch_std 来自训练时统计的那份数据，换一批数据重训就变。
 * 混进运行时的话，"换了模型忘了换归一化参数"这种错会没有任何征兆。
 */

#ifndef TM_PREP_H
#define TM_PREP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* **用 double 而不是 float**：Python 那一侧这三个数是 float64（.json 里是
     * 十进制文本，读出来就是 double），存成 float 会先丢一次精度，
     * 两边就不可能逐位一致了。8 个通道 = 128 字节，省这个没有意义。 */
    const double *ch_mean; /* [n_ch] 训练集上的逐通道均值 */
    const double *ch_std;  /* [n_ch] 逐通道标准差，必须全为正 */
    double in_scale;       /* 输入量化的 scale */
    int8_t in_zp;          /* 输入量化的 zero_point */
    int16_t n_ch;
    int16_t n_t;
} tm_prep_t;

/* x   : float [n_ch][n_t]，**原始量纲**（加速度 m/s²、角速度 dps），channel-first
 * out : int8 [n_ch][n_t]，直接喂给 tm_invoke
 */
void tm_prep(const tm_prep_t *p, const float *x, int8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TM_PREP_H */
