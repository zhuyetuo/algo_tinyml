/* 端侧 int8 推理运行时。目标：GR5513（Cortex-M4F），但这份代码不依赖任何芯片头文件，
 * 所以在 PC 上用 gcc 直接编译、跟 Python 参考实现逐位对答案——那正是 tests/ 里做的事。
 *
 * 没有 malloc：所有中间缓冲由调用方给，大小在编译期算得出来（见 tm_model.h 里的
 * TM_ARENA_BYTES）。端侧动态分配的问题不是慢，是**失败得很晚很难查**——跑几小时后
 * 某次分配失败，表现成偶发死机。
 *
 * 没有 float：GR5513 有 FPU，但定点全程 int32 更省电、也更容易保证两边一致。
 */

#ifndef TM_RUNTIME_H
#define TM_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TM_CONV1D = 0,
    TM_MAXPOOL1D = 1,
    TM_DENSE = 2
} tm_op_t;

typedef struct {
    tm_op_t op;
    /* conv/dense */
    const int8_t *w;      /* conv: [out][in][k]（行主序）；dense: [out][in] */
    const int32_t *bias;  /* [out] */
    const int32_t *mult;  /* [out] 每个输出通道一个定点乘子 */
    const int32_t *shift; /* [out] 配套的移位 */
    int16_t out_ch;
    int16_t in_ch;
    int16_t k;            /* 卷积核长度；dense 忽略 */
    int16_t pad;          /* 两端各补多少个 0（在减 zero_point 之后的域里）；
                           * pad = k/2 即 PyTorch 的 padding='same'（k 为奇数） */
    int8_t in_zp;
    int8_t out_zp;
    uint8_t relu;
    /* maxpool */
    int16_t pool;
} tm_layer_t;

typedef struct {
    const tm_layer_t *layers;
    int n_layers;
    int16_t n_ch;       /* 输入通道数 */
    int16_t n_t;        /* 输入窗口点数 */
    int16_t n_classes;
    int8_t in_zp;
    /* in_scale/out_scale 只在跟 float 对照时用得上，端侧判类别用不到，
     * 放这里是为了让固件能把它打印出来核对模型版本 */
    float in_scale;
    float out_scale;
    int8_t out_zp;
} tm_model_t;

/* 前向。
 *   input : int8 [n_ch][n_t]，channel-first
 *   out   : int8 [n_classes]
 *   arena : 至少 TM_ARENA_BYTES 字节的临时缓冲，调用方提供（可以是栈上数组）
 * 返回 0 成功，非 0 表示 arena 不够（这时 out 不会被写）。
 */
int tm_invoke(const tm_model_t *m, const int8_t *input, int8_t *out,
              int8_t *arena, int arena_bytes);

/* argmax。并列时返回**下标最小**的那个——跟 Python 的 np.argmax 一致，
 * 不定死的话两边偶尔会给出不同类别，而那种不一致极难复现。 */
int tm_argmax(const int8_t *v, int n);

/* 下面两个导出来是为了能被单独测：重量化的舍入规则是两边最容易写岔的地方。 */
int32_t tm_saturating_rounding_doubling_high_mul(int32_t a, int32_t b);
int32_t tm_rounding_divide_by_pot(int32_t x, int exponent);
int32_t tm_multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int shift);

#ifdef __cplusplus
}
#endif

#endif /* TM_RUNTIME_H */
