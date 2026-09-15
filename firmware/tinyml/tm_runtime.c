#include "tm_runtime.h"

#include <string.h>

/* ── 重量化：跟 python/tinyml/fixedpoint.py 逐位一致 ──────────────────────── */

int32_t tm_saturating_rounding_doubling_high_mul(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN) {
        return INT32_MAX;  /* 真值 2^31，int32 装不下 */
    }
    int64_t ab = (int64_t)a * (int64_t)b;
    /* 四舍五入要分正负：统一 +2^30 的话负数会整体偏一个 LSB */
    int64_t nudge = (ab >= 0) ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) >> 31);
}

int32_t tm_rounding_divide_by_pot(int32_t x, int exponent)
{
    if (exponent == 0) {
        return x;
    }
    /* 走 uint32 再转回来：exponent 取到 31 时 (1<<31) 在 int 上是未定义行为。
     * x86 上照样"算得出来"，换个编译器 / 换到 Cortex-M 上就可能不是你想的那个数。
     * 这个坑是 UBSan 逮出来的，不是看出来的。 */
    const int32_t mask = (int32_t)(((uint32_t)1u << exponent) - 1u);
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t tm_multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int shift)
{
    const int left_shift = shift > 0 ? shift : 0;
    const int right_shift = shift < 0 ? -shift : 0;
    /* 同样走 uint32：有符号数左移溢出是未定义行为。真实模型里 shift 恒为负
     * （重量化总是在缩小），这条左移分支跑不到，但不能靠"跑不到"来保证正确。
     * 这里的语义定为**按 32 位回绕**，Python 那边照着做，两边才对得上。 */
    int32_t v = tm_saturating_rounding_doubling_high_mul(
        (int32_t)((uint32_t)x << left_shift), multiplier);
    return tm_rounding_divide_by_pot(v, right_shift);
}

static int8_t requant(int32_t acc, int32_t mult, int32_t shift, int8_t out_zp, int relu)
{
    int32_t v = tm_multiply_by_quantized_multiplier(acc, mult, (int)shift) + out_zp;
    const int32_t lo = relu ? out_zp : -128;
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    return (int8_t)v;
}

/* ── 算子 ─────────────────────────────────────────────────────────────────
 *
 * 全部是最朴素的三重循环，没有 CMSIS-NN、没有 SIMD。**这是有意的**：第一版要的是
 * "板上结果跟 PC 一模一样"，不是快。快是后面的事——GR5513 的 M4F 带 DSP 指令，
 * 换 CMSIS-NN 的 arm_convolve_* 能再快几倍，到那时这套 golden vector 正好用来
 * 证明换完之后结果没变。先有对照，再谈优化。
 */

static void conv1d(const tm_layer_t *L, const int8_t *in, int t_in, int8_t *out, int *t_out)
{
    const int k = L->k;
    const int pad = L->pad;
    const int to = t_in + 2 * pad - k + 1;
    for (int o = 0; o < L->out_ch; o++) {
        const int8_t *wo = L->w + (size_t)o * L->in_ch * k;
        for (int t = 0; t < to; t++) {
            int32_t acc = L->bias[o];
            for (int c = 0; c < L->in_ch; c++) {
                const int8_t *wc = wo + (size_t)c * k;
                const int8_t *xc = in + (size_t)c * t_in;
                for (int j = 0; j < k; j++) {
                    /* padding 位置的贡献是 0。**不是补 in_zp 再减 in_zp** ——
                     * 那样写结果一样但多一次读越界内存；直接跳过既对又安全。
                     * 越界的判断放在最内层看着浪费，但 pad 通常是 1，
                     * 分支预测几乎全中，而把边界拆成三段循环的写法
                     * 是这套代码里最容易写岔、又最不容易被测出来的地方。 */
                    const int ti = t - pad + j;
                    if (ti < 0 || ti >= t_in) {
                        continue;
                    }
                    acc += (int32_t)wc[j] * ((int32_t)xc[ti] - L->in_zp);
                }
            }
            out[(size_t)o * to + t] = requant(acc, L->mult[o], L->shift[o], L->out_zp, L->relu);
        }
    }
    *t_out = to;
}

static void maxpool1d(const tm_layer_t *L, const int8_t *in, int ch, int t_in,
                      int8_t *out, int *t_out)
{
    const int to = t_in / L->pool;  /* 尾巴不够一格就丢，不补零——跟 Python 一致 */
    for (int c = 0; c < ch; c++) {
        for (int t = 0; t < to; t++) {
            const int8_t *p = in + (size_t)c * t_in + t * L->pool;
            int8_t m = p[0];
            for (int j = 1; j < L->pool; j++) {
                if (p[j] > m) m = p[j];
            }
            out[(size_t)c * to + t] = m;
        }
    }
    *t_out = to;
}

static void dense(const tm_layer_t *L, const int8_t *in, int n_in, int8_t *out)
{
    for (int o = 0; o < L->out_ch; o++) {
        const int8_t *wo = L->w + (size_t)o * n_in;
        int32_t acc = L->bias[o];
        for (int i = 0; i < n_in; i++) {
            acc += (int32_t)wo[i] * ((int32_t)in[i] - L->in_zp);
        }
        out[o] = requant(acc, L->mult[o], L->shift[o], L->out_zp, L->relu);
    }
}

/* ── 前向 ─────────────────────────────────────────────────────────────── */

int tm_invoke(const tm_model_t *m, const int8_t *input, int8_t *out,
              int8_t *arena, int arena_bytes)
{
    /* arena 切成两块乒乓缓冲。一块的大小按"最大中间张量"算，见 tm_model.h。 */
    const int half = arena_bytes / 2;
    if (half <= 0) {
        return -1;
    }
    int8_t *bufs[2] = { arena, arena + half };
    int cur = 0;

    int ch = m->n_ch;
    int t = m->n_t;
    if ((size_t)ch * t > (size_t)half) {
        return -1;
    }
    memcpy(bufs[cur], input, (size_t)ch * t);

    for (int i = 0; i < m->n_layers; i++) {
        const tm_layer_t *L = &m->layers[i];
        const int8_t *src = bufs[cur];
        int8_t *dst = bufs[1 - cur];
        int t_next = t, ch_next = ch;
        switch (L->op) {
        case TM_CONV1D:
            if ((size_t)L->out_ch * (t + 2 * L->pad - L->k + 1) > (size_t)half) return -1;
            conv1d(L, src, t, dst, &t_next);
            ch_next = L->out_ch;
            break;
        case TM_MAXPOOL1D:
            if ((size_t)ch * (t / L->pool) > (size_t)half) return -1;
            maxpool1d(L, src, ch, t, dst, &t_next);
            break;
        case TM_DENSE:
            if ((size_t)L->out_ch > (size_t)half) return -1;
            dense(L, src, ch * t, dst);
            ch_next = L->out_ch;
            t_next = 1;
            break;
        default:
            return -1;
        }
        ch = ch_next;
        t = t_next;
        cur = 1 - cur;
    }

    memcpy(out, bufs[cur], (size_t)m->n_classes);
    return 0;
}

int tm_argmax(const int8_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {  /* 严格大于 → 并列取下标最小的，跟 np.argmax 一致 */
            best = i;
        }
    }
    return best;
}
