#include "tm_runtime.h"

#include <stdint.h>
#include <string.h>

#if TM_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

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

#if !TM_CMSIS_NN
static int8_t requant(int32_t acc, int32_t mult, int32_t shift, int8_t out_zp, int relu)
{
    int32_t v = tm_multiply_by_quantized_multiplier(acc, mult, (int)shift) + out_zp;
    const int32_t lo = relu ? out_zp : -128;
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    return (int8_t)v;
}
#endif

/* ── 算子 ─────────────────────────────────────────────────────────────────
 *
 * 全部是最朴素的三重循环，没有 CMSIS-NN、没有 SIMD。**这是有意的**：第一版要的是
 * "板上结果跟 PC 一模一样"，不是快。快是后面的事——GR5513 的 M4F 带 DSP 指令，
 * 换 CMSIS-NN 的 arm_convolve_* 能再快几倍，到那时这套 golden vector 正好用来
 * 证明换完之后结果没变。先有对照，再谈优化。
 */

#if !TM_CMSIS_NN
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
#endif /* !TM_CMSIS_NN */

/* ── 前向 ─────────────────────────────────────────────────────────────── */

#if TM_CMSIS_NN
/* CMSIS-NN 路：张量按 [T][C]（NHWC，H=1）放，权重用导出脚本给的 [out][k][in] 排法。
 * 全连接当成 k = T、输出长度 1 的卷积来算——跟我们 [out][c*T+t] 的展平顺序一一对应，
 * 这样 per-channel 乘子也能用（CMSIS 的 arm_fully_connected_s8 只支持 per-tensor）。
 * 重量化都是 gemmlowp 那套（乘子 + 移位、两次舍入），但平局处理不同：arm_nn_doubling_high_mult_no_sat
 * 对负数也是 +2^30（向上），tm_saturating_rounding_doubling_high_mul 是向远离零。所以个别输出
 * 会差 1 LSB——golden 自检在这条路上按 ±1 比（见 tm_accel.h）。 */

static int cmsis_scratch_bytes(const tm_model_t *m)
{
    int ch = m->n_ch, t = m->n_t, best = 0;
    for (int i = 0; i < m->n_layers; i++) {
        const tm_layer_t *L = &m->layers[i];
        cmsis_nn_dims in_d = { 1, 1, t, ch }, f_d = { L->out_ch, 1, L->k, ch };
        switch (L->op) {
        case TM_CONV1D:
            t = t + 2 * L->pad - L->k + 1;
            ch = L->out_ch;
            break;
        case TM_MAXPOOL1D:
            t = t / L->pool;
            continue;
        case TM_DENSE:
            f_d.w = t;
            t = 1;
            ch = L->out_ch;
            break;
        default:
            return -1;
        }
        int32_t b = arm_convolve_s8_get_buffer_size(&in_d, &f_d);
        if (b > best) best = (int)b;
    }
    return (best + 3) & ~3;
}

static int cmsis_conv(const tm_layer_t *L, const int8_t *in, int t_in, int ch_in, int k, int pad,
                      int8_t *out, int *t_out, void *scratch, int scratch_bytes)
{
    const int to = t_in + 2 * pad - k + 1;
    cmsis_nn_context ctx = { scratch, scratch_bytes };
    cmsis_nn_conv_params cp;
    memset(&cp, 0, sizeof cp);
    cp.input_offset = -(int32_t)L->in_zp;
    cp.output_offset = (int32_t)L->out_zp;
    cp.stride.w = 1; cp.stride.h = 1;
    cp.padding.w = pad; cp.padding.h = 0;
    cp.dilation.w = 1; cp.dilation.h = 1;
    cp.activation.min = L->relu ? (int32_t)L->out_zp : -128;
    cp.activation.max = 127;
    cmsis_nn_per_channel_quant_params qp = { (int32_t *)L->mult, (int32_t *)L->shift };
    cmsis_nn_dims in_d = { 1, 1, t_in, ch_in };
    cmsis_nn_dims f_d = { L->out_ch, 1, k, ch_in };
    cmsis_nn_dims b_d = { 1, 1, 1, L->out_ch };
    cmsis_nn_dims up_d = { 1, 1, 1, 1 };
    cmsis_nn_dims out_d = { 1, 1, to, L->out_ch };
    if (arm_convolve_s8(&ctx, &cp, &qp, &in_d, in, &f_d, L->w, &b_d, L->bias, &up_d, &out_d, out)
        != ARM_CMSIS_NN_SUCCESS) {
        return -1;
    }
    *t_out = to;
    return 0;
}

static int cmsis_maxpool(const tm_layer_t *L, const int8_t *in, int ch, int t_in,
                         int8_t *out, int *t_out)
{
    const int to = t_in / L->pool;
    cmsis_nn_context ctx = { 0, 0 };
    cmsis_nn_pool_params pp;
    memset(&pp, 0, sizeof pp);
    pp.stride.w = L->pool; pp.stride.h = 1;
    pp.padding.w = 0; pp.padding.h = 0;
    pp.activation.min = -128; pp.activation.max = 127;
    cmsis_nn_dims in_d = { 1, 1, t_in, ch };
    cmsis_nn_dims f_d = { 1, 1, L->pool, 1 };
    cmsis_nn_dims out_d = { 1, 1, to, ch };
    if (arm_max_pool_s8(&ctx, &pp, &in_d, in, &f_d, &out_d, out) != ARM_CMSIS_NN_SUCCESS) {
        return -1;
    }
    *t_out = to;
    return 0;
}

int tm_invoke(const tm_model_t *m, const int8_t *input, int8_t *out,
              int8_t *arena, int arena_bytes)
{
    /* CMSIS 的 im2col 缓冲是 int16，要 4 字节对齐；arena 是 int8 数组，不保证对齐，
     * 这里自己往后挪最多 3 字节（导出的 TM_ARENA_BYTES 已经多给了 4 字节） */
    const int align = (int)((4u - ((uintptr_t)arena & 3u)) & 3u);
    const int scratch = cmsis_scratch_bytes(m);
    if (scratch < 0) {
        return -1;
    }
    const int half = (arena_bytes - align - scratch) / 2;
    if (half <= 0) {
        return -1;
    }
    int8_t *scratch_p = arena + align;
    int8_t *bufs[2] = { scratch_p + scratch, scratch_p + scratch + half };
    int cur = 0;

    int ch = m->n_ch;
    int t = m->n_t;
    if ((size_t)ch * t > (size_t)half) {
        return -1;
    }
    /* [C][T] → [T][C]。输入很小（8×16），这一次转置不值一提 */
    for (int c = 0; c < ch; c++) {
        for (int i = 0; i < t; i++) {
            bufs[0][(size_t)i * ch + c] = input[(size_t)c * t + i];
        }
    }

    for (int i = 0; i < m->n_layers; i++) {
        const tm_layer_t *L = &m->layers[i];
        const int8_t *src = bufs[cur];
        int8_t *dst = bufs[1 - cur];
        int t_next = t, ch_next = ch;
        switch (L->op) {
        case TM_CONV1D:
            if ((size_t)L->out_ch * (t + 2 * L->pad - L->k + 1) > (size_t)half) return -1;
            if (cmsis_conv(L, src, t, ch, L->k, L->pad, dst, &t_next, scratch_p, scratch)) return -1;
            ch_next = L->out_ch;
            break;
        case TM_MAXPOOL1D:
            if ((size_t)ch * (t / L->pool) > (size_t)half) return -1;
            if (cmsis_maxpool(L, src, ch, t, dst, &t_next)) return -1;
            break;
        case TM_DENSE:
            if ((size_t)L->out_ch > (size_t)half || L->in_ch != ch * t) return -1;
            if (cmsis_conv(L, src, t, ch, t, 0, dst, &t_next, scratch_p, scratch)) return -1;
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

#else /* 朴素实现 */

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

#endif /* TM_CMSIS_NN */

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
