/* 把固件那份 C 编成一个 .so，让服务端直接调。
 *
 * **关键是这里一行推理代码都没有重写。** tm_prep.c / tm_runtime.c /
 * 导出的 tm_model.c 原样编进来，这个文件只做三件事：把 C 的数组接口
 * 翻译成能跨 ctypes 传的形状、跑自检、报一下模型的形状。
 *
 * 重写一份"服务端版推理"是最省事也最没用的做法：那样跑出来的效果跟板上
 * 是什么关系谁也说不清，而"说不清"正是这整个仓库要消灭的东西。
 *
 * x86 上跑这份 C，跟 Cortex-M4 上跑，哪些部分保证一样：
 *   · tm_invoke 全程整数（int8 权重、int32 累加、定点重量化）——
 *     整数运算跟指令集无关，**逐位一定一样**。
 *   · tm_prep 里有 double 除法和 round()。两边都是 IEEE-754 双精度，
 *     且编译时带 -ffp-contract=off（禁掉 FMA 收缩），理论上也一样，
 *     但 round() 走的是各自的 libm，**这一段只有在真板子上跑过才算验过**。
 * 所以这个服务能验的是：模型导出对不对、整条链接得通不通、效果好不好。
 * 它**不能**替代板上那次 golden vector 比对。
 */

#include <string.h>

#include "tm_prep.h"
#include "tm_runtime.h"
#include "tm_model.h"
#include "tm_golden.h"

static int8_t g_arena[TM_ARENA_BYTES];
static int8_t g_xq[TM_N_CH * TM_N_T];
static int8_t g_out[TM_N_CLASSES];

int th_n_ch(void)      { return TM_N_CH; }
int th_n_t(void)       { return TM_N_T; }
int th_n_classes(void) { return TM_N_CLASSES; }
int th_golden_n(void)  { return TM_GOLDEN_N; }

const char *th_class_name(int i)
{
    if (i < 0 || i >= TM_N_CLASSES) {
        return "";
    }
    return TM_CLASS_NAMES[i];
}

/* win: float [n_ch][n_t]，**原始量纲**（不是归一化过的）。
 * scores: int8 [n_classes]，调用方给。
 * 返回类别下标；-1 表示 arena 不够（导出时算错了才可能发生）。 */
int th_infer(const float *win, int8_t *scores)
{
    tm_prep(&tm_model_prep, win, g_xq);
    if (tm_invoke(&tm_model, g_xq, g_out, g_arena, (int)sizeof g_arena)) {
        return -1;
    }
    memcpy(scores, g_out, TM_N_CLASSES);
    return tm_argmax(g_out, TM_N_CLASSES);
}

/* 一次算 n 个窗口。逐个调 th_infer 也行，但每次跨 ctypes 边界都有开销，
 * 回放几千个窗口时那个开销会盖过推理本身，让"端侧有多快"这个问题失真。 */
int th_infer_batch(const float *wins, int n, int8_t *classes, int8_t *scores)
{
    const int stride = TM_N_CH * TM_N_T;
    for (int i = 0; i < n; i++) {
        int c = th_infer(wins + (size_t)i * stride, scores + (size_t)i * TM_N_CLASSES);
        if (c < 0) {
            return -1;
        }
        classes[i] = (int8_t)c;
    }
    return 0;
}

/* golden vector 自检：拿导出时 Python 算好的输入和输出，用**这份 C** 重算一遍。
 *
 * 注意它从 tm_invoke 开始，不含 tm_prep —— golden 存的是已经量化好的 int8 输入。
 * 这是有意的：tm_prep 那一段有它自己的对照测试（tests/test_prep_c.py），
 * 混在一起的话，一旦对不上，分不清是量化错了还是推理错了。
 *
 * 返回不一致的**字节数**，0 表示逐位相同。 */
int th_selftest(void)
{
    int bad = 0;
    for (int i = 0; i < TM_GOLDEN_N; i++) {
        const int8_t *in = tm_golden_in + (size_t)i * TM_N_CH * TM_N_T;
        const int8_t *want = tm_golden_out + (size_t)i * TM_N_CLASSES;
        if (tm_invoke(&tm_model, in, g_out, g_arena, (int)sizeof g_arena)) {
            return -1;
        }
        for (int c = 0; c < TM_N_CLASSES; c++) {
            if (g_out[c] != want[c]) {
                bad++;
            }
        }
    }
    return bad;
}

/* 给服务端报一下归一化参数，方便核对"用的是不是配套的那份模型"。
 * 版本对不上是这类系统最常见的故障，而它不会报错，只会效果变差。 */
double th_ch_mean(int c) { return (c >= 0 && c < TM_N_CH) ? tm_model_prep.ch_mean[c] : 0.0; }
double th_ch_std(int c)  { return (c >= 0 && c < TM_N_CH) ? tm_model_prep.ch_std[c] : 0.0; }
double th_in_scale(void) { return tm_model_prep.in_scale; }
int    th_in_zp(void)    { return tm_model_prep.in_zp; }
int    th_arena_bytes(void) { return TM_ARENA_BYTES; }
