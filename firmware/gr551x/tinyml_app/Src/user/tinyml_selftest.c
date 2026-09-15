/* 上板自检：跑 golden vector，逐字节核对。
 *
 * **这是上板之后的第一件事**，不是直接喂真实数据看准确率。原因：准确率低有十几种
 * 可能（模型不行、特征错、编译选项错、传感器量程不对……），而 golden vector 只有
 * 一种解释——对不上就是工具链或编译选项的问题，跟模型无关。先把这一层排除掉，
 * 后面看到的准确率才有意义。
 *
 * 不依赖 SDK，也不依赖 BLE：可以先在 PC 上用 gcc 编出来跑一遍（tests/ 就是这么做的），
 * 再交叉编译进固件。两边都过，才说明"同一份 C 在两种编译器下行为一致"。
 */

#include "tinyml_selftest.h"

#include <string.h>

/* 两条路线各自的 golden vector。哪条都可能没导出，所以都用宏隔开——
 * 只带 RF 的固件不该因为缺 CNN 的头文件而编不过。 */
#if defined(TM_HAS_CNN)
#include "tm_runtime.h"
#include "tm_model.h"
#include "tm_golden.h"
#endif

#if defined(TM_HAS_RF)
#include "tm_forest.h"
#include "tm_features.h"
#include "tm_forest_model.h"
#include "tm_feat_cfg.h"
#if defined(TM_HAS_RF_GOLDEN)
#include "tm_forest_golden.h"
#endif
#if defined(TM_HAS_PIPELINE_GOLDEN)
#include "tm_pipeline_golden.h"
#endif
#endif

#if defined(TM_HAS_CNN)
static int selftest_cnn(tm_selftest_report_t *r)
{
    static int8_t arena[TM_ARENA_BYTES];
    int8_t out[TM_N_CLASSES];

    for (int i = 0; i < TM_GOLDEN_N; i++) {
        const int8_t *in = tm_golden_in + (size_t)i * TM_N_CH * TM_N_T;
        if (tm_invoke(&tm_model, in, out, arena, (int)sizeof(arena)) != 0) {
            r->fail_index = i;
            r->fail_kind = TM_SELFTEST_INVOKE_FAILED;
            return -1;
        }
        const int8_t *want = tm_golden_out + (size_t)i * TM_N_CLASSES;
        if (memcmp(out, want, TM_N_CLASSES) != 0) {
            /* 逐字节比，不是比 argmax。只比 argmax 的话，一个已经算错、只是恰好
             * 还没把类别翻过去的实现能一路混到量产。 */
            r->fail_index = i;
            r->fail_kind = TM_SELFTEST_MISMATCH;
            r->got = out[0];
            r->want = want[0];
            return -1;
        }
        r->n_checked++;
    }
    return 0;
}
#endif

#if defined(TM_HAS_RF) && defined(TM_HAS_RF_GOLDEN)
static int selftest_rf(tm_selftest_report_t *r)
{
    static float proba[TM_F_N_CLASSES];

    for (int i = 0; i < TM_F_GOLDEN_N; i++) {
        const float *x = tm_forest_golden_in + (size_t)i * TM_F_N_FEATURES;
        tm_forest_predict(&tm_forest, x, proba);
        const float *want = tm_forest_golden_out + (size_t)i * TM_F_N_CLASSES;
        for (int c = 0; c < TM_F_N_CLASSES; c++) {
            /* 比的是 float 的**位模式**，不是差值小于某个阈值。
             * 用容差的话，"编译器开了 -ffast-math" 这种问题会被放过去——
             * 它造成的差异往往正好在容差里面，但它会随输入放大。 */
            uint32_t a, b;
            memcpy(&a, &proba[c], sizeof a);
            memcpy(&b, &want[c], sizeof b);
            if (a != b) {
                r->fail_index = i;
                r->fail_kind = TM_SELFTEST_MISMATCH;
                r->got_bits = a;
                r->want_bits = b;
                return -1;
            }
        }
        r->n_checked++;
    }
    return 0;
}
#endif

#if defined(TM_HAS_RF) && defined(TM_HAS_PIPELINE_GOLDEN)
/* 整条链：原始窗口 → 特征 → 森林。
 *
 * 这一段**不只是多验一层**。链接时 --gc-sections 会把没人调的代码整段丢掉——
 * 自检不走完整条链的话，tm_features 压根不会被链进镜像，量出来的固件体积是假的
 * （实测过：不走整条链时 tm_features/tm_invoke 在最终镜像里根本不存在）。
 * 而且它验的是中间那道接缝：特征排列顺序跟模型训练时对不对得上。 */
static int selftest_pipeline(tm_selftest_report_t *r)
{
    static float feat[TM_FEAT_DIM];
    static float proba[TM_F_N_CLASSES];

    for (int i = 0; i < TM_P_GOLDEN_N; i++) {
        const float *x = tm_pipeline_in + (size_t)i * TM_P_N_CH * TM_P_N_T;
        if (tm_features(&tm_feat_cfg, x, feat) != 0) {
            r->fail_index = i;
            r->fail_kind = TM_SELFTEST_INVOKE_FAILED;
            return -1;
        }
        tm_forest_predict(&tm_forest, feat, proba);
        const float *want = tm_pipeline_proba + (size_t)i * TM_F_N_CLASSES;
        for (int c = 0; c < TM_F_N_CLASSES; c++) {
            uint32_t a, b;
            memcpy(&a, &proba[c], sizeof a);
            memcpy(&b, &want[c], sizeof b);
            if (a != b) {
                r->fail_index = i;
                r->fail_kind = TM_SELFTEST_MISMATCH;
                r->got_bits = a;
                r->want_bits = b;
                return -1;
            }
        }
        r->n_checked++;
    }
    return 0;
}
#endif

int tm_selftest_run(tm_selftest_report_t *r)
{
    memset(r, 0, sizeof(*r));
    r->fail_index = -1;

#if defined(TM_HAS_CNN)
    if (selftest_cnn(r) != 0) return -1;
#endif
#if defined(TM_HAS_RF) && defined(TM_HAS_RF_GOLDEN)
    if (selftest_rf(r) != 0) return -1;
#endif
#if defined(TM_HAS_RF) && defined(TM_HAS_PIPELINE_GOLDEN)
    if (selftest_pipeline(r) != 0) return -1;
#endif

    if (r->n_checked == 0) {
        /* 一条都没验到不能当成"通过"。没导 golden vector、或者宏没定义对的时候
         * 会走到这里——而那正是最需要有人知道的情况。 */
        r->fail_kind = TM_SELFTEST_NO_VECTORS;
        return -1;
    }
    return 0;
}

const char *tm_selftest_explain(const tm_selftest_report_t *r)
{
    switch (r->fail_kind) {
    case TM_SELFTEST_OK:
        return "golden vector 全部通过";
    case TM_SELFTEST_NO_VECTORS:
        return "一条 golden vector 都没有——没导出，或者 TM_HAS_CNN/TM_HAS_RF 没定义。"
               "这不算通过。";
    case TM_SELFTEST_INVOKE_FAILED:
        return "推理直接失败了（多半是 arena 不够）。看 TM_ARENA_BYTES 是不是跟"
               "导出的模型配套。";
    case TM_SELFTEST_MISMATCH:
        return "结果跟 PC 上算的不一样。**先别怀疑模型**——按这个顺序查："
               "①编译选项漏了 -ffp-contract=off，或者别处塞了 -ffast-math；"
               "②导出的 tm_*.c 跟 PC 上验过的不是同一份；"
               "③int 宽度/对齐的假设在这个编译器上不成立。";
    default:
        return "未知";
    }
}
