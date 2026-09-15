/* RF 紧凑编码那条路线的 .so：tm_features + tm_forest_c（7 字节节点 + uint8 叶子）。
 *
 * 跟 tm_host_rf.c（老的 SoA 编码）**导出完全相同的符号名**，所以 Python 那边
 * 一个 RfEngine 就能同时用——它不需要知道底下是哪种编码。
 * 编译哪一份由导出目录里有什么文件决定（见 serve.build_rf）。
 *
 * 为什么不合成一个带 #ifdef 的文件：两种编码的模型头文件名、结构体、
 * 遍历函数全都不同，合起来就是一半代码被宏包着，而"只导了其中一种"时
 * 那一半会以一堆 include 错误的形式炸出来，而不是一句话说清缺什么。
 */

#include <string.h>

#include "tm_features.h"
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"
#include "tm_feat_cfg.h"

#if defined(__has_include)
#if __has_include("tm_forest_c_golden.h")
#include "tm_forest_c_golden.h"
#define TM_HAS_FC_GOLDEN 1
#endif
#endif

int thr_n_ch(void)       { return tm_feat_cfg.n_ch; }
int thr_n_t(void)        { return tm_feat_cfg.n_t; }
int thr_n_classes(void)  { return TM_FC_N_CLASSES; }
int thr_n_features(void) { return TM_FC_N_FEATURES; }
int thr_feat_dim(void)   { return TM_FEAT_DIM; }

const char *thr_class_name(int i)
{
    if (i < 0 || i >= TM_FC_N_CLASSES) {
        return "";
    }
    return TM_FC_CLASS_NAMES[i];
}

/* wins  : float [n][n_ch][n_t]，重力对齐后、**原始量纲**（森林的阈值就是按
 *         原始特征值训的，不做归一化）
 * proba : float [n][n_classes]
 * 返回 0 成功；-1 = 特征提取失败；-2 = 特征维度跟森林对不上 */
int thr_infer_batch(const float *wins, int n, float *proba, int8_t *classes)
{
    static float feat[TM_FEAT_DIM];
    static int32_t votes[TM_FC_N_CLASSES];
    const int stride = tm_feat_cfg.n_ch * tm_feat_cfg.n_t;

    if (TM_FEAT_DIM != TM_FC_N_FEATURES) {
        /* 对不上**不会崩**——森林按下标读越界的特征，读到相邻内存，
         * 给出一个看起来完全正常的概率。所以在这里拦。 */
        return -2;
    }

    for (int i = 0; i < n; i++) {
        if (tm_features(&tm_feat_cfg, wins + (size_t)i * stride, feat) != 0) {
            return -1;
        }
        classes[i] = (int8_t)tm_forest_c_predict(&tm_forest_c, feat, votes);

        /* 端上不做这一步：argmax 对正的常数缩放不变，所以板子直接对整数票数
         * 取 argmax。这里还原成概率只是为了给平台一个置信度。 */
        int32_t total = 0;
        for (int c = 0; c < TM_FC_N_CLASSES; c++) {
            total += votes[c];
        }
        float *out = proba + (size_t)i * TM_FC_N_CLASSES;
        for (int c = 0; c < TM_FC_N_CLASSES; c++) {
            out[c] = total > 0 ? (float)votes[c] / (float)total
                               : 1.0f / (float)TM_FC_N_CLASSES;
        }
    }
    return 0;
}

int thr_features(const float *win, float *out)
{
    return tm_features(&tm_feat_cfg, win, out);
}

/* golden 自检。紧凑版存的是**整数票数**——整数累加跟指令集无关，
 * 所以对不上一定是编码或解析错了，不可能是"数值误差"。
 * 这比浮点的 golden 更有诊断力。
 *
 * 返回不一致的个数；-2 = 没有 golden（**不算通过**）。 */
int thr_selftest_forest(void)
{
#if defined(TM_HAS_FC_GOLDEN)
    static int32_t votes[TM_FC_N_CLASSES];
    int bad = 0;
    for (int i = 0; i < TM_FC_GOLDEN_N; i++) {
        const float *x = tm_forest_c_golden_in + (size_t)i * TM_FC_N_FEATURES;
        const int32_t *want = tm_forest_c_golden_out + (size_t)i * TM_FC_N_CLASSES;
        (void)tm_forest_c_predict(&tm_forest_c, x, votes);
        for (int c = 0; c < TM_FC_N_CLASSES; c++) {
            if (votes[c] != want[c]) {
                bad++;
            }
        }
    }
    return bad;
#else
    return -2;
#endif
}

int thr_golden_n(void)
{
#if defined(TM_HAS_FC_GOLDEN)
    return TM_FC_GOLDEN_N;
#else
    return 0;
#endif
}

/* 紧凑版暂时没有"整条链"的 golden（从窗口进那份）。
 * 返回 -2 而不是 0，**"没有"不能当成"通过"**。 */
int thr_selftest_pipeline(void) { return -2; }
int thr_pipeline_golden_n(void) { return 0; }
