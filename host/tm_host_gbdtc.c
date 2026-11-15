/* GBDT（XGBoost）紧凑编码那条路线的 .so：tm_features + tm_gbdt_c（6 字节节点）。
 *
 * 跟 tm_host_rfc.c **导出完全相同的符号名**，Python 那边同一个 RfEngine 就能用——
 * 它不需要知道底下是森林还是 GBDT。编译哪一份由导出目录里有什么文件决定
 * （见 serve.build_gbdt）。
 *
 * 跟 RF 的区别只有一处：GBDT 出的是每类一个 margin（分数），不是票数。
 * 板上直接对 margin 取 argmax（softmax 保序，不用算）；这里给平台的概率
 * 才过一次 softmax。golden 比的是 margin 的位模式——同样的树、同样的累加顺序，
 * float 加法结果逐位确定，对不上就一定是编码 / 解析 / 特征错了。
 */

#include <math.h>
#include <string.h>

#include "tm_features.h"
#include "tm_gbdt_c.h"
#include "tm_gbdt_c_model.h"
#include "tm_feat_cfg.h"

#if defined(__has_include)
#if __has_include("tm_gbdt_c_golden.h")
#include "tm_gbdt_c_golden.h"
#define TM_HAS_GC_GOLDEN 1
#endif
#if __has_include("tm_gbdt_c_pipeline_golden.h")
#include "tm_gbdt_c_pipeline_golden.h"
#define TM_HAS_GCP_GOLDEN 1
#endif
#endif

int thr_n_ch(void)       { return tm_feat_cfg.n_ch; }
int thr_n_t(void)        { return tm_feat_cfg.n_t; }
int thr_n_classes(void)  { return TM_GC_N_CLASSES; }
int thr_n_features(void) { return TM_GC_N_FEATURES; }
int thr_feat_dim(void)   { return TM_FEAT_DIM; }

const char *thr_class_name(int i)
{
    if (i < 0 || i >= TM_GC_N_CLASSES) {
        return "";
    }
    return TM_GC_CLASS_NAMES[i];
}

#if defined(TM_HAS_GC_GOLDEN)
static int same_bits(float a, float b)
{
    uint32_t x, y;
    memcpy(&x, &a, sizeof x);
    memcpy(&y, &b, sizeof y);
    return x == y;
}
#endif

/* wins  : float [n][n_ch][n_t]，跟 RF 那条一样是原始量纲（树的阈值按原始特征值训的）
 * proba : float [n][n_classes]，margin 过 softmax（只给平台当置信度，板上不算）
 * 返回 0 成功；-1 = 特征提取失败；-2 = 特征维度跟模型对不上 */
int thr_infer_batch(const float *wins, int n, float *proba, int8_t *classes)
{
    static float feat[TM_FEAT_DIM];
    static float margin[TM_GC_N_CLASSES];
    const int stride = tm_feat_cfg.n_ch * tm_feat_cfg.n_t;

    if (TM_FEAT_DIM != TM_GC_N_FEATURES) {
        return -2;
    }
    for (int i = 0; i < n; i++) {
        if (tm_features(&tm_feat_cfg, wins + (size_t)i * stride, feat) != 0) {
            return -1;
        }
        classes[i] = (int8_t)tm_gbdt_c_predict(&tm_gbdt_c_model, feat, margin);
        float mx = margin[0];
        for (int c = 1; c < TM_GC_N_CLASSES; c++) {
            if (margin[c] > mx) {
                mx = margin[c];
            }
        }
        float total = 0.0f;
        float *out = proba + (size_t)i * TM_GC_N_CLASSES;
        for (int c = 0; c < TM_GC_N_CLASSES; c++) {
            out[c] = expf(margin[c] - mx);
            total += out[c];
        }
        for (int c = 0; c < TM_GC_N_CLASSES; c++) {
            out[c] /= total;
        }
    }
    return 0;
}

int thr_features(const float *win, float *out)
{
    return tm_features(&tm_feat_cfg, win, out);
}

/* 从特征进：margin 逐位比。返回不一致的个数；-2 = 没有 golden（**不算通过**） */
int thr_selftest_forest(void)
{
#if defined(TM_HAS_GC_GOLDEN)
    static float margin[TM_GC_N_CLASSES];
    int bad = 0;
    for (int i = 0; i < TM_GC_GOLDEN_N; i++) {
        const float *x = tm_gc_golden_in + (size_t)i * TM_GC_N_FEATURES;
        const float *want = tm_gc_golden_margin + (size_t)i * TM_GC_N_CLASSES;
        (void)tm_gbdt_c_predict(&tm_gbdt_c_model, x, margin);
        for (int c = 0; c < TM_GC_N_CLASSES; c++) {
            if (!same_bits(margin[c], want[c])) {
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
#if defined(TM_HAS_GC_GOLDEN)
    return TM_GC_GOLDEN_N;
#else
    return 0;
#endif
}

/* 整条链：原始窗口 → 特征 → GBDT。比只验模型多覆盖**特征的排列顺序**。
 * 比的是判决（argmax）：特征那一段是 float，Python 参考和 C 末位可能差一点，
 * 只要没有恰好落在阈值上，走的路径就一样、判决就一样。 */
int thr_selftest_pipeline(void)
{
#if defined(TM_HAS_GCP_GOLDEN)
    static float feat[TM_FEAT_DIM];
    static float margin[TM_GC_N_CLASSES];
    int bad = 0;
    for (int i = 0; i < TM_GCP_GOLDEN_N; i++) {
        const float *x = tm_gbdt_c_pipeline_in + (size_t)i * TM_GCP_N_CH * TM_GCP_N_T;
        if (tm_features(&tm_feat_cfg, x, feat) != 0) {
            return -1;
        }
        if (tm_gbdt_c_predict(&tm_gbdt_c_model, feat, margin) != tm_gbdt_c_pipeline_cls[i]) {
            bad++;
        }
    }
    return bad;
#else
    return -2;
#endif
}

int thr_pipeline_golden_n(void)
{
#if defined(TM_HAS_GCP_GOLDEN)
    return TM_GCP_GOLDEN_N;
#else
    return 0;
#endif
}
