/* RF 那条路线的 .so：tm_features（193 维手工特征）+ tm_forest（森林）。
 *
 * 跟 tm_host.c（CNN 那条）分开两个编译单元，不是合成一个带开关的——
 * 两条路线的导出文件名不同、符号不同，合在一起就得到处写 #ifdef，
 * 而"只带其中一条时另一条编不过"这种问题会在最不方便的时候冒出来。
 *
 * **特征在 C 里算，不在 Python 里算**。这是整件事的重点：
 * 平台上看到的结果必须是板子会算出来的结果，而板子算的是 tm_features 这一份。
 * 用 imu_train 的 extract_features（scipy、float64）算出来再喂给森林，
 * 得到的是"服务器上的 RF"，跟板上差多少没人知道——那正是要消灭的东西。
 */

#include <string.h>

#include "tm_features.h"
#include "tm_forest.h"
#include "tm_forest_model.h"
#include "tm_feat_cfg.h"

int thr_n_ch(void)       { return tm_feat_cfg.n_ch; }
int thr_n_t(void)        { return tm_feat_cfg.n_t; }
int thr_n_classes(void)  { return TM_F_N_CLASSES; }
int thr_n_features(void) { return TM_F_N_FEATURES; }
int thr_feat_dim(void)   { return TM_FEAT_DIM; }

const char *thr_class_name(int i)
{
    if (i < 0 || i >= TM_F_N_CLASSES) {
        return "";
    }
    return TM_F_CLASS_NAMES[i];
}

/* wins  : float [n][n_ch][n_t]，**重力对齐之后、原始量纲**（不归一化——
 *         森林的阈值就是按原始特征值训的）
 * proba : float [n][n_classes]，调用方给
 * 返回 0 成功；-1 表示窗口尺寸超出 tm_features 的编译期上限。 */
int thr_infer_batch(const float *wins, int n, float *proba, int8_t *classes)
{
    static float feat[TM_FEAT_DIM];
    const int stride = tm_feat_cfg.n_ch * tm_feat_cfg.n_t;

    for (int i = 0; i < n; i++) {
        if (tm_features(&tm_feat_cfg, wins + (size_t)i * stride, feat) != 0) {
            return -1;
        }
        /* 特征维度必须跟森林训练时一致。对不上的话森林会按下标去读越界的特征，
         * 而那**不会崩**——它只是读到相邻内存，给出一个看起来正常的概率。 */
        if (TM_FEAT_DIM != TM_F_N_FEATURES) {
            return -2;
        }
        classes[i] = (int8_t)tm_forest_predict(&tm_forest, feat,
                                               proba + (size_t)i * TM_F_N_CLASSES);
    }
    return 0;
}

/* 只算特征，不跑森林。用来单独验"C 的特征跟 Python 参考一致"——
 * 混在一起的话，一旦对不上，分不清是特征错了还是森林错了。 */
int thr_features(const float *win, float *out)
{
    return tm_features(&tm_feat_cfg, win, out);
}
