#include "tm_forest.h"

#include <stddef.h>

int tm_forest_predict(const tm_forest_t *f, const float *x, float *proba)
{
    for (int c = 0; c < f->n_classes; c++) {
        proba[c] = 0.0f;
    }

    for (int t = 0; t < f->n_trees; t++) {
        int32_t node = f->tree_offset[t];
        while (f->node_left[node] != -1) {
            /* sklearn 的判决是 <= 走左。写成 < 的话，特征值正好等于阈值的样本会走反——
             * 而阈值本来就是从样本值来的，"正好等于"一点也不罕见。 */
            node = (x[f->node_feature[node]] <= f->node_threshold[node])
                   ? f->node_left[node] : f->node_right[node];
        }
        const float *p = f->leaf_proba + (size_t)f->node_feature[node] * f->n_classes;
        for (int c = 0; c < f->n_classes; c++) {
            proba[c] += p[c];
        }
    }

    int best = 0;
    for (int c = 0; c < f->n_classes; c++) {
        /* 除以棵数而不是乘以倒数：Python 那边是除法，乘倒数在末位上跟它不等价。
         * 这种一位的差别在阈值附近就会把 argmax 翻过去。 */
        proba[c] = proba[c] / (float)f->n_trees;
        if (c > 0 && proba[c] > proba[best]) {
            best = c;  /* 严格大于 → 并列取下标最小的，跟 np.argmax 一致 */
        }
    }
    return best;
}
