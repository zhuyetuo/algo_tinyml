#include "tm_gbdt.h"

#include <math.h>
#include <stddef.h>

int tm_gbdt_predict(const tm_gbdt_t *m, const float *x, float *margin)
{
    for (int c = 0; c < m->n_classes; c++) {
        margin[c] = m->base_score;
    }

    for (int t = 0; t < m->n_trees; t++) {
        int32_t node = m->tree_offset[t];
        while (m->node_left[node] != -1) {
            const float v = x[m->node_feature[node]];
            int go_left;
            if (v != v) {
                /* NaN。我们的特征不该有 NaN，但真出了（比如某个通道传感器坏了，
                 * 方差为 0 除出 nan），行为必须跟训练时一致——XGBoost 每个节点
                 * 都记了 missing 往哪边走。不处理的话 `v < thr` 对 NaN 恒为假，
                 * 会一律走右边，跟训练时可能相反。 */
                go_left = m->node_missing_left[node];
            } else {
                /* **XGBoost 是 `<`**，不是 sklearn 的 `<=`。写错的话，特征值
                 * 正好等于阈值的样本会走反——而阈值本来就是从样本值来的。 */
                go_left = (v < m->node_threshold[node]);
            }
            node = go_left ? m->node_left[node] : m->node_right[node];
        }
        /* 多分类：每轮给每个类别各一棵，树 t 属于类别 t % n_classes */
        const int c = t % m->n_classes;
        margin[c] += m->leaf_value[m->node_feature[node]];
    }

    int best = 0;
    for (int c = 1; c < m->n_classes; c++) {
        if (margin[c] > margin[best]) {   /* 严格大于 → 并列取下标最小，跟 np.argmax 一致 */
            best = c;
        }
    }
    return best;
}

void tm_gbdt_softmax(float *margin, int n)
{
    float mx = margin[0];
    for (int i = 1; i < n; i++) {
        if (margin[i] > mx) mx = margin[i];
    }
    /* 减最大值再取指数：不减的话 margin 稍大一点 expf 就溢出成 inf，
     * 后面除出来全是 nan，而 nan 比较永远是假、argmax 会返回 0——
     * 表现成"模型总投第一类" */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        margin[i] = expf(margin[i] - mx);
        sum += margin[i];
    }
    for (int i = 0; i < n; i++) {
        margin[i] /= sum;
    }
}
