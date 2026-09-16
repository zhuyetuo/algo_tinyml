#include "tm_gbdt_c.h"

#include <string.h>

/* 一个节点 6 字节，非对齐。用 memcpy 取 float——编译器在 Cortex-M4 上会把它
 * 编成一条 ldr.w（硬件支持非对齐字加载）。直接强转指针是未定义行为，
 * 而且在别的编译器上可能真的出错，memcpy 是既安全又不慢的写法。 */
static inline float node_value(const uint8_t *p)
{
    float v;
    memcpy(&v, p + 1, sizeof v);
    return v;
}

int tm_gbdt_c_predict(const tm_gbdt_c_t *m, const float *x, float *margin)
{
    for (int c = 0; c < m->n_classes; c++) {
        margin[c] = m->base_score;
    }

    for (int t = 0; t < m->n_trees; t++) {
        const uint8_t *p = m->nodes + (size_t)m->tree_offset[t] * TM_GC_NODE_BYTES;
        for (;;) {
            const uint8_t r = p[5];
            const uint8_t off = (uint8_t)(r & TM_GC_RIGHT_MASK);
            if (off == 0u) {
                break;              /* 叶子：value 槽里是叶子分数 */
            }
            const float v = x[p[0]];
            int go_left;
            if (v != v) {
                /* NaN：按训练时记下的方向走。不处理的话 `v < thr` 对 NaN 恒为假，
                 * 会一律走右边，可能跟训练时相反。 */
                go_left = (r & TM_GC_MISSING_LEFT) != 0u;
            } else {
                /* **XGBoost 是 `<`**，不是 sklearn 的 `<=` */
                go_left = (v < node_value(p));
            }
            /* 左孩子恒为下一个节点——先序存储的直接结果，也是这套布局
             * 在 cache 上占便宜的地方 */
            p += go_left ? TM_GC_NODE_BYTES : (size_t)off * TM_GC_NODE_BYTES;
        }
        margin[t % m->n_classes] += node_value(p);
    }

    int best = 0;
    for (int c = 1; c < m->n_classes; c++) {
        if (margin[c] > margin[best]) {   /* 并列取下标最小，跟 np.argmax 一致 */
            best = c;
        }
    }
    return best;
}
