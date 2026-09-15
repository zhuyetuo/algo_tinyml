/* GBDT（XGBoost）的端侧推理。
 *
 * **跟 tm_forest.c 的树遍历长得几乎一样，但故意没有合并成一份。** 三个区别每一个
 * 写错都不报错、只让结果悄悄不对，合在一起用 if 切换的话，改 RF 那条很容易碰坏
 * GBDT 这条，而两边的 golden vector 又是分开的，不一定当场发现：
 *
 *   1. 判决符号是 `<`（XGBoost），不是 `<=`（sklearn）；
 *   2. 叶子存一个分数，所有树**相加**（不是平均），再加 base_score；
 *   3. 多分类每轮给每个类别各训一棵，树 t 属于类别 t % n_classes。
 *
 * **不做 softmax。** softmax 保序，argmax 直接在 margin 上取结果完全一样，
 * 还省掉一堆 expf（libm 的末位不保证一致，不用它就少一处对不齐的来源）。
 * 真要概率再调 tm_gbdt_softmax。
 *
 * 编译要带 -ffp-contract=off。
 */

#ifndef TM_GBDT_H
#define TM_GBDT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const int32_t *tree_offset;     /* [n_trees + 1] */
    const uint16_t *node_feature;   /* [n_nodes] 内部=特征下标；叶子=leaf_value 行号 */
    const float *node_threshold;    /* [n_nodes] */
    const int32_t *node_left;       /* [n_nodes] -1 表示叶子 */
    const int32_t *node_right;      /* [n_nodes] */
    const uint8_t *node_missing_left; /* [n_nodes] 特征是 NaN 时走左边吗 */
    const float *leaf_value;        /* [n_leaves] 一个分数 */
    int32_t n_trees;
    int32_t n_features;
    int32_t n_classes;
    float base_score;
} tm_gbdt_t;

/* x: float [n_features]，margin: float [n_classes]（调用方给，原始分数不是概率）。
 * 返回 argmax；并列取下标最小的，跟 np.argmax 一致。 */
int tm_gbdt_predict(const tm_gbdt_t *m, const float *x, float *margin);

/* 要概率再调这个。就地把 margin 变成概率。用了 expf（libm），
 * 所以**这一步不保证跟 PC 逐位一致**——判类别不需要它，别为了打印概率
 * 把整条链的一致性搭进去。 */
void tm_gbdt_softmax(float *margin, int n);

#ifdef __cplusplus
}
#endif

#endif /* TM_GBDT_H */
