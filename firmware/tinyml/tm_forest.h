/* 随机森林的端侧推理。跟 tm_runtime（int8 CNN）是两条独立的路线，互不依赖——
 * 哪条能塞进 flash、哪条跟平台结论一致，是拿数据决定的，不是拿架构决定的。
 *
 * 判决方式照抄 sklearn 的 RandomForestClassifier：把每棵树叶子上的类别概率**求平均**，
 * 不是各棵树 argmax 之后投票。两者结果不一样，而端上要跟平台给同一个结论。
 *
 * 全部是比较，没有乘法——比 CNN 还便宜。贵的是 flash：叶子要存 n_classes 个 float32。
 *
 * 浮点：累加顺序写死成按树下标从小到大，跟 Python 参考实现一致。浮点加法不满足
 * 结合律，换个顺序末位就可能不同。编译时务必带 -ffp-contract=off。
 */

#ifndef TM_FOREST_H
#define TM_FOREST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const int32_t *tree_offset;    /* [n_trees + 1] */
    const uint16_t *node_feature;  /* [n_nodes] 内部节点=特征下标；叶子=leaf_proba 的行号 */
    const float *node_threshold;   /* [n_nodes] 叶子处无意义 */
    const int32_t *node_left;      /* [n_nodes] -1 表示叶子 */
    const int32_t *node_right;     /* [n_nodes] */
    const float *leaf_proba;       /* [n_leaves * n_classes] 行主序 */
    int32_t n_trees;
    int32_t n_features;
    int32_t n_classes;
} tm_forest_t;

/* x: float [n_features]，proba: float [n_classes]（调用方给）。
 * 返回 argmax；并列取下标最小的，跟 np.argmax 一致。 */
int tm_forest_predict(const tm_forest_t *f, const float *x, float *proba);

#ifdef __cplusplus
}
#endif

#endif /* TM_FOREST_H */
