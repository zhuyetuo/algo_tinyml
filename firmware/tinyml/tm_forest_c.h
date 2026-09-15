/* 紧凑随机森林：7 字节一个节点 + uint8 叶子。
 *
 * 跟 tm_forest.c（SoA、float32 叶子）是两份实现，判决**不完全相同**：
 * 叶子量化成 uint8 之后，相差不到 1/255 的两类会翻。那个差异是要实测的，
 * 不是"几乎没有"——python/prune_rf.py --grid 报的 macro-F1 就是量化之后的。
 *
 * 换来的是：
 *   · 体积 24 → 约 9.5 B/节点（含叶子摊销）；
 *   · 一个节点 7 字节连续，一条 32 字节 cache line 放得下 4 个，
 *     而且左孩子就是下一个节点——沿左分支走常常直接命中。
 *     GR5513 只有 8KB cache，这一项比体积更值钱；
 *   · 累加全程整数，板上和 PC **逐位一定一样**（原版是 float32 求和，
 *     换个累加顺序末位就可能不同）。
 */

#ifndef TM_FOREST_C_H
#define TM_FOREST_C_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TM_FC_NODE_BYTES 7

/* 节点字节布局（小端）：
 *   [0]    uint8   内部=特征下标
 *   [1..4] uint32  内部=阈值的位模式；**叶子=叶子表下标**
 *   [5..6] uint16  右孩子相对偏移；**0 表示这是叶子**
 * 左孩子恒为 idx+1，不存。
 */
typedef struct {
    const uint8_t *nodes;       /* [n_nodes * 7] */
    const uint8_t *leaves;      /* [n_leaves * n_classes]，概率 × 255 */
    const int32_t *tree_offset; /* [n_trees + 1] */
    int32_t n_trees;
    int32_t n_features;
    int32_t n_classes;
} tm_forest_c_t;

/* x: float [n_features]；votes: int32 [n_classes]（调用方给）。
 * 返回 argmax。votes 是各树叶子 uint8 之和，**不除以棵数**——
 * argmax 对正的常数缩放不变，端上没必要做那一步除法。 */
int tm_forest_c_predict(const tm_forest_c_t *f, const float *x, int32_t *votes);

#ifdef __cplusplus
}
#endif

#endif /* TM_FOREST_C_H */
