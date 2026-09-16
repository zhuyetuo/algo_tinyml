/* GBDT 的**紧凑 + AoS** 推理。跟 tm_gbdt.c 判决完全一样，只是节点换了存法。
 *
 * 为什么另开一份而不是改 tm_gbdt.c：那份已经逐位验过、还在用。两份并存的好处是
 * **可以互相对照**——同一个模型两种表示跑出来的 margin 必须逐位相同，
 * 这是"换存法不改判决"最直接的证明。测试里就是这么验的。
 *
 * 节点 6 字节，packed：
 *     uint8_t feature   内部=特征下标（≤255）
 *     float   value     内部=阈值；**叶子=叶子分数**（复用同一个槽）
 *     uint8_t right     bit0-6 右孩子相对偏移；bit7 NaN 往左；整字节 0 = 叶子
 *
 * 左孩子恒等于 idx+1（先序存储），所以不用存——这也让沿左分支走天然是
 * 顺序访存，GR5513 那 8KB cache 上差别很大。
 *
 * 非对齐的 float 读在 Cortex-M4 上就是一条 ldr.w（硬件支持非对齐字加载），
 * 跟对齐的一样快——实测过，不是猜的。
 */

#ifndef TM_GBDT_C_H
#define TM_GBDT_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TM_GC_NODE_BYTES 6
#define TM_GC_MISSING_LEFT 0x80u
#define TM_GC_RIGHT_MASK 0x7Fu

typedef struct {
    const uint8_t *nodes;        /* [n_nodes * 6]，见上面的布局 */
    const int32_t *tree_offset;  /* [n_trees + 1]，单位是**节点**不是字节 */
    int32_t n_trees;
    int32_t n_features;
    int32_t n_classes;
    float base_score;
} tm_gbdt_c_t;

/* x: float [n_features]，margin: float [n_classes]（调用方给）。返回 argmax。 */
int tm_gbdt_c_predict(const tm_gbdt_c_t *m, const float *x, float *margin);

#ifdef __cplusplus
}
#endif

#endif
