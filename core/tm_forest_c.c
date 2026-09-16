#include "tm_forest_c.h"

#include <string.h>

/* 紧凑森林的遍历。
 *
 * 节点是 7 字节的字节流，不是结构体数组——**故意不用 struct**：
 * packed struct 在不同编译器上的字段布局和访问代码不完全一样，而这份代码
 * 要在 gcc(x86)、arm-none-eabi、以及将来可能的 Keil/IAR 上给出逐位相同的结果。
 * 直接按字节偏移取，谁也没得发挥。
 *
 * 读 threshold 是从偏移 1 开始的**非对齐** 4 字节。Cortex-M4 硬件支持非对齐
 * 字访问（一条 ldr.w），所以下面用 memcpy 让编译器自己去合并——
 * memcpy 4 字节在 -Os 下会被编译成单条加载，既对又快。
 * 手写 (b[1] | b[2]<<8 | ...) 反而会强制拆成 4 次字节加载。
 * （M0 不支持非对齐访问，换核要重新验。）
 */

static uint32_t load_u32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof v);   /* 小端；ARM 和 x86 都是小端 */
    return v;
}

static float as_float(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

int tm_forest_c_predict(const tm_forest_c_t *f, const float *x, int32_t *votes)
{
    for (int c = 0; c < f->n_classes; c++) {
        votes[c] = 0;
    }

    for (int t = 0; t < f->n_trees; t++) {
        int32_t node = f->tree_offset[t];
        for (;;) {
            const uint8_t *p = f->nodes + (size_t)node * TM_FC_NODE_BYTES;
            const uint16_t right = (uint16_t)(p[5] | ((uint16_t)p[6] << 8));
            if (right == 0) {
                /* 叶子：那 4 个字节是叶子表下标，不是阈值 */
                const uint32_t li = load_u32(p + 1);
                const uint8_t *leaf = f->leaves + (size_t)li * f->n_classes;
                for (int c = 0; c < f->n_classes; c++) {
                    votes[c] += leaf[c];
                }
                break;
            }
            /* **<= 走左**，跟 sklearn 一致。写成 < 的话，特征值正好等于阈值的
             * 样本会走反——而阈值本来就是从样本值来的，"正好等于"不罕见 */
            node = (x[p[0]] <= as_float(load_u32(p + 1)))
                   ? node + 1 : node + right;
        }
    }

    /* argmax 在整数上是精确的。并列取下标最小的，跟 np.argmax 一致 */
    int best = 0;
    for (int c = 1; c < f->n_classes; c++) {
        if (votes[c] > votes[best]) {
            best = c;
        }
    }
    return best;
}
