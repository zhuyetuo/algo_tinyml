/* 把定点原语在一组扫描值上的结果打出来，交给 Python 逐个比对。
 *
 * 为什么不靠"端到端对答案"覆盖这些：实测过——把 C 里负数的舍入 nudge 故意改错，
 * 整网 golden vector **照样全过**。因为那一位偏差在后面的右移里被吃掉了。
 * 端到端只能证明"这组输入下一致"，证明不了算子本身对。所以算子要单独扫。
 *
 * 输入值从 stdin 读（每行 "x multiplier shift"），避免两边各写一份扫描列表、
 * 然后慢慢地不一样。
 */

#include <stdio.h>

#include "tm_runtime.h"

int main(void)
{
    long long x, m;
    int s;
    while (scanf("%lld %lld %d", &x, &m, &s) == 3) {
        printf("%d %d %d\n",
               tm_saturating_rounding_doubling_high_mul((int32_t)x, (int32_t)m),
               tm_rounding_divide_by_pot((int32_t)x, s < 0 ? -s : s),
               tm_multiply_by_quantized_multiplier((int32_t)x, (int32_t)m, s));
    }
    return 0;
}
