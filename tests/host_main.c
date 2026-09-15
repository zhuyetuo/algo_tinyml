/* 在 PC 上跑板上那份 C，把每条 golden vector 的输出打到 stdout。
 * 由 tests/test_c_consistency.py 编译并调用，输出交给 Python 逐位比对。
 *
 * 为什么要在 PC 上跑板子的代码：等烧进 GR5513 才发现对不上，调试手段就只剩
 * 串口打印，一轮几分钟。在 PC 上同一份 .c 几秒钟一轮，而且能上 UBSan——
 * 定点代码最容易出的就是移位溢出这类未定义行为，它在 x86 上"看起来能跑"，
 * 换到 Cortex-M 上才换一种错法。
 */

#include <stdio.h>

#include "tm_runtime.h"
#include "tm_model.h"
#include "tm_golden.h"

int main(void)
{
    static int8_t arena[TM_ARENA_BYTES];
    int8_t out[TM_N_CLASSES];

    for (int i = 0; i < TM_GOLDEN_N; i++) {
        const int8_t *in = tm_golden_in + (size_t)i * TM_N_CH * TM_N_T;
        if (tm_invoke(&tm_model, in, out, arena, (int)sizeof(arena)) != 0) {
            fprintf(stderr, "tm_invoke 失败（arena 不够？）第 %d 条\n", i);
            return 1;
        }
        for (int c = 0; c < TM_N_CLASSES; c++) {
            printf("%d%s", (int)out[c], c + 1 == TM_N_CLASSES ? "" : " ");
        }
        printf("\n");
    }
    return 0;
}
