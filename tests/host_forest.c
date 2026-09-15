/* 跑 golden vector，把每条的概率按**位模式**打出来（%08x），交给 Python 比对。
 *
 * 打位模式不打小数：浮点用文本来回转本身就会丢信息，那样比的是"打印出来像不像"，
 * 不是"是不是同一个数"。而差一位就足以在阈值附近把 argmax 翻过去。
 */

#include <stdio.h>
#include <string.h>

#include "tm_forest.h"
#include "tm_forest_model.h"
#include "tm_forest_golden.h"

int main(void)
{
    float proba[TM_F_N_CLASSES];
    for (int i = 0; i < TM_F_GOLDEN_N; i++) {
        const float *x = tm_forest_golden_in + (size_t)i * TM_F_N_FEATURES;
        int cls = tm_forest_predict(&tm_forest, x, proba);
        printf("%d", cls);
        for (int c = 0; c < TM_F_N_CLASSES; c++) {
            uint32_t bits;
            memcpy(&bits, &proba[c], sizeof bits);
            printf(" %08x", bits);
        }
        printf("\n");
    }
    return 0;
}
