/* 跑 GBDT 的 golden vector，把 margin 按位模式打出来。 */

#include <stdio.h>
#include <string.h>

#include "tm_gbdt.h"
#include "tm_gbdt_model.h"
#include "tm_gbdt_golden.h"

int main(void)
{
    float margin[TM_G_N_CLASSES];
    for (int i = 0; i < TM_G_GOLDEN_N; i++) {
        const float *x = tm_gbdt_golden_in + (size_t)i * TM_G_N_FEATURES;
        int cls = tm_gbdt_predict(&tm_gbdt, x, margin);
        printf("%d", cls);
        for (int c = 0; c < TM_G_N_CLASSES; c++) {
            uint32_t bits;
            memcpy(&bits, &margin[c], sizeof bits);
            printf(" %08x", bits);
        }
        printf("\n");
    }
    return 0;
}
