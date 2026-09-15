#include <stdio.h>
#include <string.h>
#include "tm_gbdt_c.h"
#include "tm_gbdt_c_model.h"
#include "tm_gbdt_c_golden.h"

int main(void)
{
    float margin[TM_GC_N_CLASSES];
    for (int i = 0; i < TM_GC_GOLDEN_N; i++) {
        const float *x = tm_gc_golden_in + (size_t)i * TM_GC_N_FEATURES;
        int cls = tm_gbdt_c_predict(&tm_gbdt_c_model, x, margin);
        printf("%d", cls);
        for (int c = 0; c < TM_GC_N_CLASSES; c++) {
            uint32_t b; memcpy(&b, &margin[c], sizeof b);
            printf(" %08x", b);
        }
        printf("\n");
    }
    return 0;
}
