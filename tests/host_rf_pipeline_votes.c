/* 紧凑森林整条链：stdin 读窗口（通道在前），stdout 打每类整数票数。
 * tests/test_cmsis_c.py 用它比"朴素 vs CMSIS-DSP"两个编译版本的判决。 */

#include <stdio.h>

#include "tm_features.h"
#include "tm_feat_cfg.h"
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"

int main(void)
{
    static float x[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float feat[TM_FEAT_DIM];
    static int32_t votes[TM_FC_N_CLASSES];

    while (1) {
        int ok = 1;
        for (int i = 0; i < TM_FEAT_N_CH * TM_FEAT_N_T; i++) {
            if (scanf("%f", &x[i]) != 1) { ok = 0; break; }
        }
        if (!ok) break;
        if (tm_features(&tm_feat_cfg, x, feat) != 0) {
            fprintf(stderr, "tm_features 失败\n");
            return 1;
        }
        tm_forest_c_predict(&tm_forest_c, feat, votes);
        for (int c = 0; c < TM_FC_N_CLASSES; c++) {
            printf("%d%s", (int)votes[c], c + 1 == TM_FC_N_CLASSES ? "" : " ");
        }
        printf("\n");
    }
    return 0;
}
