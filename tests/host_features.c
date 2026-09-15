/* 从 stdin 读窗口（n_ch*n_t 个浮点，通道在前），把特征按**位模式**打出来。
 * 打位模式不打小数：文本来回转会丢信息，那样比的是"打印出来像不像"。 */

#include <stdio.h>
#include <string.h>

#include "tm_features.h"
#include "tm_feat_cfg.h"

int main(void)
{
    static float x[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float out[TM_FEAT_DIM];

    while (1) {
        int ok = 1;
        for (int i = 0; i < TM_FEAT_N_CH * TM_FEAT_N_T; i++) {
            if (scanf("%f", &x[i]) != 1) { ok = 0; break; }
        }
        if (!ok) break;
        if (tm_features(&tm_feat_cfg, x, out) != 0) {
            fprintf(stderr, "tm_features 失败（超出编译期上限？）\n");
            return 1;
        }
        for (int i = 0; i < TM_FEAT_DIM; i++) {
            uint32_t bits;
            memcpy(&bits, &out[i], sizeof bits);
            printf("%s%08x", i ? " " : "", bits);
        }
        printf("\n");
    }
    return 0;
}
