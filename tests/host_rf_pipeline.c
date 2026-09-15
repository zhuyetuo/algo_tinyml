/* 整条 RF 链：原始窗口 → 193 维特征 → 森林 → 类别。
 *
 * 单独测特征、单独测森林都过了，不等于接起来就对——中间那道接缝（特征的排列
 * 顺序跟模型训练时是不是同一个）恰恰是最容易错、又最不会报错的地方：顺序错了
 * 每一维都对到别的特征上，模型照样给得出一个类别，只是准确率莫名其妙地差。
 */

#include <stdio.h>
#include <string.h>

#include "tm_features.h"
#include "tm_feat_cfg.h"
#include "tm_forest.h"
#include "tm_forest_model.h"

int main(void)
{
    static float x[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float feat[TM_FEAT_DIM];
    static float proba[TM_F_N_CLASSES];

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
        int cls = tm_forest_predict(&tm_forest, feat, proba);
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
