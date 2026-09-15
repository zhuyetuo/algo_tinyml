/* 把 tm_prep 的输出打到 stdout，给 Python 侧逐位对答案。 */
#include <stdio.h>

#include "tm_prep.h"

/* 由测试生成：TP_N_CH / TP_N_T / TP_MEAN / TP_STD / TP_SCALE / TP_ZP /
 * TP_N_CASES / TP_INPUT */
#include "prep_case.h"

int main(void)
{
    static const double mean[] = TP_MEAN;
    static const double std_[] = TP_STD;
    static const float input[TP_N_CASES][TP_N_CH * TP_N_T] = TP_INPUT;
    static int8_t out[TP_N_CH * TP_N_T];

    const tm_prep_t p = {
        .ch_mean = mean, .ch_std = std_,
        .in_scale = TP_SCALE, .in_zp = TP_ZP,
        .n_ch = TP_N_CH, .n_t = TP_N_T,
    };

    for (int i = 0; i < TP_N_CASES; i++) {
        tm_prep(&p, input[i], out);
        for (int j = 0; j < TP_N_CH * TP_N_T; j++) {
            printf("%d%s", (int)out[j], j + 1 == TP_N_CH * TP_N_T ? "" : " ");
        }
        printf("\n");
    }
    return 0;
}
