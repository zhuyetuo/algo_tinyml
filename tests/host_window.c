/* 喂样本进 tm_window，把每次出窗的内容打出来，交给 Python 比对。
 * 样本从 stdin 读（每行 n_ch 个浮点数）。 */

#include <stdio.h>
#include <stdlib.h>

#include "tm_window.h"

#define N_CH 6
#define N_T 16
#define HOP 5

int main(int argc, char **argv)
{
    const float scale = (argc > 1) ? (float)atof(argv[1]) : 0.05f;
    const int zp = (argc > 2) ? atoi(argv[2]) : -7;

    static int8_t ring[N_CH * N_T];
    static int8_t out[N_CH * N_T];
    tm_window_t w;
    tm_window_init(&w, ring, N_CH, N_T, HOP, scale, (int8_t)zp);

    float s[N_CH];
    int n = 0;
    while (1) {
        int got = 0;
        for (int c = 0; c < N_CH; c++) {
            if (scanf("%f", &s[c]) != 1) { got = -1; break; }
            got++;
        }
        if (got != N_CH) break;
        if (tm_window_push(&w, s, out)) {
            printf("%d", n);
            for (int i = 0; i < N_CH * N_T; i++) printf(" %d", (int)out[i]);
            printf("\n");
        }
        n++;
    }
    return 0;
}
