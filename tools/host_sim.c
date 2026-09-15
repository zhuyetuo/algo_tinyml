/* 在 Ubuntu 上跑**板上那份一模一样的 C**，喂真实数据，看效果和耗时。
 *
 * 编译的是 firmware/tinyml 下那几个 .c 本身，不是"等价实现"——所以这里看到的判决结果
 * 跟板上逐位相同（编译选项也跟固件一致，尤其 -ffp-contract=off）。唯一不一样的是
 * **耗时**：x86 的绝对时间跟 Cortex-M4F 没有可比性，只能用来比"RF 和 CNN 哪个贵"。
 * 板上的真实耗时要用 DWT 周期计数器测，那个得有板子。
 *
 * 输入：stdin，每行 n_ch 个浮点（一个采样点），顺序 acc_x acc_y acc_z gyr_x gyr_y gyr_z [pitch roll]
 * 输出：每凑满一个窗口打一行 "样本序号 类别 概率..."
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tm_features.h"
#include "tm_feat_cfg.h"
#include "tm_forest.h"
#include "tm_forest_model.h"
#include "tm_window.h"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const int hop = (argc > 1) ? atoi(argv[1]) : TM_FEAT_N_T / 2;

    static int8_t ring_i8[TM_FEAT_N_CH * TM_FEAT_N_T];  /* 只为复用 tm_window 的环形逻辑 */
    static float ring[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float win[TM_FEAT_N_CH * TM_FEAT_N_T];
    static float feat[TM_FEAT_DIM];
    static float proba[TM_F_N_CLASSES];
    (void)ring_i8;

    /* RF 这条路不做量化，所以自己维护 float 环形缓冲，不走 tm_window
     * （tm_window 是 CNN 那条用的，它顺带做 int8 量化）。 */
    int head = 0, filled = 0, since = 0, n = 0;
    double t_feat = 0.0, t_pred = 0.0;
    long n_win = 0;

    float s[TM_FEAT_N_CH];
    while (1) {
        int ok = 1;
        for (int c = 0; c < TM_FEAT_N_CH; c++) {
            if (scanf("%f", &s[c]) != 1) { ok = 0; break; }
        }
        if (!ok) break;

        for (int c = 0; c < TM_FEAT_N_CH; c++) {
            ring[(size_t)c * TM_FEAT_N_T + head] = s[c];
        }
        head = (head + 1) % TM_FEAT_N_T;
        if (filled < TM_FEAT_N_T) filled++;
        since++;
        n++;

        if (filled < TM_FEAT_N_T || since < hop) continue;
        since = 0;

        /* 环形展平成时间从旧到新 */
        for (int c = 0; c < TM_FEAT_N_CH; c++) {
            const float *src = ring + (size_t)c * TM_FEAT_N_T;
            float *dst = win + (size_t)c * TM_FEAT_N_T;
            for (int t = 0; t < TM_FEAT_N_T; t++) {
                dst[t] = src[(head + t) % TM_FEAT_N_T];
            }
        }

        double t0 = now_sec();
        if (tm_features(&tm_feat_cfg, win, feat) != 0) {
            fprintf(stderr, "tm_features 失败\n");
            return 1;
        }
        double t1 = now_sec();
        int cls = tm_forest_predict(&tm_forest, feat, proba);
        double t2 = now_sec();
        t_feat += t1 - t0;
        t_pred += t2 - t1;
        n_win++;

        printf("%d %d", n - 1, cls);
        for (int c = 0; c < TM_F_N_CLASSES; c++) printf(" %.9g", proba[c]);
        printf("\n");
    }

    fprintf(stderr, "窗口数 %ld\n", n_win);
    if (n_win) {
        fprintf(stderr, "特征提取 %.1f us/窗口\n", 1e6 * t_feat / n_win);
        fprintf(stderr, "森林推理 %.1f us/窗口\n", 1e6 * t_pred / n_win);
        fprintf(stderr, "合计     %.1f us/窗口   ← **x86 的数**，跟 M4F 没有可比性，\n",
                1e6 * (t_feat + t_pred) / n_win);
        fprintf(stderr, "                          只能用来比 RF 和 CNN 哪个贵\n");
    }
    return 0;
}
