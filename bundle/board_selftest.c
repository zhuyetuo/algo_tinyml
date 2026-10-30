/* 板上自检：库在这块板子、这套编译选项下算得对不对。
 *
 * 和库一起编进固件，开机调一次 tm_board_selftest()，返回 0 即通过。
 * 用 include/ 里的 golden 头文件（出厂时算好的输入和标准答案）逐位比对；
 * 结果经 printf 输出（请把 printf 重定向到串口）。正确输出见 README「板上验证」。
 *
 * 编译：和调用库的其它源文件一样，加 lib/BUILD_FLAGS.txt 里的 -D 选项。
 * golden 数据约占几 KB flash，量产固件可以不带本文件。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tm_imu.h"
#if defined(TM_KIND_RF)
#include "tm_feat_cfg.h"
#include "tm_features.h"
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"
#include "tm_forest_c_pipeline_golden.h"
#define N_T TM_FEAT_N_T
#define N_CH TM_FEAT_N_CH
#define N_CLASSES TM_FC_N_CLASSES
#else
#include "tm_golden.h"
#include "tm_model.h"
#include "tm_runtime.h"
#define N_T TM_N_T
#define N_CH TM_N_CH
#define N_CLASSES TM_N_CLASSES
#endif
#define N_SENSOR (N_CH - 2)

int tm_board_selftest(void)
{
    int bad = 0, n = 0;
#if defined(TM_KIND_RF)
    static float feat[TM_FEAT_DIM];
    for (int i = 0; i < TM_FCP_GOLDEN_N; i++, n++) {
        int32_t votes[N_CLASSES];
        tm_features(&tm_feat_cfg, tm_forest_c_pipeline_in + i * TM_FCP_N_CH * TM_FCP_N_T, feat);
        tm_forest_c_predict(&tm_forest_c, feat, votes);
        if (memcmp(votes, tm_forest_c_pipeline_votes + i * N_CLASSES, sizeof votes) != 0) {
            printf("golden #%d 不一致：", i);
            for (int c = 0; c < N_CLASSES; c++)
                printf(" %ld/%ld", (long)votes[c], (long)tm_forest_c_pipeline_votes[i * N_CLASSES + c]);
            printf("（本板/期望）\n");
            bad++;
        }
    }
#else
    static int8_t arena[TM_ARENA_BYTES];
    for (int i = 0; i < TM_GOLDEN_N; i++, n++) {
        int8_t out[N_CLASSES];
        tm_invoke(&tm_model, tm_golden_in + i * N_CH * N_T, out, arena, TM_ARENA_BYTES);
        if (memcmp(out, tm_golden_out + i * N_CLASSES, N_CLASSES) != 0) {
            printf("golden #%d 不一致\n", i);
            bad++;
        }
    }
#endif
    /* 输入映射：水平静止（重力 = +Z）时不转、pitch/roll 为 0 */
    {
        static float raw[N_T * N_SENSOR], out[N_CH * N_T];
        for (int t = 0; t < N_T; t++)
            for (int k = 0; k < N_SENSOR; k++) raw[t * N_SENSOR + k] = (k == 2) ? 1.0f : 0.0f;
        tm_imu_channels(raw, N_T, N_SENSOR, out);
        if (fabsf(out[2 * N_T] - 1.0f) > 1e-6f || fabsf(out[(N_CH - 2) * N_T]) > 1e-6f) {
            printf("输入映射不一致\n");
            bad++;
        }
        n++;
    }
    printf("selftest: %d/%d 通过\n", n - bad, n);
    return bad;
}

#ifdef TM_SELFTEST_MAIN
int main(void) { return tm_board_selftest() ? 1 : 0; }
#endif
