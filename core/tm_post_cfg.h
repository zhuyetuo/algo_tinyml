/* 按类别名字配后处理。见 tm_post_cfg.c 的说明。 */

#ifndef TM_POST_CFG_H
#define TM_POST_CFG_H

#include "tm_post.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 名字跟 imu_train 的类别名一致。要改的话两边一起改——
 * 这里改了那边没改，表现是"找不到抓挠"，而不是算错，看日志能发现 */
#ifndef TM_POST_SCRATCH_NAME
#define TM_POST_SCRATCH_NAME "抓挠"
#endif
#ifndef TM_POST_SHAKE_NAME
#define TM_POST_SHAKE_NAME "甩身体"
#endif

/* names 里找 want，返回下标；找不到返回 -1。 */
int tm_post_class_of(const char *const *names, int n, const char *want);

/* 精准版起止（平移）：编译时 -DTM_POST_PRECISE=1 才在 tm_post_cfg_from_names 里打开。
 * 默认关——平台上「板上整条链」那一列要跟稳定版 v2 对拍，打开的话那一列的含义就悄悄变了。
 * 给设备的包（export_train --bundle）对 8 s 窗口的模型自动打开（footprint.imu_defines）。 */
#ifndef TM_POST_PRECISE
#define TM_POST_PRECISE 0
#endif

/* 按类别名填精准版的平移量：名字以「抓挠」开头的 +750 / -1700 ms，以「甩」开头的 +500 / -1450 ms。
 * 只对 8 s 窗口（train_v2 训的、label_mode=segment）的模型生效——偏差是 2026-11-18 在 8 s 窗口上
 * 5 折量出来的，别的窗口长度没量过，不动。返回填了几类。 */
int tm_post_cfg_precise(tm_post_cfg_t *cfg, const char *const *names, int n, float window_s);

/* 用类别名把 cfg 配好。返回 bit0=找到抓挠，bit1=找到甩身体。 */
int tm_post_cfg_from_names(tm_post_cfg_t *cfg, const char *const *names, int n,
                           float window_s, float stride_s);

#ifdef __cplusplus
}
#endif

#endif
