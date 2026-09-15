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

/* 用类别名把 cfg 配好。返回 bit0=找到抓挠，bit1=找到甩身体。 */
int tm_post_cfg_from_names(tm_post_cfg_t *cfg, const char *const *names, int n,
                           float window_s, float stride_s);

#ifdef __cplusplus
}
#endif

#endif
