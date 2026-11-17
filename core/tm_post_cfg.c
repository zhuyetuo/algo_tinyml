/* 按类别**名字**去配后处理，而不是在别处写死下标。
 *
 * 下标写死的后果：重训一次模型、类别顺序变了（训练脚本按出现顺序编号，
 * 换一批数据就可能变），板子会把"睡觉"当成抓挠去吞并、去过滤。
 * 不报错，只是所有事件都错。名字在导出的 tm_model.h 里就有
 * （TM_CLASS_NAMES），没有理由再手写一遍下标。
 *
 * 找不到就返回 -1，对应的那一步自动跳过——比猜一个下标安全。
 */

#include "tm_post_cfg.h"

#include <string.h>

int tm_post_class_of(const char *const *names, int n, const char *want)
{
    if (!names || !want) return -1;
    for (int i = 0; i < n; ++i)
        if (names[i] && strcmp(names[i], want) == 0) return i;
    return -1;
}

int tm_post_cfg_from_names(tm_post_cfg_t *cfg, const char *const *names, int n,
                           float window_s, float stride_s)
{
    const int sl = tm_post_class_of(names, n, TM_POST_SCRATCH_NAME);
    const int sh = tm_post_class_of(names, n, TM_POST_SHAKE_NAME);
    tm_post_cfg_default(cfg, n, sl, sh, window_s, stride_s);
#if TM_POST_PRECISE
    tm_post_cfg_precise(cfg, names, n, window_s);
#endif
    /* 返回值让调用方能记一条日志。抓挠都找不到的话这份后处理基本没用，
     * 悄悄跑下去只会让人以为它在工作 */
    return (sl >= 0 ? 1 : 0) | (sh >= 0 ? 2 : 0);
}

static int starts_with(const char *s, const char *p)
{
    return s && strncmp(s, p, strlen(p)) == 0;
}

int tm_post_cfg_precise(tm_post_cfg_t *cfg, const char *const *names, int n, float window_s)
{
    const float d = window_s - 8.0f;
    if (!cfg || !names || d > 0.5f || d < -0.5f) return 0;
    int k = 0;
    for (int i = 0; i < n && i < TM_POST_MAX_CLASSES; ++i) {
        if (starts_with(names[i], "抓挠")) {
            cfg->shift_start_ms[i] = 750;
            cfg->shift_end_ms[i] = -1700;
            ++k;
        } else if (starts_with(names[i], "甩")) {
            cfg->shift_start_ms[i] = 500;
            cfg->shift_end_ms[i] = -1450;
            ++k;
        }
    }
    return k;
}
