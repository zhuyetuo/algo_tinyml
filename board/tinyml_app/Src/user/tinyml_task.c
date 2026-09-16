/* 项圈上的推理任务：IMU 样本进来 → 攒窗口 → 推理 → **聚合成事件** → 上报。
 *
 * 最后那一步"聚合"才是端侧推理省电的地方。不聚合、每个窗口发一条 BLE 通知的话，
 * 射频开销比直接把原始数据传上去还大——那就把端侧推理的意义整个抵消了。
 * 一天几百个窗口判成抓挠，人关心的是"今天抓了几次、什么时候"，不是每 1 秒一条。
 *
 * 这一层**没有** SDK 依赖：时间靠调用方传进来的毫秒数，上报靠一个回调。
 * 这样它能在 PC 上跑测试（tests/test_task_c.py），而不用把整个 BLE 协议栈拖进来。
 */

#include "tinyml_task.h"

#include <string.h>

void tm_task_init(tm_task_t *t, const tm_task_cfg_t *cfg)
{
    memset(t, 0, sizeof(*t));
    t->cfg = *cfg;
    t->cur_class = -1;
}

/* 一个窗口的判决进来。返回 1 表示刚刚产生了一个事件（写在 *ev 里）。 */
int tm_task_on_window(tm_task_t *t, int cls, uint32_t t_ms, tm_event_t *ev)
{
    t->n_windows++;

    if (cls != t->cfg.event_class) {
        /* 不是目标类别。**不立刻结束事件**，先记着空了多久——抓挠这种动作中间
         * 会停顿一两秒（换个姿势、挠另一边），一停就切断的话，一次连续的抓挠
         * 会被拆成五六个事件，上报次数反而更多，统计出来的"次数"也是错的。 */
        if (t->in_event) {
            t->gap_windows++;
            if (t->gap_windows > t->cfg.max_gap_windows) {
                const int hit = t->hit_windows;
                const uint32_t start = t->event_start_ms;
                const uint32_t end = t->last_hit_ms;
                t->in_event = 0;
                t->hit_windows = 0;
                t->gap_windows = 0;
                /* 太短的不报。一两个窗口的命中多半是别的动作蹭上来的
                 * （甩头、被抱起来），报上去只会污染统计 */
                if (hit >= t->cfg.min_windows) {
                    ev->class_id = t->cfg.event_class;
                    ev->start_ms = start;
                    ev->end_ms = end;
                    ev->n_windows = hit;
                    t->n_events++;
                    return 1;
                }
            }
        }
        return 0;
    }

    /* 命中 */
    if (!t->in_event) {
        t->in_event = 1;
        t->event_start_ms = t_ms;
        t->hit_windows = 0;
    }
    t->hit_windows++;
    t->gap_windows = 0;
    t->last_hit_ms = t_ms;
    return 0;
}

/* 主动收尾：要睡了、或者要上报当天汇总之前调一次，否则最后一个事件会卡在
 * 缓冲里永远发不出去——而它恰好是"刚刚发生的那次"，最值得报的一个。 */
int tm_task_flush(tm_task_t *t, tm_event_t *ev)
{
    if (!t->in_event) {
        return 0;
    }
    const int hit = t->hit_windows;
    const uint32_t start = t->event_start_ms;
    const uint32_t end = t->last_hit_ms;
    t->in_event = 0;
    t->hit_windows = 0;
    t->gap_windows = 0;
    if (hit < t->cfg.min_windows) {
        return 0;
    }
    ev->class_id = t->cfg.event_class;
    ev->start_ms = start;
    ev->end_ms = end;
    ev->n_windows = hit;
    t->n_events++;
    return 1;
}
