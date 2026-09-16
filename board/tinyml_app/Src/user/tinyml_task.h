/* 逐窗口判决 → 事件。这一层决定射频开销，是端侧推理省电的关键。
 *
 * 没有 SDK 依赖：时间由调用方传毫秒数，事件通过返回值交出去。
 * 这样它能在 PC 上编译、测试，不用把 BLE 协议栈拖进来。
 */

#ifndef TINYML_TASK_H
#define TINYML_TASK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int event_class;        /* 要聚合的类别（比如抓挠的类别号） */
    int min_windows;        /* 累计命中够几个窗口才算一次事件 */
    int max_gap_windows;    /* 中间最多允许空几个窗口还算同一次 */
} tm_task_cfg_t;

typedef struct {
    int class_id;
    uint32_t start_ms;
    uint32_t end_ms;
    int n_windows;          /* 这次事件里命中的窗口数，可当强度用 */
} tm_event_t;

typedef struct {
    tm_task_cfg_t cfg;
    int in_event;
    int hit_windows;
    int gap_windows;
    int cur_class;
    uint32_t event_start_ms;
    uint32_t last_hit_ms;
    uint32_t n_windows;     /* 统计用 */
    uint32_t n_events;
} tm_task_t;

void tm_task_init(tm_task_t *t, const tm_task_cfg_t *cfg);

/* 喂一个窗口的判决。返回 1 表示产生了一个事件，写在 *ev。 */
int tm_task_on_window(tm_task_t *t, int cls, uint32_t t_ms, tm_event_t *ev);

/* 收尾。要睡、要上报汇总之前必须调一次，否则最后一个事件永远发不出去——
 * 而那恰好是刚刚发生、最值得报的一个。 */
int tm_task_flush(tm_task_t *t, tm_event_t *ev);

#ifdef __cplusplus
}
#endif

#endif
