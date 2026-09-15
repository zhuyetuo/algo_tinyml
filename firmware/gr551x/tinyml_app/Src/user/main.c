/* GR551x 上的 tinyml 固件入口。
 *
 * 结构照 SDK 示例（ble_app_hrs/Src/user/main.c）：起外设 → 起协议栈 → 主循环里
 * 刷日志 + 让电源管理调度。项圈的功耗几乎全在 pwr_mgmt_schedule() 能不能真的
 * 让芯片睡下去，所以主循环里不能有忙等。
 *
 * 跟示例不同的一件事：**协议栈起来之后第一件事是跑 golden vector 自检**，
 * 不是直接开始推理。自检没过就把推理关掉——让一块算错的板子安安静静地上报
 * 错误数据，比它直接不工作糟得多，而且会污染后端统计。
 */

#include "app_log.h"
#include "custom_config.h"
#include "gr_includes.h"
#include "patch.h"
#include "scatter_common.h"
#include "user_periph_setup.h"

#include "tinyml_bench.h"
#include "tinyml_selftest.h"
#include "tinyml_task.h"

#include "tm_model.h"
#include "tm_post.h"
#include "tm_post_cfg.h"

/* 协议栈的堆。这个宏来自 SDK，必须有，而且只能有一份 */
STACK_HEAP_INIT(heaps_table);

/* 自检结论。推理任务只有在它为真的时候才往外报结果 */
static bool s_inference_trusted = false;

static tm_task_t s_task;
static const tm_task_cfg_t s_task_cfg = {
    .event_class = 2,       /* 抓挠。类别号看导出的 TM_CLASS_NAMES，别硬记 */
    .min_windows = 3,
    .max_gap_windows = 2,
};

/* 稳定版 v2 后处理。跟平台上「稳定版 v2」是同一套规则
 * （firmware/tinyml/tm_post.c，对着 label_service/postprocess.py 逐段验过），
 * 所以板子报的次数和时长跟平台上看到的能对上——两边用的是同一个模型，
 * 以前对不上的差异全在后处理这一层。 */
static tm_post_t s_post;
static bool s_post_ready = false;

/* 窗口几何：要跟训练/导出时一致。对不上的话片段时间戳整体错位，
 * 而每一段看起来都正常 */
#define TINYML_WINDOW_S 1.0f
#define TINYML_STRIDE_S 0.5f

/* 上电就把真实耗时打出来。**这个数不能估**——它直接决定占空比和功耗，
 * 而 PC 上量的 286 µs/窗口对 64MHz 的 M4F 没有参考价值（指令集不同、
 * 频率差 50 倍）。让板子自己数，插上就知道。
 *
 * 放在自检**之后**：自检没过的话算出来的东西本来就不可信，测它的速度没有意义。 */
static void tinyml_report_speed(void)
{
    tm_bench_report_t b;

    if (tm_bench_run(&b) != 0) {
        APP_LOG_WARNING("测速跳过：DWT 周期计数器没走（没接调试时钟？）。"
                        "别把耗时当成 0——是没测到，不是很快。");
        return;
    }

    APP_LOG_INFO("── 推理耗时（实测，主频 %u Hz）──", (unsigned)b.cpu_hz);
    if (b.cnn_first) {
        APP_LOG_INFO("  CNN   首次 %u 周期 = %u us（占空比 %u/1000，按每秒一窗）",
                     (unsigned)b.cnn_first,
                     (unsigned)tm_bench_report_us(b.cnn_first, &b),
                     (unsigned)tm_bench_duty_permille(b.cnn_first, &b, 1000));
        APP_LOG_INFO("  CNN   均值 %u 周期 = %u us",
                     (unsigned)b.cnn_mean,
                     (unsigned)tm_bench_report_us(b.cnn_mean, &b));
    }
    if (b.feat_first || b.rf_first) {
        /* 特征和森林分开报：RF 这条路上特征提取往往比模型本身还贵，
         * 合在一起的话"该优化哪一半"就没法回答 */
        APP_LOG_INFO("  特征  首次 %u 周期 = %u us",
                     (unsigned)b.feat_first,
                     (unsigned)tm_bench_report_us(b.feat_first, &b));
        APP_LOG_INFO("  森林  首次 %u 周期 = %u us",
                     (unsigned)b.rf_first,
                     (unsigned)tm_bench_report_us(b.rf_first, &b));
        APP_LOG_INFO("  RF合计 %u us（占空比 %u/1000，按每秒一窗）",
                     (unsigned)tm_bench_report_us(b.feat_first + b.rf_first, &b),
                     (unsigned)tm_bench_duty_permille(b.feat_first + b.rf_first,
                                                      &b, 1000));
    }
}


static void tinyml_boot_check(void)
{
    tm_selftest_report_t rep;

    if (tm_selftest_run(&rep) == 0) {
        s_inference_trusted = true;
        APP_LOG_INFO("tinyml 自检通过：%d 条 golden vector 逐位一致", rep.n_checked);
        tinyml_report_speed();
        tm_task_init(&s_task, &s_task_cfg);

        {
            tm_post_cfg_t pc;
            const int found = tm_post_cfg_from_names(
                &pc, TM_CLASS_NAMES, TM_N_CLASSES,
                TINYML_WINDOW_S, TINYML_STRIDE_S);
            tm_post_init(&s_post, &pc);
            s_post_ready = ((found & 1) != 0);
            if (s_post_ready) {
                APP_LOG_INFO("后处理：稳定版 v2（跟平台同一套规则）");
            } else {
                /* 抓挠都找不到说明类别名对不上，后处理等于没有。
                 * **说出来**——不说的话表现是"事件比平台上多很多"，
                 * 而那会被当成模型的问题去查 */
                APP_LOG_ERROR("后处理关了：类别名里找不到 %s，"
                              "退回逐窗口聚合（结果会比平台碎）",
                              TM_POST_SCRATCH_NAME);
            }
        }
        return;
    }

    s_inference_trusted = false;
    APP_LOG_ERROR("tinyml 自检失败（第 %d 条）：%s", rep.fail_index,
                  tm_selftest_explain(&rep));
    /* 故意不往下走。自检没过说明这块板子算出来的东西不可信，这时候继续推理、
     * 继续上报，得到的是一批看起来正常、实际错误的数据——那比不工作难查得多。 */
}

void ble_evt_handler(const ble_evt_t *p_evt)
{
    switch (p_evt->evt_id) {
    case BLE_COMMON_EVT_STACK_INIT:
        /* 协议栈就绪。先自检再谈别的 */
        tinyml_boot_check();
        break;

    case BLE_GAPM_EVT_ADV_START:
        if (p_evt->evt_status) {
            APP_LOG_DEBUG("广播启动失败 (0x%02X)", p_evt->evt_status);
        }
        break;

    case BLE_GAPC_EVT_CONNECTED:
        APP_LOG_INFO("已连接");
        break;

    case BLE_GAPC_EVT_DISCONNECTED:
        APP_LOG_INFO("已断开 (0x%02X)", p_evt->evt.gapc_evt.params.disconnected.reason);
        break;

    default:
        break;
    }
}

/* 老入口的前向声明：概率那条路在后处理没配起来时会退回它 */
void tinyml_on_window_result(int cls, uint32_t t_ms);

/* 一个窗口的**概率**出来之后调这里。QMI8658B 的驱动接上之后这就是入口。
 *
 * 要概率不要 argmax：稳定版 v2 整套都建立在概率上——viterbi 的发射项是
 * log(p)，段的过滤看段内平均/最大概率。只给一个 argmax 的话这些全做不了，
 * 而那正是平台上效果好的原因。 */
void tinyml_on_window_probs(const float *probs, uint32_t t_ms)
{
    tm_seg_t segs[8];

    if (!s_inference_trusted) {
        return;
    }
    if (!s_post_ready) {
        /* 后处理没配起来：退回老路子，至少还能报事件 */
        int best = 0;
        for (int c = 1; c < TM_N_CLASSES; ++c)
            if (probs[c] > probs[best]) best = c;
        tinyml_on_window_result(best, t_ms);
        return;
    }

    const int n = tm_post_on_window(&s_post, probs, t_ms, segs,
                                    (int)(sizeof segs / sizeof segs[0]));
    if (n < 0) {
        APP_LOG_ERROR("后处理配置不合法（延迟线太短？），这一窗丢了");
        return;
    }
    for (int i = 0; i < n; ++i) {
        /* 到这里才值得动射频。逐窗口发通知的话，射频开销比直接把原始数据
         * 传上去还大，端侧推理就白做了。
         * 状态段（活动/睡觉）也一起报——平台那条时间轴要靠它 */
        APP_LOG_INFO("片段：%s %u→%u ms，%u 窗，最大 %d%%，平均 %d%%",
                     TM_CLASS_NAMES[segs[i].cls], segs[i].start_ms,
                     segs[i].end_ms, (unsigned)segs[i].n_windows,
                     (int)(segs[i].conf_max * 100.0f + 0.5f),
                     (int)(segs[i].conf_mean * 100.0f + 0.5f));
    }
}

/* 要睡、要上报汇总之前调一次，把压在延迟线里的片段吐出来。
 * 不调的话最后一段永远发不出去——而那恰好是刚刚发生、最值得报的一段。 */
void tinyml_flush(void)
{
    tm_seg_t segs[8];
    tm_event_t ev;

    if (s_post_ready) {
        const int n = tm_post_flush(&s_post, segs,
                                    (int)(sizeof segs / sizeof segs[0]));
        for (int i = 0; i < n && n > 0; ++i)
            APP_LOG_INFO("片段：%s %u→%u ms，%u 窗",
                         TM_CLASS_NAMES[segs[i].cls], segs[i].start_ms,
                         segs[i].end_ms, (unsigned)segs[i].n_windows);
    } else if (tm_task_flush(&s_task, &ev)) {
        APP_LOG_INFO("事件：类别 %d，%u→%u ms，命中 %d 个窗口",
                     ev.class_id, ev.start_ms, ev.end_ms, ev.n_windows);
    }
}

/* 老入口：只有 argmax 的时候用。**保留**——已经接上的调用方不该因为
 * 加了后处理就编不过。但它没有稳定版 v2 那套，结果会比平台碎。 */
void tinyml_on_window_result(int cls, uint32_t t_ms)
{
    tm_event_t ev;

    if (!s_inference_trusted) {
        return;
    }
    if (tm_task_on_window(&s_task, cls, t_ms, &ev)) {
        APP_LOG_INFO("事件：类别 %d，%u→%u ms，命中 %d 个窗口",
                     ev.class_id, ev.start_ms, ev.end_ms, ev.n_windows);
    }
}

int main(void)
{
    app_periph_init();

    ble_stack_init(ble_evt_handler, &heaps_table);

    while (1) {
        app_log_flush();
        pwr_mgmt_schedule();
    }
}
