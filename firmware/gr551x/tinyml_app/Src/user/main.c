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

/* IMU 攒满一个窗口、推理出类别之后调这里。
 * 还没有 QMI8658B 的驱动，所以目前没有人调它——驱动接上之后它就是入口。 */
void tinyml_on_window_result(int cls, uint32_t t_ms)
{
    tm_event_t ev;

    if (!s_inference_trusted) {
        return;
    }
    if (tm_task_on_window(&s_task, cls, t_ms, &ev)) {
        /* 到这里才值得动射频。逐窗口发通知的话，射频开销比直接把原始数据传上去
         * 还大，端侧推理就白做了 */
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
