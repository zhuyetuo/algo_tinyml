/* 板级初始化。照 SDK 示例工程（ble_app_hrs/Src/platform/user_periph_setup.c）的最小子集写，
 * 只保留起 BLE 必需的三件事：设 BD 地址、起日志、设低功耗模式。
 *
 * 不复制 SDK 那份而是自己写一个最小版，是因为它带了一堆这个工程用不到的东西
 * （DTM 测试触发引脚、按键、fault_trace）。留着它们不是"多几 KB"的问题——
 * 是以后有人来读的时候，分不清哪些是这个项目需要的、哪些是示例带的。
 */

#include "user_periph_setup.h"

#include "app_log.h"
#include "board_SK.h"
#include "custom_config.h"
#include "gr_includes.h"

/* BD 地址。**量产必须换掉**：整批设备用同一个地址的话，手机端一次只能认出一只，
 * 而且配对信息会互相顶掉。正式做法是从芯片 UID 派生或走产测写入。 */
static const uint8_t s_bd_addr[SYS_BD_ADDR_LEN] = {0x11, 0x00, 0xcf, 0x3e, 0xcb, 0xea};

static void app_log_assert_init(void)
{
    app_log_init_t log_init;

    log_init.filter.level                 = APP_LOG_LVL_DEBUG;
    log_init.fmt_set[APP_LOG_LVL_ERROR]   = APP_LOG_FMT_ALL & (~APP_LOG_FMT_TAG);
    log_init.fmt_set[APP_LOG_LVL_WARNING] = APP_LOG_FMT_LVL;
    log_init.fmt_set[APP_LOG_LVL_INFO]    = APP_LOG_FMT_LVL;
    log_init.fmt_set[APP_LOG_LVL_DEBUG]   = APP_LOG_FMT_LVL;

    app_log_init(&log_init, bsp_uart_send, bsp_uart_flush);
}

void app_periph_init(void)
{
    SYS_SET_BD_ADDR(s_bd_addr);
    board_init();
    app_log_assert_init();
    /* 自动进睡眠。项圈的功耗全在这一条上——不设的话 CPU 一直醒着，
     * 端侧推理省下来的那点射频完全不够抵。 */
    pwr_mgmt_mode_set(PMR_MGMT_SLEEP_MODE);
}
