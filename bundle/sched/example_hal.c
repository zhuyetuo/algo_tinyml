/* pet_behavior 接入示例：硬件接口层（嵌入式侧实现）+ 事件转发。
 * 把 my_* 换成工程里的驱动即可；下面的写法只是示意，按 SDK 实际接口改。 */
#include "pet_behavior.h"

/* ── 硬件接口 ─────────────────────────────────────────────────────────── */
static int my_imu_config(uint16_t rate_hz, uint8_t range_g)
{
    /* 只开加速度计：ODR = rate_hz，量程 ±range_g，开 FIFO（建议水位 ≥ 5 秒的数据）；成功返回 0 */
    (void)rate_hz; (void)range_g;
    return 0;
}

static uint16_t my_imu_read(pb_sample_t *buf, uint16_t max)
{
    /* 从 FIFO 读最多 max 个样本（x/y/z 原始计数），按采样先后填 buf，返回个数；空返回 0 */
    (void)buf; (void)max;
    return 0;
}

static uint32_t my_now_ms(void) { return 0; /* 单调毫秒，休眠不停（RTC / 低功耗定时器） */ }
static uint32_t my_unix_time(void) { return 0; /* Unix 秒，未校时返回 0 */ }
static void my_set_wakeup(uint32_t at_ms) { (void)at_ms; /* 到 at_ms 时投递 PB_EVT_WAKEUP，再次调用覆盖 */ }

#define PB_FLASH_BASE  0x00000000u   /* 给库划的 Flash 区域起始地址 */
static int my_flash_write(uint32_t off, const void *d, uint32_t n) { (void)off; (void)d; (void)n; return 0; }
static int my_flash_read(uint32_t off, void *d, uint32_t n) { (void)off; (void)d; (void)n; return 0; }
static int my_flash_erase(uint32_t off, uint32_t n) { (void)off; (void)n; return 0; }

static int my_ble_send(const uint8_t *d, uint16_t n) { (void)d; (void)n; return -1; /* 未连接或忙返回负数 */ }
static uint16_t my_ble_mtu(void) { return 244; /* 当前单包最大字节数（ATT MTU − 3） */ }

static const pb_hal_t hal = {
    .imu_config = my_imu_config,
    .imu_read = my_imu_read,
    .now_ms = my_now_ms,
    .unix_time = my_unix_time,
    .set_wakeup = my_set_wakeup,
    .flash_write = my_flash_write,
    .flash_read = my_flash_read,
    .flash_erase = my_flash_erase,
    .flash_size = 64 * 1024,
    .flash_erase_unit = 4096,
    .ble_send = my_ble_send,
    .ble_mtu = my_ble_mtu,
};

static uint8_t pb_mem[PB_WORK_SIZE] __attribute__((aligned(4)));

/* ── 开机 ─────────────────────────────────────────────────────────────── */
int app_pb_start(void)
{
    return pb_init(&hal, pb_mem, sizeof pb_mem);   /* 0 = 成功（内含自检） */
}

/* ── 事件：中断里只置标志 / 投递消息，在任务（或主循环）里调 pb_on_event ── */
void app_on_wakeup_timer(void)   { pb_on_event(PB_EVT_WAKEUP, 0, 0); }
void app_on_imu_watermark(void)  { pb_on_event(PB_EVT_IMU_FIFO, 0, 0); }
void app_on_ble_connected(void)  { pb_on_event(PB_EVT_BLE_CONNECTED, 0, 0); }
void app_on_ble_disconnected(void) { pb_on_event(PB_EVT_BLE_DISCONNECTED, 0, 0); }
void app_on_ble_tx_done(void)    { pb_on_event(PB_EVT_BLE_SENT, 0, 0); }
void app_on_ble_rx(const uint8_t *d, uint16_t n) { pb_on_event(PB_EVT_BLE_RECEIVED, d, n); }
void app_on_time_synced(void)    { pb_on_event(PB_EVT_TIME_SYNCED, 0, 0); }
