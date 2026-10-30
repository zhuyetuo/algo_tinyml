/* 宠物行为识别库 pet_behavior（调度层 + 算法 + 模型）
 *
 * 嵌入式侧只做两件事：实现 pb_hal_t 里的硬件接口；事件发生时调 pb_on_event()。
 * 什么时候取数、怎么识别、存什么、什么时候发、断连补发，都在库里。说明见 README.md。
 * 本文件随库一起发布，请勿修改。
 */
#ifndef PET_BEHAVIOR_H
#define PET_BEHAVIOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PB_HEADER_VERSION   @HEADER_VERSION@u   /* 头文件与库须同一版本，否则 pb_init 返回 PB_ERR_VERSION */
#define PB_WORK_SIZE        @WORK_SIZE@u        /* 库的工作内存（字节），4 字节对齐 */

/* 错误码 */
#define PB_ERR_PARAM        (-1)   /* 参数错误（hal 缺函数、内存不够、Flash 区域不合规） */
#define PB_ERR_SELFTEST     (-4)   /* 自检未通过 */
#define PB_ERR_VERSION      (-5)   /* 头文件与库版本不一致 */
#define PB_ERR_IMU          (-6)   /* imu_config 失败 */

/* 事件（pb_on_event 的 event） */
#define PB_EVT_WAKEUP            1   /* set_wakeup 设定的时间到 */
#define PB_EVT_IMU_FIFO          2   /* IMU FIFO 达到水位 */
#define PB_EVT_BLE_CONNECTED     3   /* 手机连上 */
#define PB_EVT_BLE_DISCONNECTED  4   /* 手机断开 */
#define PB_EVT_BLE_SENT          5   /* 上一包发送完成 */
#define PB_EVT_BLE_RECEIVED      6   /* 收到手机发来的数据，data/len 为原始字节 */
#define PB_EVT_TIME_SYNCED       7   /* 系统时间已校准（unix_time 开始返回非 0） */

typedef struct {
    int16_t x, y, z;           /* 加速度计原始计数 */
} pb_sample_t;

typedef struct {
    /* IMU：只开加速度计，按 rate_hz / range_g 配置并启动 FIFO，成功返回 0 */
    int      (*imu_config)(uint16_t rate_hz, uint8_t range_g);
    /* 从 FIFO 读出最多 max 个样本（按采样先后），返回实际个数；FIFO 空返回 0 */
    uint16_t (*imu_read)(pb_sample_t *buf, uint16_t max);

    /* 时间 */
    uint32_t (*now_ms)(void);              /* 单调递增毫秒，休眠期间不停 */
    uint32_t (*unix_time)(void);           /* Unix 秒；未校时返回 0 */
    void     (*set_wakeup)(uint32_t at_ms); /* 在 at_ms（now_ms 基准）产生 PB_EVT_WAKEUP；再调覆盖上一次 */

    /* 库专用 Flash 区域，offset 为区域内偏移；成功返回 0 */
    int      (*flash_write)(uint32_t offset, const void *data, uint32_t len);
    int      (*flash_read)(uint32_t offset, void *data, uint32_t len);
    int      (*flash_erase)(uint32_t offset, uint32_t len);   /* offset/len 按 flash_erase_unit 对齐 */
    uint32_t flash_size;                   /* 区域大小（字节），至少 2 个擦除单位 */
    uint32_t flash_erase_unit;             /* 擦除单位（字节），如 4096 */

    /* BLE：发一包到手机，len <= ble_mtu()；成功返回 0，未连接或忙返回负数 */
    int      (*ble_send)(const uint8_t *data, uint16_t len);
    uint16_t (*ble_mtu)(void);
} pb_hal_t;

/* 开机调用一次（内含自检）。mem 至少 PB_WORK_SIZE 字节、4 字节对齐，之后不得释放或改写。
 * 返回 0 成功，负数为错误码。 */
#define pb_init(hal, mem, mem_size) pb_init_v((hal), (mem), (mem_size), PB_HEADER_VERSION)
int      pb_init_v(const pb_hal_t *hal, void *mem, uint32_t mem_size, uint32_t header_version);

/* 事件发生时调用：在任务或主循环里调，不在中断里调；hal 函数里不得回调本函数。 */
void     pb_on_event(uint8_t event, const uint8_t *data, uint16_t len);

/* 高 16 位库版本，低 16 位模型版本。 */
uint32_t pb_version(void);

/* 诊断计数（可选，调试用） */
typedef struct {
    uint32_t samples;          /* 已处理的样本数 */
    uint32_t windows;          /* 已推理的窗口数 */
    uint32_t segments;         /* 已存的行为片段数 */
    uint32_t unacked;          /* Flash 里还没被 App 确认的片段数 */
    uint32_t lost;             /* 存储满被覆盖、还没确认就丢掉的片段数 */
    uint32_t gaps;             /* 数据断档次数（FIFO 溢出 / 长时间没取数） */
} pb_stats_t;
void     pb_get_stats(pb_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PET_BEHAVIOR_H */
