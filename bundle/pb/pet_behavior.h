/* 宠物行为识别库 pet_behavior
 * 输入三轴加速度原始数据，输出行为片段（类别 + 起止时间 + 置信度）。
 * 接口说明见 README.md。本文件随库一起发布，请勿修改。
 */
#ifndef PET_BEHAVIOR_H
#define PET_BEHAVIOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PB_HEADER_VERSION   @HEADER_VERSION@u   /* 头文件与库须同一版本，否则 pb_init 返回 PB_ERR_VERSION */
#define PB_WORK_SIZE        @WORK_SIZE@u        /* 库的工作内存（字节），4 字节对齐 */
#define PB_SAMPLE_RATE_HZ   @HZ@               /* 采样率，pb_config_t.sample_rate_hz 填这个值 */

/* 错误码 */
#define PB_ERR_PARAM        (-1)   /* 参数错误 */
#define PB_ERR_NOT_INIT     (-2)   /* 未初始化 */
#define PB_ERR_TIME         (-3)   /* 时间戳倒退 */
#define PB_ERR_SELFTEST     (-4)   /* 自检未通过，拒绝推理 */
#define PB_ERR_VERSION      (-5)   /* 头文件与库版本不一致 */

/* 行为类别（只增不改；收到未知编号按 PB_TYPE_OTHER 处理） */
#define PB_TYPE_REST        0      /* 休息 */
#define PB_TYPE_EAT         1      /* 进食 */
#define PB_TYPE_DRINK       2      /* 喝水 */
#define PB_TYPE_LICK        3      /* 舔舐 */
#define PB_TYPE_SCRATCH     4      /* 抓挠 */
#define PB_TYPE_BARK        5      /* 吠叫 */
#define PB_TYPE_OTHER       6      /* 其他 */
#define PB_TYPE_NOT_WORN    7      /* 未佩戴 */

typedef struct {
    uint16_t sample_rate_hz;   /* 采样率，固定填 PB_SAMPLE_RATE_HZ */
    uint8_t  acc_range_g;      /* 加速度计满量程：2 / 4 / 8 / 16（单位 g） */
    uint8_t  reserved;
} pb_config_t;

typedef struct {
    int16_t x, y, z;           /* 加速度计原始计数，FIFO 读出直接填 */
} pb_sample_t;                 /* 6 字节 */

typedef struct {
    uint32_t start_ms;         /* 开始时间，与 t0_ms 同一时间基准 */
    uint32_t end_ms;           /* 结束时间 */
    uint8_t  type;             /* 行为类别 PB_TYPE_* */
    uint8_t  confidence;       /* 置信度 0~100 */
    uint16_t reserved;
} pb_segment_t;                /* 12 字节，小端 */

/* 初始化（内含自检）。mem 至少 PB_WORK_SIZE 字节、4 字节对齐，库运行期间不得释放或改写。 */
#define pb_init(cfg, mem, mem_size) pb_init_v((cfg), (mem), (mem_size), PB_HEADER_VERSION)
int      pb_init_v(const pb_config_t *cfg, void *mem, uint32_t mem_size, uint32_t header_version);

/* 喂一批样本。t0_ms 为该批第一个样本的时间，其余按采样间隔推算；相邻两批须连续。
 * 返回本次写入 out 的片段数（>=0），负数为错误码。out 放不下的片段留到下次调用输出。 */
int      pb_push(const pb_sample_t *samples, uint16_t n, uint32_t t0_ms,
                 pb_segment_t *out, uint8_t max_out);

/* 休眠前、数据断档前、上报前调用：取出剩余片段。之后的数据视为新的一段。 */
int      pb_flush(pb_segment_t *out, uint8_t max_out);

/* 当前状态：最近一个窗口（约 1 秒）的判断，不经后处理，供 App 实时显示。
 * 返回 0 成功；还没有足够数据时返回 1。 */
int      pb_current(uint8_t *type, uint8_t *confidence);

/* 自检：返回 0 表示库在本板运行正确。 */
int      pb_selftest(void);

/* 高 16 位库版本，低 16 位模型版本。 */
uint32_t pb_version(void);

/* 片段缓冲溢出丢掉的片段数（max_out 太小且长期不取时才会发生）。 */
uint32_t pb_lost(void);

#ifdef __cplusplus
}
#endif

#endif /* PET_BEHAVIOR_H */
