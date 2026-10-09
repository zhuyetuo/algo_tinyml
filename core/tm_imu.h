/* IMU 原始采样 → 模型输入通道（"8 通道输入映射"），RF 和 CNN 两条路线共用。
 *
 * 训练（imu_train 的 infer_csv_scratch.py / gravity_align.py）对每个窗口做的事，
 * 端上必须照做一遍，否则模型看到的是另一种输入——不报错，只是准确率掉：
 *
 *   IMU 每个样本 6 个数（3 轴模型是 3 个）：
 *       acc_x acc_y acc_z [gyr_x gyr_y gyr_z]
 *       单位：加速度 **g**，角速度 **°/s**（int16 原始计数要先按量程换算）
 *   凑满一个窗口（n_t 个样本）后：
 *       1. pitch/roll：用**未旋转**的加速度逐样本算（弧度）
 *              pitch = atan2(-ax, sqrt(ay² + az²))   roll = atan2(ay, az)
 *       2. 重力对齐：窗口内加速度均值当重力方向 g，求把 g 转到 +Z 的旋转 R
 *          （Rodrigues；|g|≈0 不转；g 已朝 +Z 不转；g 朝 -Z 绕 X 轴转 180°），
 *          R 同时作用到这个窗口的 acc 和 gyr
 *       3. 拼成 n_sensor + 2 个通道（6 轴 → 8，3 轴 → 5），**通道在前** [n_ch][n_t]：
 *              0..2  对齐后的 acc_x acc_y acc_z
 *              3..5  对齐后的 gyr_x gyr_y gyr_z   （3 轴模型没有这三路）
 *              最后两路  pitch roll（对齐前算的）
 *   这个 [n_ch][n_t] 就是 tm_features（RF）/ tm_prep（CNN）的输入。
 *
 * 跟 Python 不是逐位一致（那边旋转是 float64、atan2 用的是 numpy），差在 float32 末位；
 * tests/test_imu_c.py 量的是这一段的最大误差和整条链判决是否一致。
 */

#ifndef TM_IMU_H
#define TM_IMU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 一个窗口做映射。
 *   raw : float [n_t][n_sensor]，**时间在前**（就是采样顺序），n_sensor = 3 或 6
 *   out : float [n_sensor + 2][n_t]，**通道在前**
 * 返回输出通道数（5 或 8）；n_sensor 不是 3/6 返回 -1。 */
int tm_imu_channels(const float *raw, int n_t, int n_sensor, float *out);

/* 流式：每来一个 IMU 样本喂一次，每 hop 个样本吐一个映射好的窗口。 */
typedef struct {
    float *ring;        /* 调用方给：n_t * n_sensor 个 float */
    float *lin;         /* 调用方给：n_t * n_sensor 个 float（出窗时按时间顺序排好） */
    int16_t n_sensor;
    int16_t n_t;
    int16_t hop;
    int16_t head;
    int16_t filled;
    int16_t since;
} tm_imu_stream_t;

void tm_imu_stream_init(tm_imu_stream_t *s, float *ring, float *lin, int n_sensor, int n_t, int hop);

/* sample: float [n_sensor]（g / °/s）。返回 1 表示凑齐一个窗口、已写进
 * out（float [n_sensor + 2][n_t]）；返回 0 表示还没到。 */
int tm_imu_push(tm_imu_stream_t *s, const float *sample, float *out);

#ifdef __cplusplus
}
#endif

#endif /* TM_IMU_H */
