/* 见 tm_imu.h。照 imu_train/src/gravity_align.py 逐行翻的。 */

#include "tm_imu.h"

#include <math.h>
#include <string.h>

int tm_imu_channels(const float *raw, int n_t, int n_sensor, float *out)
{
    if (n_sensor != 3 && n_sensor != 6) return -1;
    const int n_ch = n_sensor + 2;
    float *pitch = out + (size_t)(n_ch - 2) * n_t;
    float *roll = out + (size_t)(n_ch - 1) * n_t;

    /* 1. 姿态角：用对齐前的加速度（Python 是 float32 数组上的 np.arctan2） */
    double gx = 0.0, gy = 0.0, gz = 0.0;
    for (int t = 0; t < n_t; t++) {
        const float *s = raw + (size_t)t * n_sensor;
        const float ax = s[0], ay = s[1], az = s[2];
        pitch[t] = atan2f(-ax, sqrtf(ay * ay + az * az));
        roll[t] = atan2f(ay, az);
        gx += ax;
        gy += ay;
        gz += az;
    }

    /* 2. 重力对齐的旋转 R（float64，跟 Python 一样） */
    gx /= n_t;
    gy /= n_t;
    gz /= n_t;
    double R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const double norm = sqrt(gx * gx + gy * gy + gz * gz);
    if (norm >= 1e-6) {
        const double ux = gx / norm, uy = gy / norm, uz = gz / norm;
        double dot = uz; /* 跟 (0,0,1) 点积 */
        if (dot > 1.0) dot = 1.0;
        if (dot < -1.0) dot = -1.0;
        if (dot > 0.9999) {
            /* 已经朝 +Z，不转 */
        } else if (dot < -0.9999) {
            R[1][1] = -1.0;
            R[2][2] = -1.0;
        } else {
            /* axis = u × (0,0,1) = (uy, -ux, 0)，归一化 */
            double kx = uy, ky = -ux;
            const double kn = sqrt(kx * kx + ky * ky);
            kx /= kn;
            ky /= kn;
            const double kz = 0.0;
            const double ang = acos(dot);
            const double sn = sin(ang), c1 = 1.0 - cos(ang);
            const double K[3][3] = {{0.0, -kz, ky}, {kz, 0.0, -kx}, {-ky, kx, 0.0}};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double kk = 0.0;
                    for (int m = 0; m < 3; m++) kk += K[i][m] * K[m][j];
                    R[i][j] = (i == j ? 1.0 : 0.0) + sn * K[i][j] + c1 * kk;
                }
            }
        }
    }

    /* 3. 旋转 acc（和 gyr），写成通道在前 */
    for (int t = 0; t < n_t; t++) {
        const float *s = raw + (size_t)t * n_sensor;
        for (int blk = 0; blk < n_sensor; blk += 3) {
            const double v0 = s[blk], v1 = s[blk + 1], v2 = s[blk + 2];
            for (int i = 0; i < 3; i++) {
                out[(size_t)(blk + i) * n_t + t] = (float)(R[i][0] * v0 + R[i][1] * v1 + R[i][2] * v2);
            }
        }
    }
    return n_ch;
}

void tm_imu_stream_init(tm_imu_stream_t *s, float *ring, float *lin, int n_sensor, int n_t, int hop)
{
    s->ring = ring;
    s->lin = lin;
    s->n_sensor = (int16_t)n_sensor;
    s->n_t = (int16_t)n_t;
    s->hop = (int16_t)(hop > 0 ? hop : 1);
    s->head = 0;
    s->filled = 0;
    s->since = 0;
}

int tm_imu_push(tm_imu_stream_t *s, const float *sample, float *out)
{
    memcpy(s->ring + (size_t)s->head * s->n_sensor, sample, sizeof(float) * (size_t)s->n_sensor);
    s->head = (int16_t)((s->head + 1) % s->n_t);
    /* 窗口起点跟训练时的滑窗一致：0, hop, 2hop…——攒满 n_t 出第一个，之后每 hop 个出一个 */
    if (s->filled < s->n_t) {
        if (++s->filled < s->n_t) return 0;
        s->since = 0;
    } else if (++s->since < s->hop) {
        return 0;
    } else {
        s->since = 0;
    }
    /* head 指向最旧的样本 */
    for (int t = 0; t < s->n_t; t++) {
        const int src = (s->head + t) % s->n_t;
        memcpy(s->lin + (size_t)t * s->n_sensor, s->ring + (size_t)src * s->n_sensor,
               sizeof(float) * (size_t)s->n_sensor);
    }
    tm_imu_channels(s->lin, s->n_t, s->n_sensor, out);
    return 1;
}
