/* 见 tm_imu.h。照 imu_train/src/gravity_align.py 逐行翻的。 */

#include "tm_imu.h"

#include <math.h>
#include <string.h>

/* 方向无关通道（训练时 channel_mode=inv）：跟 imu_train gravity_align.invariant_channels 逐行对应。
 * 每一路对设备的任意固定旋转都不变——用户怎么戴都一样。u = 窗口内加速度均值方向：
 *   0 v=a·u  1 h=|a−v·u|  2 |a|  [3 ωv  4 ωh  5 |ω|]  倒数第二 θ=atan2(h,v)  最后 jerk=|a_t−a_{t−1}|
 * 按模型编译：这一版模型是 inv 的话库里带 -DTM_IMU_INVARIANT=1（见 lib/BUILD_FLAGS.txt），调用方不用管。 */
#ifndef TM_IMU_INVARIANT
#define TM_IMU_INVARIANT 0
#endif

#if TM_IMU_INVARIANT
static void split3(const float *s, double ux, double uy, double uz, double *v, double *h, double *m)
{
    const double x = s[0], y = s[1], z = s[2];
    *v = x * ux + y * uy + z * uz;
    *m = sqrt(x * x + y * y + z * z);
    const double h2 = *m * *m - *v * *v;
    *h = h2 > 0.0 ? sqrt(h2) : 0.0;
}

static int imu_invariant(const float *raw, int n_t, int n_sensor, float *out)
{
    const int n_ch = n_sensor + 2;
    double gx = 0.0, gy = 0.0, gz = 0.0;
    for (int t = 0; t < n_t; t++) {
        const float *s = raw + (size_t)t * n_sensor;
        gx += s[0];
        gy += s[1];
        gz += s[2];
    }
    gx /= n_t;
    gy /= n_t;
    gz /= n_t;
    double ux = 0.0, uy = 0.0, uz = 1.0;
    const double gn = sqrt(gx * gx + gy * gy + gz * gz);
    if (gn >= 1e-6) {
        ux = gx / gn;
        uy = gy / gn;
        uz = gz / gn;
    }
    for (int t = 0; t < n_t; t++) {
        const float *s = raw + (size_t)t * n_sensor;
        double v, h, m;
        split3(s, ux, uy, uz, &v, &h, &m);
        out[(size_t)0 * n_t + t] = (float)v;
        out[(size_t)1 * n_t + t] = (float)h;
        out[(size_t)2 * n_t + t] = (float)m;
        if (n_sensor == 6) {
            double wv, wh, wm;
            split3(s + 3, ux, uy, uz, &wv, &wh, &wm);
            out[(size_t)3 * n_t + t] = (float)wv;
            out[(size_t)4 * n_t + t] = (float)wh;
            out[(size_t)5 * n_t + t] = (float)wm;
        }
        out[(size_t)(n_ch - 2) * n_t + t] = (float)atan2(h, v);
        double j = 0.0;
        if (t > 0) {
            const float *p = s - n_sensor;
            const double dx = (double)s[0] - p[0], dy = (double)s[1] - p[1], dz = (double)s[2] - p[2];
            j = sqrt(dx * dx + dy * dy + dz * dz);
        }
        out[(size_t)(n_ch - 1) * n_t + t] = (float)j;
    }
    return n_ch;
}
#endif

int tm_imu_channels(const float *raw, int n_t, int n_sensor, float *out)
{
    if (n_sensor != 3 && n_sensor != 6) return -1;
#if TM_IMU_INVARIANT
    return imu_invariant(raw, n_t, n_sensor, out);
#endif
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
