/* tm_imu 的宿主测试：stdin 读 "n_sensor n_t hop n" 再读 n 个样本，
 * 走流式接口 tm_imu_push，每出一个窗口打一行：出窗时的样本序号 + n_ch*n_t 个 float。 */
#include <stdio.h>
#include <stdlib.h>

#include "tm_imu.h"

int main(void)
{
    int n_sensor, n_t, hop, n;
    if (scanf("%d %d %d %d", &n_sensor, &n_t, &hop, &n) != 4) return 2;
    float *ring = malloc(sizeof(float) * n_t * n_sensor);
    float *lin = malloc(sizeof(float) * n_t * n_sensor);
    float *out = malloc(sizeof(float) * n_t * (n_sensor + 2));
    tm_imu_stream_t s;
    tm_imu_stream_init(&s, ring, lin, n_sensor, n_t, hop);
    float smp[6];
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n_sensor; k++)
            if (scanf("%f", &smp[k]) != 1) return 3;
        if (tm_imu_push(&s, smp, out)) {
            printf("%d", i);
            for (int j = 0; j < n_t * (n_sensor + 2); j++) printf(" %.9g", out[j]);
            printf("\n");
        }
    }
    return 0;
}
