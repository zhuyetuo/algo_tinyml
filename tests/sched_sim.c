/* 调度层 PC 模拟：假 IMU（按虚拟时钟从文件出样本，FIFO 有容量会溢出）、假 Flash（RAM，
 * 可存盘模拟断电重启）、假 BLE（发出的帧解码打到 stdout）。stdin 读脚本：
 *   run S        虚拟时间往前走 S 秒，按 set_wakeup 触发 PB_EVT_WAKEUP
 *   stall S      往前走 S 秒但不触发（模拟取数停掉，FIFO 溢出）
 *   connect / disconnect
 *   ack on|off   收到 0x09 后自动回确认（最大 seq）
 *   sync U       校时：unix_time 从此返回 U + 经过的秒数，触发 PB_EVT_TIME_SYNCED
 *   reboot       Flash 保留，内存清空重新 pb_init
 *   stats
 * 用法：sched_sim imu.bin flash_size erase_unit mtu fifo_cap < script */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pet_behavior.h"
#include "pb_model_cfg.h"

static int16_t *imu_data;
static long imu_n, imu_pos;          /* imu_pos：下一个要进 FIFO 的样本 */
static long fifo_head;               /* FIFO 里最旧的样本号 */
static long fifo_cap;
static uint32_t now, wake_at;
static int wake_set;
static uint8_t *flash;
static uint32_t flash_size, erase_unit;
static int connected, ack_on;
static uint16_t mtu;
static uint32_t unix_base, unix_ms;
static int synced;
static uint8_t mem[PB_WORK_SIZE] __attribute__((aligned(8)));
static uint32_t pending_ack;

static long due(void) { return (long)((uint64_t)now * PB_HZ / 1000u); }

static void fill_fifo(void)
{
    long d = due();
    if (d > imu_n) d = imu_n;
    imu_pos = d;
    if (imu_pos - fifo_head > fifo_cap) fifo_head = imu_pos - fifo_cap;   /* 溢出：丢最旧的 */
}

static int h_imu_config(uint16_t hz, uint8_t g) { printf("IMU %u %u\n", hz, g); return 0; }
static uint16_t h_imu_read(pb_sample_t *buf, uint16_t max)
{
    fill_fifo();
    uint16_t n = 0;
    while (n < max && fifo_head < imu_pos) {
        buf[n].x = imu_data[fifo_head * 3];
        buf[n].y = imu_data[fifo_head * 3 + 1];
        buf[n].z = imu_data[fifo_head * 3 + 2];
        n++;
        fifo_head++;
    }
    return n;
}
static uint32_t h_now(void) { return now; }
static uint32_t h_unix(void) { return synced ? unix_base + (now - unix_ms) / 1000u : 0; }
static void h_wake(uint32_t at) { wake_at = at; wake_set = 1; }
static int h_fw(uint32_t off, const void *d, uint32_t len)
{
    if (off + len > flash_size) return -1;
    for (uint32_t i = 0; i < len; i++) flash[off + i] &= ((const uint8_t *)d)[i];   /* NOR：只能 1→0 */
    return 0;
}
static int h_fr(uint32_t off, void *d, uint32_t len)
{
    if (off + len > flash_size) return -1;
    memcpy(d, flash + off, len);
    return 0;
}
static int h_fe(uint32_t off, uint32_t len)
{
    if (off % erase_unit || len % erase_unit || off + len > flash_size) { printf("BAD_ERASE\n"); return -1; }
    memset(flash + off, 0xFF, len);
    return 0;
}
static uint32_t be32(const uint8_t *p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static int h_send(const uint8_t *d, uint16_t len)
{
    if (!connected) return -1;
    if (len > mtu) { printf("TOO_LONG %u\n", len); return -1; }
    uint8_t sum = 0;
    for (int i = 0; i < len - 1; i++) sum = (uint8_t)(sum + d[i]);
    if (d[0] != 0x55 || d[1] != 0xAA || sum != d[len - 1]) { printf("BAD_FRAME\n"); return -1; }
    if (d[3] == 0x07) {
        printf("CUR %u %u %u\n", now, d[6 + 16], d[6 + 22]);
    } else if (d[3] == 0x09) {
        const int n = d[6];
        for (int i = 0; i < n; i++) {
            const uint8_t *e = d + 7 + i * 13;
            printf("SEG %u %u %u %u %u %u\n", be32(e), be32(e + 4), (e[8] << 8) | e[9], e[10], e[11], e[12]);
            if (be32(e) > pending_ack) pending_ack = be32(e);
        }
    }
    return 0;
}
static uint16_t h_mtu(void) { return mtu; }

static pb_hal_t hal = {h_imu_config, h_imu_read, h_now, h_unix, h_wake, h_fw, h_fr, h_fe, 0, 0, h_send, h_mtu};

static void after_send(void)
{
    /* 发完立刻回 SENT；开着自动确认就回 0x0A */
    for (int k = 0; k < 10000 && connected; k++) {
        uint32_t before = pending_ack;
        pb_on_event(PB_EVT_BLE_SENT, NULL, 0);
        if (ack_on && pending_ack) {
            uint8_t f[11] = {0x55, 0xAA, 0x00, 0x0A, 0x00, 0x04, 0, 0, 0, 0, 0};
            f[6] = (uint8_t)(pending_ack >> 24); f[7] = (uint8_t)(pending_ack >> 16);
            f[8] = (uint8_t)(pending_ack >> 8); f[9] = (uint8_t)pending_ack;
            uint8_t s = 0;
            for (int i = 0; i < 10; i++) s = (uint8_t)(s + f[i]);
            f[10] = s;
            pb_on_event(PB_EVT_BLE_RECEIVED, f, sizeof f);
        }
        if (pending_ack == before) break;
    }
}

static int do_init(void)
{
    hal.flash_size = flash_size;
    hal.flash_erase_unit = erase_unit;
    memset(mem, 0xA5, sizeof mem);
    const int r = pb_init(&hal, mem, sizeof mem);
    printf("INIT %d\n", r);
    return r;
}

int main(int argc, char **argv)
{
    if (argc < 6) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    fseek(f, 0, SEEK_END);
    imu_n = ftell(f) / 6;
    fseek(f, 0, SEEK_SET);
    imu_data = malloc(imu_n * 6);
    if (fread(imu_data, 6, imu_n, f) != (size_t)imu_n) return 2;
    fclose(f);
    flash_size = (uint32_t)atol(argv[2]);
    erase_unit = (uint32_t)atol(argv[3]);
    mtu = (uint16_t)atoi(argv[4]);
    fifo_cap = atol(argv[5]);
    flash = malloc(flash_size);
    memset(flash, 0xFF, flash_size);
    if (do_init() != 0) return 1;

    char cmd[32];
    double arg;
    while (scanf("%31s", cmd) == 1) {
        if (!strcmp(cmd, "run") || !strcmp(cmd, "stall")) {
            if (scanf("%lf", &arg) != 1) return 2;
            const uint32_t end = now + (uint32_t)(arg * 1000);
            const int fire = !strcmp(cmd, "run");
            while (now < end) {
                uint32_t next = end;
                if (fire && wake_set && wake_at < next) next = wake_at;
                now = next < now ? now : next;
                if (fire && wake_set && now >= wake_at) {
                    wake_set = 0;
                    pb_on_event(PB_EVT_WAKEUP, NULL, 0);
                    after_send();
                    if (!wake_set) { printf("NO_REWAKE\n"); return 1; }
                }
            }
            fill_fifo();
        } else if (!strcmp(cmd, "connect")) {
            connected = 1;
            pending_ack = 0;
            pb_on_event(PB_EVT_BLE_CONNECTED, NULL, 0);
            after_send();
        } else if (!strcmp(cmd, "disconnect")) {
            connected = 0;
            pb_on_event(PB_EVT_BLE_DISCONNECTED, NULL, 0);
        } else if (!strcmp(cmd, "ack")) {
            if (scanf("%31s", cmd) != 1) return 2;
            ack_on = !strcmp(cmd, "on");
        } else if (!strcmp(cmd, "sync")) {
            if (scanf("%lf", &arg) != 1) return 2;
            synced = 1;
            unix_base = (uint32_t)arg;
            unix_ms = now;
            pb_on_event(PB_EVT_TIME_SYNCED, NULL, 0);
        } else if (!strcmp(cmd, "reboot")) {
            /* 断电：内存全丢，Flash 留着；IMU 接着出数据 */
            connected = 0;
            synced = 0;
            wake_set = 0;
            if (do_init() != 0) return 1;
        } else if (!strcmp(cmd, "stats")) {
            pb_stats_t s;
            pb_get_stats(&s);
            printf("STATS samples=%u windows=%u segments=%u unacked=%u lost=%u gaps=%u\n",
                   s.samples, s.windows, s.segments, s.unacked, s.lost, s.gaps);
        }
        fflush(stdout);
    }
    return 0;
}
