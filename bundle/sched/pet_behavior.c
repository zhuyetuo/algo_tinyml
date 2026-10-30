/* pet_behavior 调度层：事件驱动，取数 → 推理 → 后处理成片段 → 写 Flash → 连上手机就补发。
 *
 * 每个模型版本编一次，模型相关常量在 pb_model_cfg.h（打包时生成，不随包发布）。
 * 全部状态在调用方给的 mem 里（pb_state_t），库自己只有推理那几块静态缓冲。
 *
 * ── Flash 记录（16 字节，环形，按擦除单位分块）──────────────────────────────
 *   seq u32   单调递增，0xFFFFFFFF = 空槽
 *   a   u32   片段：开始时间（flags&PB_F_UNIX ? Unix 秒 : 本次开机的 now_ms）；确认：已确认到的 seq
 *   dur u16   片段时长（0.1 秒）
 *   kind u8   1 = 片段，2 = 确认
 *   cls u8, conf u8, flags u8, boot u8（开机序号低 8 位）, crc u8（前 15 字节 CRC-8）
 * 开机扫一遍：最大的有效 seq 之后接着写；最大的「确认」记录给出已确认位置。
 * 写满一块就擦下一块（最旧的）；里面没确认的片段计入 lost。擦完先补一条确认记录，保证它不会被擦丢。
 *
 * ── BLE 帧（沿用项圈现有格式）55 AA 00 cmd len(2,大端) data 校验和（前面全部字节求和低 8 位）──
 *   0x07 设备→App 当前行为：DP 0x0A 时间戳(8) / 0x30 类别(enum 1) / 0x31 置信度(int 2)   每个窗口一条
 *   0x09 设备→App 行为片段：n(1) + n × 13 字节 {seq u32, start u32, dur_ds u16, cls u8, conf u8, flags u8}（大端）
 *        flags bit0=1：start 是 Unix 秒；bit0=0、bit1=0：未校时，start 为「发送时距片段开始的秒数」；bit1=1：时间未知
 *   0x0A App→设备 确认：seq u32（大端），≤ seq 的片段都已收到，可以覆盖
 */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pet_behavior.h"
#include "pb_model_cfg.h"
#include "tm_imu.h"
#include "tm_post.h"
#include "tm_post_cfg.h"

#if defined(PB_KIND_RF)
#include "tm_feat_cfg.h"
#include "tm_features.h"
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"
#include "tm_forest_c_pipeline_golden.h"
#define PB_N_T TM_FEAT_N_T
#define PB_N_CH TM_FEAT_N_CH
#define PB_N_CLASSES TM_FC_N_CLASSES
#elif defined(PB_KIND_CNN)
#include "tm_golden.h"
#include "tm_model.h"
#include "tm_prep.h"
#include "tm_runtime.h"
#define PB_N_T TM_N_T
#define PB_N_CH TM_N_CH
#define PB_N_CLASSES TM_N_CLASSES
#else
#error "PB_KIND_RF 或 PB_KIND_CNN"
#endif

#define PB_LIB_VERSION 0x0002u   /* v0.2：调度层 */
#define PB_N_SENSOR 3
#define PB_RANGE_G 8
#define PB_POLL_MS 5000u         /* 取数间隔 */
#define PB_GAP_MS 3000           /* 样本时间跟 now_ms 差这么多就当断档 */
#define PB_READ_MAX 128         /* 一次最多读这么多（5 秒 @16Hz 是 80 个），读完一整批再判断有没有断档 */
#define PB_DRIFT_MS 500         /* 样本时间比 now_ms 超前这么多就往回校（IMU 时钟比标称快） */
#define PB_TMP 48
#define PB_MAGIC 0x50425332u     /* "PBS2" */
#define PB_REC 16u
#define PB_TX 256

#define PB_K_SEG 1
#define PB_K_ACK 2
#define PB_F_UNIX 1
#define PB_F_UNKNOWN 2

#if PB_N_CH != PB_N_SENSOR + 2
#error "pet_behavior 只支持三轴加速度模型（5 通道）"
#endif

typedef struct {
    uint32_t seq, a;
    uint16_t dur;
    uint8_t kind, cls, conf, flags, boot, crc;
} pb_rec_t;

typedef struct {
    uint32_t magic;
    const pb_hal_t *hal;
    float acc_scale;

    /* 推理 */
    tm_imu_stream_t imu;
    float ring[PB_N_T * PB_N_SENSOR], lin[PB_N_T * PB_N_SENSOR], x[PB_N_CH * PB_N_T];
    tm_post_t post;
    tm_seg_t tmp[PB_TMP];
    pb_sample_t rd[PB_READ_MAX];
    uint32_t g;                 /* 自上次重置以来的样本数 */
    uint32_t anchor_g, anchor_ms;
    int anchored;

    /* Flash */
    uint32_t n_slots, slots_per_unit;
    uint32_t wpos;              /* 下一个写的槽 */
    uint32_t seq;               /* 最后用掉的 seq */
    uint32_t acked;             /* 已确认到的 seq */
    uint8_t boot;

    /* 校时 */
    int synced;
    uint32_t sync_unix, sync_ms;

    /* BLE */
    int connected, waiting;
    uint32_t wait_since;
    uint32_t send_seq;          /* 下一个要发的片段 seq 下限 */
    uint32_t cursor;            /* 从哪个槽开始找 */
    uint8_t tx[PB_TX];

    pb_stats_t st;
} pb_state_t;

#ifdef PB_SIZE_PROBE
char pb_state_size_probe[sizeof(pb_state_t)];
#else
typedef char pb_work_size_check[(sizeof(pb_state_t) <= PB_WORK_SIZE) ? 1 : -1];
#endif

static pb_state_t *S;

/* ── 推理（同 pet_behavior 算法层） ─────────────────────────────────────── */
static int classify(const float *x, float *prob)
{
#if defined(PB_KIND_RF)
    static float feat[TM_FEAT_DIM];
    int32_t votes[PB_N_CLASSES];
    if (tm_features(&tm_feat_cfg, x, feat) != 0) return -1;
    const int cls = tm_forest_c_predict(&tm_forest_c, feat, votes);
    double sum = 0.0;
    for (int c = 0; c < PB_N_CLASSES; c++) sum += votes[c];
    for (int c = 0; c < PB_N_CLASSES; c++) prob[c] = sum > 0 ? (float)(votes[c] / sum) : 0.0f;
    return cls;
#else
    static int8_t xi[PB_N_CH * PB_N_T];
    static int8_t arena[TM_ARENA_BYTES];
    int8_t out[PB_N_CLASSES];
    tm_prep(&tm_model_prep, x, xi);
    if (tm_invoke(&tm_model, xi, out, arena, TM_ARENA_BYTES) != 0) return -1;
    double mx = -1e30, e[PB_N_CLASSES], sum = 0.0;
    for (int c = 0; c < PB_N_CLASSES; c++) {
        e[c] = (out[c] - tm_model.out_zp) * (double)tm_model.out_scale;
        if (e[c] > mx) mx = e[c];
    }
    for (int c = 0; c < PB_N_CLASSES; c++) sum += (e[c] = exp(e[c] - mx));
    for (int c = 0; c < PB_N_CLASSES; c++) prob[c] = (float)(e[c] / sum);
    return tm_argmax(out, PB_N_CLASSES);
#endif
}

static int run_selftest(void)
{
#if defined(PB_KIND_RF)
    static float feat[TM_FEAT_DIM];
    for (int i = 0; i < TM_FCP_GOLDEN_N; i++) {
        int32_t votes[PB_N_CLASSES];
        tm_features(&tm_feat_cfg, tm_forest_c_pipeline_in + (size_t)i * TM_FCP_N_CH * TM_FCP_N_T, feat);
        tm_forest_c_predict(&tm_forest_c, feat, votes);
        if (memcmp(votes, tm_forest_c_pipeline_votes + (size_t)i * PB_N_CLASSES, sizeof votes) != 0) return -1;
    }
#else
    static int8_t arena[TM_ARENA_BYTES];
    for (int i = 0; i < TM_GOLDEN_N; i++) {
        int8_t out[PB_N_CLASSES];
        tm_invoke(&tm_model, tm_golden_in + (size_t)i * PB_N_CH * PB_N_T, out, arena, TM_ARENA_BYTES);
        if (memcmp(out, tm_golden_out + (size_t)i * PB_N_CLASSES, PB_N_CLASSES) != 0) return -1;
    }
#endif
    return 0;
}

static uint8_t pct(float p)
{
    const float v = p * 100.0f + 0.5f;
    return (uint8_t)(v < 0.0f ? 0 : v > 100.0f ? 100 : v);
}

/* ── 帧 ──────────────────────────────────────────────────────────────────── */
static void be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static uint32_t rd_be32(const uint8_t *p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }

static uint16_t frame(uint8_t cmd, uint16_t dlen)
{
    uint8_t *f = S->tx;
    f[0] = 0x55; f[1] = 0xAA; f[2] = 0x00; f[3] = cmd;
    f[4] = (uint8_t)(dlen >> 8); f[5] = (uint8_t)dlen;
    uint8_t sum = 0;
    for (uint16_t i = 0; i < 6u + dlen; i++) sum = (uint8_t)(sum + f[i]);
    f[6 + dlen] = sum;
    return (uint16_t)(7 + dlen);
}

static int send_raw(uint16_t len)
{
    if (!S->connected || S->waiting) return -1;
    const uint16_t mtu = S->hal->ble_mtu ? S->hal->ble_mtu() : 20;
    if (len > mtu) return -2;
    if (S->hal->ble_send(S->tx, len) != 0) return -1;
    S->waiting = 1;
    S->wait_since = S->hal->now_ms();
    return 0;
}

/* 当前行为（0x07，跟现有 App 协议一致）：时间戳用 Unix 毫秒，未校时用 now_ms */
static void send_current(int cls, uint8_t conf)
{
    uint8_t *d = S->tx + 6;
    uint64_t ts = S->synced ? (uint64_t)S->sync_unix * 1000u + (uint32_t)(S->hal->now_ms() - S->sync_ms)
                            : S->hal->now_ms();
    d[0] = 0x0A; d[1] = 0x02; d[2] = 0x00; d[3] = 0x08;
    be32(d + 4, (uint32_t)(ts >> 32));
    be32(d + 8, (uint32_t)ts);
    d[12] = 0x30; d[13] = 0x04; d[14] = 0x00; d[15] = 0x01; d[16] = (uint8_t)cls;
    d[17] = 0x31; d[18] = 0x02; d[19] = 0x00; d[20] = 0x02; d[21] = 0x00; d[22] = conf;
    send_raw(frame(0x07, 23));
}

/* ── Flash ──────────────────────────────────────────────────────────────── */
static uint8_t crc8(const uint8_t *p, int n)
{
    uint8_t c = 0;
    for (int i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (uint8_t)((c & 0x80) ? (c << 1) ^ 0x07 : c << 1);
    }
    return c;
}

static void pack(const pb_rec_t *r, uint8_t *b)
{
    memcpy(b, &r->seq, 4);
    memcpy(b + 4, &r->a, 4);
    memcpy(b + 8, &r->dur, 2);
    b[10] = r->kind; b[11] = r->cls; b[12] = r->conf; b[13] = r->flags; b[14] = r->boot;
    b[15] = crc8(b, 15);
}

/* 读一个槽；空或坏返回 0 */
static int read_rec(uint32_t slot, pb_rec_t *r)
{
    uint8_t b[PB_REC];
    if (S->hal->flash_read(slot * PB_REC, b, PB_REC) != 0) return 0;
    memcpy(&r->seq, b, 4);
    if (r->seq == 0xFFFFFFFFu || crc8(b, 15) != b[15]) return 0;
    memcpy(&r->a, b + 4, 4);
    memcpy(&r->dur, b + 8, 2);
    r->kind = b[10]; r->cls = b[11]; r->conf = b[12]; r->flags = b[13]; r->boot = b[14]; r->crc = b[15];
    return 1;
}

static void append(pb_rec_t *r);

static void enter_slot(void)
{
    /* 进入新的一块：先擦（擦掉的是最旧的），没确认的片段算丢 */
    if (S->wpos % S->slots_per_unit) return;
    for (uint32_t i = 0; i < S->slots_per_unit; i++) {
        pb_rec_t o;
        if (read_rec(S->wpos + i, &o) && o.kind == PB_K_SEG && o.seq > S->acked) {
            S->st.lost++;
            if (S->st.unacked) S->st.unacked--;
        }
    }
    S->hal->flash_erase(S->wpos * PB_REC, S->slots_per_unit * PB_REC);
    if (S->cursor / S->slots_per_unit == S->wpos / S->slots_per_unit)
        S->cursor = (S->wpos + S->slots_per_unit) % S->n_slots;
}

static void append(pb_rec_t *r)
{
    const int fresh = (S->wpos % S->slots_per_unit) == 0;
    enter_slot();
    if (fresh && S->acked && r->kind != PB_K_ACK) {
        /* 新块的第一条先补确认记录，最新的确认位置永远不会被擦掉 */
        pb_rec_t a = {0};
        a.seq = ++S->seq; a.a = S->acked; a.kind = PB_K_ACK; a.boot = S->boot;
        uint8_t b[PB_REC];
        pack(&a, b);
        S->hal->flash_write(S->wpos * PB_REC, b, PB_REC);
        S->wpos = (S->wpos + 1) % S->n_slots;
    }
    r->seq = ++S->seq;
    r->boot = S->boot;
    uint8_t b[PB_REC];
    pack(r, b);
    S->hal->flash_write(S->wpos * PB_REC, b, PB_REC);
    S->wpos = (S->wpos + 1) % S->n_slots;
}

static void scan_flash(void)
{
    uint32_t max_seq = 0, max_slot = 0, acked = 0;
    int max_boot = -1, any = 0;
    for (uint32_t i = 0; i < S->n_slots; i++) {
        pb_rec_t r;
        if (!read_rec(i, &r)) continue;
        if (!any || r.seq > max_seq) {
            max_seq = r.seq;
            max_slot = i;
            max_boot = r.boot;
            any = 1;
        }
        if (r.kind == PB_K_ACK && r.a > acked) acked = r.a;
    }
    S->seq = max_seq;
    S->acked = acked;
    S->boot = (uint8_t)(max_boot + 1);
    S->wpos = any ? (max_slot + 1) % S->n_slots : 0;
    /* 下一块要是半写过（上次掉电时），接着写会撞上旧数据：跳到下一块开头 */
    if (any && S->wpos % S->slots_per_unit) {
        uint8_t b[PB_REC];
        S->hal->flash_read(S->wpos * PB_REC, b, PB_REC);
        for (int k = 0; k < (int)PB_REC; k++)
            if (b[k] != 0xFF) {
                S->wpos = (S->wpos / S->slots_per_unit + 1) * S->slots_per_unit % S->n_slots;
                break;
            }
    }
    S->st.unacked = 0;
    for (uint32_t i = 0; i < S->n_slots; i++) {
        pb_rec_t r;
        if (read_rec(i, &r) && r.kind == PB_K_SEG && r.seq > acked) S->st.unacked++;
    }
    /* 最旧的数据在写指针的下一块 */
    S->cursor = ((S->wpos / S->slots_per_unit + 1) * S->slots_per_unit) % S->n_slots;
}

/* ── 片段 → Flash ───────────────────────────────────────────────────────── */
static void store_segments(const tm_seg_t *s, int n)
{
    const uint32_t now = S->hal->now_ms();
    for (int i = 0; i < n; i++) {
        if (s[i].cls < 0 || s[i].cls >= PB_N_CLASSES) continue;
        pb_rec_t r = {0};
        r.kind = PB_K_SEG;
        r.cls = (uint8_t)s[i].cls;
        r.conf = pct(s[i].conf_mean);
        const uint32_t d = (s[i].end_ms - s[i].start_ms + 50u) / 100u;
        r.dur = (uint16_t)(d > 0xFFFFu ? 0xFFFFu : d);
        if (S->synced) {
            r.a = S->sync_unix + (uint32_t)((int32_t)(s[i].start_ms - S->sync_ms) / 1000);
            r.flags = PB_F_UNIX;
        } else {
            r.a = s[i].start_ms;
        }
        (void)now;
        append(&r);
        S->st.segments++;
        S->st.unacked++;
    }
}

/* ── 补发 ──────────────────────────────────────────────────────────────── */
static void pump(void)
{
    if (!S->connected) return;
    if (S->waiting && (uint32_t)(S->hal->now_ms() - S->wait_since) > 2000u) S->waiting = 0;   /* 没等到 SENT，兜底 */
    if (S->waiting) return;
    const uint16_t mtu = S->hal->ble_mtu ? S->hal->ble_mtu() : 20;
    int per = (mtu > 8 ? (mtu - 8) : 0) / 13;
    if (per > (PB_TX - 8) / 13) per = (PB_TX - 8) / 13;
    if (per <= 0) return;
    uint8_t *d = S->tx + 6;
    int n = 0;
    uint32_t pos = S->cursor, last_seq = 0, next_pos = pos;
    const uint32_t now = S->hal->now_ms();
    while (pos != S->wpos && n < per) {
        pb_rec_t r;
        if (read_rec(pos, &r) && r.kind == PB_K_SEG && r.seq >= S->send_seq && r.seq > S->acked) {
            uint8_t *e = d + 1 + n * 13;
            uint32_t start = r.a;
            uint8_t fl = r.flags & PB_F_UNIX;
            if (!fl) {
                if (r.boot != S->boot) {
                    fl = PB_F_UNKNOWN;
                    start = 0;
                } else if (S->synced) {
                    start = S->sync_unix + (uint32_t)((int32_t)(r.a - S->sync_ms) / 1000);
                    fl = PB_F_UNIX;
                } else {
                    start = (now - r.a) / 1000u;   /* 距今多少秒 */
                }
            }
            be32(e, r.seq);
            be32(e + 4, start);
            e[8] = (uint8_t)(r.dur >> 8); e[9] = (uint8_t)r.dur;
            e[10] = r.cls; e[11] = r.conf; e[12] = fl;
            last_seq = r.seq;
            n++;
        }
        pos = (pos + 1) % S->n_slots;
        next_pos = pos;
    }
    if (n == 0) {
        S->cursor = pos;   /* 没有要发的了，下次从写指针接着找，不用再扫一遍 */
        return;
    }
    d[0] = (uint8_t)n;
    if (send_raw(frame(0x09, (uint16_t)(1 + n * 13))) == 0) {
        S->send_seq = last_seq + 1;
        S->cursor = next_pos;
    }
}

static void on_ack(uint32_t seq)
{
    if (seq <= S->acked || seq > S->seq) return;
    /* 数一下这次确认掉多少片段 */
    for (uint32_t i = 0; i < S->n_slots; i++) {
        pb_rec_t r;
        if (read_rec(i, &r) && r.kind == PB_K_SEG && r.seq > S->acked && r.seq <= seq && S->st.unacked)
            S->st.unacked--;
    }
    S->acked = seq;
    pb_rec_t a = {0};
    a.kind = PB_K_ACK;
    a.a = seq;
    append(&a);
}

static void on_rx(const uint8_t *p, uint16_t len)
{
    while (len >= 7) {
        if (p[0] != 0x55 || p[1] != 0xAA) { p++; len--; continue; }
        const uint16_t dl = (uint16_t)((p[4] << 8) | p[5]);
        if (len < 7u + dl) return;
        uint8_t sum = 0;
        for (uint16_t i = 0; i < 6u + dl; i++) sum = (uint8_t)(sum + p[i]);
        if (sum == p[6 + dl] && p[3] == 0x0A && dl >= 4) on_ack(rd_be32(p + 6));
        p += 7u + dl;
        len = (uint16_t)(len - (7u + dl));
    }
}

/* ── 取数 + 推理 ──────────────────────────────────────────────────────── */
static uint32_t ms_of(uint32_t g)
{
    const int64_t off = (int64_t)(int32_t)(g - S->anchor_g);
    return S->anchor_ms + (uint32_t)(int32_t)((off * 1000) / PB_HZ);
}

static void reset_stream(void)
{
    tm_post_cfg_t pc;
    tm_post_cfg_from_names(&pc, PB_CLASS_NAMES, PB_N_CLASSES, (float)PB_N_T / PB_HZ, (float)PB_HOP / PB_HZ);
    tm_post_init(&S->post, &pc);
    tm_imu_stream_init(&S->imu, S->ring, S->lin, PB_N_SENSOR, PB_N_T, PB_HOP);
    S->g = 0;
    S->anchored = 0;
}

static void flush_post(void)
{
    const int k = tm_post_flush(&S->post, S->tmp, PB_TMP);
    if (k > 0) store_segments(S->tmp, k);
}

static void feed(const pb_sample_t *smp, uint16_t n)
{
    float prob[PB_N_CLASSES];
    for (uint16_t i = 0; i < n; i++, S->g++) {
        const float a[PB_N_SENSOR] = {smp[i].x * S->acc_scale, smp[i].y * S->acc_scale, smp[i].z * S->acc_scale};
        S->st.samples++;
        if (!tm_imu_push(&S->imu, a, S->x)) continue;
        const int cls = classify(S->x, prob);
        if (cls < 0) continue;
        S->st.windows++;
        if (S->connected) send_current(cls, pct(prob[cls]));
        const int k = tm_post_on_window(&S->post, prob, ms_of(S->g + 1 - PB_N_T), S->tmp, PB_TMP);
        if (k > 0) store_segments(S->tmp, k);
    }
}

static void anchor_last(uint32_t g_last, uint32_t t)
{
    S->anchor_g = g_last;
    S->anchor_ms = t;
    S->anchored = 1;
}

static void poll_imu(void)
{
    uint16_t n = S->hal->imu_read(S->rd, PB_READ_MAX);
    if (n == 0) return;
    if (n > PB_READ_MAX) n = PB_READ_MAX;
    const uint32_t now = S->hal->now_ms();
    int more = n == PB_READ_MAX;
    /* FIFO 读空了（没有更多）：这批最后一个样本就是「现在」。按采样间隔推算它的时间，
     * 比现在早很多 = 中间丢了数据（FIFO 溢出 / 太久没取）：前面的收尾、从这批重新开始 */
    if (S->anchored && !more && (int32_t)(now - ms_of(S->g + n - 1)) > PB_GAP_MS) {
        flush_post();
        reset_stream();
        S->st.gaps++;
    }
    if (!S->anchored) anchor_last(S->g + n - 1, now);
    feed(S->rd, n);
    for (int loops = 0; more && loops < 64; loops++) {
        n = S->hal->imu_read(S->rd, PB_READ_MAX);
        if (n > PB_READ_MAX) n = PB_READ_MAX;
        more = n == PB_READ_MAX;
        feed(S->rd, n);
    }
    /* IMU 时钟比标称快：样本时间跑到 now 前面去了，往回校（只会跳回不到一秒） */
    if ((int32_t)(now - ms_of(S->g - 1)) < -PB_DRIFT_MS) anchor_last(S->g - 1, now);
}

/* ── 对外 ──────────────────────────────────────────────────────────────── */
int pb_init_v(const pb_hal_t *hal, void *mem, uint32_t mem_size, uint32_t header_version)
{
    S = NULL;
    if (header_version != PB_HEADER_VERSION) return PB_ERR_VERSION;
    if (!hal || !mem || ((uintptr_t)mem & 3u) || mem_size < sizeof(pb_state_t)) return PB_ERR_PARAM;
    if (!hal->imu_config || !hal->imu_read || !hal->now_ms || !hal->unix_time || !hal->set_wakeup ||
        !hal->flash_write || !hal->flash_read || !hal->flash_erase || !hal->ble_send)
        return PB_ERR_PARAM;
    if (hal->flash_erase_unit < PB_REC || hal->flash_erase_unit % PB_REC ||
        hal->flash_size % hal->flash_erase_unit || hal->flash_size < 2 * hal->flash_erase_unit)
        return PB_ERR_PARAM;
    pb_state_t *s = (pb_state_t *)mem;
    memset(s, 0, sizeof *s);
    s->hal = hal;
    s->acc_scale = (float)PB_RANGE_G / 32768.0f;
    s->n_slots = hal->flash_size / PB_REC;
    s->slots_per_unit = hal->flash_erase_unit / PB_REC;
    S = s;
    if (run_selftest() != 0) {
        S = NULL;
        return PB_ERR_SELFTEST;
    }
    reset_stream();
    scan_flash();
    const uint32_t u = hal->unix_time();
    if (u) {
        s->synced = 1;
        s->sync_unix = u;
        s->sync_ms = hal->now_ms();
    }
    if (hal->imu_config(PB_HZ, PB_RANGE_G) != 0) {
        S = NULL;
        return PB_ERR_IMU;
    }
    s->magic = PB_MAGIC;
    hal->set_wakeup(hal->now_ms() + PB_POLL_MS);
    return 0;
}

void pb_on_event(uint8_t event, const uint8_t *data, uint16_t len)
{
    if (!S || S->magic != PB_MAGIC) return;
    switch (event) {
    case PB_EVT_WAKEUP:
    case PB_EVT_IMU_FIFO:
        poll_imu();
        pump();
        S->hal->set_wakeup(S->hal->now_ms() + PB_POLL_MS);
        break;
    case PB_EVT_BLE_CONNECTED:
        S->connected = 1;
        S->waiting = 0;
        S->send_seq = S->acked + 1;      /* 每次连上都从没确认的开始发 */
        S->cursor = ((S->wpos / S->slots_per_unit + 1) * S->slots_per_unit) % S->n_slots;
        pump();
        break;
    case PB_EVT_BLE_DISCONNECTED:
        S->connected = 0;
        S->waiting = 0;
        break;
    case PB_EVT_BLE_SENT:
        S->waiting = 0;
        pump();
        break;
    case PB_EVT_BLE_RECEIVED:
        if (data && len) on_rx(data, len);
        pump();
        break;
    case PB_EVT_TIME_SYNCED: {
        const uint32_t u = S->hal->unix_time();
        if (u) {
            S->synced = 1;
            S->sync_unix = u;
            S->sync_ms = S->hal->now_ms();
        }
        break;
    }
    default:
        break;
    }
}

uint32_t pb_version(void) { return (PB_LIB_VERSION << 16) | (PB_MODEL_VERSION & 0xFFFFu); }

void pb_get_stats(pb_stats_t *out)
{
    if (!out) return;
    if (S) *out = S->st;
    else memset(out, 0, sizeof *out);
}
