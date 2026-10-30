/* pet_behavior：协议接口（pet_behavior.h）→ 端侧运行时（tm_imu → 模型 → tm_post）。
 *
 * 每个模型版本编一次，模型相关的常量在 pb_model_cfg.h（打包时生成）：
 *   PB_MODEL_VERSION   训练任务号（低 16 位）
 *   PB_HZ / PB_HOP     模型采样率、窗口步长（样本数）
 *   PB_TYPE_OF_CLASS   模型类别下标 → 协议类别 PB_TYPE_*
 *   PB_CLASS_NAMES     模型类别名（后处理按名字认抓挠/甩身体）
 *   PB_KIND_RF / PB_KIND_CNN
 *
 * 全部状态在调用方给的 mem 里（pb_state_t），库自己只有模型推理那几块静态缓冲。
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

#define PB_LIB_VERSION 0x0001u   /* v0.1 */
#define PB_N_SENSOR 3
#define PB_MAGIC 0x50424C31u     /* "PBL1" */
#define PB_QUEUE 64
#define PB_TMP 48                /* 后处理一次定稿的片段数上限（超出的会被后处理丢掉，所以给足） */

#if PB_N_CH != PB_N_SENSOR + 2
#error "pet_behavior 只支持三轴加速度模型（5 通道）"
#endif

typedef struct {
    uint32_t magic;
    float    acc_scale;                 /* 原始计数 → g */
    int      selftest_ok;

    tm_imu_stream_t imu;
    float    ring[PB_N_T * PB_N_SENSOR];
    float    lin[PB_N_T * PB_N_SENSOR];
    float    x[PB_N_CH * PB_N_T];

    /* 时间：当前这一批的第一个样本是第 batch_g0 个（自上次 flush 起），时间 batch_t0 */
    uint32_t g;                         /* 已喂的样本数 */
    uint32_t batch_g0, batch_t0;
    uint32_t last_ms;                   /* 上一批最后一个样本的时间 */
    int      has_batch;

    int      cur_valid;
    uint8_t  cur_type, cur_conf;

    tm_post_t post;
    tm_seg_t tmp[PB_TMP];

    pb_segment_t q[PB_QUEUE];           /* 待取出的片段 */
    uint8_t  q_head, q_n;
    uint32_t lost;
} pb_state_t;

#ifdef PB_SIZE_PROBE
/* 打包时用 nm -S 读它的大小，填进 pet_behavior.h 的 PB_WORK_SIZE */
char pb_state_size_probe[sizeof(pb_state_t)];
#else
typedef char pb_work_size_check[(sizeof(pb_state_t) <= PB_WORK_SIZE) ? 1 : -1];
#endif

static pb_state_t *S;

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
    /* 输入映射冒烟：水平静止时不转、pitch/roll 为 0 */
    float raw[PB_N_T * PB_N_SENSOR], out[PB_N_CH * PB_N_T];
    for (int t = 0; t < PB_N_T; t++)
        for (int k = 0; k < PB_N_SENSOR; k++) raw[t * PB_N_SENSOR + k] = (k == 2) ? 1.0f : 0.0f;
    tm_imu_channels(raw, PB_N_T, PB_N_SENSOR, out);
    if (fabsf(out[2 * PB_N_T] - 1.0f) > 1e-6f || fabsf(out[(PB_N_CH - 2) * PB_N_T]) > 1e-6f) return -1;
    return 0;
}

/* 第 g 个样本（自上次 flush 起）的时间 */
static uint32_t ms_of(uint32_t g)
{
    const int64_t off = (int64_t)(int32_t)(g - S->batch_g0);
    const int64_t d = off >= 0 ? (off * 1000 + PB_HZ / 2) / PB_HZ : -((-off * 1000 + PB_HZ / 2) / PB_HZ);
    return S->batch_t0 + (uint32_t)(int32_t)d;
}

static uint8_t conf_pct(float p)
{
    const float v = p * 100.0f + 0.5f;
    return (uint8_t)(v < 0.0f ? 0 : v > 100.0f ? 100 : v);
}

static void enqueue(const tm_seg_t *s, int n)
{
    for (int i = 0; i < n; i++) {
        if (s[i].cls < 0 || s[i].cls >= PB_N_CLASSES) continue;
        if (S->q_n == PB_QUEUE) {           /* 满了丢最旧的 */
            S->q_head = (uint8_t)((S->q_head + 1) % PB_QUEUE);
            S->q_n--;
            S->lost++;
        }
        pb_segment_t *o = &S->q[(S->q_head + S->q_n) % PB_QUEUE];
        o->start_ms = s[i].start_ms;
        o->end_ms = s[i].end_ms;
        o->type = PB_TYPE_OF_CLASS[s[i].cls];
        o->confidence = conf_pct(s[i].conf_mean);
        o->reserved = 0;
        S->q_n++;
    }
}

static int dequeue(pb_segment_t *out, uint8_t max_out)
{
    int n = 0;
    while (n < max_out && S->q_n) {
        out[n++] = S->q[S->q_head];
        S->q_head = (uint8_t)((S->q_head + 1) % PB_QUEUE);
        S->q_n--;
    }
    return n;
}

static void reset_stream(void)
{
    tm_post_cfg_t pc;
    tm_post_cfg_from_names(&pc, PB_CLASS_NAMES, PB_N_CLASSES, (float)PB_N_T / PB_HZ, (float)PB_HOP / PB_HZ);
    tm_post_init(&S->post, &pc);
    tm_imu_stream_init(&S->imu, S->ring, S->lin, PB_N_SENSOR, PB_N_T, PB_HOP);
    S->g = 0;
    S->has_batch = 0;
}

int pb_init_v(const pb_config_t *cfg, void *mem, uint32_t mem_size, uint32_t header_version)
{
    S = NULL;
    if (header_version != PB_HEADER_VERSION) return PB_ERR_VERSION;
    if (!cfg || !mem || ((uintptr_t)mem & 3u) || mem_size < sizeof(pb_state_t)) return PB_ERR_PARAM;
    if (cfg->sample_rate_hz != PB_HZ) return PB_ERR_PARAM;
    if (cfg->acc_range_g != 2 && cfg->acc_range_g != 4 && cfg->acc_range_g != 8 && cfg->acc_range_g != 16)
        return PB_ERR_PARAM;
    pb_state_t *s = (pb_state_t *)mem;
    memset(s, 0, sizeof *s);
    s->acc_scale = (float)cfg->acc_range_g / 32768.0f;
    S = s;
    reset_stream();
    s->selftest_ok = run_selftest() == 0;
    s->magic = PB_MAGIC;
    return s->selftest_ok ? 0 : PB_ERR_SELFTEST;
}

int pb_push(const pb_sample_t *samples, uint16_t n, uint32_t t0_ms, pb_segment_t *out, uint8_t max_out)
{
    if (!S || S->magic != PB_MAGIC) return PB_ERR_NOT_INIT;
    if (!S->selftest_ok) return PB_ERR_SELFTEST;
    if ((n && !samples) || (max_out && !out)) return PB_ERR_PARAM;
    if (n == 0) return dequeue(out, max_out);
    if (S->has_batch && (int32_t)(t0_ms - S->last_ms) <= 0) return PB_ERR_TIME;

    S->batch_g0 = S->g;
    S->batch_t0 = t0_ms;
    S->has_batch = 1;
    float prob[PB_N_CLASSES];
    for (uint16_t i = 0; i < n; i++, S->g++) {
        const float a[PB_N_SENSOR] = {samples[i].x * S->acc_scale, samples[i].y * S->acc_scale,
                                      samples[i].z * S->acc_scale};
        if (!tm_imu_push(&S->imu, a, S->x)) continue;
        const int cls = classify(S->x, prob);
        if (cls < 0) continue;
        S->cur_valid = 1;
        S->cur_type = PB_TYPE_OF_CLASS[cls];
        S->cur_conf = conf_pct(prob[cls]);
        const int k = tm_post_on_window(&S->post, prob, ms_of(S->g + 1 - PB_N_T), S->tmp, PB_TMP);
        if (k > 0) enqueue(S->tmp, k);
    }
    S->last_ms = ms_of(S->g - 1);
    return dequeue(out, max_out);
}

int pb_flush(pb_segment_t *out, uint8_t max_out)
{
    if (!S || S->magic != PB_MAGIC) return PB_ERR_NOT_INIT;
    if (max_out && !out) return PB_ERR_PARAM;
    const int k = tm_post_flush(&S->post, S->tmp, PB_TMP);
    if (k > 0) enqueue(S->tmp, k);
    const uint32_t last = S->last_ms;
    const int had = S->has_batch;
    reset_stream();
    S->has_batch = had;   /* 断档后时间仍不许倒退 */
    S->last_ms = last;
    return dequeue(out, max_out);
}

int pb_current(uint8_t *type, uint8_t *confidence)
{
    if (!S || S->magic != PB_MAGIC) return PB_ERR_NOT_INIT;
    if (!type || !confidence) return PB_ERR_PARAM;
    if (!S->cur_valid) return 1;
    *type = S->cur_type;
    *confidence = S->cur_conf;
    return 0;
}

int pb_selftest(void)
{
    const int r = run_selftest();
    if (S && S->magic == PB_MAGIC) S->selftest_ok = r == 0;
    return r == 0 ? 0 : PB_ERR_SELFTEST;
}

uint32_t pb_version(void) { return (PB_LIB_VERSION << 16) | (PB_MODEL_VERSION & 0xFFFFu); }

uint32_t pb_lost(void) { return S ? S->lost : 0; }
