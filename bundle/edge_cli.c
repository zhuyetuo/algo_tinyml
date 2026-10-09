/* PC 参考实现 / 命令行测试：采集 CSV → 逐窗口状态序列。
 *
 * 编的是包里 core/ 和 model/ 下**板上那份 C**，不是另写的等价实现：
 *   CSV → （可选）按量程换算成 g / °/s → （可选）整数倍降采样 → 滑窗
 *       → tm_imu_channels（重力对齐 + pitch/roll，8 通道）
 *       → RF: tm_features + tm_forest_c_predict   /   CNN: tm_prep + tm_invoke
 * 所以这里的判决就是板上会给的判决（降采样那一步除外，见 --in-hz）。
 *
 *   ./edge_cli --selftest              golden 自检（跟导出时 Python 算的逐位对答案）
 *   ./edge_cli sample.csv              逐窗口结果打到 stdout（CSV）
 *   ./edge_cli --help
 *
 * 编译期选路线：-DTM_KIND_RF 或 -DTM_KIND_CNN（Makefile 已经按模型选好）。
 */

#define _POSIX_C_SOURCE 200809L /* clock_gettime；-std=c99 下默认不给 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tm_edge_cfg.h"
#include "tm_imu.h"

#if defined(TM_KIND_RF)
#include "tm_feat_cfg.h"
#include "tm_features.h"
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"
#include "tm_forest_c_pipeline_golden.h"
#define N_T TM_FEAT_N_T
#define N_CH TM_FEAT_N_CH
#define N_CLASSES TM_FC_N_CLASSES
#define CLASS_NAMES TM_FC_CLASS_NAMES
#elif defined(TM_KIND_CNN)
#include "tm_golden.h"
#include "tm_model.h"
#include "tm_prep.h"
#include "tm_runtime.h"
#define N_T TM_N_T
#define N_CH TM_N_CH
#define N_CLASSES TM_N_CLASSES
#define CLASS_NAMES TM_CLASS_NAMES
#else
#error "编译时要 -DTM_KIND_RF 或 -DTM_KIND_CNN"
#endif

#define N_SENSOR (N_CH - 2)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── 一个窗口：x 是 [N_CH][N_T]（tm_imu_channels 的输出），返回类别，score 写每类置信度 ── */
static int classify(const float *x, float *score)
{
#if defined(TM_KIND_RF)
    static float feat[TM_FEAT_DIM];
    int32_t votes[N_CLASSES];
    if (tm_features(&tm_feat_cfg, x, feat) != 0) return -1;
    const int cls = tm_forest_c_predict(&tm_forest_c, feat, votes);
    double sum = 0.0;
    for (int c = 0; c < N_CLASSES; c++) sum += votes[c];
    for (int c = 0; c < N_CLASSES; c++) score[c] = sum > 0 ? (float)(votes[c] / sum) : 0.0f;
    return cls;
#else
    static int8_t xi[N_CH * N_T];
    static int8_t arena[TM_ARENA_BYTES];
    int8_t out[N_CLASSES];
    tm_prep(&tm_model_prep, x, xi);
    if (tm_invoke(&tm_model, xi, out, arena, TM_ARENA_BYTES) != 0) return -1;
    /* int8 分数反量化后 softmax，只为了打一个 0~1 的置信度；判类别只看 argmax */
    double mx = -1e30, e[N_CLASSES], sum = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        e[c] = (out[c] - tm_model.out_zp) * (double)tm_model.out_scale;
        if (e[c] > mx) mx = e[c];
    }
    for (int c = 0; c < N_CLASSES; c++) sum += (e[c] = exp(e[c] - mx));
    for (int c = 0; c < N_CLASSES; c++) score[c] = (float)(e[c] / sum);
    return tm_argmax(out, N_CLASSES);
#endif
}

/* ── golden 自检 ─────────────────────────────────────────────────────────── */
static int selftest(void)
{
    int bad = 0, n = 0;
#if defined(TM_KIND_RF)
    static float feat[TM_FEAT_DIM];
    for (int i = 0; i < TM_FCP_GOLDEN_N; i++, n++) {
        int32_t votes[N_CLASSES];
        tm_features(&tm_feat_cfg, tm_forest_c_pipeline_in + (size_t)i * TM_FCP_N_CH * TM_FCP_N_T, feat);
        tm_forest_c_predict(&tm_forest_c, feat, votes);
        if (memcmp(votes, tm_forest_c_pipeline_votes + (size_t)i * N_CLASSES, sizeof votes) != 0) {
            fprintf(stderr, "golden #%d 不一致：votes", i);
            for (int c = 0; c < N_CLASSES; c++)
                fprintf(stderr, " %d/%d", (int)votes[c], (int)tm_forest_c_pipeline_votes[i * N_CLASSES + c]);
            fprintf(stderr, "（本机/期望）\n");
            bad++;
        }
    }
#else
    static int8_t arena[TM_ARENA_BYTES];
    for (int i = 0; i < TM_GOLDEN_N; i++, n++) {
        int8_t out[N_CLASSES];
        tm_invoke(&tm_model, tm_golden_in + (size_t)i * N_CH * N_T, out, arena, TM_ARENA_BYTES);
        if (memcmp(out, tm_golden_out + (size_t)i * N_CLASSES, N_CLASSES) != 0) {
            fprintf(stderr, "golden #%d 不一致\n", i);
            bad++;
        }
    }
#endif
    /* tm_imu 冒烟：水平静止（重力 = +Z）时对齐不转、pitch/roll 为 0 */
    {
        float raw[N_T * N_SENSOR], out[N_CH * N_T];
        for (int t = 0; t < N_T; t++)
            for (int k = 0; k < N_SENSOR; k++) raw[t * N_SENSOR + k] = (k == 2) ? 1.0f : 0.0f;
        tm_imu_channels(raw, N_T, N_SENSOR, out);
        if (fabsf(out[2 * N_T] - 1.0f) > 1e-6f || fabsf(out[(N_CH - 2) * N_T]) > 1e-6f) {
            fprintf(stderr, "tm_imu 冒烟不过\n");
            bad++;
        }
        n++;
    }
    printf("selftest: %d/%d 通过%s\n", n - bad, n, bad ? "  ← 有不一致：检查编译选项是否带 -ffp-contract=off" : "");
    return bad ? 1 : 0;
}

/* ── CSV ─────────────────────────────────────────────────────────────────── */
static const char *ACC_NAMES[][3] = {{"acc_x", "acc_y", "acc_z"}, {"accx", "accy", "accz"},
                                     {"ax", "ay", "az"}};
static const char *GYR_NAMES[][3] = {{"gyro_x", "gyro_y", "gyro_z"}, {"gyr_x", "gyr_y", "gyr_z"},
                                     {"gyrox", "gyroy", "gyroz"}, {"gx", "gy", "gz"}};

static void lower_trim(char *s)
{
    char *p = s, *q = s;
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3;
    while (*p && isspace((unsigned char)*p)) p++;
    while (*p) *q++ = (char)tolower((unsigned char)*p++);
    *q = 0;
    while (q > s && isspace((unsigned char)q[-1])) *--q = 0;
}

static int split(char *line, char **f, int max)
{
    int n = 0;
    char *p = line;
    while (n < max) {
        f[n++] = p;
        char *c = strchr(p, ',');
        if (!c) break;
        *c = 0;
        p = c + 1;
    }
    for (int i = 0; i < n; i++) {
        char *e = f[i] + strlen(f[i]);
        while (e > f[i] && (e[-1] == '\n' || e[-1] == '\r')) *--e = 0;
    }
    return n;
}

static int find_cols(char **f, int nf, const char *names[][3], int ngroups, int *col)
{
    for (int g = 0; g < ngroups; g++) {
        int hit = 0;
        for (int k = 0; k < 3; k++) {
            col[k] = -1;
            for (int i = 0; i < nf; i++)
                if (strcmp(f[i], names[g][k]) == 0) col[k] = i;
            hit += col[k] >= 0;
        }
        if (hit == 3) return 1;
    }
    return 0;
}

static int is_missing(const char *s, float *v)
{
    char *end;
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return 1;
    *v = strtof(s, &end);
    return end == s || isnan(*v);
}

typedef struct {
    float *x;      /* [n][N_SENSOR] */
    unsigned char *valid;
    long n, cap;
} series_t;

static int read_csv(const char *path, float acc_scale, float gyr_scale, series_t *out)
{
    FILE *fp = strcmp(path, "-") == 0 ? stdin : fopen(path, "r");
    if (!fp) {
        perror(path);
        return -1;
    }
    static char line[1 << 16];
    char *f[256];
    int col[6] = {0, 1, 2, 3, 4, 5};
    int have_header = 0;
    float last[N_SENSOR] = {0};
    int seen_valid = 0;
    long first_valid = -1;
    out->n = 0;
    out->cap = 4096;
    out->x = malloc(sizeof(float) * N_SENSOR * out->cap);
    out->valid = malloc(out->cap);
    while (fgets(line, sizeof line, fp)) {
        int nf = split(line, f, 256);
        if (!have_header) {
            have_header = 1;
            float tmp;
            if (is_missing(f[0], &tmp)) { /* 第一格不是数 → 当表头 */
                for (int i = 0; i < nf; i++) lower_trim(f[i]);
                if (!find_cols(f, nf, ACC_NAMES, 3, col)) {
                    fprintf(stderr, "找不到加速度列（acc_x/acc_y/acc_z、AccX…、ax…）\n");
                    return -1;
                }
                if (N_SENSOR == 6 && !find_cols(f, nf, GYR_NAMES, 4, col + 3)) {
                    fprintf(stderr, "这是 6 轴模型，找不到角速度列（gyro_x…、gyr_x…、GyroX…、gx…）\n");
                    return -1;
                }
                continue;
            }
            /* 没表头：按 ax ay az [gx gy gz] 的列序读 */
        }
        if (out->n == out->cap) {
            out->cap *= 2;
            out->x = realloc(out->x, sizeof(float) * N_SENSOR * out->cap);
            out->valid = realloc(out->valid, out->cap);
        }
        float s[N_SENSOR];
        int ok = 1;
        for (int k = 0; k < N_SENSOR; k++) {
            if (col[k] >= nf || is_missing(f[col[k]], &s[k])) {
                ok = 0;
                break;
            }
            s[k] *= (k < 3) ? acc_scale : gyr_scale;
        }
        /* 缺失行（蓝牙断联）：前向填充，跟平台一致；同时记下来，缺得多的窗口跳过 */
        if (ok) {
            memcpy(last, s, sizeof s);
            if (!seen_valid) first_valid = out->n;
            seen_valid = 1;
        }
        memcpy(out->x + out->n * N_SENSOR, last, sizeof last);
        out->valid[out->n] = (unsigned char)ok;
        out->n++;
    }
    if (fp != stdin) fclose(fp);
    /* 开头的缺失行用第一条有效值回填（pandas 的 bfill） */
    for (long i = 0; i < first_valid; i++)
        memcpy(out->x + i * N_SENSOR, out->x + first_valid * N_SENSOR, sizeof(float) * N_SENSOR);
    return 0;
}

/* 整数倍降采样：块平均。平台用的是 scipy resample_poly（FIR），两者在边沿上会有小差别 */
static void decimate(series_t *s, int k)
{
    if (k <= 1) return;
    long m = s->n / k;
    for (long i = 0; i < m; i++) {
        int nv = 0;
        for (int c = 0; c < N_SENSOR; c++) {
            double acc = 0.0;
            for (int j = 0; j < k; j++) acc += s->x[(i * k + j) * N_SENSOR + c];
            s->x[i * N_SENSOR + c] = (float)(acc / k);
        }
        for (int j = 0; j < k; j++) nv += s->valid[i * k + j];
        s->valid[i] = (unsigned char)(nv * 2 > k);
    }
    s->n = m;
}

static void usage(const char *argv0)
{
    printf("用法：\n"
           "  %s --selftest\n"
           "  %s [选项] input.csv        （input 用 - 表示 stdin）\n\n"
           "模型：%d 类，%d 通道 × %d 点 @ %d Hz，步长 %d 点\n"
           "CSV：表头里要有 acc_x,acc_y,acc_z%s（AccX/ax 这类写法也认），其它列忽略；\n"
           "     没表头就按 ax,ay,az%s 的列序读。空格 / NaN 当缺失（前向填充，缺失 > 30%% 的窗口跳过）。\n\n"
           "选项：\n"
           "  --in-hz N        CSV 的采样率，默认 = 模型的 %d Hz；必须是它的整数倍（块平均降采样）\n"
           "  --acc-scale F    加速度乘 F 换成 g。int16 原始计数用 量程/32768，例如 ±16g → 0.00048828125\n"
           "  --gyr-scale F    角速度乘 F 换成 °/s，例如 ±2000dps → 0.06103515625\n"
           "  --hop N          窗口步长（点），默认 %d\n"
           "  --out FILE       结果写到文件，默认 stdout\n\n"
           "输出 CSV：window,start_s,end_s,class_id,class,conf,<每类置信度…>；汇总打到 stderr。\n",
           argv0, argv0, N_CLASSES, N_CH, N_T, TM_EDGE_HZ, TM_EDGE_HOP,
           N_SENSOR == 6 ? ",gyro_x,gyro_y,gyro_z" : "", N_SENSOR == 6 ? ",gx,gy,gz" : "", TM_EDGE_HZ,
           TM_EDGE_HOP);
}

int main(int argc, char **argv)
{
    int in_hz = TM_EDGE_HZ, hop = TM_EDGE_HOP;
    float acc_scale = 1.0f, gyr_scale = 1.0f;
    const char *path = NULL, *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--selftest")) return selftest();
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        }
        if (i + 1 < argc && !strcmp(a, "--in-hz")) in_hz = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(a, "--acc-scale")) acc_scale = strtof(argv[++i], NULL);
        else if (i + 1 < argc && !strcmp(a, "--gyr-scale")) gyr_scale = strtof(argv[++i], NULL);
        else if (i + 1 < argc && !strcmp(a, "--hop")) hop = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(a, "--out")) out_path = argv[++i];
        else if (a[0] == '-' && a[1]) {
            fprintf(stderr, "不认识的参数 %s（--help 看用法）\n", a);
            return 2;
        } else path = a;
    }
    if (!path) {
        usage(argv[0]);
        return 2;
    }
    if (in_hz <= 0 || in_hz % TM_EDGE_HZ != 0 || hop <= 0) {
        fprintf(stderr, "--in-hz 要是 %d 的整数倍、--hop 要 > 0\n", TM_EDGE_HZ);
        return 2;
    }

    series_t s;
    if (read_csv(path, acc_scale, gyr_scale, &s) != 0) return 1;
    decimate(&s, in_hz / TM_EDGE_HZ);
    /* 量纲自检：重力是 1g。原始计数没换算的话这里是几千 */
    if (s.n > 0) {
        double g = 0.0;
        long m = s.n < 256 ? s.n : 256;
        for (long i = 0; i < m; i++) {
            const float *v = s.x + i * N_SENSOR;
            g += sqrt((double)v[0] * v[0] + (double)v[1] * v[1] + (double)v[2] * v[2]);
        }
        g /= m;
        if (g < 0.3 || g > 3.0)
            fprintf(stderr, "[警告] 加速度模长均值 %.3g，不像以 g 为单位（应 ≈ 1）。"
                            "原始计数请用 --acc-scale 换算；单位是 m/s² 用 --acc-scale 0.10197\n", g);
    }

    FILE *fo = out_path ? fopen(out_path, "w") : stdout;
    if (!fo) {
        perror(out_path);
        return 1;
    }
    fprintf(fo, "window,start_s,end_s,class_id,class,conf");
    for (int c = 0; c < N_CLASSES; c++) fprintf(fo, ",p_%s", CLASS_NAMES[c]);
    fprintf(fo, "\n");

    static float x[N_CH * N_T];
    float score[N_CLASSES];
    long count[N_CLASSES] = {0}, n_win = 0, n_skip = 0;
    double t_infer = 0.0;
    for (long st = 0; st + N_T <= s.n; st += hop) {
        int nv = 0;
        for (int t = 0; t < N_T; t++) nv += s.valid[st + t];
        if (nv < 0.7 * N_T) {
            n_skip++;
            continue;
        }
        double t0 = now_sec();
        tm_imu_channels(s.x + st * N_SENSOR, N_T, N_SENSOR, x);
        int cls = classify(x, score);
        t_infer += now_sec() - t0;
        if (cls < 0) {
            fprintf(stderr, "推理失败（窗口 %ld）\n", n_win);
            return 1;
        }
        count[cls]++;
        fprintf(fo, "%ld,%.3f,%.3f,%d,%s,%.4f", n_win, (double)st / TM_EDGE_HZ,
                (double)(st + N_T) / TM_EDGE_HZ, cls, CLASS_NAMES[cls], score[cls]);
        for (int c = 0; c < N_CLASSES; c++) fprintf(fo, ",%.4f", score[c]);
        fprintf(fo, "\n");
        n_win++;
    }
    if (fo != stdout) fclose(fo);

    fprintf(stderr, "样本 %ld 个（@%d Hz，%.1f s），窗口 %ld 个，因缺失跳过 %ld 个\n", s.n, TM_EDGE_HZ,
            (double)s.n / TM_EDGE_HZ, n_win, n_skip);
    for (int c = 0; c < N_CLASSES; c++)
        fprintf(stderr, "  %-8s %6ld  %5.1f%%\n", CLASS_NAMES[c], count[c], n_win ? 100.0 * count[c] / n_win : 0.0);
    if (n_win)
        fprintf(stderr, "每窗 %.1f µs（x86，只能横向比，跟 Cortex-M4F 没有可比性）\n", t_infer / n_win * 1e6);
    free(s.x);
    free(s.valid);
    return 0;
}
