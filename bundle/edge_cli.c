/* PC 参考实现 / 命令行测试：采集 CSV → 逐窗口状态序列。
 *
 * 编的是包里 core/ 和 model/ 下**板上那份 C**，不是另写的等价实现：
 *   CSV → （可选）按量程换算成 g / °/s → （可选）整数倍降采样 → 滑窗
 *       → tm_imu_channels（重力对齐 + pitch/roll，8 通道）
 *       → RF: tm_features + tm_forest_c_predict   /   CNN: tm_prep + tm_invoke
 * 所以这里的判决就是板上会给的判决；--in-hz 50 这种重采样也照抄平台（resample_poly）。
 *
 *   ./edge_cli --selftest              golden 自检（跟导出时 Python 算的逐位对答案）
 *   ./edge_cli sample.csv              逐窗口结果打到 stdout（CSV）
 *   ./edge_cli --help
 *
 * 编译期选路线：-DTM_KIND_RF 或 -DTM_KIND_CNN（Makefile 已经按模型选好）。
 */

#define _POSIX_C_SOURCE 200809L /* clock_gettime；-std=c99 下默认不给 */
#define _XOPEN_SOURCE 700       /* M_PI */

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

/* ── 重采样：照抄平台（imu_train infer_csv_scratch.downsample）──────────────────
 *   采样率相同      → 不动
 *   整数倍（up==1） → 直接隔点抽（data[::down]，平台就是这么做的，不滤波）
 *   其它（50→16）  → scipy.signal.resample_poly(x, up, down)：Kaiser(β=5) 窗 FIR 低通，
 *                     半长 10×max(up,down)，零填充边界，取中心对齐的那段输出
 * 平台对缺失掩码也走同一个函数再 > 0.5，这里一样。 */
static double bessel_i0(double x)
{
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 200; k++) {
        const double q = x / (2.0 * k);
        t *= q * q;
        s += t;
        if (t < 1e-17 * s) break;
    }
    return s;
}

/* scipy firwin(2*half+1, 1/max_rate, window=('kaiser', 5.0))，再 × up。
 * 平台的输入是 float32，scipy 会把 h 转成 float32 再乘 up——这里一样 */
static float *design_fir(int up, int down, int *half_len)
{
    const int max_rate = up > down ? up : down;
    const int half = 10 * max_rate, n = 2 * half + 1;
    const double fc = 1.0 / max_rate, beta = 5.0, i0b = bessel_i0(beta);
    double *hd = malloc(sizeof(double) * n), sum = 0.0;
    float *h = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        const double m = i - 0.5 * (n - 1);
        const double xs = fc * m;
        const double sinc = xs == 0.0 ? 1.0 : sin(M_PI * xs) / (M_PI * xs);
        const double r = 2.0 * i / (n - 1) - 1.0;
        hd[i] = fc * sinc * bessel_i0(beta * sqrt(1.0 - r * r)) / i0b;
        sum += hd[i];
    }
    for (int i = 0; i < n; i++) h[i] = (float)(hd[i] / sum) * (float)up;
    free(hd);
    *half_len = half;
    return h;
}

/* 一路信号：x[k*stride]，k < n_in → y[i*stride]，i < 返回值 */
static long resample_poly(const float *x, long n_in, int stride, float *y, int up, int down,
                          const float *h, int half)
{
    const long n_out = (n_in * up) / down + ((n_in * up) % down != 0);
    const int pre_pad = down - half % down;
    const long pre_remove = (half + pre_pad) / down;
    const long hlen = 2L * half + 1;
    for (long i = 0; i < n_out; i++) {
        /* upfirdn：y[o] = Σ_j hp[j]·xu[o·down − j]，hp = [pre_pad 个 0, h]，xu 是插零上采样 */
        const long m0 = (i + pre_remove) * down - pre_pad;   /* 对 h 本身的下标偏移 */
        long k_lo = m0 - (hlen - 1);
        k_lo = k_lo <= 0 ? 0 : (k_lo + up - 1) / up;
        long k_hi = m0 < 0 ? -1 : m0 / up;
        if (k_hi > n_in - 1) k_hi = n_in - 1;
        double acc = 0.0;
        for (long k = k_lo; k <= k_hi; k++) acc += (double)h[m0 - k * up] * x[k * stride];
        y[i * stride] = (float)acc;
    }
    return n_out;
}

static long gcd_l(long a, long b)
{
    while (b) {
        long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static void resample_series(series_t *s, int in_hz, int model_hz)
{
    if (in_hz == model_hz || s->n == 0) return;
    const long g = gcd_l(in_hz, model_hz);
    const int up = (int)(model_hz / g), down = (int)(in_hz / g);
    float *vf = malloc(sizeof(float) * s->n);
    for (long i = 0; i < s->n; i++) vf[i] = s->valid[i] ? 1.0f : 0.0f;
    long m;
    if (up == 1) {
        m = (s->n + down - 1) / down;
        for (long i = 0; i < m; i++) {
            memmove(s->x + i * N_SENSOR, s->x + i * down * N_SENSOR, sizeof(float) * N_SENSOR);
            vf[i] = vf[i * down];
        }
    } else {
        int half;
        float *h = design_fir(up, down, &half);
        const long cap = (s->n * up) / down + 1;
        float *y = malloc(sizeof(float) * N_SENSOR * cap), *vy = malloc(sizeof(float) * cap);
        m = 0;
        for (int c = 0; c < N_SENSOR; c++) m = resample_poly(s->x + c, s->n, N_SENSOR, y + c, up, down, h, half);
        resample_poly(vf, s->n, 1, vy, up, down, h, half);
        free(s->x);
        free(h);
        free(vf);
        s->x = y;
        vf = vy;
        s->valid = realloc(s->valid, cap);
    }
    for (long i = 0; i < m; i++) s->valid[i] = (unsigned char)(vf[i] > 0.5f);
    free(vf);
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
           "  --in-hz N        CSV 的采样率（如 50），默认 = 模型的 %d Hz；按平台同一套方法重采样\n"
           "                   （整数倍隔点抽，其它比例用 resample_poly 那套 FIR，跟平台逐窗口对得上）\n"
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
    if (in_hz <= 0 || in_hz > 10000 || hop <= 0) {
        fprintf(stderr, "--in-hz 要 > 0、--hop 要 > 0\n");
        return 2;
    }

    series_t s;
    if (read_csv(path, acc_scale, gyr_scale, &s) != 0) return 1;
    resample_series(&s, in_hz, TM_EDGE_HZ);
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
