#include "tm_features.h"

#include <math.h>
#include <string.h>

#if TM_CMSIS_DSP
#include "arm_math.h"
#endif

/* 所有求和都是从左到右逐个累加，跟 Python 参考实现的 fsum 一样。
 * 别"顺手"改成分块累加或者 SIMD——浮点加法不满足结合律，一改两边就对不上了，
 * 而且差别小到肉眼看不出来。真要提速，先改这里再重跑 golden vector。 */
static float fsum(const float *a, int n)
{
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += a[i];
    }
    return acc;
}

/* 下面几个是"可换成 CMSIS"的向量原语。默认走上面那种逐个累加（逐位一致）；
 * TM_CMSIS_DSP=1 时换成 CMSIS-DSP 的实现——M4F 上两路并行累加、循环展开，
 * 结果跟逐个累加差在末几位。 */
static float vmean(const float *a, int n)
{
#if TM_CMSIS_DSP
    float r;
    arm_mean_f32(a, (uint32_t)n, &r);
    return r;
#else
    return fsum(a, n) / (float)n;
#endif
}

/* Σ a[i]*a[i] */
static float vpower(const float *a, int n)
{
#if TM_CMSIS_DSP
    float r;
    arm_power_f32(a, (uint32_t)n, &r);
    return r;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i] * a[i];
    return acc;
#endif
}

/* Σ a[i]*b[i] */
static float vdot(const float *a, const float *b, int n)
{
#if TM_CMSIS_DSP
    float r;
    arm_dot_prod_f32(a, b, (uint32_t)n, &r);
    return r;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}

static void vminmax(const float *a, int n, float *mn, float *mx)
{
#if TM_CMSIS_DSP
    arm_min_no_idx_f32(a, (uint32_t)n, mn);
    arm_max_no_idx_f32(a, (uint32_t)n, mx);
#else
    float lo = a[0], hi = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] < lo) lo = a[i];
        if (a[i] > hi) hi = a[i];
    }
    *mn = lo;
    *mx = hi;
#endif
}

static void sort_asc(float *a, int n)
{
    /* 插入排序。n=32 的时候它比快排还快，而且没有递归、没有栈深度问题。 */
    for (int i = 1; i < n; i++) {
        float v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

static float percentile_linear(const float *s, int n, float q)
{
    /* np.percentile 默认是线性插值，不是取最近的那个样本点。
     * 取最近的话 IQR 在小窗口上会明显偏，而且不报错。 */
    float pos = q * (float)(n - 1);
    int lo = (int)floorf(pos);
    if (lo >= n - 1) {
        return s[n - 1];
    }
    float frac = pos - (float)lo;
    return s[lo] + frac * (s[lo + 1] - s[lo]);
}

static int count_peaks(const float *x, int n)
{
    /* scipy.signal.find_peaks(x) 无参数时的峰数：平台（连续相等的一段）算一个峰。
     * 每个点算一个的话，量化造成的平顶会让"动了几下"凭空翻倍。 */
    int cnt = 0, i = 1;
    while (i < n - 1) {
        if (x[i - 1] < x[i]) {
            int j = i;
            while (j < n - 1 && x[j + 1] == x[i]) {
                j++;
            }
            if (j < n - 1 && x[j + 1] < x[i]) {
                cnt++;
            }
            i = j + 1;
        } else {
            i++;
        }
    }
    return cnt;
}

static float sign_of(float v)
{
    /* np.sign：正好等于 0 给 0。样本正好落在均值上会算作两次穿越——
     * 这不是 bug，是要跟 imu_train 一致。 */
    return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
}

/* 时域 11 维。顺序：mean std min max range rms skew kurt mcr iqr peaks */
static void time_stats(const float *x, int n, float *out)
{
    static float d[TM_FEAT_MAX_T];
    static float tmp[TM_FEAT_MAX_T];

    float mean = vmean(x, n);
    for (int i = 0; i < n; i++) {
        d[i] = x[i] - mean;
    }

    float m2 = vpower(d, n) / (float)n;
    float std = sqrtf(m2);

    float xmin, xmax;
    vminmax(x, n, &xmin, &xmax);

    float rms = sqrtf(vpower(x, n) / (float)n);

    float skew = 0.0f, kurt = 0.0f;
    if (std > 1e-8f) {
        for (int i = 0; i < n; i++) tmp[i] = (d[i] * d[i]) * d[i];
        float m3 = fsum(tmp, n) / (float)n;
        for (int i = 0; i < n; i++) tmp[i] = ((d[i] * d[i]) * d[i]) * d[i];
        float m4 = fsum(tmp, n) / (float)n;
        /* 用 m2*std 而不是 powf(m2, 1.5f)：powf 在不同 libm 上末位可能不同，
         * sqrtf 则是 IEEE 精确定义的。 */
        skew = m3 / (m2 * std);
        kurt = m4 / (m2 * m2) - 3.0f;
    }

    int mcr = 0;
    float prev = sign_of(d[0]);
    for (int i = 1; i < n; i++) {
        float s = sign_of(d[i]);
        if (s != prev) mcr++;
        prev = s;
    }

    memcpy(tmp, x, (size_t)n * sizeof(float));
    sort_asc(tmp, n);
    float iqr = percentile_linear(tmp, n, 0.75f) - percentile_linear(tmp, n, 0.25f);

    out[0] = mean;
    out[1] = std;
    out[2] = xmin;
    out[3] = xmax;
    out[4] = xmax - xmin;
    out[5] = rms;
    out[6] = skew;
    out[7] = kurt;
    out[8] = (float)mcr;
    out[9] = iqr;
    out[10] = (float)count_peaks(x, n);
}

#if TM_CMSIS_DSP
/* CMSIS 的 FFT：复数交错存放（re0 im0 re1 im1 …），实例按段长初始化一次后缓存。
 * 旋转因子和位反序表由 CMSIS 自带（编译时用 ARM_TABLE_* 只带需要的长度），
 * cfg 里从 Python 导出的 cos/sin/bitrev 表在这条路上用不到。 */
static void fft_r2(const tm_feat_cfg_t *cfg, float *re, float *im)
{
    const int n = cfg->nperseg;
    static arm_cfft_instance_f32 inst;
    static int inst_len = 0;
    static float buf[2 * TM_FEAT_MAX_NPERSEG];
    if (inst_len != n) {
        /* 按长度调各自的 init，而不是 arm_cfft_init_f32：后者的 switch 把 16～4096 点的
         * 表全引用进来，链接器就没法丢掉没用的那些（光 4096 点的旋转因子就 32 KB）。
         * 再按 TM_FEAT_MAX_NPERSEG 把用不到的长度砍掉——编译时给 -DTM_FEAT_MAX_NPERSEG=16
         * 就只带 16 点那一张表（0.2 KB）。 */
        arm_status st;
        switch (n) {
        case 16:  st = arm_cfft_init_16_f32(&inst); break;
#if TM_FEAT_MAX_NPERSEG >= 32
        case 32:  st = arm_cfft_init_32_f32(&inst); break;
#endif
#if TM_FEAT_MAX_NPERSEG >= 64
        case 64:  st = arm_cfft_init_64_f32(&inst); break;
#endif
#if TM_FEAT_MAX_NPERSEG >= 128
        case 128: st = arm_cfft_init_128_f32(&inst); break;
#endif
#if TM_FEAT_MAX_NPERSEG >= 256
        case 256: st = arm_cfft_init_256_f32(&inst); break;
#endif
        default:  st = ARM_MATH_ARGUMENT_ERROR; break;
        }
        if (st != ARM_MATH_SUCCESS) {
            /* 这个段长没带表（third_party/cmsis 只带 16～256 点）：退回零输出，
             * 让 golden 自检当场失败，而不是板上悄悄算出垃圾 */
            for (int i = 0; i < n; i++) { re[i] = 0.0f; im[i] = 0.0f; }
            return;
        }
        inst_len = n;
    }
    for (int i = 0; i < n; i++) { buf[2 * i] = re[i]; buf[2 * i + 1] = im[i]; }
    arm_cfft_f32(&inst, buf, 0, 1);
    for (int i = 0; i < n; i++) { re[i] = buf[2 * i]; im[i] = buf[2 * i + 1]; }
}
#else
/* 就地基-2 DIT FFT。循环结构跟 Python 的 fft_r2 完全一致。 */
static void fft_r2(const tm_feat_cfg_t *cfg, float *re, float *im)
{
    const int n = cfg->nperseg;
    static float br_re[TM_FEAT_MAX_NPERSEG];
    static float br_im[TM_FEAT_MAX_NPERSEG];
    for (int i = 0; i < n; i++) {
        br_re[i] = re[cfg->bitrev[i]];
        br_im[i] = im[cfg->bitrev[i]];
    }
    memcpy(re, br_re, (size_t)n * sizeof(float));
    memcpy(im, br_im, (size_t)n * sizeof(float));

    for (int size = 2; size <= n; size <<= 1) {
        const int half = size / 2;
        const int step = n / size;
        for (int i = 0; i < n; i += size) {
            for (int j = 0; j < half; j++) {
                const float c = cfg->cos_t[j * step];
                const float s = cfg->sin_t[j * step];
                const int a = i + j, b = i + j + half;
                const float tr = c * re[b] - s * im[b];
                const float ti = c * im[b] + s * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] = re[a] + tr;
                im[a] = im[a] + ti;
            }
        }
    }
}
#endif /* TM_CMSIS_DSP */

/* scipy.signal.welch 的默认参数：density / onesided / mean / detrend='constant'。 */
static void welch_psd(const tm_feat_cfg_t *cfg, const float *x, int n, float *psd)
{
    const int nps = cfg->nperseg;
    const int noverlap = nps / 2;
    const int step = nps - noverlap;
    const int n_out = nps / 2 + 1;

    const float scale = 1.0f / (cfg->fs * vpower(cfg->win, nps));

    for (int k = 0; k < n_out; k++) psd[k] = 0.0f;

    int n_seg = 0;
    static float re[TM_FEAT_MAX_NPERSEG], im[TM_FEAT_MAX_NPERSEG];
    for (int s0 = 0; s0 + nps <= n; s0 += step) {
        float m = vmean(x + s0, nps);
        for (int i = 0; i < nps; i++) {
            re[i] = (x[s0 + i] - m) * cfg->win[i];
            im[i] = 0.0f;
        }
        fft_r2(cfg, re, im);
        for (int k = 0; k < n_out; k++) {
            float p = (re[k] * re[k] + im[k] * im[k]) * scale;
            /* 单边谱：除直流和 Nyquist 外乘 2，把负频率那一半折过来 */
            if (k > 0 && k < n_out - 1) p *= 2.0f;
            psd[k] += p;
        }
        n_seg++;
    }
    if (n_seg == 0) n_seg = 1;  /* n < nperseg 时不该发生，但除零要挡住 */
    for (int k = 0; k < n_out; k++) psd[k] /= (float)n_seg;
}

static const float BAND_LO[4] = {0.0f, 0.125f, 0.375f, 0.75f};
static const float BAND_HI[4] = {0.125f, 0.375f, 0.75f, 1.0f};

/* 频域 8 维：spec_mean spec_std peak_freq entropy + 4 个分频段能量占比 */
static void freq_stats(const tm_feat_cfg_t *cfg, const float *x, int n, float *out)
{
    const int n_out = cfg->nperseg / 2 + 1;
    static float psd[TM_FEAT_MAX_NPERSEG / 2 + 1];
    static float freqs[TM_FEAT_MAX_NPERSEG / 2 + 1];
    static float pn[TM_FEAT_MAX_NPERSEG / 2 + 1];
    static float tmp[TM_FEAT_MAX_NPERSEG / 2 + 1];

    welch_psd(cfg, x, n, psd);
    for (int k = 0; k < n_out; k++) {
        freqs[k] = (float)k * cfg->fs / (float)cfg->nperseg;
    }

    const float total = fsum(psd, n_out) + 1e-8f;
    for (int k = 0; k < n_out; k++) pn[k] = psd[k] / total;

    const float spec_mean = vdot(freqs, pn, n_out);

    for (int k = 0; k < n_out; k++) {
        const float dv = freqs[k] - spec_mean;
        tmp[k] = (dv * dv) * pn[k];
    }
    const float spec_std = sqrtf(fsum(tmp, n_out));

    int kmax = 0;
    for (int k = 1; k < n_out; k++) {
        if (psd[k] > psd[kmax]) kmax = k;  /* 并列取下标小的，跟 np.argmax 一致 */
    }

    /* 这一项是**唯一**可能跟 Python 差最后一位的：logf 属于 libm，各实现不保证
     * 正确舍入。测试里对它单独放一个 ULP 的容差，其余各项要求逐位相同。 */
    for (int k = 0; k < n_out; k++) tmp[k] = pn[k] * logf(pn[k] + 1e-8f);
    const float ent = -fsum(tmp, n_out);

    out[0] = spec_mean;
    out[1] = spec_std;
    out[2] = freqs[kmax];
    out[3] = ent;

    const float nyq = cfg->fs / 2.0f;
    for (int b = 0; b < 4; b++) {
        const float lo = BAND_LO[b] * nyq, hi = BAND_HI[b] * nyq;
        int cnt = 0;
        for (int k = 0; k < n_out; k++) {
            if (freqs[k] >= lo && freqs[k] < hi) tmp[cnt++] = pn[k];
        }
        out[4 + b] = fsum(tmp, cnt);
    }
}

static void magnitude(const float *x, int n_t, int c0, float *out, int n)
{
    const float *a = x + (size_t)(c0 + 0) * n_t;
    const float *b = x + (size_t)(c0 + 1) * n_t;
    const float *c = x + (size_t)(c0 + 2) * n_t;
    for (int i = 0; i < n; i++) {
        out[i] = sqrtf(a[i] * a[i] + b[i] * b[i] + c[i] * c[i]);
    }
}

static float corr(const float *xi, const float *xj, int n)
{
    static float di[TM_FEAT_MAX_T], dj[TM_FEAT_MAX_T];
    const float mi = vmean(xi, n);
    const float mj = vmean(xj, n);
    for (int i = 0; i < n; i++) { di[i] = xi[i] - mi; dj[i] = xj[i] - mj; }
    const float si = sqrtf(vpower(di, n) / (float)n);
    const float sj = sqrtf(vpower(dj, n) / (float)n);
    if (si <= 1e-8f || sj <= 1e-8f) {
        return 0.0f;  /* 常数通道（传感器卡死）。不挡的话是 0/0 = nan */
    }
    return (vdot(di, dj, n) / (float)n) / (si * sj);
}

#ifdef TM_BENCH
/* 只在做性能剖析时导出内部函数（tools/bench_features.c 用）。
 * 平时不导出：它们是实现细节，暴露出去就会有人绕过 tm_features 直接调，
 * 那样特征的拼接顺序就没有唯一的出处了。 */
void tm_bench_time_stats(const tm_feat_cfg_t *c, const float *x, int n, float *out)
{
    (void)c;
    time_stats(x, n, out);
}

void tm_bench_freq_stats(const tm_feat_cfg_t *c, const float *x, int n, float *out)
{
    freq_stats(c, x, n, out);
}
#endif

/* 传感器通道数：按三个一组（acc3 / acc3+gyro3），余 2 就是后面追加了
 * pitch/roll。5 → 3（3 轴），8 → 6，6 → 6。频域特征只对传感器通道算——
 * 姿态角是慢变量，跟 python/tinyml/features.py 的 n_sensor 一致。 */
static int n_sensor_ch(int n_ch)
{
    return (n_ch % 3 == 2) ? n_ch - 2 : n_ch;
}

int tm_feat_dim(const tm_feat_cfg_t *cfg)
{
    int n = 11 * cfg->n_ch + 8 * n_sensor_ch(cfg->n_ch);
    if (cfg->n_ch >= 6) n += 8 + 2 * (11 + 8) + 11;
    return n;
}

int tm_features(const tm_feat_cfg_t *cfg, const float *x, float *out)
{
    const int n_t = cfg->n_t;
    if (n_t > TM_FEAT_MAX_T || cfg->nperseg > TM_FEAT_MAX_NPERSEG || cfg->nperseg > n_t) {
        return -1;
    }
    int p = 0;

    for (int c = 0; c < cfg->n_ch; c++) {
        time_stats(x + (size_t)c * n_t, n_t, out + p);
        p += 11;
    }
    const int n_freq_ch = n_sensor_ch(cfg->n_ch);
    for (int c = 0; c < n_freq_ch; c++) {
        freq_stats(cfg, x + (size_t)c * n_t, n_t, out + p);
        p += 8;
    }

    if (cfg->n_ch >= 6) {
        static float tmp[TM_FEAT_MAX_T];
        static float mag[TM_FEAT_MAX_T];

        for (int base = 0; base <= 3; base += 3) {
            const float *a = x + (size_t)(base + 0) * n_t;
            const float *b = x + (size_t)(base + 1) * n_t;
            const float *c = x + (size_t)(base + 2) * n_t;
            for (int i = 0; i < n_t; i++) {
                tmp[i] = fabsf(a[i]) + fabsf(b[i]) + fabsf(c[i]);
            }
            out[p++] = fsum(tmp, n_t) / (float)n_t;
        }
        for (int base = 0; base <= 3; base += 3) {
            static const int PAIRS[3][2] = {{0, 1}, {1, 2}, {0, 2}};
            for (int k = 0; k < 3; k++) {
                out[p++] = corr(x + (size_t)(base + PAIRS[k][0]) * n_t,
                                x + (size_t)(base + PAIRS[k][1]) * n_t, n_t);
            }
        }
        for (int base = 0; base <= 3; base += 3) {
            magnitude(x, n_t, base, mag, n_t);
            time_stats(mag, n_t, out + p);
            p += 11;
            freq_stats(cfg, mag, n_t, out + p);
            p += 8;
        }
        /* jerk = Δacc * hz，长度比窗口少一个点 */
        static float jerk[3][TM_FEAT_MAX_T];
        static float jmag[TM_FEAT_MAX_T];
        for (int c = 0; c < 3; c++) {
            const float *a = x + (size_t)c * n_t;
            for (int i = 0; i < n_t - 1; i++) {
                jerk[c][i] = (a[i + 1] - a[i]) * cfg->fs;
            }
        }
        for (int i = 0; i < n_t - 1; i++) {
            jmag[i] = sqrtf(jerk[0][i] * jerk[0][i] + jerk[1][i] * jerk[1][i]
                            + jerk[2][i] * jerk[2][i]);
        }
        time_stats(jmag, n_t - 1, out + p);
        p += 11;
    }
    return 0;
}
