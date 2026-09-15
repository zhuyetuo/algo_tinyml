"""193 维手工特征的端侧参考实现，对应 imu_train 的 `src/ml/features.py`。

**这份是给板子用的参考实现，不是 imu_train 那份的替代品。** 两者的关系要说清楚，
否则后面会有人拿错的东西去对：

    imu_train/src/ml/features.py   scipy + float64   ← 训练和平台在用的那份，是基准
    tinyml/features.py             纯 numpy + float32 ← 这份，逐行对着上面写的
    firmware/tinyml/tm_features.c  C + float          ← 板上跑的那份

**这份 ↔ C 是逐位一致的**（tests/test_features_c.py 现场编译对答案）。
**这份 ↔ scipy 不是**，而且做不到：

  - scipy 全程 float64，板上只能 float32（M4F 的 FPU 是单精度，double 要软件模拟）；
  - FFT 的算法不同（pocketfft vs 这里的基-2），浮点加法不满足结合律，加法顺序
    不同结果就不同。

所以"板上跟平台一致"这件事**不能靠逐位**，只能靠测量。`verify_against_scipy.py`
在有 scipy 的机器上跑：拿真实窗口，两边各算一遍特征、各跑一遍森林，报告
**最终判别不一致的比例**。那个数才是决策依据——特征差第几位小数不重要，
判别翻没翻才重要。

几处照抄 scipy 语义、写错了不会报错的地方（都写成了测试）：
  - `np.percentile` 默认是**线性插值**，不是取最近的样本点；
  - `scipy.stats.skew/kurtosis` 是**有偏**估计（分母用 n，不是 n-1），峰度是 Fisher 的（减 3）；
  - `np.sign` 对**正好等于 0** 给 0，所以样本正好落在均值上会算作两次穿越；
  - `find_peaks` 无参数时，平台（连续相等的一段）算**一个**峰，不是每个点一个；
  - `welch` 的窗是**周期** Hann（sym=False），去趋势是逐段减均值，单边谱除首尾外乘 2。
"""

import numpy as np

F32 = np.float32

# 频段边界，按 Nyquist 的比例给。跟 imu_train 的 FREQ_BANDS 一致
FREQ_BANDS = ((0.0, 0.125), (0.125, 0.375), (0.375, 0.75), (0.75, 1.0))

N_TIME_FEATS = 11
N_FREQ_FEATS = 8


def fsum(a):
    """从左到右逐个累加的 float32 求和。

    **不能用 np.sum**：numpy 对 float32 数组用的是成对求和（pairwise），
    32 个点会走 8 路部分和再合并——加法顺序跟 C 里一个朴素 for 循环不一样，
    浮点加法不满足结合律，结果末位就会差。这种差别小到看不出来，却足以让
    "板上跟 PC 逐位一致"这个目标直接失守，而且查起来毫无线索。
    """
    acc = F32(0.0)
    for v in np.asarray(a, F32):
        acc = F32(acc + v)
    return acc


# ── 基础件 ────────────────────────────────────────────────────────────────


def hann_periodic(n):
    """周期 Hann（scipy 的 get_window('hann', n)，sym=False）。

    对称版（sym=True）分母是 n-1，两者差一个点——用错的话谱会整体偏一点点，
    不报错，只是特征跟训练时对不上。
    """
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(F32)


def fft_twiddles(n):
    """基-2 FFT 用的旋转因子。**导出到 C 时是当常量表写进去的**，不让 C 自己算 ——
    `cosf` 在不同 libm 上末位可能不同，而那就是两边对不上的一个来源。"""
    k = np.arange(n // 2, dtype=np.float64)
    ang = -2.0 * np.pi * k / n
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32)


def _bit_reverse(n):
    bits = int(np.log2(n))
    idx = np.zeros(n, dtype=np.int32)
    for i in range(n):
        r, v = 0, i
        for _ in range(bits):
            r = (r << 1) | (v & 1)
            v >>= 1
        idx[i] = r
    return idx


def fft_r2(re, im, cos_t, sin_t):
    """就地基-2 DIT FFT，float32。C 那边是同一套循环、同一个顺序。

    不用 np.fft：pocketfft 的算法和加法顺序都不一样，结果在末位上跟 C 对不上。
    这里要的是"跟 C 一样"，不是"最准"。
    """
    n = len(re)
    idx = _bit_reverse(n)
    re = re[idx].astype(F32).copy()
    im = im[idx].astype(F32).copy()
    size = 2
    while size <= n:
        half = size // 2
        step = n // size
        for i in range(0, n, size):
            for j in range(half):
                c = cos_t[j * step]
                s = sin_t[j * step]
                a, b = i + j, i + j + half
                tr = F32(c * re[b] - s * im[b])
                ti = F32(c * im[b] + s * re[b])
                re[b] = F32(re[a] - tr)
                im[b] = F32(im[a] - ti)
                re[a] = F32(re[a] + tr)
                im[a] = F32(im[a] + ti)
        size *= 2
    return re, im


# ── 时域 11 维 ────────────────────────────────────────────────────────────


def percentile_linear(x_sorted, q):
    """np.percentile 的默认插值（linear）。x_sorted 必须已经升序。"""
    n = len(x_sorted)
    pos = F32(q * (n - 1))
    lo = int(np.floor(float(pos)))
    if lo >= n - 1:
        return F32(x_sorted[n - 1])
    frac = F32(pos - F32(lo))
    return F32(x_sorted[lo] + frac * F32(x_sorted[lo + 1] - x_sorted[lo]))


def count_peaks(x):
    """scipy.signal.find_peaks(x) 无参数时的峰数。

    平台（连续相等的一段）只算**一个**峰——每个点算一个的话，采样量化造成的
    平顶会让"动了几下"凭空翻倍，而这个特征在 EDA 里判别力很强。
    """
    n = len(x)
    cnt = 0
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i]:
            j = i
            while j < n - 1 and x[j + 1] == x[i]:
                j += 1
            if j < n - 1 and x[j + 1] < x[i]:
                cnt += 1
            i = j + 1
        else:
            i += 1
    return cnt


def time_stats(x):
    """11 个时域统计量，顺序跟 imu_train 的 TIME_FEAT_NAMES 完全一致。"""
    x = np.asarray(x, F32)
    n = len(x)
    mean = F32(fsum(x) / F32(n))
    d = (x - mean).astype(F32)
    m2 = F32(fsum(d * d) / F32(n))
    std = F32(np.sqrt(float(m2)))
    xmin, xmax = F32(x.min()), F32(x.max())
    rms = F32(np.sqrt(float(F32(fsum(x * x) / F32(n)))))

    if std > F32(1e-8):
        m3 = F32(fsum((d * d) * d) / F32(n))
        m4 = F32(fsum(((d * d) * d) * d) / F32(n))
        # 用 m2*std 而不是 pow(m2,1.5)：powf 在不同 libm 上末位可能不同，
        # 而 sqrtf 是 IEEE 精确定义的
        skew = F32(m3 / F32(m2 * std))
        kurt = F32(F32(m4 / F32(m2 * m2)) - F32(3.0))
    else:
        skew = kurt = F32(0.0)

    # 均值穿越率。np.sign 对正好等于 0 给 0 —— 样本正好落在均值上会产生
    # 两次"变化"。这不是 bug，是要跟 imu_train 一致
    sgn = np.sign(d).astype(F32)
    mcr = F32(int(np.count_nonzero(np.diff(sgn) != 0)))

    s = np.sort(x)
    iqr = F32(percentile_linear(s, 0.75) - percentile_linear(s, 0.25))
    peaks = F32(count_peaks(x))

    return [mean, std, xmin, xmax, F32(xmax - xmin), rms, skew, kurt, mcr, iqr, peaks]


# ── 频域 8 维 ─────────────────────────────────────────────────────────────


def welch_psd(x, fs, nperseg, win, cos_t, sin_t):
    """scipy.signal.welch(x, fs, nperseg=nperseg) 的 float32 复刻。

    scaling='density'、return_onesided=True、average='mean'、detrend='constant'，
    都是 scipy 的默认值。nperseg 必须是 2 的幂（基-2 FFT 的限制），
    在实际配置下是 32（16Hz × 2 秒）。
    """
    x = np.asarray(x, F32)
    n = len(x)
    assert nperseg & (nperseg - 1) == 0, f"nperseg={nperseg} 不是 2 的幂"
    noverlap = nperseg // 2
    step = nperseg - noverlap
    starts = list(range(0, n - nperseg + 1, step)) or [0]

    win_pow = F32(fsum(win * win))
    scale = F32(F32(1.0) / F32(F32(fs) * win_pow))
    n_out = nperseg // 2 + 1
    acc = np.zeros(n_out, dtype=F32)

    for s0 in starts:
        seg = x[s0:s0 + nperseg].astype(F32)
        m = F32(fsum(seg) / F32(nperseg))
        seg = ((seg - m) * win).astype(F32)          # 去趋势 + 加窗
        re, im = fft_r2(seg, np.zeros(nperseg, F32), cos_t, sin_t)
        for k in range(n_out):
            p = F32(F32(re[k] * re[k] + im[k] * im[k]) * scale)
            # 单边谱：除了直流和 Nyquist，其它都要乘 2（把负频率那一半折过来）
            if 0 < k < n_out - 1:
                p = F32(p * F32(2.0))
            acc[k] = F32(acc[k] + p)

    return (acc / F32(len(starts))).astype(F32)


def freq_stats(x, fs, nperseg, win, cos_t, sin_t):
    psd = welch_psd(x, fs, nperseg, win, cos_t, sin_t)
    n_out = len(psd)
    freqs = np.array([F32(F32(k) * F32(fs) / F32(nperseg)) for k in range(n_out)], F32)

    total = F32(fsum(psd) + F32(1e-8))
    pn = (psd / total).astype(F32)

    spec_mean = F32(fsum(freqs * pn))
    dev = (freqs - spec_mean).astype(F32)
    spec_std = F32(np.sqrt(float(F32(fsum((dev * dev) * pn)))))
    peak_freq = F32(freqs[int(np.argmax(psd))])
    ent = F32(-fsum(pn * np.log(pn + F32(1e-8), dtype=F32)))

    out = [spec_mean, spec_std, peak_freq, ent]
    nyq = F32(F32(fs) / F32(2.0))
    for lo, hi in FREQ_BANDS:
        m = (freqs >= F32(F32(lo) * nyq)) & (freqs < F32(F32(hi) * nyq))
        out.append(F32(fsum(pn[m])))
    return out


# ── 拼装 ──────────────────────────────────────────────────────────────────


def magnitude(triplet):
    """三轴合成模长。逐点 sqrt(x²+y²+z²)，sqrtf 是 IEEE 精确的，两边不会差。"""
    t = np.asarray(triplet, F32)
    s = (t[:, 0] * t[:, 0] + t[:, 1] * t[:, 1] + t[:, 2] * t[:, 2]).astype(F32)
    return np.sqrt(s).astype(F32)


def _corr(xi, xj):
    xi = np.asarray(xi, F32)
    xj = np.asarray(xj, F32)
    n = len(xi)
    mi = F32(fsum(xi) / F32(n))
    mj = F32(fsum(xj) / F32(n))
    di = (xi - mi).astype(F32)
    dj = (xj - mj).astype(F32)
    vi = F32(fsum(di * di) / F32(n))
    vj = F32(fsum(dj * dj) / F32(n))
    si = F32(np.sqrt(float(vi)))
    sj = F32(np.sqrt(float(vj)))
    # imu_train 那边是 std > 1e-8 才算，否则给 0。常数通道（传感器卡死）会走到这里
    if si <= F32(1e-8) or sj <= F32(1e-8):
        return F32(0.0)
    cov = F32(fsum(di * dj) / F32(n))
    return F32(cov / F32(si * sj))


def extract_one(window, hz, nperseg=32):
    """window: [T, C] float32 → 特征向量 float32 [n_features]。

    拼接顺序必须跟 imu_train 的 `_extract_one` 一模一样：
    时域(全部通道) → 频域(前 6 通道) → 全局 → acc/gyro 模长(时域+频域) → jerk 模长(时域)。
    顺序错了不会报错，只会让每一维都对到别的特征上，而模型照样给得出结果。
    """
    w = np.asarray(window, F32)
    t_len, n_ch = w.shape
    nps = min(nperseg, t_len)
    win = hann_periodic(nps)
    cos_t, sin_t = fft_twiddles(nps)

    feats = []
    for c in range(n_ch):
        feats.extend(time_stats(w[:, c]))
    for c in range(min(6, n_ch)):
        feats.extend(freq_stats(w[:, c], hz, nps, win, cos_t, sin_t))

    if n_ch >= 6:
        acc, gyr = w[:, 0:3], w[:, 3:6]
        for trip in (acc, gyr):
            a = np.abs(trip).astype(F32)
            sma = F32(fsum((a[:, 0] + a[:, 1] + a[:, 2]).astype(F32)) / F32(t_len))
            feats.append(sma)
        for trip in (acc, gyr):
            for i, j in ((0, 1), (1, 2), (0, 2)):
                feats.append(_corr(trip[:, i], trip[:, j]))
        for trip in (acc, gyr):
            mag = magnitude(trip)
            feats.extend(time_stats(mag))
            feats.extend(freq_stats(mag, hz, nps, win, cos_t, sin_t))
        jerk = (np.diff(acc, axis=0) * F32(hz)).astype(F32)
        feats.extend(time_stats(magnitude(jerk)))

    return np.asarray(feats, F32)


def n_features(n_ch):
    """维度。6 通道 171 维，8 通道 193 维。"""
    n = N_TIME_FEATS * n_ch + N_FREQ_FEATS * min(6, n_ch)
    if n_ch >= 6:
        n += 8 + 2 * (N_TIME_FEATS + N_FREQ_FEATS) + N_TIME_FEATS
    return n


# ── 特征分组：砍特征只能按「计算组」砍，不能按单个特征砍 ────────────────────
#
# 这一点是砍特征时最容易搞错的地方：同一个通道的 11 个时域统计量**共享同一趟
# 循环和同一次排序**，砍掉其中几个几乎不省时间；要省就得整组砍掉（那个通道
# 的时域特征全不要），才能跳过整趟计算。频域 8 维共享一次 FFT，同理。
#
# 所以下面按「一次计算产出哪些维度」来分组，而不是按语义分。

def feature_groups(n_ch=8):
    """→ [(组名, start, stop, 计算类型)]，start/stop 是特征向量里的下标区间。

    计算类型：time = 一趟时域统计（含排序）；freq = 一次 FFT + Welch；
    cheap = 只是几次加减乘（SMA、相关系数）；derive = 先要派生出一路新信号。

    顺序必须跟 extract_one 的拼接顺序一致。这里用循环生成而不是写死一张表，
    是为了改通道数时它自己跟着变——写死的表迟早跟代码分家，而分家之后
    "砍掉第 100 维"会砍到完全不相干的东西上。
    """
    names = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "pitch", "roll"]
    g, p = [], 0
    for c in range(n_ch):
        nm = names[c] if c < len(names) else f"ch{c}"
        g.append((f"{nm} 时域", p, p + N_TIME_FEATS, "time"))
        p += N_TIME_FEATS
    for c in range(min(6, n_ch)):
        nm = names[c] if c < len(names) else f"ch{c}"
        g.append((f"{nm} 频域", p, p + N_FREQ_FEATS, "freq"))
        p += N_FREQ_FEATS
    if n_ch >= 6:
        g.append(("全局 SMA+相关系数", p, p + 8, "cheap"))
        p += 8
        for nm in ("acc 模长", "gyro 模长"):
            g.append((f"{nm} 时域", p, p + N_TIME_FEATS, "derive+time"))
            p += N_TIME_FEATS
            g.append((f"{nm} 频域", p, p + N_FREQ_FEATS, "freq"))
            p += N_FREQ_FEATS
        g.append(("jerk 模长 时域", p, p + N_TIME_FEATS, "derive+time"))
        p += N_TIME_FEATS
    assert p == n_features(n_ch), f"分组覆盖 {p} 维，但总共 {n_features(n_ch)} 维"
    return g
