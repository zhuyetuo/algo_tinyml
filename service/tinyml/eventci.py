"""事件级指标的不确定区间。

为什么必须有：留出集里「抓挠」只聚合出 **17 次事件**。两个模型报出 0.788 和
0.848，看着差 6 个点，实际差的是**一个事件**（对 13 次 vs 对 14 次）。
17 个样本下，这个差异完全在噪声里——两边的 95% 区间是 [0.63, 0.91] 和
[0.70, 0.97]，重叠得几乎完全。

不把区间打出来的话，人（包括写这段代码的我）会拿 6 个点的差去做选型决定，
而那个差经不起再采一批数据。窗口级有几百上千个样本，事件级只有十几个——
这两张表的可信度差一个数量级，但它们打印出来长得一模一样。

区间怎么算：召回是「17 次真事件里报中了几次」，精确率是「16 次报出里对了几次」，
两个都是二项的。这里对两者各自做参数自助（binomial bootstrap）再合成 F1。
比正态近似好：F1 是两个比例的调和平均，在样本少、比例接近 1 时分布明显偏斜。
"""

import numpy as np


def f1_from(p, r):
    """精确率、召回率 → F1。两个都是 0 时给 0，不是 nan。"""
    p = np.asarray(p, np.float64)
    r = np.asarray(r, np.float64)
    d = p + r
    return np.where(d > 0, 2 * p * r / np.where(d > 0, d, 1.0), 0.0)


def event_ci(n_true, n_pred, tp, level=0.95, n_boot=20000, seed=0):
    """事件级 P / R / F1 的自助区间。

    返回 {"p": (lo, hi), "r": (lo, hi), "f1": (lo, hi), "n_true": ...}。

    n_true=0 或 n_pred=0 时区间没有意义（分母是 0），返回 None 而不是编一个数——
    "没有数据"和"数据说是 0"必须分得开。
    """
    if n_true <= 0 or n_pred <= 0:
        return None
    if not (0 <= tp <= min(n_true, n_pred)):
        raise ValueError(f"tp={tp} 超出范围（真值 {n_true}、报出 {n_pred}）")
    rng = np.random.default_rng(seed)
    r = rng.binomial(n_true, tp / n_true, n_boot) / n_true
    p = rng.binomial(n_pred, tp / n_pred, n_boot) / n_pred
    lo_q = 100 * (1 - level) / 2
    hi_q = 100 - lo_q
    return {
        "p": tuple(np.percentile(p, [lo_q, hi_q])),
        "r": tuple(np.percentile(r, [lo_q, hi_q])),
        "f1": tuple(np.percentile(f1_from(p, r), [lo_q, hi_q])),
        "n_true": int(n_true),
        "level": level,
    }


def describe(ci, n_true):
    """一行话，说清这张表能不能用来做决定。

    阈值不是拍的：事件级 F1 的区间宽度大约按 1/sqrt(n) 收敛，30 次事件时
    半宽仍有 ~0.1。所以 30 以下一律明说"分辨不了小差距"。
    """
    if ci is None:
        return "  ⚠ 没有事件，算不出区间。"
    lo, hi = ci["f1"]
    half = (hi - lo) / 2
    s = (f"  事件 F1 的 {int(ci['level'] * 100)}% 区间 [{lo:.3f}, {hi:.3f}]"
         f"（半宽 ±{half:.3f}，基于 {n_true} 次真值事件）")
    if n_true < 30:
        s += (f"\n  ⚠ **只有 {n_true} 次事件，这张表分辨不了小于 ±{half:.2f} 的差距。**"
              "\n    两个模型差几个点的话，那多半就是差一两个事件，换一批数据就会翻过来。"
              "\n    要在事件级上做选型，先把留出集里的目标事件攒到几十次以上；"
              "\n    在那之前，**窗口级的数更可信**（样本数多一两个数量级）。")
    return s


def gap_is_meaningful(n_true, n_pred, tp_a, tp_b, level=0.95, n_boot=20000, seed=0):
    """两个模型的事件 F1 差异是否超出噪声。

    做法是配对自助：同一批重采样下同时算两个模型的 F1，看差值的区间是否跨 0。
    分别算两个区间再看"有没有重叠"是**错的**——那个判据过于保守，
    而且忽略了两个模型是在同一批事件上评的。
    """
    rng = np.random.default_rng(seed)
    out = []
    for tp in (tp_a, tp_b):
        r = rng.binomial(n_true, tp / n_true, n_boot) / n_true
        p = rng.binomial(n_pred, tp / n_pred, n_boot) / n_pred
        out.append(f1_from(p, r))
    d = out[1] - out[0]
    lo_q = 100 * (1 - level) / 2
    lo, hi = np.percentile(d, [lo_q, 100 - lo_q])
    return {"diff": float(np.mean(d)), "ci": (float(lo), float(hi)),
            "meaningful": bool(lo > 0 or hi < 0)}
