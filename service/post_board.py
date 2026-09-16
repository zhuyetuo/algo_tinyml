"""用**板子上那份后处理**（core/tm_post.c）跑片段，而不是服务端的 Python。

为什么要有这个：平台上选 `edge:<标签>` 时，模型和推理确实已经是板上那份 C
了，但**后处理还是服务端的 Python**（label_service/postprocess.py，离线
viterbi + 整条序列回溯）。板子上跑的是 tm_post.c——流式、有界回溯，
结构上就不一样。

我量过这两份：18 万个窗口上窗口级和片段级都是 100% 一致。但那是在**我造的
数据**上量的。手里没有板子的时候，唯一能拿真实数据回答"板子会报什么"的办法，
就是让服务把这最后一段也换成板上那份 C。

于是平台上多一个版本串：`edge:<标签>@board`
  · edge:<标签>        模型=板上C  推理=板上C  后处理=服务端 Python
  · edge:<标签>@board  三段全是板上那份 C

两列一起跑，差多少是**你的数据**说了算，不是我的测试说了算。
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(ROOT, "core")

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
_lock = threading.Lock()
_lib = None


def _build():
    """把 core/tm_post.c 编成 .so。

    编译开关跟板子那边、跟别的对照测试是**同一套**：
    -ffp-contract=off 不是可选项——允许 FMA 合并的话中间结果少一次舍入，
    跟板上算出来的末位就不同，而那足以在阈值附近把判决翻过去。
    那样这里跑出来的就不是"板子会报什么"了。
    """
    so = os.path.join(tempfile.mkdtemp(prefix="tm_post_"), "tm_post.so")
    cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", "-fno-math-errno", "-fPIC", "-shared",
           f"-I{CORE}", os.path.join(CORE, "tm_post.c"),
           os.path.join(HERE, "host_post.c"), "-lm", "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"编不出板上后处理：\n{r.stderr}")
    d = ctypes.CDLL(so)
    f32 = ctypes.POINTER(ctypes.c_float)
    u32 = ctypes.POINTER(ctypes.c_uint)
    i32 = ctypes.POINTER(ctypes.c_int)
    d.tm_post_run.restype = ctypes.c_int
    d.tm_post_run.argtypes = [
        f32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        i32, u32, u32, i32, f32, f32, ctypes.c_int, u32]
    return d


def lib():
    global _lib
    with _lock:
        if _lib is None:
            _lib = _build()
        return _lib


def _parse_ts(s: str) -> _dt.datetime:
    return _dt.datetime.strptime(s, _TS_FMT)


def _fmt_ts(t: _dt.datetime) -> str:
    return t.strftime(_TS_FMT)[:-3]


def stabilize(windows, classes, target_labels, window_s, stride_s,
              params, scratch="抓挠", shake="甩身体"):
    """跟 postprocess.stabilize 同样的入参和返回结构，但跑的是 tm_post.c。

    返回 {label: [{start_ts, end_ts, conf_max, conf_mean, n_windows, spec}]}。
    """
    out_empty = {lab: [] for lab in target_labels}
    if not windows:
        return out_empty

    n, m = len(windows), len(classes)
    probs = np.zeros((n, m), np.float32)
    for i, w in enumerate(windows):
        d = w.get("probs") or {}
        for c, name in enumerate(classes):
            probs[i, c] = float(d.get(name, 0.0))

    # 时间戳转成**相对第一个窗口的毫秒**：tm_post 用的是 uint32 的板上 tick，
    # 直接塞 Unix 毫秒会溢出（1.7e12 > 2^32）。溢出之后所有间隔判断都是错的，
    # 而且不报错
    t0 = _parse_ts(windows[0]["ts"])
    ts = np.array([int((_parse_ts(w["ts"]) - t0).total_seconds() * 1000)
                   for w in windows], np.uint32)

    try:
        sl = classes.index(scratch)
    except ValueError:
        sl = -1
    try:
        sh = classes.index(shake)
    except ValueError:
        sh = -1

    d = lib()
    mx = n + 16
    cls = np.zeros(mx, np.int32)
    s_ms = np.zeros(mx, np.uint32)
    e_ms = np.zeros(mx, np.uint32)
    nw = np.zeros(mx, np.int32)
    cmax = np.zeros(mx, np.float32)
    cmean = np.zeros(mx, np.float32)
    forced = np.zeros(2, np.uint32)

    def p(a, t):
        return a.ctypes.data_as(ctypes.POINTER(t))

    # tm_post 里有静态缓冲吗？没有——状态全在调用方给的 tm_post_t 里。
    # 但 host_post.c 每次调用自己建一个，所以这里不用加锁
    k = d.tm_post_run(
        p(probs, ctypes.c_float), p(ts, ctypes.c_uint), n, m, sl, sh,
        float(window_s), float(stride_s),
        float(params.viterbi_switch), float(params.event_gap_s),
        float(params.shake_absorb_s), int(params.event_min_windows),
        float(params.event_min_mean), float(params.event_single_conf),
        p(cls, ctypes.c_int), p(s_ms, ctypes.c_uint), p(e_ms, ctypes.c_uint),
        p(nw, ctypes.c_int), p(cmax, ctypes.c_float), p(cmean, ctypes.c_float),
        mx, p(forced, ctypes.c_uint))
    _last_forced[0], _last_forced[1] = int(forced[0]), int(forced[1])
    if k < 0:
        raise RuntimeError(
            "板上后处理拒绝了这套参数（延迟线装不下 event_gap_s + "
            "shake_absorb_s）。板子上也会拒——这不是服务这边的问题。")

    out = {lab: [] for lab in target_labels}
    for i in range(k):
        lab = classes[int(cls[i])]
        if lab not in out:
            continue
        out[lab].append({
            "start_ts": _fmt_ts(t0 + _dt.timedelta(milliseconds=int(s_ms[i]))),
            "end_ts": _fmt_ts(t0 + _dt.timedelta(milliseconds=int(e_ms[i]))),
            "conf_max": float(cmax[i]),
            "conf_mean": float(cmean[i]),
            "n_windows": int(nw[i]),
            # spec 是服务端的频谱占比，板上没有这个输入。
            # **给 None 而不是 0**——0 会被下游当成"算过了，是 0"
            "spec": None,
        })
    return out


# 上一次调用的两个"强制"计数。模块级变量看着糙，但服务这边每个模型的推理
# 本来就被 EdgeRunner.lock 串起来了，读的时候不会串台。
_last_forced = [0, 0]


def forced_counts():
    """上一次跑里"可能跟离线算法不同"的两个计数。

      forced_settle  viterbi 回溯缓冲满了还没汇合，按当前最优强制定稿
      forced_split   事件密集到延迟线装不下，块被从中间切开

    这两处是有界回溯**唯一**可能跟离线结果分家的地方。读得到才谈得上
    "两边一致"——读不到的话那句话没有任何可核对的依据。
    正常数据上实测都是 0。
    """
    return {"forced_settle": _last_forced[0], "forced_split": _last_forced[1]}
