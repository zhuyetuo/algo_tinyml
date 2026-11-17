"""板上精准版起止（只做平移）vs 服务端 label_service/postprocess.py 的 precise_boundaries（不给能量包络 = 只平移）。

基准还是服务端那份：先用同一套稳定版 v2 解出片段，再各自平移，逐段对答案。"""
import ctypes
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_post_c import CLASSES, _load_postprocess, _py_segs_flat, c_run, py_stabilize, synth, ts_of  # noqa: E402

BIAS = {"抓挠": (-0.75, 1.7), "甩身体": (-0.5, 1.45)}     # 服务端的写法：(起点偏差, 终点偏差)，正 = 偏晚


from test_post_c import lib  # noqa: E402,F401  同一份编译好的库（argtypes 在那边设好了）


def _set_shift(lib, shifts):
    st = (ctypes.c_int * len(CLASSES))(*[shifts.get(c, (0, 0))[0] for c in CLASSES])
    en = (ctypes.c_int * len(CLASSES))(*[shifts.get(c, (0, 0))[1] for c in CLASSES])
    lib.host_post_set_shift(st, en, len(CLASSES))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_平移跟服务端精准版一致(lib, seed):
    _, pp = _load_postprocess()
    probs = synth(500, seed)
    ts = ts_of(len(probs))
    _set_shift(lib, {c: (int(round(-b[0] * 1000)), int(round(-b[1] * 1000))) for c, b in BIAS.items()})
    try:
        got, _ = c_run(lib, probs, ts)
    finally:
        _set_shift(lib, {})
    segs, t0 = py_stabilize(probs, ts)
    for lab, b in BIAS.items():
        pp.precise_boundaries(segs.get(lab) or [], None, b)
    mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got], key=lambda x: (x[1], x[0]))
    assert mine == _py_segs_flat(segs, t0)


def test_不设平移就跟稳定版v2一样(lib):
    probs = synth(300, 7)
    ts = ts_of(len(probs))
    _set_shift(lib, {})
    got, _ = c_run(lib, probs, ts)
    segs, t0 = py_stabilize(probs, ts)
    assert sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got], key=lambda x: (x[1], x[0])) == _py_segs_flat(segs, t0)


def _cfg_precise(lib, names, window_s):
    arr = (ctypes.c_char_p * len(names))(*[n.encode("utf-8") for n in names])
    st, en = (ctypes.c_int * len(names))(), (ctypes.c_int * len(names))()
    lib.host_cfg_precise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
    k = lib.host_cfg_precise(arr, len(names), 8.0 if window_s is None else window_s, st, en)
    return k, list(st), list(en)


def test_按类别名填平移量(lib):
    names = ["未佩戴", "静止/休息", "活动", "抓挠", "甩头/抖身"]
    k, st, en = _cfg_precise(lib, names, 8.0)
    assert k == 2
    assert (st[3], en[3]) == (750, -1700) and (st[4], en[4]) == (500, -1450)
    assert st[:3] == [0, 0, 0] and en[:3] == [0, 0, 0]


def test_别的窗口长度不平移(lib):
    k, st, en = _cfg_precise(lib, ["活动", "抓挠"], 4.0)
    assert k == 0 and st == [0, 0] and en == [0, 0]


def test_跟服务端的校正量是同一组数():
    """板上写死的数要跟服务端 PRECISE_BIAS_8S 一致：改了一边忘了另一边，两边的起止就分家了"""
    _, pp = _load_postprocess()
    assert pp.PRECISE_BIAS_8S["抓挠"] == (-0.75, 1.7)
    assert pp.PRECISE_BIAS_8S["_other"] == (-0.5, 1.45)
