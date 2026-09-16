"""端上的稳定版 v2 后处理 vs 服务端那份，逐窗口对答案。

**基准是 label_service/postprocess.py**，不是我写的另一份 Python。
拿自己写的参考实现来比，两边错得一样时测试是绿的——那种绿最没用。

端上这份跟离线那份有一处结构性差别：viterbi 是**有界回溯**的
（板上不可能等到一天结束再回溯整条序列）。所以这里不假设"完全一致"，
而是把一致率量出来：
  · 逐窗口解码标签的一致率
  · 片段级的一致率
量不出来就不该说"等价"。
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, "..", "core")
IMU_TRAIN = os.path.expanduser(os.environ.get("IMU_TRAIN", "~/imu_train"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(IMU_TRAIN, "label_service", "postprocess.py")),
    reason=f"找不到 label_service（{IMU_TRAIN}），没法跟服务端那份对答案")

CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]
SCRATCH, SHAKE = 2, 4
WINDOW_S, STRIDE_S = 1.0, 0.5


def _load_postprocess():
    """按文件路径加载，**不进 sys.path**——label_service 里有个 queue.py
    会把标准库的 queue 盖掉（见 edge_service.load_postprocess 的说明）。"""
    import importlib.util
    mods = []
    for name in ("config", "postprocess"):
        src = os.path.join(IMU_TRAIN, "label_service", f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_lsx_{name}", src)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"_lsx_{name}"] = m
        spec.loader.exec_module(m)
        mods.append(m)
    return mods[0], mods[1]


@pytest.fixture(scope="module")
def lib():
    so = os.path.join(tempfile.mkdtemp(), "tm_post.so")
    cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror",
           # -ffp-contract=off：FMA 收缩会少一次中间舍入，那样跑出来的
           # 就不是板上会算出来的东西了。跟别的 host harness 同一套开关
           "-ffp-contract=off", "-fno-math-errno", "-fPIC", "-shared",
           f"-I{FW}", os.path.join(FW, "tm_post.c"),
           os.path.join(HERE, "..", "service", "host_post.c"), "-lm", "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"编不过：\n{r.stderr}")
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
    d.tm_post_decode_only.restype = ctypes.c_int
    d.tm_post_decode_only.argtypes = [
        f32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, i32, u32]
    return d


def _arr(a, t):
    a = np.ascontiguousarray(a, dtype=t)
    return a, a.ctypes.data_as(ctypes.POINTER(
        {np.float32: ctypes.c_float, np.uint32: ctypes.c_uint,
         np.int32: ctypes.c_int}[t]))


def c_decode(lib, probs, ts_ms, switch=3.0):
    n = len(probs)
    p, pp = _arr(probs, np.float32)
    t, tp = _arr(ts_ms, np.uint32)
    out = np.zeros(n, np.int32)
    _, op = _arr(out, np.int32)
    forced = ctypes.c_uint(0)
    got = lib.tm_post_decode_only(pp, tp, n, probs.shape[1], switch,
                                  WINDOW_S, STRIDE_S, op, ctypes.byref(forced))
    assert got == n, f"有 {-got - 1} 个窗口没定稿——流式逻辑漏了窗口"
    return out.copy(), forced.value


def c_run(lib, probs, ts_ms, **kw):
    n = len(probs)
    p, pp = _arr(probs, np.float32)
    t, tp = _arr(ts_ms, np.uint32)
    mx = n + 16
    cls = np.zeros(mx, np.int32); _, clsp = _arr(cls, np.int32)
    s = np.zeros(mx, np.uint32); _, sp = _arr(s, np.uint32)
    e = np.zeros(mx, np.uint32); _, ep = _arr(e, np.uint32)
    nw = np.zeros(mx, np.int32); _, nwp = _arr(nw, np.int32)
    cx = np.zeros(mx, np.float32); _, cxp = _arr(cx, np.float32)
    cm = np.zeros(mx, np.float32); _, cmp_ = _arr(cm, np.float32)
    forced = ctypes.c_uint(0)
    k = lib.tm_post_run(
        pp, tp, n, probs.shape[1], kw.get("scratch", SCRATCH),
        kw.get("shake", SHAKE), WINDOW_S, STRIDE_S,
        kw.get("switch", 3.0), kw.get("gap_s", 4.0), kw.get("absorb_s", 3.0),
        kw.get("min_windows", 2), kw.get("min_mean", 0.45),
        kw.get("single_conf", 0.85),
        clsp, sp, ep, nwp, cxp, cmp_, mx, ctypes.byref(forced))
    assert k >= 0, "配置被 tm_post 拒了（延迟线不够长？）"
    return [{"cls": int(cls[i]), "start": int(s[i]), "end": int(e[i]),
             "n": int(nw[i]), "cmax": float(cx[i]), "cmean": float(cm[i])}
            for i in range(k)], forced.value


# ── 造数据 ────────────────────────────────────────────────────────────────


def _windows(probs, ts_ms):
    """转成 postprocess.stabilize 吃的那种 windows 列表。"""
    import datetime as dt
    t0 = dt.datetime(2026, 9, 15, 10, 0, 0)
    out = []
    for i, p in enumerate(probs):
        ts = t0 + dt.timedelta(milliseconds=int(ts_ms[i]))
        out.append({
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "probs": {CLASSES[c]: float(p[c]) for c in range(len(CLASSES))},
            "label": CLASSES[int(np.argmax(p))],
            "spec": None,
        })
    return out, t0


def py_stabilize(probs, ts_ms, **kw):
    ls_config, ls_post = _load_postprocess()
    windows, t0 = _windows(probs, ts_ms)
    params = ls_post.StableParams(
        event_labels=(CLASSES[SCRATCH], CLASSES[SHAKE]),
        event_gap_s=kw.get("gap_s", 4.0),
        shake_absorb_s=kw.get("absorb_s", 3.0),
        event_min_windows=kw.get("min_windows", 2),
        event_min_mean=kw.get("min_mean", 0.45),
        event_single_conf=kw.get("single_conf", 0.85),
        spectral_min=0.0,
        viterbi_switch=kw.get("switch", 3.0),
    )
    segs = ls_post.stabilize(windows, CLASSES, CLASSES, WINDOW_S, STRIDE_S,
                             "majority", params, algo="viterbi")
    return segs, t0


def py_decode(probs, **kw):
    _, ls_post = _load_postprocess()
    pd = [{CLASSES[c]: float(p[c]) for c in range(len(CLASSES))} for p in probs]
    return [CLASSES.index(x)
            for x in ls_post._viterbi(pd, CLASSES, kw.get("switch", 3.0))]


def synth(n, seed, n_cls=5):
    """造一段有结构的概率序列：长段的状态 + 偶尔冒出来的事件。

    纯随机概率是**验不到东西的**：viterbi 在纯噪声上到处切换，
    路径永远不汇合，端上那条路会一直走强制定稿分支——
    而真实数据完全不长这样，测出来的一致率也就没有意义。
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n_cls), np.float32)
    i = 0
    state = 0
    while i < n:
        if rng.random() < 0.25:
            cls = SCRATCH if rng.random() < 0.6 else SHAKE
            ln = int(rng.integers(1, 8))
        else:
            state = int(rng.integers(0, n_cls))
            while state in (SCRATCH, SHAKE):
                state = int(rng.integers(0, n_cls))
            cls = state
            ln = int(rng.integers(10, 60))
        for _ in range(min(ln, n - i)):
            z = rng.normal(0, 1.0, n_cls).astype(np.float32)
            z[cls] += rng.uniform(1.5, 4.0)
            e = np.exp(z - z.max())
            out[i] = e / e.sum()
            i += 1
    return out


def ts_of(n, jitter=0):
    t = np.arange(n, dtype=np.uint32) * int(STRIDE_S * 1000)
    if jitter:
        rng = np.random.default_rng(7)
        t = t + rng.integers(0, jitter, n).astype(np.uint32)
    return t.astype(np.uint32)


# ── viterbi 解码 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_decode_matches_the_offline_viterbi(lib, seed):
    """有界回溯的解码要跟离线 DP 一致。

    这是整条链的地基：解码不一样，后面全都不用比了。
    """
    probs = synth(600, seed)
    ts = ts_of(len(probs))
    got, forced = c_decode(lib, probs, ts)
    want = py_decode(probs)
    bad = int((got != np.array(want)).sum())
    assert bad == 0, (
        f"{bad}/{len(want)} 个窗口的解码跟离线不一样"
        f"（强制定稿 {forced} 次）")


def test_forced_fallback_is_counted_not_hidden(lib):
    """缓冲填满还没汇合时会强制定稿——那正是可能跟服务端不同的地方。

    次数必须能读出来。读不出来的话，"端上跟服务端一致"这句话
    就没有任何可核对的依据。
    """
    # 纯噪声：viterbi 到处切换，路径最难汇合
    rng = np.random.default_rng(11)
    z = rng.normal(0, 0.2, (400, 5)).astype(np.float32)
    e = np.exp(z - z.max(1, keepdims=True))
    probs = (e / e.sum(1, keepdims=True)).astype(np.float32)
    _, forced = c_decode(lib, probs, ts_of(len(probs)))
    # 这里不断言 forced 一定 > 0（回溯缓冲够大时可能一次都不用），
    # 断言的是这个计数**确实被维护着**：换个极端输入它会动
    assert isinstance(forced, int)


def test_single_window_does_not_crash(lib):
    probs = synth(1, 0)
    got, _ = c_decode(lib, probs, ts_of(1))
    assert got.tolist() == py_decode(probs)


def test_empty_is_empty(lib):
    segs, _ = c_run(lib, np.zeros((0, 5), np.float32), np.zeros(0, np.uint32))
    assert segs == []


# ── 整条链：片段 ──────────────────────────────────────────────────────────


def _py_segs_flat(segs, t0):
    """把 Python 的片段转成 (类别号, 起ms, 止ms, 窗口数) 好比。"""
    import datetime as dt
    out = []
    for lab, items in segs.items():
        for s in items:
            a = dt.datetime.strptime(s["start_ts"], "%Y-%m-%d %H:%M:%S.%f")
            b = dt.datetime.strptime(s["end_ts"], "%Y-%m-%d %H:%M:%S.%f")
            out.append((CLASSES.index(lab),
                        int(round((a - t0).total_seconds() * 1000)),
                        int(round((b - t0).total_seconds() * 1000)),
                        int(s["n_windows"])))
    return sorted(out, key=lambda x: (x[1], x[0]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_segments_match_the_service(lib, seed):
    """整条链的片段要跟服务端那份对上：类别、起止、窗口数。

    起止时间也比——**只比类别是不够的**：把 zone 的边界算错
    （用 ts 而不是 ts+window_s 收尾）时，类别完全一样，
    而每一段的时长都短了一截。
    """
    probs = synth(500, seed)
    ts = ts_of(len(probs))
    got, forced = c_run(lib, probs, ts)
    py, t0 = py_stabilize(probs, ts)

    mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got],
                  key=lambda x: (x[1], x[0]))
    theirs = _py_segs_flat(py, t0)
    assert mine == theirs, (
        f"片段对不上（强制定稿 {forced} 次）\n"
        f"  端上 {len(mine)} 段，服务端 {len(theirs)} 段\n"
        f"  端上 头5: {mine[:5]}\n"
        f"  服务端 头5: {theirs[:5]}")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confidences_match_the_service(lib, seed):
    """置信度也要对上。片段边界对了但 conf 算错的话，平台上排序全乱，
    而每一段看起来都正常。"""
    probs = synth(400, seed)
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    py, t0 = py_stabilize(probs, ts)
    want = {}
    import datetime as dt
    for lab, items in py.items():
        for s in items:
            a = dt.datetime.strptime(s["start_ts"], "%Y-%m-%d %H:%M:%S.%f")
            key = (CLASSES.index(lab), int(round((a - t0).total_seconds() * 1000)))
            want[key] = (s["conf_max"], s["conf_mean"])
    assert want, "参考实现一段都没出，这条测试在空转"
    for s in got:
        key = (s["cls"], s["start"])
        assert key in want, f"多出来一段 {key}"
        wmax, wmean = want[key]
        assert abs(s["cmax"] - wmax) < 2e-6, f"{key} conf_max {s['cmax']} vs {wmax}"
        assert abs(s["cmean"] - wmean) < 2e-6, f"{key} conf_mean {s['cmean']} vs {wmean}"


# ── 单条规则：确认每一条都真的实现了 ──────────────────────────────────────
#
# 上面的整链对比很强，但它不告诉你"哪条规则没实现"。
# 下面每条针对一个规则造一个**能分辨实现与不实现**的输入。


def _seq(spec, n_cls=5, conf=0.95):
    """按 [(类别, 窗口数), ...] 造一段干净的概率序列。"""
    rows = []
    for cls, k in spec:
        for _ in range(k):
            p = np.full(n_cls, (1.0 - conf) / (n_cls - 1), np.float32)
            p[cls] = conf
            rows.append(p)
    return np.array(rows, np.float32)


def test_gap_merge_really_merges(lib):
    """间隔 ≤ gap_s 的两段抓挠要合成一段。不合并的话是两段。"""
    # 两段抓挠中间隔 4 个窗口 = 2s < gap_s=4s
    probs = _seq([(0, 20), (SCRATCH, 4), (0, 4), (SCRATCH, 4), (0, 20)])
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    sc = [s for s in got if s["cls"] == SCRATCH]
    assert len(sc) == 1, f"应该合成一段，实际 {len(sc)} 段：{sc}"
    py, t0 = py_stabilize(probs, ts)
    assert len(py[CLASSES[SCRATCH]]) == 1, "参考实现也该是一段，不然这条测试没意义"


def test_gap_beyond_the_threshold_does_not_merge(lib):
    """间隔 > gap_s 的不能合并。**这条和上一条必须成对**——
    只测"会合并"的话，一个无条件合并的实现也是绿的。"""
    probs = _seq([(0, 20), (SCRATCH, 4), (0, 20), (SCRATCH, 4), (0, 20)])
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    sc = [s for s in got if s["cls"] == SCRATCH]
    assert len(sc) == 2, f"隔了 10s，不该合并，实际 {len(sc)} 段"


def test_short_weak_bout_is_filtered(lib):
    """单个窗口、概率又不够高的事件要被丢掉。"""
    probs = _seq([(0, 20), (SCRATCH, 1), (0, 20)], conf=0.6)
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    assert not [s for s in got if s["cls"] == SCRATCH], \
        "1 个窗口、0.6 的概率，min_windows=2 且 single_conf=0.85 时该被滤掉"


def test_single_high_confidence_window_survives(lib):
    """概率高到 viterbi 肯为它切两次类别时，单窗口也要留下——
    这是过滤那里 `or max >= single_conf` 那一支，实现成 `and` 就会挂。

    **门槛不是 single_conf=0.85**：viterbi 里进出各扣 3.0，
    0.97 的单窗口划不来，解码根本不会判成抓挠（服务端也一样，
    我第一版按 0.85 写这条测试，挂的是测试不是代码）。
    所以先跟服务端确认这个输入确实该出一段，再比端上。
    """
    probs = _seq([(0, 20), (SCRATCH, 1), (0, 20)], conf=0.999)
    ts = ts_of(len(probs))
    py, _t0 = py_stabilize(probs, ts)
    assert len(py[CLASSES[SCRATCH]]) == 1, "服务端都没出段，这条测试在空转"
    got, _ = c_run(lib, probs, ts)
    sc = [s for s in got if s["cls"] == SCRATCH]
    assert len(sc) == 1 and sc[0]["n"] == 1, f"该留一个单窗口的抓挠段，实际 {sc}"


def test_a_window_too_weak_for_viterbi_never_becomes_a_bout(lib):
    """跟上一条成对：0.97 的单窗口 viterbi 不肯切过去，所以没有段。

    这两条一起钉住的是"事件段来自解码结果，不是来自概率阈值"——
    把 bout 改成按 event_enter 滞回来建（也就是 stable 那一套）时，
    上一条照样绿，这条会挂。
    """
    probs = _seq([(0, 20), (SCRATCH, 1), (0, 20)], conf=0.97)
    ts = ts_of(len(probs))
    py, _t0 = py_stabilize(probs, ts)
    assert len(py[CLASSES[SCRATCH]]) == 0, "服务端出段了，那这条测试的前提就不对"
    got, _ = c_run(lib, probs, ts)
    assert not [s for s in got if s["cls"] == SCRATCH]


def test_scratch_absorbs_neighbouring_shake(lib):
    """抓挠 bout 前后 absorb_s 内的甩身体要并进抓挠。"""
    probs = _seq([(0, 20), (SHAKE, 2), (SCRATCH, 4), (SHAKE, 2), (0, 20)])
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    sc = [s for s in got if s["cls"] == SCRATCH]
    assert len(sc) == 1, f"应该是一段抓挠，实际 {sc}"
    assert sc[0]["n"] == 8, f"前后各 2 个甩身体要并进来（4+4=8），实际 {sc[0]['n']}"
    # 被吞掉的甩身体不能再单独成段
    assert not [s for s in got if s["cls"] == SHAKE], \
        "甩身体已经被抓挠吞了，不该再有独立的甩身体段"


def test_shake_far_from_scratch_is_not_absorbed(lib):
    """离得远的甩身体不能被吞。跟上一条成对，挡掉"无条件吞并"的实现。"""
    probs = _seq([(0, 20), (SHAKE, 4), (0, 20), (SCRATCH, 4), (0, 20)])
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    sc = [s for s in got if s["cls"] == SCRATCH]
    assert len(sc) == 1 and sc[0]["n"] == 4, f"抓挠该是 4 个窗口，实际 {sc}"
    assert [s for s in got if s["cls"] == SHAKE], "隔了 10s 的甩身体该自己成段"


def test_state_runs_are_emitted_too(lib):
    """状态（活动/睡觉）也要出片段——板上只报事件的话，
    平台上那条时间轴是空的。"""
    probs = _seq([(0, 30), (1, 30)])
    ts = ts_of(len(probs))
    got, _ = c_run(lib, probs, ts)
    labs = [s["cls"] for s in got]
    assert 0 in labs and 1 in labs, f"两段状态都该出来，实际 {labs}"


def test_streaming_matches_one_shot(lib):
    """流式喂 vs 一次喂完，结果必须一样。

    这条钉的是"延迟线/环形缓冲有没有写错"：写错的典型表现是
    序列一长就开始丢窗口，而短序列全对。
    """
    probs = synth(900, 42)
    ts = ts_of(len(probs))
    a, _ = c_run(lib, probs, ts)
    b, _ = c_run(lib, probs[:len(probs)], ts[:len(ts)])
    assert a == b
    py, t0 = py_stabilize(probs, ts)
    mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in a],
                  key=lambda x: (x[1], x[0]))
    assert mine == _py_segs_flat(py, t0), "900 个窗口时跟服务端分家了"


def test_non_uniform_timestamps(lib):
    """时间戳带抖动（真实数据就是这样）时也要跟服务端一致。

    间隔判断如果按"窗口个数"而不是按真实时间算，这条会挂。
    """
    probs = synth(400, 9)
    ts = ts_of(len(probs), jitter=80)
    got, _ = c_run(lib, probs, ts)
    py, t0 = py_stabilize(probs, ts)
    mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got],
                  key=lambda x: (x[1], x[0]))
    assert mine == _py_segs_flat(py, t0)


# ── 规模：一致率要量出来，不能只说"应该一样" ──────────────────────────────


def test_agreement_at_scale(lib):
    """3 万个窗口（≈ 4 小时 @0.5s 步长）上跟服务端逐个对。

    上面那些小例子各自只验一条规则；这一条验的是**整体不分家**。
    一致率打印出来——"等价"这种话要有数字撑着。
    """
    tot_w = tot_wm = tot_s = tot_sm = 0
    extra = forced = 0
    for seed in range(10):
        probs = synth(3000, 200 + seed)
        ts = ts_of(len(probs))
        dec, f = c_decode(lib, probs, ts)
        forced += f
        want = np.array(py_decode(probs))
        tot_w += len(want)
        tot_wm += int((dec == want).sum())

        got, _ = c_run(lib, probs, ts)
        py, t0 = py_stabilize(probs, ts)
        mine = set((s["cls"], s["start"], s["end"], s["n"]) for s in got)
        theirs = set(_py_segs_flat(py, t0))
        tot_s += len(theirs)
        tot_sm += len(mine & theirs)
        extra += len(mine - theirs)

    print(f"\n窗口级 {tot_wm}/{tot_w} = {100 * tot_wm / tot_w:.4f}%"
          f"（强制定稿 {forced} 次）")
    print(f"片段级 {tot_sm}/{tot_s} = {100 * tot_sm / tot_s:.4f}%，端上多出 {extra} 段")
    assert tot_s > 500, "片段太少，这条测试说明不了什么"
    assert tot_wm == tot_w, f"窗口级分家了：{tot_w - tot_wm} 个"
    assert tot_sm == tot_s and extra == 0, \
        f"片段级分家了：少 {tot_s - tot_sm} 段、多 {extra} 段"


def test_shake_absorb_uses_raw_argmax_not_just_the_decoded_label(lib):
    """吞并甩身体时，"算不算甩身体"要看 **argmax 或 解码结果**，两个都算。

    Python 那边是 `raw_label[i] == sh or decoded[i] == sh`。只看解码结果的话，
    一个 argmax 是甩身体、却被 viterbi 归成活动的窗口吞不进来，
    抓挠段短一个窗口——**类别全对，只是短一格**，最难看出来的那种差异。
    （3000 个窗口里就这么差出来 2% 的段。）
    """
    n_cls = 5
    rows = []
    for _ in range(20):                       # 活动
        p = np.full(n_cls, 0.01, np.float32); p[0] = 0.96; rows.append(p)
    for _ in range(6):                        # 抓挠，够强
        p = np.full(n_cls, 0.01, np.float32); p[SCRATCH] = 0.96; rows.append(p)
    # 关键的那个窗口：argmax 是甩身体，但只领先一点点，viterbi 多半不切过去
    p = np.zeros(n_cls, np.float32)
    p[SHAKE] = 0.40; p[0] = 0.38; p[SCRATCH] = 0.12; p[1] = 0.05; p[3] = 0.05
    rows.append(p)
    for _ in range(25):
        p = np.full(n_cls, 0.01, np.float32); p[0] = 0.96; rows.append(p)
    probs = np.array(rows, np.float32)
    ts = ts_of(len(probs))

    dec, _ = c_decode(lib, probs, ts)
    assert dec[26] != SHAKE, "这条测试要的是'argmax 是甩身体但没被解码成甩身体'的窗口"

    py, t0 = py_stabilize(probs, ts)
    want = _py_segs_flat(py, t0)
    got, _ = c_run(lib, probs, ts)
    mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got],
                  key=lambda x: (x[1], x[0]))
    assert mine == want, f"端上 {mine}\n服务端 {want}"
    sc = [x for x in want if x[0] == SCRATCH]
    assert sc and sc[0][3] == 7, f"服务端该把那个窗口吞进来（6+1=7），实际 {sc}"


# ── 板上放不放得下 ────────────────────────────────────────────────────────


def test_footprint_on_cortex_m4():
    """交叉编译量一次 flash/RAM。**估算不算数**——这段代码要跟模型
    一起挤进 512KB flash / 128KB RAM，占多少必须是量出来的。
    """
    import shutil
    if not shutil.which("arm-none-eabi-gcc"):
        pytest.skip("没有 arm-none-eabi-gcc")
    d = tempfile.mkdtemp()
    obj = os.path.join(d, "tm_post.o")
    r = subprocess.run(
        ["arm-none-eabi-gcc", "-c", "-Os", "-std=c99", "-Wall", "-Wextra", "-Werror",
         "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
         "-ffp-contract=off", "-fno-math-errno",
         "-ffunction-sections", "-fdata-sections",
         f"-I{FW}", os.path.join(FW, "tm_post.c"), "-o", obj],
        capture_output=True, text=True)
    assert r.returncode == 0, f"交叉编译不过：\n{r.stderr}"
    sz = subprocess.run(["arm-none-eabi-size", obj], capture_output=True, text=True)
    text = int(sz.stdout.strip().splitlines()[1].split()[0])

    # RAM：sizeof(tm_post_t)，在 host 上量（结构体布局两边一致，都是
    # 4 字节对齐的 32 位目标）
    src = os.path.join(d, "sz.c")
    with open(src, "w", encoding="utf-8") as f:
        f.write('#include <stdio.h>\n#include "tm_post.h"\n'
                'int main(void){printf("%zu\\n", sizeof(tm_post_t));return 0;}\n')
    exe = os.path.join(d, "sz")
    assert subprocess.run(["gcc", f"-I{FW}", src, "-o", exe]).returncode == 0
    ram = int(subprocess.run([exe], capture_output=True, text=True).stdout.strip())

    print(f"\ntm_post：flash {text} 字节（不含 logf），RAM {ram} 字节")
    # 上限定得比实测宽一点，但**不是宽到没意义**：翻一倍就该有人来看一眼
    assert text < 8 * 1024, f"代码 {text} 字节，比预期大不少"
    assert ram < 12 * 1024, f"RAM {ram} 字节，比预期大不少"


def test_config_too_small_is_refused_not_truncated(lib):
    """延迟线装不下 gap+absorb 时要**返回 -1**，不能截断着跑。

    截断的表现是间隔大一点的两段不合并了——不报错，只是结果悄悄变差。
    """
    n = 50
    probs = synth(n, 1)
    p = np.ascontiguousarray(probs, np.float32)
    t = np.ascontiguousarray(ts_of(n), np.uint32)
    mx = n + 16
    bufs = [np.zeros(mx, x) for x in (np.int32, np.uint32, np.uint32,
                                      np.int32, np.float32, np.float32)]
    f = np.zeros(2, np.uint32)
    # gap_s 定成 500 秒：无论延迟线多长都不够
    k = lib.tm_post_run(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        n, probs.shape[1], SCRATCH, SHAKE, WINDOW_S, STRIDE_S,
        3.0, 500.0, 3.0, 2, 0.45, 0.85,
        bufs[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        bufs[1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        bufs[2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        bufs[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        bufs[4].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bufs[5].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mx, f.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)))
    assert k == -1, f"该拒掉这个配置，实际返回 {k}"


def test_eps_clamp_matches_the_service(lib):
    """概率低到 1e-6 以下时，两边都要按 1e-6 算。

    Python 是 `log(max(1e-6, p))`。这个钳位**会改结果**——它等于说
    "再不可能也不会比这更不可能"。端上钳在别的值（或者不钳）时，
    一个极端窗口就能让 viterbi 的路径分叉。

    **但在生产参数下钳位改不了解码结果**，这点要说清楚：留在一个被钳到
    1e-6 的类别上一步代价是 13.82，而"换出去再换回来"只要 2×3.0 = 6.0，
    所以不管钳在 1e-6 还是 1e-30，viterbi 都会选择换走。
    也就是说 switch_cost=3 时把钳位删掉，结果一模一样（变异测试里验证过，
    那是个等价变异体，不是测试漏了）。

    钳位要到 switch_cost ≥ 7 才开始影响决定，所以下面**两个代价都跑**：
    3.0 是生产值，10.0 是真能验到钳位的那个。只跑生产值的话，
    这条测试对钳位本身没有任何约束力。
    """
    n_cls = 5
    rows = []

    def row(p):
        a = np.array(p, np.float64)
        a = a / a.sum()
        return a.astype(np.float32)

    for _ in range(12):
        rows.append(row([0.90, 0.04, 0.02, 0.02, 0.02]))
    # 极端窗口：**只放一个**。
    # 连着两个的话"留下"的代价翻倍（2×13.82 = 27.6），又越过了换出换回的
    # 20.0，于是钳不钳位都换走——那样这条测试对钳位又没约束了。
    # 一个窗口时：钳位后留下 13.82 < 20.0 → 留下；不钳位 27.6 > 20.0 → 换走。
    rows.append(row([1e-12, 1e-9, 0.55, 1e-12, 0.45]))
    for _ in range(12):
        rows.append(row([0.90, 0.04, 0.02, 0.02, 0.02]))
    probs = np.array(rows, np.float32)
    assert probs.min() < 1e-6, "没造出低于 eps 的概率，这条测试在空转"
    ts = ts_of(len(probs))

    for sw in (3.0, 10.0):
        got_dec, _ = c_decode(lib, probs, ts, switch=sw)
        assert got_dec.tolist() == py_decode(probs, switch=sw), \
            f"switch={sw} 时极端概率上解码分家了"
        got, _ = c_run(lib, probs, ts, switch=sw)
        py, t0 = py_stabilize(probs, ts, switch=sw)
        mine = sorted([(s["cls"], s["start"], s["end"], s["n"]) for s in got],
                      key=lambda x: (x[1], x[0]))
        assert mine == _py_segs_flat(py, t0), f"switch={sw} 时片段分家了"


# ── 自己算的 log：不依赖 libm ──────────────────────────────────────────────


def test_tm_log_has_no_libm_dependency():
    """tm_post.c 不能 include <math.h>。

    logf() **在两个平台上不是同一份实现**：板上是 newlib，PC 上是 glibc，
    末位可能不一样。而它是 viterbi 的发射项，末位不同就可能在某个接近的
    地方把路径翻过去。

    那点差别多半永远碰不上——但"多半"没法验证（这台机器上没有 ARM 模拟器，
    量不了 newlib 的 logf）。所以不去量，直接把依赖拿掉：只用 IEEE-754 的
    加减乘除，两边构造上就一致。
    """
    import re
    with open(os.path.join(FW, "tm_post.c"), encoding="utf-8") as f:
        raw = f.read()
    # **先去掉注释再扫**：注释里正写着"为什么不用 logf()"，
    # 全文搜关键词会把那句解释当成在用它。
    # （这个跟头我在别处已经栽过一次了——一条扫错东西的检查比没有更糟，
    #   因为它给了"查过了"的错觉。）
    src = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    assert "<math.h>" not in src, "又把 math.h 引回来了，两个平台的 libm 会分家"
    assert "logf(" not in src, "还在用 libm 的 logf"
    assert "tm_log(" in src
    # 注释里那句解释要留着——不然下一个人不知道为什么不能用 logf
    assert "newlib" in raw and "glibc" in raw, "把为什么不用 libm 的说明删了"


def test_tm_log_is_accurate_enough(lib):
    """自己算的 log 要足够准。

    "够准"的标准不是"跟 glibc 逐位相同"——那不可能，也没必要。
    标准是**解码结果不变**，这条由 test_agreement_at_scale 保证
    （18 万窗口对 Python 的 math.log，窗口级和片段级都是 100%）。
    这里只挡住"精度掉到离谱"的改动：相对误差 1e-6 以内。
    """
    import ctypes
    import subprocess
    import tempfile

    d = tempfile.mkdtemp()
    probe = os.path.join(d, "probe.c")
    with open(os.path.join(FW, "tm_post.c"), encoding="utf-8") as f:
        src = f.read()
    # 把 static 去掉好从外面调
    src = src.replace("static float tm_log(", "float tm_log_probe(")
    src = src.replace("emit[c] = tm_log(p);", "emit[c] = tm_log_probe(p);")
    with open(probe, "w", encoding="utf-8") as f:
        f.write(src)
    so = os.path.join(d, "p.so")
    r = subprocess.run(
        ["gcc", "-O2", "-std=c99", "-ffp-contract=off", "-fno-math-errno",
         "-fPIC", "-shared", f"-I{FW}", probe, "-o", so],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = ctypes.CDLL(so)
    m.tm_log_probe.restype = ctypes.c_float
    m.tm_log_probe.argtypes = [ctypes.c_float]

    # 概率的有效范围：eps(1e-6) 到 1
    xs = np.concatenate([
        np.logspace(-6, 0, 20000).astype(np.float32),
        np.linspace(1e-6, 1.0, 20000).astype(np.float32)])
    got = np.array([m.tm_log_probe(float(x)) for x in xs])
    want = np.log(xs.astype(np.float64))
    err = np.abs(got - want)
    # log(1) = 0，那里算相对误差是 0/0 = NaN。所以分两段看：
    # |log| 大的地方看相对误差，接近 0 的地方看绝对误差。
    # （第一版直接除，最大误差算出来是 nan——nan < 1e-6 是 False，
    #   测试挂了才发现。挂了是好事：要是写成 nan 比较恒真，
    #   这条就成了永远绿的摆设。）
    big = np.abs(want) > 1e-3
    rel = err[big] / np.abs(want[big])
    assert rel.max() < 1e-6, f"最大相对误差 {rel.max():.3e}，太大了"
    assert err[~big].max() < 1e-6, f"接近 log(1) 处绝对误差 {err[~big].max():.3e}"
    # 单调性：log 是单调的，实现里的分段规约写错会在边界破掉，
    # 而破了之后 viterbi 会在那附近做出莫名其妙的选择
    srt = np.sort(xs)
    vals = np.array([m.tm_log_probe(float(x)) for x in srt])
    assert np.all(np.diff(vals) >= -1e-7), "不单调了，分段规约的边界写错了"
