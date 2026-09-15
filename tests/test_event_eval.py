"""事件聚合的评估逻辑。

这一段的结论会直接决定"能减到多少轮"，所以每条规则都要钉住——算错了不会报错，
只会给出一个看起来合理、实际错误的事件级指标。
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load():
    path = os.path.join(ROOT, "python", "event_eval.py")
    spec = importlib.util.spec_from_file_location("_ev", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load()


def test_聚合规则跟固件一致_命中不够数不算事件():
    assert ev.to_events([0, 1, 1, 0, 0, 0], 3, 2) == []
    assert ev.to_events([0, 1, 1, 1, 0, 0, 0], 3, 2) == [(1, 3)]


def test_短暂停顿不拆成两次():
    """抓挠中间会停顿。一停就切断的话，一次连续抓挠会被拆成好几个事件——
    事件级 precision 会被这些"多出来的"事件拉低，而那不是模型的问题。"""
    hits = [1, 1, 0, 0, 1, 1, 0, 0, 0]
    assert len(ev.to_events(hits, 3, 2)) == 1


def test_停顿超过上限就分成两次():
    hits = [1, 1, 1] + [0] * 4 + [1, 1, 1] + [0] * 3
    assert len(ev.to_events(hits, 3, 2)) == 2


def test_结尾还在事件里也要收出来():
    """数据末尾正好在一次事件中间。不收的话最后一次永远不算——
    而真值和预测都会被这个影响，方向还不一定一致。"""
    assert ev.to_events([0, 1, 1, 1], 3, 2) == [(1, 3)]


def test_事件按重叠配对不要求边界对齐():
    """一次抓挠持续多久，人标的和模型判的本来就差几个窗口。
    按严格边界配的话，本来算对的会被判成一错一漏（precision 和 recall 双杀）。"""
    true_ev = [(10, 20)]
    assert ev.match_events([(12, 15)], true_ev) == (1, 0, 0)   # 预测包在里面
    assert ev.match_events([(5, 25)], true_ev) == (1, 0, 0)    # 预测包住真值
    assert ev.match_events([(20, 30)], true_ev) == (1, 0, 0)   # 只挨着一个窗口
    assert ev.match_events([(21, 30)], true_ev) == (0, 1, 1)   # 完全不重叠


def test_一个真值不会被多个预测重复算对():
    """两个预测事件都压在同一次真值上，只能算一次 TP，另一个是 FP。
    不去重的话 precision 会被虚高——模型把一次抓挠拆成三段反而"更准"了。"""
    tp, fp, fn = ev.match_events([(10, 12), (14, 16)], [(10, 20)])
    assert (tp, fp, fn) == (1, 1, 0)


def test_打乱过的数据会被拒绝():
    """事件聚合要求窗口按时间排。打乱过的话"连续命中"这个概念就不存在，
    算出来的事件级指标没有任何意义——所以宁可拒绝跑，也不给一个假的数。"""
    ordered = np.array([0] * 20 + [2] * 8 + [0] * 20 + [2] * 6)
    run, err = ev.check_ordered(ordered, 2)
    assert err is None and run == 7.0

    rng = np.random.default_rng(0)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    run2, _ = ev.check_ordered(shuffled, 2)
    assert run2 < 1.5, f"打乱之后游程 {run2}，检查判据不灵了"


def test_一条目标类别都没有时明确报错():
    _, err = ev.check_ordered(np.zeros(10, int), 2)
    assert err and "一条" in err


def test_类别偏置只挪工作点_不改模型():
    """给某个类别的 margin 加常数，等价于把判决门槛挪一挪。
    板上就是一次加法（或折进 base_score），**零 flash 成本**。

    这里钉住两条性质：①偏置足够负时该类别一条都不报；②足够正时全报。
    中间是单调的——单调性是"这个旋钮可调"的前提，不单调的话调参就没意义了。
    """
    import numpy as np
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, size=(500, 5)).astype(np.float32)
    fi = 2
    counts = []
    for bias in (-10.0, -1.0, 0.0, 1.0, 10.0):
        adj = M.copy()
        adj[:, fi] += np.float32(bias)
        counts.append(int((adj.argmax(axis=1) == fi).sum()))
    assert counts[0] == 0, "偏置 -10 还有报的"
    assert counts[-1] == len(M), "偏置 +10 还没全报"
    assert counts == sorted(counts), f"命中数随偏置不单调：{counts}"


def test_min_windows_调大只会让事件变少():
    """min_windows 是 precision/recall 之间的旋钮。调大只能让事件更少——
    不单调的话这个旋钮就没法用了。"""
    import numpy as np
    rng = np.random.default_rng(3)
    hits = rng.random(2000) < 0.25
    counts = [len(ev.to_events(hits, mw, 2)) for mw in (1, 2, 3, 5, 8, 12)]
    assert counts == sorted(counts, reverse=True), counts


def test_min_windows_变了真值事件数也会变():
    """容易踩的坑：调 min_windows 的时候，**真值那一侧也要用同一个参数**重新聚合。
    只改预测侧的话，比较的就是两套不同定义下的事件，P/R 全是错的。"""
    import numpy as np
    y = np.array([0] * 10 + [2] * 4 + [0] * 10 + [2] * 15 + [0] * 5)
    assert len(ev.to_events(y == 2, 3, 2)) == 2
    assert len(ev.to_events(y == 2, 8, 2)) == 1, "min_windows 调大后短的真事件应该消失"
