"""事件级区间的验证。

为什么这个模块值得测：它的唯一作用是**阻止一个错误的结论**——
留出集里只有 17 次抓挠事件，两个模型报 0.788 和 0.848 看着差 6 个点，
实际差的是一个事件。区间算错、或者该警告的时候不警告，这个模块就白写了，
而且失效的方式是安静的（照样打印一行看起来很专业的数）。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.eventci import (  # noqa: E402
    describe, event_ci, f1_from, gap_is_meaningful,
)


def test_f1_matches_the_textbook_formula():
    assert f1_from(1.0, 1.0) == pytest.approx(1.0)
    assert f1_from(0.5, 0.5) == pytest.approx(0.5)
    assert f1_from(1.0, 0.0) == pytest.approx(0.0)


def test_f1_gives_zero_not_nan_when_both_are_zero():
    """0/0 给 nan 的话会一路传到"最好的一格"的比较里，
    而 nan 的比较永远是 False —— 表现成"那一格永远选不中"。"""
    v = f1_from(0.0, 0.0)
    assert v == 0.0 and not np.isnan(v)


def test_ci_brackets_the_point_estimate():
    ci = event_ci(n_true=17, n_pred=16, tp=14)
    point = f1_from(14 / 16, 14 / 17)
    lo, hi = ci["f1"]
    assert lo <= point <= hi


def test_ci_is_wide_when_there_are_few_events():
    """17 次事件下半宽应该在 0.1 以上——**这正是要让人看见的那件事**。"""
    lo, hi = event_ci(n_true=17, n_pred=16, tp=14)["f1"]
    assert (hi - lo) / 2 > 0.1


def test_ci_narrows_as_events_accumulate():
    """区间宽度大致按 1/sqrt(n) 收敛。不收敛的话说明自助写错了，
    而那种错会给出一个"看起来很确定"的假区间。"""
    w = []
    for n in (17, 170, 1700):
        lo, hi = event_ci(n_true=n, n_pred=n, tp=int(0.82 * n))["f1"]
        w.append(hi - lo)
    assert w[0] > w[1] > w[2]
    assert w[0] / w[2] > 5          # 100 倍样本 → 宽度约 1/10，给足余量


def test_ci_is_none_when_there_are_no_events():
    """没有事件和"事件级得分是 0"必须分得开。编一个数出来最糟。"""
    assert event_ci(n_true=0, n_pred=5, tp=0) is None
    assert event_ci(n_true=5, n_pred=0, tp=0) is None


def test_ci_rejects_impossible_tp():
    with pytest.raises(ValueError):
        event_ci(n_true=17, n_pred=16, tp=17)   # 报出只有 16，对不可能有 17


def test_describe_warns_loudly_below_thirty_events():
    s = describe(event_ci(17, 16, 14), 17)
    assert "17" in s and "分辨不了" in s and "窗口级" in s


def test_describe_stays_quiet_when_there_is_enough_data():
    s = describe(event_ci(500, 480, 410), 500)
    assert "分辨不了" not in s


def test_the_real_cnn_vs_rf_gap_is_not_meaningful():
    """本次选型的关键结论，钉成测试。

    CNN 对 13 次、RF 对 14 次，同一批 17 个真值事件、各报 16 次。
    差一个事件，配对自助的差值区间跨 0 —— **分不出高下**。
    这条要是变绿变红了，说明谁动了区间的算法，那正是需要重新审视结论的时候。
    """
    g = gap_is_meaningful(n_true=17, n_pred=16, tp_a=13, tp_b=14)
    assert g["diff"] > 0                      # RF 名义上确实高一点
    assert not g["meaningful"], f"区间 {g['ci']} 不该排除 0"


def test_gap_is_meaningful_when_the_difference_is_large():
    """反向用例：差距真的大时必须判为显著，否则这个函数只会说"都一样"。"""
    g = gap_is_meaningful(n_true=200, n_pred=200, tp_a=100, tp_b=180)
    assert g["meaningful"]


def test_gap_uses_paired_resampling_not_overlapping_intervals():
    """配对自助比"两个区间有没有重叠"敏感。

    构造一个区间**确实重叠**、但配对差值显著的情形。参数是搜出来的，
    不是拍的：第一版写 240 vs 270，那两个区间根本不重叠，前提就不成立，
    这条测试等于什么都没验（而它会是绿的——因为显著性那一半成立）。

    300 次事件下 240 vs 254：区间 [0.767,0.832] 和 [0.817,0.875] 重叠，
    但配对之后差值区间不跨 0。
    """
    a = event_ci(300, 300, 240)["f1"]
    b = event_ci(300, 300, 254)["f1"]
    assert a[1] > b[0], f"前提不成立：{a} 和 {b} 不重叠"
    assert gap_is_meaningful(300, 300, 240, 254)["meaningful"]
