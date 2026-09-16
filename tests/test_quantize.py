"""量化本身的测试：定点算子的边界、以及 int8 前向跟 float 前向差多少。

"差多少"必须有个数字钉住。量化掉点是正常的，掉多少不正常得有据可依——
没有基线的话，某次改动把掉点从 1% 推到 8%，没有任何东西会拦住它。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml import forward_int, make_net, quantize  # noqa: E402
from tinyml.fixedpoint import (  # noqa: E402
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    rounding_divide_by_pot,
)
from tinyml.export_c import _arena_bytes  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_c_consistency import _fake_windows, N_CH, N_CLASSES  # noqa: E402


def test_定点乘子往返误差在可接受范围():
    for real in (1e-5, 1e-3, 0.01, 0.3, 0.5, 0.99):
        m, s = quantize_multiplier(real)
        x = 1 << 20
        got = int(multiply_by_quantized_multiplier(np.int64(x), m, s))
        assert abs(got - x * real) <= max(2.0, x * real * 1e-6), (real, got)


def test_乘子为零不死循环():
    # 某个输出通道权重全零时会走到这里。不特判的话 quantize_multiplier 里的
    # while real < 0.5 会一直乘 2，永远到不了 0.5
    assert quantize_multiplier(0.0) == (0, 0)


def test_负数除法是四舍五入不是向下取整():
    """直接写 `>>` 就是向下取整，负数会系统性偏一位——表现成「板上整体比 PC 低一点点」，
    而且两边代码看起来都对。这里钉住 gemmlowp 的规则：四舍五入，.5 远离零。

    -5/4 是关键用例：向下取整给 -2，正确答案是 -1。-3/2 两种规则都给 -2，
    单看它区分不出实现对不对，所以三个用例缺一不可。
    """
    assert int(rounding_divide_by_pot(np.int64(-5), 2)) == -1   # -1.25 → -1（>> 会给 -2）
    assert int(rounding_divide_by_pot(np.int64(3), 1)) == 2     # 1.5 → 2（远离零）
    assert int(rounding_divide_by_pot(np.int64(-3), 1)) == -2   # -1.5 → -2（远离零）


def test_量化后与_float_的判别结果基本一致():
    net = make_net(N_CH, N_CLASSES, seed=3)
    calib = _fake_windows(64, seed=11)
    qnet = quantize(net, calib, class_names=["sleep", "active", "scratch"])

    xs = _fake_windows(128, seed=12)
    f_pred = [int(np.argmax(net.forward(x))) for x in xs]
    q_pred = [int(np.argmax(forward_int(qnet, qnet.quantize_input(x))[0])) for x in xs]
    agree = np.mean([a == b for a, b in zip(f_pred, q_pred)])
    # 随机权重、合成输入下，int8 跟 float 的判别一致率应该很高。定这个门槛不是
    # 为了"通过"，是为了让某天它掉到 0.8 时有人知道
    assert agree >= 0.95, f"一致率只有 {agree:.3f}"


def test_校准集不覆盖剧烈动作时会饱和():
    """把"校准集必须覆盖剧烈动作"这条从注释变成可执行的事实。

    只拿安静片段校准，再喂剧烈片段进去，输入量化就会大面积撞到 ±127——
    这时候模型在最该判对的时候是瞎的，而且**不会报任何错**。
    """
    quiet = _fake_windows(32, seed=21) * 0.05
    loud = _fake_windows(8, seed=22) * 4.0
    net = make_net(N_CH, N_CLASSES, seed=5)

    q_quiet = quantize(net, quiet)
    sat = np.mean([np.mean(np.abs(q_quiet.quantize_input(x)) >= 127) for x in loud])
    assert sat > 0.2, f"没复现出饱和（{sat:.3f}），这个测试就失去意义了"

    q_all = quantize(net, np.concatenate([quiet, loud]))
    sat2 = np.mean([np.mean(np.abs(q_all.quantize_input(x)) >= 127) for x in loud])
    assert sat2 < 0.01, f"校准集覆盖了剧烈动作却还在饱和（{sat2:.3f}）"


def test_arena_够装最大的中间张量():
    net = make_net(N_CH, N_CLASSES, seed=9)
    qnet = quantize(net, _fake_windows(16, seed=31))
    # 手算：输入 6x64=384；conv1 出 8x60=480；pool 8x15=120；
    # conv2 出 16x13=208；pool 16x3=48；dense 3
    assert _arena_bytes(qnet) == 480


@pytest.mark.parametrize("n_t", [16, 32, 64, 128])
def test_不同窗口长度都能导出(n_t):
    """窗口长度是要调的（2 秒 @25Hz = 50 点，4 秒 = 100 点）。改了之后
    pool 两次会不会把时间维压没，得有东西拦着。"""
    if n_t < 32:
        # T→conv(-4)→/4→conv(-2)→/4，太短会把时间维压成 0。要的是在**建网络时**
        # 就说清楚「窗口太短」，而不是等到矩阵乘那里报一个形状不匹配
        with pytest.raises(ValueError, match="太短"):
            make_net(N_CH, N_CLASSES, seed=1, n_t=n_t)
        return
    net = make_net(N_CH, N_CLASSES, seed=1, n_t=n_t)
    xs = np.random.default_rng(0).normal(size=(8, N_CH, n_t)).astype(np.float32)
    qnet = quantize(net, xs)
    out, _ = forward_int(qnet, qnet.quantize_input(xs[0]))
    assert out.shape == (N_CLASSES,)
