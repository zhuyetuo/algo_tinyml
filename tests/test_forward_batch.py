"""批量前向必须跟单条前向**逐位相同**。

批量化是为了能在真实规模上跑完（23712 条 × [64,128,256] 的网络，单条版本要几分钟，
而第一版纯循环要几十小时）。但"快"如果以"结果不同"为代价就毫无意义——
板上跑的是单条那一套语义，评估报的却是批量算出来的数，两者一旦分家，
你看到的准确率就不是板子的准确率。

能逐位相同的依据：整数加法满足结合律，把 N 条摊进同一个矩阵乘不改变任何结果。
浮点不是这样，所以 float 那一侧不做同样的承诺（见 test_float_forward_shape）。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml import forward_int, forward_int_batch, quantize  # noqa: E402
from tinyml.net import Conv1D, Dense, FloatNet, MaxPool1D, im2col  # noqa: E402


def _net(n_ch, n_t, n_cls, filters, seed=0, pad=True, k=3):
    rng = np.random.default_rng(seed)
    layers, in_ch, t = [], n_ch, n_t
    for oc in filters:
        w = rng.normal(0, np.sqrt(2.0 / (in_ch * k)), (oc, in_ch, k)).astype(np.float32)
        layers += [Conv1D(w, rng.normal(0, 0.1, oc).astype(np.float32),
                          relu=True, pad=k // 2 if pad else 0),
                   MaxPool1D(2)]
        in_ch = oc
        t = (t + (2 * (k // 2) if pad else 0) - k + 1) // 2
    nf = in_ch * t
    layers.append(Dense(rng.normal(0, np.sqrt(2.0 / nf), (n_cls, nf)).astype(np.float32),
                        np.zeros(n_cls, np.float32), relu=False))
    return FloatNet(layers)


def _calib(n, n_ch, n_t, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, (n, n_ch, n_t)).astype(np.float32)
    x[:, 2] += 9.8
    return x


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("filters", [[8, 16], [8, 16, 32]])
def test_batch_matches_single_bitwise(pad, filters):
    # 不补零时每层要吃掉 k-1 个点，三层下 16 点会被压到 0——那是窗口太短，
    # 不是实现的问题，所以这里给 VALID 的情形一个够长的窗口
    n_ch, n_cls = 8, 5
    n_t = 16 if pad else 48
    net = _net(n_ch, n_t, n_cls, filters, pad=pad)
    calib = _calib(40, n_ch, n_t)
    q = quantize(net, calib)

    X = np.stack([q.quantize_input(x) for x in _calib(37, n_ch, n_t, seed=9)])
    one = np.stack([forward_int(q, x)[0] for x in X])
    many = forward_int_batch(q, X)
    assert many.shape == one.shape
    bad = np.argwhere(many != one)
    assert not len(bad), (f"{len(bad)} 处不一致，头几处 {bad[:5].tolist()}\n"
                          f"批量={many[bad[0][0]]}  单条={one[bad[0][0]]}")


def test_batch_matches_single_on_saturating_inputs():
    """全 127 / 全 -128 这种会把累加器推到极限的输入，最容易暴露
    reshape/transpose 写反——随机输入下写反也常常"看起来对"。"""
    n_ch, n_t, n_cls = 8, 16, 5
    net = _net(n_ch, n_t, n_cls, [8, 16, 32])
    q = quantize(net, _calib(40, n_ch, n_t))
    X = np.stack([np.full((n_ch, n_t), 127, np.int8),
                  np.full((n_ch, n_t), -128, np.int8),
                  np.zeros((n_ch, n_t), np.int8)])
    assert np.array_equal(forward_int_batch(q, X),
                          np.stack([forward_int(q, x)[0] for x in X]))


def test_batch_of_one_is_not_a_special_case():
    """N=1 要跟单条一样。批量代码里 reshape(oc, n, t) 在 n=1 时最容易蒙混过关，
    所以单独钉一条。"""
    net = _net(8, 16, 5, [8, 16, 32])
    q = quantize(net, _calib(40, 8, 16))
    x = q.quantize_input(_calib(1, 8, 16, seed=5)[0])
    assert np.array_equal(forward_int_batch(q, x[None]), forward_int(q, x)[0][None])


def test_batch_preserves_sample_order():
    """结果的第 i 行必须是第 i 条样本。顺序错了的话整体分布一模一样、
    每一条都算对了，但每个样本配错了标签——准确率会莫名其妙地掉到随机水平。"""
    net = _net(8, 16, 5, [8, 16, 32])
    q = quantize(net, _calib(40, 8, 16))
    X = np.stack([q.quantize_input(x) for x in _calib(12, 8, 16, seed=3)])
    many = forward_int_batch(q, X)
    perm = np.array([7, 0, 3, 11, 5, 2, 9, 1, 8, 4, 10, 6])
    assert np.array_equal(forward_int_batch(q, X[perm]), many[perm])


def test_batch_rejects_unbatched_input():
    """给了 [C, T] 而不是 [N, C, T] 要当场报错。numpy 会很乐意把它当成
    N=C 的一批来广播，然后安静地算出一堆没有意义的数。"""
    net = _net(8, 16, 5, [8, 16])
    q = quantize(net, _calib(40, 8, 16))
    # **必须匹配消息**：不匹配的话这条测试是靠下游的 im2col 也抛 ValueError
    # 通过的，把入口这个校验整段删掉它照样绿（变异测试发现的）。
    with pytest.raises(ValueError, match=r"\[N, C, T\]"):
        forward_int_batch(q, np.zeros((8, 16), np.int8))


# ── im2col 本身 ───────────────────────────────────────────────────────────


def test_im2col_layout_matches_weight_reshape():
    """w[o,c,j] 必须对上 cols[c*k+j, t]。摊平顺序反了的话形状完全正确、
    结果全错——这是整个改动里最危险的一处。"""
    ic, k, T = 3, 3, 8
    x = np.arange(ic * T, dtype=np.float32).reshape(ic, T)
    cols = im2col(x, k, pad=0)
    t_out = T - k + 1
    assert cols.shape == (ic * k, t_out)
    for c in range(ic):
        for j in range(k):
            assert np.array_equal(cols[c * k + j], x[c, j:j + t_out])


def test_im2col_equals_naive_triple_loop():
    """跟原来那份三重循环对答案。改写成矩阵乘之后，原来的实现就是唯一的口径。"""
    rng = np.random.default_rng(0)
    oc, ic, k, T, pad = 5, 3, 3, 12, 1
    w = rng.normal(size=(oc, ic, k)).astype(np.float32)
    b = rng.normal(size=oc).astype(np.float32)
    x = rng.normal(size=(ic, T)).astype(np.float32)

    xp = np.pad(x, ((0, 0), (pad, pad)))
    t_out = xp.shape[1] - k + 1
    want = np.empty((oc, t_out), np.float32)
    for o in range(oc):
        a = np.full(t_out, b[o], np.float32)
        for c in range(ic):
            for j in range(k):
                a += w[o, c, j] * xp[c, j:j + t_out]
        want[o] = a
    got = Conv1D(w, b, relu=False, pad=pad).forward(x)
    # float32 下换累加顺序末位会差，所以给容差；整数那边不给（见上面的逐位测试）
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_im2col_raises_when_window_shorter_than_kernel():
    with pytest.raises(ValueError, match="比卷积核"):
        im2col(np.zeros((2, 2), np.float32), k=5, pad=0)
