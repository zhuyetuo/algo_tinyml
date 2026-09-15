"""带 padding 的卷积：C ↔ Python 逐位对照，外加 BatchNorm 折叠的验证。

为什么单独开一个文件而不是塞进 test_c_consistency.py：那边的网络是 VALID 卷积
（pad=0），而 imu_train 训出来的 cnn 是 `Conv1d(padding=k//2) + BatchNorm1d +
ReLU + MaxPool1d(2)` ×3。**padding 的边界是错得最安静的地方**——中间全对，
只有头尾几个时间步差一点，看起来完全像"数值误差"而不是"实现错了"。
所以要专门用一组会踩到边界的输入去验。

结构照着 imu_train 的 cnn 搭（三段 conv-bn-relu-pool + 一个 Linear），
但通道数收窄成 [8,16,32]：要验的是**结构和边界**，而 Python 侧的参考前向是
纯循环，按真实的 [64,128,256] 跑一遍 golden vector 要几分钟。
真实宽度下的溢出另有测试（test_overflow_at_real_width）。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml import export, forward_int, quantize  # noqa: E402
from tinyml.net import (  # noqa: E402
    Conv1D, Dense, FloatNet, MaxPool1D, fold_batchnorm,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

N_CH, N_T, N_CLASSES = 8, 16, 5   # 真实配置：8 通道、16 点（1 秒 @16Hz）、5 类
WIDTHS = [8, 16, 32]


def _windows(n, seed):
    """有动态范围的假窗口。只为让 scale 不退化，不是数据。"""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / 16.0
    xs = []
    for _ in range(n):
        f, amp = rng.uniform(1.0, 6.0), rng.uniform(0.2, 3.0)
        w = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        x = np.stack([w * rng.uniform(0.3, 1.0) + rng.normal(0, 0.05, N_T)
                      for _ in range(N_CH)])
        x[2] += 9.8
        xs.append(x.astype(np.float32))
    return np.stack(xs)


def _imu_train_shaped_net(seed=0, k=3):
    """照搬 imu_train 的 cnn：conv(pad=k//2) → BN → relu → pool(2)，三段，再一个 Linear。

    BN 在这里就折进 conv —— 端上没有 BN 这个算子，也不需要有。
    """
    rng = np.random.default_rng(seed)
    layers, in_ch, t = [], N_CH, N_T
    for out_ch in WIDTHS:
        w = rng.normal(0, np.sqrt(2.0 / (in_ch * k)),
                       (out_ch, in_ch, k)).astype(np.float32)
        conv = Conv1D(w, rng.normal(0, 0.1, out_ch).astype(np.float32),
                      relu=True, pad=k // 2)
        # BN 的统计量随便给，但 var 必须为正，gamma 要有正有负——
        # gamma 全正的话，折叠里 s 的符号那一路就没被验到
        folded = fold_batchnorm(
            conv,
            gamma=rng.normal(1.0, 0.3, out_ch),
            beta=rng.normal(0.0, 0.2, out_ch),
            mean=rng.normal(0.0, 0.5, out_ch),
            var=rng.uniform(0.5, 2.0, out_ch),
        )
        layers += [folded, MaxPool1D(2)]
        in_ch, t = out_ch, t // 2
    nf = in_ch * t
    layers.append(Dense(rng.normal(0, np.sqrt(2.0 / nf), (N_CLASSES, nf)).astype(np.float32),
                        np.zeros(N_CLASSES, np.float32), relu=False))
    return FloatNet(layers)


# ── padding 本身的语义 ────────────────────────────────────────────────────


def test_same_padding_keeps_length():
    """pad=k//2（k 奇数）应该让时间维长度不变，这是 'same' 的定义。"""
    rng = np.random.default_rng(0)
    for k in (3, 5, 7):
        c = Conv1D(rng.normal(size=(4, 2, k)).astype(np.float32),
                   np.zeros(4, np.float32), relu=False, pad=k // 2)
        assert c.forward(rng.normal(size=(2, 16)).astype(np.float32)).shape == (4, 16)


def test_padding_pads_with_zero_not_edge_replication():
    """补的必须是 0，不是复制边缘。

    用一个只看最左那一拍的核（w = [1,0,0]，pad=1）：输出第 0 位取的是
    padding 那一格。补 0 → 0；复制边缘 → x[0]。给 x[0] 一个显眼的值就分得开。
    """
    w = np.zeros((1, 1, 3), np.float32)
    w[0, 0, 0] = 1.0
    c = Conv1D(w, np.zeros(1, np.float32), relu=False, pad=1)
    x = np.array([[7.0, 1.0, 2.0, 3.0]], np.float32)
    assert float(c.forward(x)[0, 0]) == 0.0      # 补 0；复制边缘会给 7.0


def test_pad_zero_is_unchanged_from_before():
    """pad 默认 0 时行为必须跟以前完全一样，否则这次改动会悄悄动到已有的网络。"""
    rng = np.random.default_rng(3)
    w = rng.normal(size=(4, 2, 3)).astype(np.float32)
    x = rng.normal(size=(2, 20)).astype(np.float32)
    a = Conv1D(w, np.zeros(4, np.float32), relu=False).forward(x)
    b = Conv1D(w, np.zeros(4, np.float32), relu=False, pad=0).forward(x)
    assert np.array_equal(a, b)
    assert a.shape == (4, 18)


# ── BatchNorm 折叠 ────────────────────────────────────────────────────────


def test_fold_batchnorm_matches_explicit_bn():
    """折叠必须跟"卷积之后显式做一遍 BN"逐值相同（浮点容差内）。

    这是恒等变换，不是近似——所以容差给得很紧。松容差会让一个真的写错
    （比如漏了 beta、或者 s 用了 var 而不是 sqrt(var)）也照样通过。
    """
    rng = np.random.default_rng(7)
    oc, ic, k, T = 5, 3, 3, 12
    conv = Conv1D(rng.normal(size=(oc, ic, k)).astype(np.float32),
                  rng.normal(size=oc).astype(np.float32), relu=False, pad=1)
    gamma = rng.normal(1.0, 0.4, oc)
    beta = rng.normal(0.0, 0.3, oc)
    mean = rng.normal(0.0, 0.6, oc)
    var = rng.uniform(0.3, 2.0, oc)
    eps = 1e-5

    x = rng.normal(size=(ic, T)).astype(np.float32)
    y = conv.forward(x).astype(np.float64)
    want = (y - mean[:, None]) / np.sqrt(var[:, None] + eps) * gamma[:, None] + beta[:, None]
    got = fold_batchnorm(conv, gamma, beta, mean, var, eps).forward(x).astype(np.float64)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_fold_batchnorm_applies_relu_after_not_before():
    """relu 必须在 BN 之后。折叠如果把 relu 留在 conv 里先夹一遍，
    负的 gamma 会把结果彻底改掉——而 gamma 全正的测试数据看不出来。"""
    rng = np.random.default_rng(11)
    conv = Conv1D(rng.normal(size=(3, 2, 3)).astype(np.float32),
                  np.zeros(3, np.float32), relu=True, pad=1)
    gamma = np.array([-1.0, 1.0, -0.5])     # 故意带负的
    beta, mean, var = np.zeros(3), np.zeros(3), np.ones(3)
    x = rng.normal(size=(2, 10)).astype(np.float32)

    folded = fold_batchnorm(conv, gamma, beta, mean, var)
    got = folded.forward(x)
    raw = Conv1D(conv.w, conv.b, relu=False, pad=1).forward(x).astype(np.float64)
    want = np.maximum(raw * gamma[:, None] / np.sqrt(1.0 + 1e-5), 0.0)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad", ["gamma", "beta", "mean", "var"])
def test_fold_batchnorm_rejects_length_mismatch(bad):
    """长度对不上要当场炸，不能靠 numpy 广播悄悄"算出来"一个错的网络。"""
    conv = Conv1D(np.zeros((4, 2, 3), np.float32), np.zeros(4, np.float32))
    kw = dict(gamma=np.ones(4), beta=np.zeros(4), mean=np.zeros(4), var=np.ones(4))
    kw[bad] = np.ones(3)
    with pytest.raises(ValueError):
        fold_batchnorm(conv, **kw)


# ── C ↔ Python 逐位对照 ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    net = _imu_train_shaped_net()
    calib = _windows(40, seed=1)
    qnet = quantize(net, calib, class_names=["活动", "睡觉", "抓挠", "未佩戴", "甩身体"])

    cand = _windows(80, seed=2)
    by_cls = {}
    for x in cand:
        xi = qnet.quantize_input(x)
        by_cls.setdefault(int(np.argmax(forward_int(qnet, xi)[0])), []).append(xi)
    picked, i = [], 0
    while len(picked) < 12:
        added = False
        for c in sorted(by_cls):
            if i < len(by_cls[c]) and len(picked) < 12:
                picked.append(by_cls[c][i])
                added = True
        if not added:
            break
        i += 1

    # **额外塞两个专挑边界的输入**：随机窗口的头尾没什么特别，padding 写错
    # 可能只差几个 LSB 就被 requant 吃掉了。一个全是 int8 最大值、一个全是最小值，
    # 能把边界那一格的贡献放到最大。
    picked.append(np.full((N_CH, N_T), 127, np.int8))
    picked.append(np.full((N_CH, N_T), -128, np.int8))
    golden_i8 = np.stack(picked)

    out_dir = tmp_path_factory.mktemp("gen_pad")
    for name, content in export(qnet, golden_x_i8=golden_i8).items():
        (out_dir / name).write_text(content, encoding="utf-8")

    exe = out_dir / "host"
    r = subprocess.run([
        "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fsanitize=undefined,address", "-fno-sanitize-recover=all",
        f"-I{FW}", f"-I{out_dir}",
        os.path.join(FW, "tm_runtime.c"), str(out_dir / "tm_model.c"),
        os.path.join(ROOT, "tests", "host_main.c"), "-o", str(exe),
    ], capture_output=True, text=True)
    assert r.returncode == 0, f"编译失败：\n{r.stderr}"
    return qnet, golden_i8, exe


def test_padded_conv_c_matches_python_bitwise(built):
    qnet, golden_i8, exe = built
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, f"跑失败：\n{r.stderr}"
    got = np.array([[int(v) for v in line.split()]
                    for line in r.stdout.strip().splitlines()], dtype=np.int8)
    want = np.stack([forward_int(qnet, x)[0] for x in golden_i8])
    assert got.shape == want.shape
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"{len(bad)} 处不一致，头几处 (样本, 类别): {bad[:5].tolist()}\n"
        f"C={got[bad[0][0]]}  Python={want[bad[0][0]]}")


def test_asan_catches_no_out_of_bounds(built):
    """上面那个测试是带 ASan 编的（padding 最容易写成读越界）。
    这条只是把这件事写成一个显式的断言，免得有人日后把 sanitizer 关掉。"""
    _, _, exe = built
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert "AddressSanitizer" not in r.stderr
    assert "runtime error" not in r.stderr


def test_time_dim_survives_three_pools(built):
    """16 点连过三次 pool(2) 只剩 2 点。padding 一旦漏掉，第一层就变成 14 点，
    三次之后是 1 点，dense 的输入维度对不上——这个断言把那条路堵死。"""
    qnet, _, _ = built
    t = qnet.n_t
    for lyr in qnet.layers:
        if hasattr(lyr, "pool"):
            t //= lyr.pool
        elif hasattr(lyr, "w") and lyr.w.ndim == 3:
            t = t + 2 * lyr.pad - lyr.w.shape[2] + 1
    assert t == 2, f"时间维走完是 {t}，应该是 2（16 → 8 → 4 → 2）"
