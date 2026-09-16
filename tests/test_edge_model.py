"""EdgeCNN 包装层：轴序、概率、以及"判决必须跟 C 一致"。

这一层只有几十行，但它是**端侧 C 和 imu_train 预处理之间唯一的接缝**，
而接缝正是最容易出错、又最难发现的地方：轴转错了不会报错（形状还是三维的），
只会让效果莫名其妙地差。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml import export, quantize  # noqa: E402
from tinyml.edge_model import EdgeCNN, softmax  # noqa: E402
from tinyml.net import Conv1D, Dense, FloatNet, MaxPool1D  # noqa: E402
import serve  # noqa: E402

N_CH, N_T, N_CLS = 8, 16, 5
CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]


def _windows(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([
        np.stack([rng.normal(0, 3, N_T) if c < 3 else rng.normal(0, 200, N_T)
                  for c in range(N_CH)])
        for _ in range(n)]).astype(np.float32)


@pytest.fixture(scope="module")
def qnet_and_norm(_built):
    return _built[1], _built[2], _built[0]


@pytest.fixture(scope="module")
def eng(_built):
    return _built[0]


@pytest.fixture(scope="module")
def _built(tmp_path_factory):
    rng = np.random.default_rng(0)
    layers, ic, t = [], N_CH, N_T
    for oc in (16, 32, 32):
        layers += [Conv1D(rng.normal(0, np.sqrt(2 / (ic * 3)),
                                     (oc, ic, 3)).astype(np.float32),
                          rng.normal(0, .05, oc).astype(np.float32),
                          relu=True, pad=1), MaxPool1D(2)]
        ic, t = oc, t // 2
    layers.append(Dense(rng.normal(0, np.sqrt(2 / (ic * t)),
                                   (N_CLS, ic * t)).astype(np.float32),
                        np.zeros(N_CLS, np.float32)))
    X = _windows(200, 1)
    meta = {"ch_mean": [float(X[:, c].mean()) for c in range(N_CH)],
            "ch_std": [float(X[:, c].std()) for c in range(N_CH)]}
    mean = np.asarray(meta["ch_mean"], np.float32).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float32).reshape(-1, 1)
    Xn = ((X - mean) / std).astype(np.float32)
    q = quantize(FloatNet(layers), Xn[:64], class_names=CLASSES)
    gen = tmp_path_factory.mktemp("gen_edge")
    for name, content in export(q, golden_x_i8=np.stack(
            [q.quantize_input(x) for x in Xn[:4]]), prep=meta).items():
        (gen / name).write_text(content, encoding="utf-8")
    return (serve.Engine(serve.build(str(gen), out_so=str(gen / "e.so"))),
            FloatNet(layers), meta)


# ── softmax ───────────────────────────────────────────────────────────────


def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], 1 / 3)


def test_softmax_survives_huge_logits():
    """减最大值不是可选的：logit 到 ±800 直接 overflow 成 inf/nan，
    而 nan 传到 argmax 会安静地变成"总是第 0 类"。"""
    p = softmax(np.array([[800.0, 799.0, -800.0]]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(), 1.0)
    assert int(np.argmax(p)) == 0


def test_softmax_is_order_preserving():
    """顺序不变是这层能用的前提——概率只影响"显示成多少分"，不影响判决。"""
    z = np.array([[0.3, -1.2, 2.5, 0.0, 1.1]])
    assert np.array_equal(np.argsort(z[0]), np.argsort(softmax(z)[0]))


# ── 轴序：这层唯一真正危险的地方 ──────────────────────────────────────────


def test_predict_proba_takes_time_first_and_matches_c(eng):
    """imu_train 给的是 [N, T, C]，C 要的是 [N, C, T]。
    转错了不会报错，只会让效果莫名其妙地差。"""
    m = EdgeCNN(eng, CLASSES)
    X = _windows(30, seed=5)                       # [N, C, T]
    p = m.predict_proba(X.transpose(0, 2, 1))      # 按 imu_train 的约定喂 [N, T, C]
    want_cls, _ = eng.infer(X)
    assert np.array_equal(np.argmax(p, axis=1), want_cls), \
        "包装之后的判决跟直接调 C 不一致——多半是轴转反了"


def test_argmax_matches_c_exactly_not_just_approximately(eng):
    """**判决必须跟 C 逐条相同**，不是"大致一样"。

    还原 logit 再 softmax 是仿射 + 单调变换，所以保序。
    注意 out_zp 对结果**完全没有影响**——它给所有类别加同一个常数，
    而 softmax 对平移不变。（这一条是变异测试发现的：把 out_zp 的符号
    写反，所有测试照样绿，因为那确实是个恒等变换。）
    真正会影响输出的是 out_scale，见 test_confidence_matches_the_float_model。
    """
    m = EdgeCNN(eng, CLASSES)
    X = _windows(200, seed=9)
    got = m.predict(X.transpose(0, 2, 1))
    want, _ = eng.infer(X)
    bad = np.flatnonzero(got != want)
    assert not len(bad), f"{len(bad)} 条判决不一致，头几条 {bad[:5].tolist()}"


def test_confidence_matches_the_float_model(eng, qnet_and_norm):
    """置信度要跟 float 模型给的概率对得上——这是唯一盯着 **out_scale** 的测试。

    out_scale 错一倍，argmax 一点不变（仿射保序），但置信度会从 0.61 变成 0.85。
    平台拿这个数当置信度显示、甚至用来过滤片段，所以它错了是真会出事的，
    而且不会有任何报错。

    比的是"跟 float 模型的概率接近"，不是"跟自己算的一样"——后者是循环论证。
    """
    m = EdgeCNN(eng, CLASSES)
    net, meta, _ = qnet_and_norm
    X = _windows(120, seed=21)
    mean = np.asarray(meta["ch_mean"], np.float32).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float32).reshape(-1, 1)
    Xn = ((X - mean) / std).astype(np.float32)

    got = m.predict_proba(X.transpose(0, 2, 1))
    want = softmax(np.stack([net.forward(x) for x in Xn]))
    mad = float(np.mean(np.abs(got - want)))
    # **阈值是实测出来的，不是拍的。** 正确时约 0.003（纯量化误差）；
    # out_scale 错一倍是 0.050、错一半是 0.073。
    # 第一版我拍了 0.05，刚好卡在"翻倍"的外面——测试是绿的，但完全靠运气，
    # 稍微换个模型就分不开了。0.01 留了 3 倍余量，同时离最近的错误情形还有 5 倍。
    assert mad < 0.01, (
        f"端侧概率跟 float 模型差太多（平均 {mad:.4f}，正常约 0.003）——"
        "多半是 out_scale 没用对，或者反量化那一步错了")


def test_probabilities_are_proper(eng):
    m = EdgeCNN(eng, CLASSES)
    p = m.predict_proba(_windows(40, seed=3).transpose(0, 2, 1))
    assert p.shape == (40, N_CLS)
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.0)


def test_confidence_tracks_the_winning_class(eng):
    """conf_max 取的是 max(prob)，所以最大概率必须对应 argmax 那一类。"""
    m = EdgeCNN(eng, CLASSES)
    X = _windows(50, seed=4).transpose(0, 2, 1)
    p = m.predict_proba(X)
    assert np.array_equal(np.argmax(p, axis=1), m.predict(X))


# ── 拦住配不上的输入 ──────────────────────────────────────────────────────


def test_wrong_window_length_is_rejected(eng):
    """窗口长度跟训练时不一致要当场报错。numpy 不会拦，C 会读越界。"""
    m = EdgeCNN(eng, CLASSES)
    with pytest.raises(ValueError, match="点"):
        m.predict_proba(np.zeros((3, N_T * 2, N_CH), np.float32))


def test_wrong_channel_count_is_rejected(eng):
    """6 通道喂给 8 通道的模型——少了 pitch/roll，这是最容易犯的一个错
    （infer_csv_scratch 里 data6 就是 6 通道的）。"""
    m = EdgeCNN(eng, CLASSES)
    with pytest.raises(ValueError, match="通道"):
        m.predict_proba(np.zeros((3, N_T, 6), np.float32))


def test_class_count_mismatch_is_rejected(eng):
    """.json 和 .pt 不配套时必须在构造时就炸，不能等到跑完一整天数据
    才发现类别名对不上下标。"""
    with pytest.raises(ValueError, match="类别数"):
        EdgeCNN(eng, ["活动", "睡觉", "抓挠"])


def test_empty_batch_returns_empty_not_crash(eng):
    """一整个 CSV 全是缺失时窗口数会是 0。返回空数组，别崩。"""
    m = EdgeCNN(eng, CLASSES)
    p = m.predict_proba(np.zeros((0, N_T, N_CH), np.float32))
    assert p.shape == (0, N_CLS)


def test_non_three_dim_input_is_rejected(eng):
    m = EdgeCNN(eng, CLASSES)
    with pytest.raises(ValueError, match=r"\[N, T, C\]"):
        m.predict_proba(np.zeros((N_T, N_CH), np.float32))


def test_is_dl_flag_is_set(eng):
    """imu_train 靠这个字段决定走不走手工特征。端侧 CNN 吃原始窗口，
    这个字段错了会去算 193 维特征然后喂给 CNN——形状对不上，但错得很远。"""
    assert EdgeCNN(eng, CLASSES).is_dl is True
