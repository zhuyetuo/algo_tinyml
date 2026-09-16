"""imu_train 的 cnn checkpoint → FloatNet 的解析。

这里**不需要 torch**：load_cnn 只在真的读 .pt 时才 import torch，
而键名解析、BN 折叠、形状自检这几件事才是会出错的地方，用一个假的
state_dict（普通 dict + numpy 数组）就能全部验到。

真正危险的失败是"解析错了但不报错"：少认一层、把 BN 当成卷积、
或者 fc 的输入维度对不上却一路走到导出。所以下面每一条都在测一种**安静的**错法。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml.net import Conv1D, Dense, MaxPool1D  # noqa: E402
from tinyml.torch_import import load_meta, normalize  # noqa: E402
import tinyml.torch_import as ti  # noqa: E402

N_CH, N_T, N_CLS = 8, 16, 5
FILTERS = [8, 16, 32]


def _fake_state_dict(seed=0, filters=FILTERS, k=3):
    """照着 nn.Sequential 的下标编号：每段 5 层，所以卷积在 0, 5, 10。"""
    rng = np.random.default_rng(seed)
    sd, in_ch, i = {}, N_CH, 0
    for out_ch in filters:
        sd[f"conv.{i}.weight"] = rng.normal(size=(out_ch, in_ch, k)).astype(np.float32)
        sd[f"conv.{i}.bias"] = rng.normal(size=out_ch).astype(np.float32)
        sd[f"conv.{i + 1}.weight"] = rng.uniform(0.5, 1.5, out_ch).astype(np.float32)
        sd[f"conv.{i + 1}.bias"] = rng.normal(size=out_ch).astype(np.float32)
        sd[f"conv.{i + 1}.running_mean"] = rng.normal(size=out_ch).astype(np.float32)
        sd[f"conv.{i + 1}.running_var"] = rng.uniform(0.5, 2.0, out_ch).astype(np.float32)
        in_ch = out_ch
        i += 5
    nf = in_ch * (N_T // 2 ** len(filters))
    sd["fc.weight"] = rng.normal(size=(N_CLS, nf)).astype(np.float32)
    sd["fc.bias"] = rng.normal(size=N_CLS).astype(np.float32)
    return sd


def _meta(**over):
    m = {
        # hz 也是必填：两条路线都要用它算特征/重采样。
        # imu_train 的 dl_*.json 和 ml_*.json 里都有
        "model": "cnn", "window_size": N_T, "n_channels": N_CH, "hz": 16,
        "classes": ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"],
        "ch_mean": [0.1] * N_CH, "ch_std": [2.0] * N_CH,
    }
    m.update(over)
    return m


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """把 torch.load 换成返回假 state_dict，这样不用装 torch 也能测解析。"""
    def _load(path, **kw):
        return _fake_state_dict()
    fake = type("T", (), {"load": staticmethod(_load)})
    monkeypatch.setitem(sys.modules, "torch", fake)

    def make(meta_over=None, sd=None):
        mp = tmp_path / "m.json"
        mp.write_text(json.dumps(_meta(**(meta_over or {})), ensure_ascii=False),
                      encoding="utf-8")
        if sd is not None:
            monkeypatch.setitem(sys.modules, "torch",
                                type("T", (), {"load": staticmethod(lambda p, **k: sd)}))
        return str(tmp_path / "m.pt"), str(mp)
    return make


# ── 结构解析 ──────────────────────────────────────────────────────────────


def test_parses_all_three_conv_blocks(patched):
    pt, js = patched()
    net, meta = ti.load_cnn(pt, js)
    convs = [l for l in net.layers if isinstance(l, Conv1D)]
    pools = [l for l in net.layers if isinstance(l, MaxPool1D)]
    dense = [l for l in net.layers if isinstance(l, Dense)]
    assert len(convs) == 3, "三段卷积要全认出来；少认一段不会报错，只会算出别的东西"
    assert len(pools) == 3 and len(dense) == 1
    assert [c.w.shape[0] for c in convs] == FILTERS


def test_conv_uses_same_padding(patched):
    pt, js = patched()
    net, _ = ti.load_cnn(pt, js)
    for c in (l for l in net.layers if isinstance(l, Conv1D)):
        assert c.pad == c.w.shape[2] // 2, "padding 必须是 k//2，跟 imu_train 的 cnn.py 一致"


def test_dropout_is_dropped_not_kept_as_a_layer(patched):
    """Dropout 推理期是恒等。要是被当成一层留下来，层数就对不上。"""
    pt, js = patched()
    net, _ = ti.load_cnn(pt, js)
    assert len(net.layers) == 3 * 2 + 1   # (conv + pool) × 3 + dense


def test_time_dim_ends_at_two(patched):
    """16 → 8 → 4 → 2。padding 漏掉的话会变成 1，dense 就对不上了。"""
    pt, js = patched()
    net, meta = ti.load_cnn(pt, js)
    out = net.forward(np.zeros((N_CH, N_T), np.float32))
    assert out.shape == (N_CLS,)


# ── 安静的错法，每一条都要炸 ───────────────────────────────────────────────


def test_rejects_non_cnn_model(patched):
    """cnn_lstm 有 LSTM，端上没有对应算子。"尽力解析"会安静地算错。"""
    pt, js = patched({"model": "cnn_lstm"})
    with pytest.raises(ValueError, match="cnn"):
        ti.load_cnn(pt, js)


def test_rejects_missing_batchnorm(patched):
    sd = _fake_state_dict()
    del sd["conv.1.running_var"]
    pt, js = patched(sd=sd)
    with pytest.raises(ValueError, match="BatchNorm"):
        ti.load_cnn(pt, js)


def test_rejects_shape_mismatch_between_meta_and_weights(patched):
    """window_size 填错 → 展平维度跟 fc 对不上。必须在这里炸，
    不能等到导出到板上才发现 dense 输入维度不对。"""
    pt, js = patched({"window_size": 32})
    with pytest.raises((ValueError, Exception)):
        ti.load_cnn(pt, js)


def test_rejects_class_count_mismatch(patched):
    """类别数对不上必须炸——而这**只有形状自检拦得住**。

    上面那条 window_size 的用例其实验不到自检：维度不对时 Dense 的矩阵乘
    自己就抛了，把自检整段删掉测试照样绿（变异测试发现的）。
    类别数不同则不一样：前向能正常跑完，只是输出 5 个 logit 而 .json 说有 3 类，
    不拦的话会一路走到导出，板上安静地多出两个类。
    """
    pt, js = patched({"classes": ["活动", "睡觉", "抓挠"]})   # 权重是 5 类
    with pytest.raises(ValueError, match="类别数"):
        ti.load_cnn(pt, js)


@pytest.mark.parametrize("missing", ["classes", "ch_mean", "ch_std", "window_size", "hz"])
def test_meta_requires_every_field_inference_needs(tmp_path, missing):
    m = _meta()
    del m[missing]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=missing):
        load_meta(str(p))


def test_meta_rejects_zero_std(tmp_path):
    """std=0 会让归一化给出 inf/nan，而 nan 传到 argmax 会安静地变成"总是第 0 类"。"""
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_meta(ch_std=[2.0] * (N_CH - 1) + [0.0])), encoding="utf-8")
    with pytest.raises(ValueError, match="ch_std"):
        load_meta(str(p))


# ── 归一化 ────────────────────────────────────────────────────────────────


def test_normalize_is_per_channel_not_global(tmp_path):
    """逐通道，不是全局。用不同的 mean/std 才分得开——全一样的话
    一个写成全局的实现也能通过。"""
    meta = _meta(ch_mean=list(range(N_CH)), ch_std=[float(i + 1) for i in range(N_CH)])
    x = np.ones((N_CH, N_T), np.float32) * 10.0
    got = normalize(x, meta)
    for c in range(N_CH):
        assert np.allclose(got[c], (10.0 - c) / (c + 1), atol=1e-5)


# ── 两条路线要的字段不一样 ────────────────────────────────────────────────


def test_rf_meta_does_not_require_normalisation_params(tmp_path):
    """**RF 不需要 ch_mean/ch_std。**

    森林吃的是 193 维手工特征，特征从原始量纲的窗口算，阈值就是按那些
    原始特征值训的——不做任何归一化。给它 ch_mean 反而是错的。

    这一条是补一个真实的错：我一开始用同一个 load_meta 去读 RF 的
    ml_rf.json，报"缺 ch_mean"。那不是 json 的问题，是我把 CNN 的加载器
    套到了 RF 上。
    """
    m = {"classes": ["活动", "抓挠"], "window_size": 16, "hz": 16,
         "stride": 8, "gravity_aligned": True, "label_mode": "majority"}
    p = tmp_path / "ml_rf.json"
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    got = load_meta(str(p), kind="rf")          # 不该抛
    assert got["classes"] == ["活动", "抓挠"]

    # 同一份 json 按 cnn 读必须报错——两者不能混着用
    with pytest.raises(ValueError, match="ch_mean"):
        load_meta(str(p), kind="cnn")


def test_rf_meta_still_requires_the_shared_fields(tmp_path):
    """类别/窗口/采样率这三样两条路线都要，缺了照样报错。"""
    for missing in ("classes", "window_size", "hz"):
        m = {"classes": ["a"], "window_size": 16, "hz": 16}
        del m[missing]
        p = tmp_path / f"no_{missing}.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(ValueError, match=missing):
            load_meta(str(p), kind="rf")


def test_unknown_kind_is_refused(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="只支持"):
        load_meta(str(p), kind="gbdt")
