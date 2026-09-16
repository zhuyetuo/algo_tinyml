"""挂一个服务器上的 sklearn 模型（kind: sk）。

这条路线是给"导出成 C 之前先看看效果值不值得导"用的。典型场景：训了一版
只用加速计的模型，想先在平台上跑真实样本看看效果掉多少。

这里盯的全是**错了不报错**的事：
  · is_dl 写反 —— C 那两条路线要原始窗口，sklearn 要特征。写反之后
    C 那条照样能跑，只是算的是另一套特征（形状正好也是 193）
  · optional 被误加到正式模型上 —— 路径写错时服务照起，平台上少一个模型
  · 类别数跟 meta 对不上 —— predict_proba 的列跟 classes 错位，
    平台上每条都标着错的行为名，而数值完全正常
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml import sk_model  # noqa: E402

CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]


class _FakeForest:
    """够 SkModel 用的最小 sklearn 接口。不装 sklearn 也能跑这些测试。"""

    def __init__(self, n_features=193, n_classes=5):
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes

    def predict_proba(self, X):
        import numpy as np
        n = len(X)
        return np.full((n, self.n_classes_), 1.0 / self.n_classes_)


# ── SkModel 本身 ──────────────────────────────────────────────────────────


def test_feature_dim_mismatch_says_which_model_is_wrong():
    """57 维模型挂到 193 维的链上，要说清是模型不对，不是特征算多了。

    sklearn 原话是 "X has 193 features, but ... expecting 57"，
    单看这句人会去查特征提取——而真正的原因是这个 pkl 不是这套特征训的。
    """
    import numpy as np
    m = sk_model.SkModel(_FakeForest(n_features=57), CLASSES, path="acc3.pkl")
    with pytest.raises(ValueError) as e:
        m.predict_proba(np.zeros((4, 193), np.float32))
    msg = str(e.value)
    assert "57" in msg and "193" in msg
    assert "acc_only" in msg, "没告诉人正确的做法是哪条路"


def test_matching_dims_pass_through():
    import numpy as np
    m = sk_model.SkModel(_FakeForest(), CLASSES)
    out = m.predict_proba(np.zeros((3, 193), np.float32))
    assert out.shape == (3, 5)


def test_class_count_mismatch_is_rejected_at_load():
    """类别数对不上**必须在加载时就炸**。

    放过去的话 predict_proba 的列跟 classes 错位，平台上每一条都标着
    错的行为名——而概率值本身完全正常，没有任何迹象。
    """
    with pytest.raises(ValueError) as e:
        sk_model.SkModel(_FakeForest(n_classes=3), CLASSES, path="x.pkl")
    assert "3" in str(e.value) and "5" in str(e.value)


def test_is_dl_is_false():
    """sklearn 吃的是**特征**，不是原始窗口。

    写成 True 的话 infer_file 会把 [N, T, 8] 的窗口直接喂进来。
    """
    assert sk_model.SkModel(_FakeForest(), CLASSES).is_dl is False


# ── EdgeRunner 传给 infer_file 的 is_dl ───────────────────────────────────
#
# 上面那条只验了 SkModel 自己的字段，**真正决定行为的是 EdgeRunner 的**。
# 只验前者的话，把 EdgeRunner 里写死成 True 测试照样全绿——变异测试里
# 这个变异活下来了，所以补这一组。


class _FakeEngine:
    n_ch, n_t, n_classes, n_features = 8, 16, 5, 193

    @property
    def classes(self):
        return CLASSES


def _runner(edge_service, kind):
    meta = {"classes": CLASSES, "window_size": 16, "hz": 16, "stride": 8,
            "label_mode": "majority", "gravity_aligned": True}
    engine = (sk_model.SkModel(_FakeForest(), CLASSES) if kind == "sk"
              else _FakeEngine())
    return edge_service.EdgeRunner("t", engine, meta, "/nonexistent", kind=kind)


def test_runner_is_dl_false_only_for_sklearn(edge_service):
    """C 那两条要**原始窗口**（特征在 C 里算），sklearn 那条要**特征**。

    写反的方向决定后果：
      sk 写成 True  → sklearn 收到 [N, T, 8]，当场抱怨维度，还算好查；
      C  写成 False → infer_file 先用 scipy 算一遍 193 维特征再喂进去，
                      **形状正好对得上，不报错**，但算的是另一套。
    """
    assert _runner(edge_service, "sk").is_dl is False
    assert _runner(edge_service, "rf").is_dl is True
    assert _runner(edge_service, "cnn").is_dl is True


def test_runner_uses_the_sklearn_model_as_is(edge_service):
    """sk 这条不能再套一层 EdgeCNN/EdgeRF——那两个会把窗口当输入。"""
    r = _runner(edge_service, "sk")
    assert isinstance(r.model, sk_model.SkModel)


def test_load_rejects_a_directory():
    """sk 的 gen 要指到 .pkl 文件；给目录是把 C 那条路线的写法照搬过来了。"""
    with pytest.raises(ValueError) as e:
        sk_model.load(os.path.dirname(__file__), CLASSES)
    assert ".pkl" in str(e.value)


# ── 模型清单里的 sk / optional ────────────────────────────────────────────


def _cfg(tmp_path, models):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"models": models}, ensure_ascii=False), encoding="utf-8")
    return str(p)


@pytest.fixture
def edge_service():
    import edge_service
    return edge_service


def test_optional_model_missing_is_skipped_not_fatal(tmp_path, edge_service, capsys):
    """实验模型还没训出来时，服务照起，其余模型不受影响。"""
    real = tmp_path / "ml_rf.pkl"
    real.write_bytes(b"x")
    meta = tmp_path / "ml_rf.json"
    meta.write_text("{}", encoding="utf-8")
    cfg = _cfg(tmp_path, [
        {"tag": "ok", "kind": "sk", "gen": str(real), "meta": str(meta)},
        {"tag": "not_trained_yet", "kind": "sk", "optional": True,
         "gen": str(tmp_path / "nope.pkl"), "meta": str(tmp_path / "nope.json")},
    ])
    out = edge_service.load_models_config(cfg)
    assert [m["tag"] for m in out] == ["ok"]
    # **要吵**：安静跳过的话平台上少一个选项，而日志里一切正常
    assert "not_trained_yet" in capsys.readouterr().err


def test_missing_model_without_optional_is_fatal(tmp_path, edge_service):
    """正式模型路径写错**必须当场报错**。

    跳过的话服务照常起来，只是少了一个模型——平台上那个版本"不存在"，
    而没有任何人会注意到。

    **旁边必须放一个挂得上的模型。** 只放这一个坏的话，就算 optional 被
    误写成"永远生效"，结果也是"一个都挂不上"那条 SystemExit——测试照样绿，
    而实际行为已经错了。第一版就是这么写的，变异测试里活下来了。
    """
    good = tmp_path / "ml_rf.pkl"
    good.write_bytes(b"x")
    cfg = _cfg(tmp_path, [
        {"tag": "ok", "kind": "sk", "gen": str(good), "meta": str(good)},
        {"tag": "prod", "kind": "sk",
         "gen": str(tmp_path / "nope.pkl"), "meta": str(tmp_path / "nope.json")},
    ])
    with pytest.raises(SystemExit) as e:
        edge_service.load_models_config(cfg)
    assert "nope" in str(e.value), (
        "退出原因不是'这个模型找不到'——可能是被当成 optional 跳过后"
        "走到了别的分支")


def test_all_optional_and_all_missing_is_fatal(tmp_path, edge_service):
    """一个都挂不上时别假装起来了——那样平台上是个空的下拉。"""
    cfg = _cfg(tmp_path, [
        {"tag": "a", "kind": "sk", "optional": True,
         "gen": str(tmp_path / "no.pkl"), "meta": str(tmp_path / "no.json")},
    ])
    with pytest.raises(SystemExit):
        edge_service.load_models_config(cfg)


def test_unknown_kind_is_rejected(tmp_path, edge_service):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    cfg = _cfg(tmp_path, [{"tag": "t", "kind": "onnx",
                           "gen": str(f), "meta": str(f)}])
    with pytest.raises(SystemExit):
        edge_service.load_models_config(cfg)


def test_sk_kind_is_not_guessed_from_the_tag(tmp_path, edge_service):
    """标签里带 rf 的 sklearn 模型不写 kind，会被猜成 C 那条 rf 路线。

    这里钉的是"猜"的行为本身没变（老配置照旧），**所以 sk 必须显式写**。
    acc_only_rf 正是这种标签，写漏了会报"找不到 tm_forest_model.c"，
    而那个错误跟"忘了写 kind"看不出关系。
    """
    f = tmp_path / "f"
    f.write_bytes(b"x")
    cfg = _cfg(tmp_path, [{"tag": "acc_only_rf", "gen": str(f), "meta": str(f)}])
    assert edge_service.load_models_config(cfg)[0]["kind"] == "rf"


# ── 仓库里那份清单 ────────────────────────────────────────────────────────


def test_repo_config_marks_the_experimental_model_optional():
    """acc_only_rf 是实验模型，没训的机器上不该起不来服务。"""
    p = os.path.join(os.path.dirname(__file__), "..", "edge_models.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    by = {m["tag"]: m for m in cfg["models"] if isinstance(m, dict) and "tag" in m}
    assert "acc_only_rf" in by, "edge_models.json 里没有 acc_only_rf"
    m = by["acc_only_rf"]
    assert m.get("optional") is True
    assert m.get("kind") == "sk", "不写 kind 会被 tag 里的 rf 猜成 C 那条路线"
    # 正式模型**不能**是 optional：路径写错时必须当场报错
    for tag in ("edge_cnn_i8", "edge_rf_d10"):
        assert not by[tag].get("optional"), f"{tag} 是正式模型，不该标 optional"
