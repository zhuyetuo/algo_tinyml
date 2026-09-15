"""--model 路径解析的报错。

这几个脚本在 algo_tinyml 目录下跑，而模型在 imu_train 里——相对路径会解析到
algo_tinyml 下面去（我自己就写错过一次）。直接交给 joblib 只会甩一个
FileNotFoundError，看不出是"路径写错了"还是"模型没训出来"。

这段是纯路径逻辑，不依赖 sklearn，所以能在这里测。
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCRIPTS = ["export_gbdt", "export_rf", "prune_rf", "rf_footprint", "feature_usage"]


def _load(name):
    path = os.path.join(ROOT, "python", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_t_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", SCRIPTS)
def test_每个脚本都有路径检查(name):
    mod = _load(name)
    assert hasattr(mod, "resolve_model"), f"{name}.py 少了 resolve_model"


@pytest.mark.parametrize("name", SCRIPTS)
def test_存在的路径原样返回(name, tmp_path):
    mod = _load(name)
    p = tmp_path / "m.pkl"
    p.write_bytes(b"x")
    assert mod.resolve_model(str(p)) == str(p)


@pytest.mark.parametrize("name", SCRIPTS)
def test_不存在时报错里带绝对路径(name):
    """报错必须把相对路径解析成什么说出来——只说"找不到 results/xxx"的话，
    人还是不知道它到底找到哪儿去了。"""
    mod = _load(name)
    with pytest.raises(SystemExit) as e:
        mod.resolve_model("results/nope/x.pkl")
    msg = str(e.value)
    assert "results/nope/x.pkl" in msg
    assert os.path.abspath("results/nope/x.pkl") in msg, "报错里没说清解析成了哪个绝对路径"
    assert "imu_train" in msg, "报错里没提示模型应该在 imu_train 下"


@pytest.mark.parametrize("name", SCRIPTS)
def test_波浪号会展开(name, tmp_path, monkeypatch):
    mod = _load(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m.pkl").write_bytes(b"x")
    assert mod.resolve_model("~/m.pkl") == str(tmp_path / "m.pkl")


def test_prune_gbdt_也有路径检查():
    mod = _load("prune_gbdt")
    with pytest.raises(SystemExit) as e:
        mod.resolve_model("results/nope/x.pkl")
    assert os.path.abspath("results/nope/x.pkl") in str(e.value)


def test_prf_和_macro_f1_算得对():
    """指标算错的话整张剪枝表就是错的，而它看起来完全正常。"""
    mod = _load("prune_gbdt")
    import numpy as np
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0])
    # 类别 1：tp=2（下标2,3），fp=1（下标1），fn=0 → P=2/3, R=1.0
    pr, rc, f1 = mod.prf(y, p, 1)
    assert pr == pytest.approx(2 / 3) and rc == pytest.approx(1.0)
    assert f1 == pytest.approx(2 * (2 / 3) / (2 / 3 + 1))
    # 类别 2 一个都没预测对 → F1=0，不该是 nan
    assert mod.prf(y, p, 2)[2] == 0.0
    assert not np.isnan(mod.macro_f1(y, p, 3))


@pytest.mark.parametrize("name", ["prune_gbdt", "prune_rf"])
def test_features_路径错时告诉人怎么生成(name):
    """--features 不是训练的产物，得先用 dump_holdout.py 导一次。
    只说"找不到"的话，人不知道这文件从哪来——我第一次就是这么让人卡住的。"""
    mod = _load(name)
    with pytest.raises(SystemExit) as e:
        mod._need("holdout_feats.npy", "--features")
    msg = str(e.value)
    assert "dump_holdout" in msg, "报错里没说怎么生成这个文件"
    assert os.path.abspath("holdout_feats.npy") in msg


def test_dump_holdout_不在_imu_train_目录时明确报错(tmp_path, monkeypatch):
    mod = _load("dump_holdout")
    monkeypatch.setattr(sys, "argv", ["x", "--processed-dir", "d",
                                      "--imu-train", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "imu_train" in str(e.value)


@pytest.mark.parametrize("name", SCRIPTS + ["prune_gbdt", "event_eval"])
def test_路径里的省略号占位符会被点出来(name):
    """我在说明里习惯用 `.../` 当占位符，照抄过来就是个找不到的路径。
    只说"找不到"的话，人会以为是自己目录搞错了——实际是占位符没替换。"""
    mod = _load(name)
    with pytest.raises(SystemExit) as e:
        mod.resolve_model("~/imu_train/results/.../xgb/ml_xgb.pkl")
    assert "占位符" in str(e.value)
