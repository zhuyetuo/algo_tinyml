"""export_train.py 里不依赖 sklearn 的那部分 + 端侧服务读本地清单 / reload。

真正的导出（sklearn → C → 留出集 F1）要训练机才跑得了，这里只守住：
指标算法、清单登记/撤销、本地清单一律 optional、reload 没起清单时不炸。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import export_train as et  # noqa: E402


def test_nperseg():
    assert et.nperseg_for(16) == 16
    assert et.nperseg_for(32) == 32
    assert et.nperseg_for(48) == 32
    assert et.nperseg_for(20) == 16


def test_per_class_report():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    r = et.per_class_report(y, p, ["a", "b", "c"])
    assert r["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert r["per_class"]["b"]["recall"] == 1.0
    assert r["per_class"]["b"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["per_class"]["a"]["support"] == 2
    assert 0 < r["macro_f1"] < 1
    # 留出集里没出现的类不进 macro
    r2 = et.per_class_report(np.array([0, 0]), np.array([0, 0]), ["a", "b"])
    assert r2["macro_f1"] == 1.0


def test_register_and_remove(tmp_path):
    lj = str(tmp_path / "local.json")
    et.register_local("train6", "/x/gen6", "/x/gen6/meta.json", lj)
    et.register_local("train7", "/x/gen7", "/x/gen7/meta.json", lj)
    et.register_local("train6", "/y/gen6", "/y/gen6/meta.json", lj)   # 同 tag 覆盖
    cfg = json.load(open(lj, encoding="utf-8"))
    assert [m["tag"] for m in cfg["models"]] == ["train7", "train6"]
    assert cfg["models"][1]["gen"] == "/y/gen6" and cfg["models"][1]["kind"] == "rf"
    et.unregister_local("train7", lj)
    assert [m["tag"] for m in json.load(open(lj, encoding="utf-8"))["models"]] == ["train6"]


def test_local_list_is_optional(tmp_path, capsys):
    """本地清单里目录没了（训练记录删了）不能让服务起不来。"""
    import edge_service as es
    gen = tmp_path / "gen_ok"
    gen.mkdir()
    (gen / "meta.json").write_text("{}", encoding="utf-8")
    main = tmp_path / "edge_models.json"
    main.write_text(json.dumps({"models": [{"tag": "m", "gen": str(gen), "meta": str(gen / "meta.json"), "kind": "rf"}]}),
                    encoding="utf-8")
    local = tmp_path / "edge_models.local.json"
    local.write_text(json.dumps({"models": [
        {"tag": "train9", "gen": str(tmp_path / "nope"), "meta": str(tmp_path / "nope/meta.json")},
        {"tag": "train8", "gen": str(gen), "meta": str(gen / "meta.json")},
    ]}), encoding="utf-8")
    specs = es.load_models_config(str(main))
    assert [s["tag"] for s in specs] == ["m", "train8"]
    assert specs[1]["kind"] == "rf"
    assert "train9" in capsys.readouterr().err


def test_reload_without_models_arg():
    import edge_service as es
    es.Handler.boot = {}
    r = es.Handler.reload()
    assert r["ok"] is False
