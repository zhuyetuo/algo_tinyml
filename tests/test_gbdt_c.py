"""GBDT：C ↔ Python 逐位对照 + XGBoost 那三条跟 sklearn 不一样的语义。

这台机器上没有 xgboost，所以用**手写的 JSON dump** 造模型。这不是将就：
要验的正是"我对 XGBoost dump 格式的理解对不对"，手写反而把每条断言写得更明确。
真模型上还可能出岔的只有版本差异，那个要在训练机上用 export_gbdt.py 跑。
"""

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.export_gbdt_c import export  # noqa: E402
from tinyml.gbdt import from_xgboost_dumps  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")


def stump(fi, thr, yes_leaf, no_leaf, missing_yes=True):
    """一个决策桩的 JSON dump，格式照 XGBoost 的 dump_format="json"。"""
    return json.dumps({
        "nodeid": 0, "depth": 0, "split": f"f{fi}", "split_condition": thr,
        "yes": 1, "no": 2, "missing": 1 if missing_yes else 2,
        "children": [{"nodeid": 1, "leaf": yes_leaf}, {"nodeid": 2, "leaf": no_leaf}],
    })


def _build(tmp, b, xs):
    for n, content in export(b, golden_x=xs).items():
        (tmp / n).write_text(content, encoding="utf-8")
    exe = tmp / "gbdt"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-fno-math-errno", "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", f"-I{tmp}", os.path.join(FW, "tm_gbdt.c"),
         str(tmp / "tm_gbdt_model.c"), os.path.join(ROOT, "tests", "host_gbdt.c"),
         "-lm", "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rows = []
    for line in out.stdout.strip().splitlines():
        p = line.split()
        rows.append((int(p[0]), [int(v, 16) for v in p[1:]]))
    return rows


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def _rand_forest(n_feat, n_cls, rounds, seed=0):
    rng = np.random.default_rng(seed)
    dumps = []
    for _ in range(rounds):
        for _ in range(n_cls):
            dumps.append(stump(int(rng.integers(0, n_feat)), float(rng.normal()),
                               float(rng.normal()), float(rng.normal())))
    return from_xgboost_dumps(dumps, n_feat, n_cls, base_score=0.5)


def test_c_与_python_margin_逐位一致(tmp_path):
    n_feat, n_cls = 12, 5
    b = _rand_forest(n_feat, n_cls, rounds=8, seed=3)
    rng = np.random.default_rng(7)
    xs = rng.normal(0, 1, size=(30, n_feat)).astype(np.float32)
    # 掺一批正好落在阈值上的：XGBoost 是 `<`，写成 `<=` 只有这些样本会走反
    thrs = b.node_threshold[b.node_left != -1]
    feats = b.node_feature[b.node_left != -1]
    for k in range(min(10, len(thrs))):
        x = rng.normal(0, 1, size=n_feat).astype(np.float32)
        x[int(feats[k])] = np.float32(thrs[k])
        xs = np.vstack([xs, x])

    rows = _build(tmp_path, b, xs)
    assert len(rows) == len(xs)
    bad = []
    for i, (x, (c_cls, c_bits)) in enumerate(zip(xs, rows)):
        m = b.margins(x)
        if c_cls != int(np.argmax(m)) or c_bits != [_bits(v) for v in m]:
            bad.append((i, c_cls, int(np.argmax(m))))
    assert not bad, f"{len(bad)}/{len(xs)} 条不一致，头一条 {bad[0]}"


def test_阈值上的样本走的是_小于_不是小于等于():
    """XGBoost 的判决是 `x < split_condition` 走 yes；sklearn 是 `<=` 走左。
    写成 `<=` 只有"正好等于"的样本会走反，日常随机数据几乎测不出来。"""
    b = from_xgboost_dumps([stump(0, 1.0, yes_leaf=10.0, no_leaf=-10.0)],
                           n_features=1, n_classes=1, base_score=0.0)
    assert b.margins(np.array([0.9], np.float32))[0] == pytest.approx(10.0)
    # 正好等于阈值：`<` 为假 → 走 no
    assert b.margins(np.array([1.0], np.float32))[0] == pytest.approx(-10.0)
    assert b.margins(np.array([1.1], np.float32))[0] == pytest.approx(-10.0)


def test_多分类的树按轮数交错分给各类():
    """树 t 属于类别 t % n_classes。摊错的话每个类别都会拿到别人的分数，
    而模型照样给得出一个类别。"""
    # 2 类 2 轮：类0 的两棵都给 +1，类1 的两棵都给 -1
    dumps = []
    for _ in range(2):
        dumps.append(stump(0, 0.0, 1.0, 1.0))    # 类 0
        dumps.append(stump(0, 0.0, -1.0, -1.0))  # 类 1
    b = from_xgboost_dumps(dumps, n_features=1, n_classes=2, base_score=0.0)
    m = b.margins(np.array([0.0], np.float32))
    assert m[0] == pytest.approx(2.0) and m[1] == pytest.approx(-2.0)
    assert b.predict(np.array([0.0], np.float32)) == 0


def test_树数除不尽类别数要报错():
    dumps = [stump(0, 0.0, 1.0, 1.0)] * 7
    with pytest.raises(ValueError, match="除不尽"):
        from_xgboost_dumps(dumps, n_features=1, n_classes=5)


def test_叶子是相加不是平均():
    """RF 是平均，GBDT 是相加。写成平均的话分数被压扁 n_trees 倍，
    argmax 大多数时候还是对的——所以这个错能潜伏很久。"""
    dumps = [stump(0, 0.0, 3.0, 3.0), stump(0, 0.0, 4.0, 4.0)]
    b = from_xgboost_dumps(dumps, n_features=1, n_classes=1, base_score=0.0)
    assert b.margins(np.array([0.0], np.float32))[0] == pytest.approx(7.0)


def test_nan_按_missing_方向走(tmp_path):
    """特征出 NaN 时要走训练时记下的方向。不处理的话 `v < thr` 对 NaN 恒为假，
    会一律走右边，可能跟训练时相反。"""
    b = from_xgboost_dumps([stump(0, 0.0, 10.0, -10.0, missing_yes=True)],
                           n_features=1, n_classes=1, base_score=0.0)
    assert b.margins(np.array([np.nan], np.float32))[0] == pytest.approx(10.0)
    b2 = from_xgboost_dumps([stump(0, 0.0, 10.0, -10.0, missing_yes=False)],
                            n_features=1, n_classes=1, base_score=0.0)
    assert b2.margins(np.array([np.nan], np.float32))[0] == pytest.approx(-10.0)
    # C 那边也要一致
    xs = np.array([[np.nan]], np.float32)
    rows = _build(tmp_path, b, xs)
    assert rows[0][1][0] == _bits(np.float32(10.0))


def test_argmax_在_margin_上取跟先_softmax_再取一样():
    """端上不做 softmax 的依据。softmax 保序，所以两边 argmax 必然相同——
    写成测试是因为"保序"这个性质靠记不靠谱，而错了会很难发现。"""
    from tinyml.gbdt import softmax_ref
    rng = np.random.default_rng(1)
    b = _rand_forest(8, 5, rounds=6, seed=11)
    for _ in range(200):
        x = rng.normal(0, 2, size=8).astype(np.float32)
        m = b.margins(x)
        assert int(np.argmax(m)) == int(np.argmax(softmax_ref(m)))


def test_截断到前_K_轮_跟只训_K_轮完全相同():
    """GBDT 的截断是**精确**的，不是近似——boosting 顺序累加，第 k 棵树拟合的是
    前 k-1 棵之后的残差，所以前 K 棵树跟总共训多少轮无关。

    这条是"不用重训就能拿到 轮数→掉点 曲线"的全部依据，所以要钉死：
    造一个 6 轮的模型，截断到 3 轮，跟直接用前 3 轮的树构建出来的必须一模一样。
    """
    n_feat, n_cls, rounds = 6, 3, 6
    rng = np.random.default_rng(21)
    dumps = [stump(int(rng.integers(0, n_feat)), float(rng.normal()),
                   float(rng.normal()), float(rng.normal()))
             for _ in range(rounds * n_cls)]
    full = from_xgboost_dumps(dumps, n_feat, n_cls, base_score=0.3)
    cut = full.truncate(3)
    direct = from_xgboost_dumps(dumps[:3 * n_cls], n_feat, n_cls, base_score=0.3)

    assert cut.n_trees == direct.n_trees == 9
    xs = rng.normal(0, 1, size=(50, n_feat)).astype(np.float32)
    for x in xs:
        a, b = cut.margins(x), direct.margins(x)
        assert np.array_equal(a, b), f"截断 {a} ≠ 直接构建 {b}"


def test_截断保留的是完整的轮不是半轮():
    """多分类一轮 = n_classes 棵树。截断到"2.5 轮"会让某些类别比别人多一棵树，
    margin 就系统性偏了——而模型照样给得出结果。"""
    n_cls = 5
    dumps = [stump(0, 0.0, 1.0, 1.0) for _ in range(4 * n_cls)]
    b = from_xgboost_dumps(dumps, 1, n_cls)
    for r in (1, 2, 3, 4):
        assert b.truncate(r).n_trees == r * n_cls
    with pytest.raises(ValueError, match="超出范围"):
        b.truncate(5)
    with pytest.raises(ValueError, match="超出范围"):
        b.truncate(0)
