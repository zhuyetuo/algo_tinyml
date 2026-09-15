"""适配层：同一套分析脚本同时吃 XGBoost 和 sklearn 随机森林。

**不能把两者的差别抹平**——它们在"怎么变小"上的性质不一样：

  · GBDT 沿轮数截断是**精确**的（boosting 顺序累加）；
  · RF 沿深度截断是**近似**的（分裂点是为深树选的）。

所以适配层除了统一接口，还要把"这个数准不准"带出来。抹平的话，
人会拿一张近似的表当最终答案用，而那张表看起来跟精确的一模一样。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

from tinyml.adapter import ModelAdapter  # noqa: E402
from test_forest_from_sklearn import _tree  # noqa: E402


def _fake_rf(depth=3):
    """仿 sklearn 的 RandomForestClassifier：有 estimators_，每个有 tree_。"""
    t = _tree(children_left=[1, 2, -1, -1, 5, -1, -1],
              children_right=[4, 3, -1, -1, 6, -1, -1],
              feature=[0, 1, -2, -2, 2, -2, -2],
              threshold=[0., 0., -2., -2., 0., -2., -2.],
              value=[[[6, 2, 2]], [[4, 1, 0]], [[3, 0, 0]], [[1, 1, 0]],
                     [[2, 1, 2]], [[2, 0, 0]], [[0, 1, 2]]])
    t.tree_.max_depth = depth
    return SimpleNamespace(estimators_=[t], n_features_in_=3,
                           classes_=["a", "b", "c"])


def test_rf_的轴是深度且标明是近似():
    ad = ModelAdapter(_fake_rf(), class_names=["a", "b", "c"])
    assert ad.kind == "rf"
    assert ad.axis_name == "max_depth"
    assert ad.exact is False, "RF 的截断是近似的，不能标成精确"
    assert "悲观" in ad.caveat() and "重训" in ad.caveat(), \
        "RF 的提醒里必须说清楚这是下界、定了参数还要重训"


def test_rf_沿深度变小_节点数单调不增():
    ad = ModelAdapter(_fake_rf(), class_names=["a", "b", "c"])
    counts = [ad.variant(d).n_nodes for d in (1, 2, 3)]
    assert counts == sorted(counts), counts
    assert ad.full().n_nodes == counts[-1]


def test_rf_的分数是概率_和为一():
    """RF 给概率、GBDT 给 margin。两者都保序，所以 argmax 和加偏置含义一致，
    但**数量级差很远**——偏置的合适范围不一样，脚本里会提示。"""
    ad = ModelAdapter(_fake_rf(), class_names=["a", "b", "c"])
    s = ad.full().scores(np.zeros(3, np.float32))
    assert abs(float(s.sum()) - 1.0) < 1e-6, f"RF 的分数应该是概率，和为 1，实际 {s.sum()}"


def test_不认识的模型直接报错():
    with pytest.raises(TypeError, match="既不是"):
        ModelAdapter(SimpleNamespace())


def test_rf_的紧凑体积把叶子表算进去():
    """GBDT 的叶子分数能塞进阈值那个槽，RF 的叶子要存 n_classes 个概率塞不下——
    所以 RF 的紧凑体积必须另外加上叶子表，不能照抄 GBDT 的 6 B/节点。"""
    ad = ModelAdapter(_fake_rf(), class_names=["a", "b", "c"])
    v = ad.full()
    n_leaves = 4       # 上面那棵树有 4 个叶子
    assert v.compact_flash() == 6 * v.n_nodes + n_leaves * 3 * 4
    assert v.compact_flash() < v.flash(), "紧凑布局应该比现在小"


# ── 端上真跑的那一格：棵数 × 深度 × uint8 叶子 ─────────────────────────────


def _fake_rf_n(n, depth=3):
    """n 棵一模一样的树。棵数是真的，所以减树的下标逻辑验得到。"""
    rf = _fake_rf(depth)
    rf.estimators_ = [_fake_rf(depth).estimators_[0] for _ in range(n)]
    return rf


def test_variant_可以同时按棵数和深度截断():
    """**只截深度得到的数对应不上任何一个塞得进 flash 的配置。**
    端上那一格是棵数和深度一起生效的，评估必须评同一个东西。"""
    ad = ModelAdapter(_fake_rf_n(5), class_names=["a", "b", "c"])
    full = ad.variant(3)
    cut = ad.variant(3, n_trees=2)
    assert cut.f.n_trees == 2 and full.f.n_trees == 5
    assert cut.n_nodes < full.n_nodes


def test_variant_量化叶子会落到_1_over_255_的格子上():
    ad = ModelAdapter(_fake_rf_n(2), class_names=["a", "b", "c"])
    q = ad.variant(3, quantize_leaves=True)
    scaled = np.asarray(q.f.leaf_proba, np.float64) * 255
    assert np.allclose(scaled, np.round(scaled), atol=1e-6)
    # 不加这个开关时**不能**被量化——否则"没量化"这条路径就没了
    assert not np.allclose(np.asarray(ad.variant(3).f.leaf_proba, np.float64) * 255,
                           np.round(np.asarray(ad.variant(3).f.leaf_proba, np.float64) * 255),
                           atol=1e-6)


def test_量化叶子确实会改判决而不是白给():
    """叶子量化不是免费的——相差不到 1/255 的两类会翻。
    如果这条永远不触发，说明量化根本没接上。"""
    ad = ModelAdapter(_fake_rf_n(3), class_names=["a", "b", "c"])
    a, b = ad.variant(3), ad.variant(3, quantize_leaves=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    sa = np.array([a.scores(x) for x in X])
    sb = np.array([b.scores(x) for x in X])
    assert not np.array_equal(sa, sb), "量化之后分数一模一样，量化没生效"


class _FakeBoosterModel:
    """只要有 get_booster，ModelAdapter 就当它是 GBDT。"""
    def get_booster(self):
        raise AssertionError("不该走到这里")


def test_gbdt_传减树参数要报错而不是静默忽略(monkeypatch):
    """**静默忽略是最坏的**：调用方会以为自己评的是减过树的模型，
    而 GBDT 减树本来就是非法的（树是顺序的，第 k 棵拟合前 k-1 棵的残差）。"""
    ad = ModelAdapter(_fake_rf_n(3), class_names=["a", "b", "c"])
    ad.kind = "gbdt"          # 只改类型，走到那条分支即可
    with pytest.raises(ValueError, match="只对 RF"):
        ad.variant(3, n_trees=2)
    with pytest.raises(ValueError, match="只对 RF"):
        ad.variant(3, quantize_leaves=True)
