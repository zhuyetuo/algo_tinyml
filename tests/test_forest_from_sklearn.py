"""`from_sklearn` 的解析。

这台机器上没有 sklearn，用跟它 `tree_` 一样布局的假对象验。这不是将就——
要验的正是"我对 sklearn 的数据结构理解对不对"，而那几个字段（children_left/right、
feature、threshold、value）的含义十来年没变，用假对象反而把断言写得更明确。

真模型上唯一还可能出岔的是版本差异，所以 rf_footprint.py 那边是直接打开 .pkl 的。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.forest import from_sklearn  # noqa: E402


def _tree(children_left, children_right, feature, threshold, value):
    return SimpleNamespace(tree_=SimpleNamespace(
        node_count=len(children_left),
        children_left=np.asarray(children_left),
        children_right=np.asarray(children_right),
        feature=np.asarray(feature),
        threshold=np.asarray(threshold, float),
        value=np.asarray(value, float),
    ))


def _stump(thr=0.5, left_counts=(10, 0, 0), right_counts=(0, 2, 8)):
    """一个决策桩：根 + 两个叶子。叶子上是各类**样本数**，不是概率。"""
    return _tree(
        children_left=[1, -1, -1], children_right=[2, -1, -1],
        feature=[0, -2, -2], threshold=[thr, -2.0, -2.0],
        value=[[[5, 1, 4]], [list(left_counts)], [list(right_counts)]],
    )


def test_叶子的样本数被归一成概率():
    m = SimpleNamespace(estimators_=[_stump()], n_features_in_=3, classes_=["a", "b", "c"])
    f = from_sklearn(m)
    assert f.n_classes == 3
    # 右叶子 (0,2,8) → (0, 0.2, 0.8)
    right_leaf = int(f.node_feature[2])
    assert np.allclose(f.leaf_proba[right_leaf], [0.0, 0.2, 0.8])


def test_多棵树的节点下标不会撞车():
    """扁平数组里第二棵树的孩子下标必须加上偏移。不加的话每棵树都从 0 开始，
    推理会在第一棵树里打转——结果自洽，看不出来。"""
    m = SimpleNamespace(estimators_=[_stump(0.5), _stump(1.5)], n_features_in_=3)
    f = from_sklearn(m)
    assert list(f.tree_offset) == [0, 3, 6]
    # 第二棵树的根在下标 3，它的孩子应该是 4 和 5，不是 1 和 2
    assert int(f.node_left[3]) == 4 and int(f.node_right[3]) == 5


def test_叶子样本数全零时给均匀分布不给_nan():
    """sample_weight 全零之类的边角情况会出现。不特判的话 0/0 = nan，
    一路传到 argmax——nan 比较永远是 False，np.argmax 会返回 0，
    于是"这棵树坏了"表现成"它总投第一类"。"""
    m = SimpleNamespace(estimators_=[_stump(right_counts=(0, 0, 0))], n_features_in_=3)
    f = from_sklearn(m)
    right_leaf = int(f.node_feature[2])
    assert np.allclose(f.leaf_proba[right_leaf], [1 / 3, 1 / 3, 1 / 3])
    assert not np.isnan(f.leaf_proba).any()


def test_不是森林就明确报错():
    with pytest.raises(TypeError, match="不是随机森林"):
        from_sklearn(SimpleNamespace())


def test_解析出来的森林推理结果对():
    m = SimpleNamespace(estimators_=[_stump(thr=0.5)], n_features_in_=3)
    f = from_sklearn(m)
    # x[0] = 0.4 <= 0.5 → 左叶子 (10,0,0) → 类别 0
    assert f.predict(np.array([0.4, 0, 0], np.float32)) == 0
    # x[0] = 0.5 正好等于阈值，sklearn 是 <= 走左
    assert f.predict(np.array([0.5, 0, 0], np.float32)) == 0
    # x[0] = 0.6 → 右叶子 (0,2,8) → 类别 2
    assert f.predict(np.array([0.6, 0, 0], np.float32)) == 2
