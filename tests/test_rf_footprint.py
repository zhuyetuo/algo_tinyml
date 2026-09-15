"""rf_footprint.py 的算术部分。

这台机器上没有 sklearn，脚本本身要在训练机上跑；但「怎么数节点、怎么折算字节」
是纯算术，可以在这里用假的树对象验。不验的话，那个脚本只会在别人的机器上
第一次跑的时候才暴露问题，而那时候人是拿它去做决策的。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from rf_footprint import ENCODINGS, tree_stats  # noqa: E402


def _fake_tree(children_left, max_depth):
    """仿 sklearn 的 tree_：children_left 里 -1 表示叶子（TREE_LEAF）。"""
    cl = np.asarray(children_left)
    return SimpleNamespace(tree_=SimpleNamespace(
        node_count=len(cl), children_left=cl, max_depth=max_depth))


def test_数节点和叶子():
    # 一棵满二叉：根 + 2 个内部 + 4 个叶子 = 7 个节点
    est = _fake_tree([1, 3, 5, -1, -1, -1, -1], 2)
    n_nodes, n_internal, n_leaves, depth = tree_stats(est)
    assert (n_nodes, n_internal, n_leaves, depth) == (7, 3, 4, 2)


def test_单节点树也算对():
    """只有一个根、而且它就是叶子——数据极度不平衡时 sklearn 真会训出这种树。
    按"节点数 - 叶子数"算内部节点，这种情况下是 0，不能算成 -1 或 1。"""
    n_nodes, n_internal, n_leaves, _ = tree_stats(_fake_tree([-1], 0))
    assert (n_nodes, n_internal, n_leaves) == (1, 0, 1)


def test_编码大小的相对关系():
    # packed 是理论下界，compact16 次之，f32 最费。顺序反了说明表被改错了
    assert ENCODINGS["packed"] < ENCODINGS["compact16"] < ENCODINGS["f32"]


def test_折算字节():
    internal, leaves, leaf_bytes = 1000, 1001, 4
    assert internal * ENCODINGS["f32"] + leaves * leaf_bytes == 12000 + 4004
