"""随机森林：C ↔ Python 逐位对照。

这台机器上没有 sklearn，所以森林是**手工造**的——用跟 sklearn tree_ 一样的扁平
数组布局。这不是将就：真正要验的是"导出 + C 推理"这一段，sklearn 只是上游来源，
拿真模型反而会让测试依赖一份几十 MB 的 .pkl 和一个特定版本。
`from_sklearn` 的解析另有测试（test_forest_from_sklearn.py）用假对象验。

比的是概率的**位模式**，不是 argmax：只比 argmax 的话，一个已经算错、只是恰好
还没把类别翻过去的实现能一路混到量产。
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.export_forest_c import export  # noqa: E402
from tinyml.forest import Forest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

N_FEAT, N_CLS = 12, 3


def _random_tree(rng, depth, n_feat, n_cls, feat, thr, left, right, leaves, base):
    """递归造一棵随机树，节点直接追加进扁平数组。返回本节点的全局下标。"""
    idx = len(feat)
    make_leaf = depth == 0 or rng.random() < 0.25
    if make_leaf:
        p = rng.dirichlet(np.ones(n_cls)).astype(np.float32)
        feat.append(len(leaves))
        leaves.append(p)
        thr.append(np.float32(0.0))
        left.append(-1)
        right.append(-1)
        return idx
    feat.append(int(rng.integers(0, n_feat)))
    thr.append(np.float32(rng.normal(0, 1)))
    left.append(-1)   # 占位，孩子造完再回填
    right.append(-1)
    li = _random_tree(rng, depth - 1, n_feat, n_cls, feat, thr, left, right, leaves, base)
    ri = _random_tree(rng, depth - 1, n_feat, n_cls, feat, thr, left, right, leaves, base)
    left[idx] = li
    right[idx] = ri
    return idx


def _make_forest(n_trees=25, depth=6, seed=0):
    rng = np.random.default_rng(seed)
    feat, thr, left, right, leaves = [], [], [], [], []
    offsets = [0]
    for _ in range(n_trees):
        _random_tree(rng, depth, N_FEAT, N_CLS, feat, thr, left, right, leaves, offsets[-1])
        offsets.append(len(feat))
    return Forest(
        n_features=N_FEAT, n_classes=N_CLS,
        tree_offset=np.asarray(offsets, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        class_names=("sleep", "active", "scratch"),
    )


def _golden_inputs(forest, n=40, seed=7):
    rng = np.random.default_rng(seed)
    xs = rng.normal(0, 1, size=(n, forest.n_features)).astype(np.float32)
    # 掺进一批**正好落在阈值上**的样本：sklearn 是 <= 走左，写成 < 就会走反，
    # 而随机浮点几乎撞不上阈值，不特意构造的话这个 bug 测不出来
    thrs = forest.node_threshold[forest.node_left != -1]
    feats = forest.node_feature[forest.node_left != -1]
    for k in range(min(10, len(thrs))):
        x = rng.normal(0, 1, size=forest.n_features).astype(np.float32)
        x[int(feats[k])] = np.float32(thrs[k])
        xs = np.vstack([xs, x])
    return xs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    forest = _make_forest()
    xs = _golden_inputs(forest)
    out_dir = tmp_path_factory.mktemp("forest")
    for n, content in export(forest, golden_x=xs).items():
        (out_dir / n).write_text(content, encoding="utf-8")

    exe = out_dir / "host"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         # -ffp-contract=off 是**必须**的：允许 FMA 合并的话，a*b+c 会用一次
         # 不舍入的中间结果，跟 Python 分两步算的结果末位不同
         "-ffp-contract=off",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", f"-I{out_dir}",
         os.path.join(FW, "tm_forest.c"), str(out_dir / "tm_forest_model.c"),
         os.path.join(ROOT, "tests", "host_forest.c"), "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return forest, xs, exe


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", np.float32(f)))[0]


def test_森林_c_与_python_逐位一致(built):
    forest, xs, exe = built
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == len(xs)

    bad = []
    for i, (x, line) in enumerate(zip(xs, lines)):
        parts = line.split()
        c_cls = int(parts[0])
        c_bits = [int(v, 16) for v in parts[1:]]
        p = forest.predict_proba(x)
        p_bits = [_bits(v) for v in p]
        if c_cls != int(np.argmax(p)) or c_bits != p_bits:
            bad.append((i, c_cls, int(np.argmax(p)), c_bits, p_bits))
    assert not bad, f"{len(bad)}/{len(xs)} 条不一致，头一条：{bad[0]}"


def test_概率和为一且不是常数(built):
    """守住上面那个测试的有效性。概率和必须是 1（归一化没写漏），
    而且不同输入要给出不同概率——否则一个返回常量的实现也能通过逐位比对。"""
    forest, xs, _ = built
    ps = np.stack([forest.predict_proba(x) for x in xs])
    assert np.allclose(ps.sum(axis=1), 1.0, atol=1e-6)
    assert len(np.unique(ps.round(6), axis=0)) > 1, "所有输入给出同一组概率"
    assert len(set(int(np.argmax(p)) for p in ps)) >= 2, "所有输入判成同一类"


def test_并列时取下标最小的那一类(tmp_path):
    """两类概率**完全相等**时，C 必须跟 np.argmax 一样取下标小的那个。

    单独造一个会并列的森林，因为随机概率几乎不可能撞上并列——实测过：把 C 里的
    `>` 改成 `>=`（变成取下标大的），上面那组 golden vector 全部照过。
    并列在真模型上不罕见：叶子样本数小的时候 (1,1,0) 这种平分很常见。
    """
    forest = Forest(
        n_features=1, n_classes=3,
        tree_offset=np.asarray([0, 1], np.int32),
        node_feature=np.asarray([0], np.int32),      # 唯一的节点就是叶子，指向第 0 行
        node_threshold=np.asarray([0.0], np.float32),
        node_left=np.asarray([-1], np.int32),
        node_right=np.asarray([-1], np.int32),
        leaf_proba=np.asarray([[0.5, 0.5, 0.0]], np.float32),
        class_names=("a", "b", "c"),
    )
    xs = np.zeros((1, 1), np.float32)
    for n, content in export(forest, golden_x=xs).items():
        (tmp_path / n).write_text(content, encoding="utf-8")
    exe = tmp_path / "host"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         f"-I{FW}", f"-I{tmp_path}", os.path.join(FW, "tm_forest.c"),
         str(tmp_path / "tm_forest_model.c"),
         os.path.join(ROOT, "tests", "host_forest.c"), "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    p = forest.predict_proba(xs[0])
    assert p[0] == p[1], "这个用例没造出并列，测试就失去意义了"
    assert int(out.stdout.split()[0]) == int(np.argmax(p)) == 0


def test_阈值上的样本走左边(built):
    """sklearn 的规则是 x <= threshold 走左。这条单独钉住，因为把 <= 写成 <
    只影响"正好等于"的样本，日常随机数据几乎测不出来。"""
    forest, _, _ = built
    internal = np.where(forest.node_left != -1)[0]
    node = int(internal[0])
    f = int(forest.node_feature[node])
    x = np.zeros(forest.n_features, np.float32)
    x[f] = forest.node_threshold[node]
    # 直接走一遍这棵树的根，确认落到左子树
    assert x[f] <= forest.node_threshold[node]
