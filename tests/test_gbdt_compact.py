"""紧凑 + AoS 布局：**换存法不改判决**，三方逐位对照。

三方是：
  1. `Booster`          —— 原来的 SoA 表示（已经跟 tm_gbdt.c 逐位对过）
  2. `CompactBooster`   —— 紧凑表示的 Python 参考
  3. `tm_gbdt_c.c`      —— 紧凑表示的板上实现

1 ↔ 2 证明"换存法不改判决"；2 ↔ 3 证明 C 写对了。
两条都过，才说明可以放心用紧凑布局替换掉原来那套。
"""

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml.export_gbdt_compact_c import export, pack_nodes  # noqa: E402
from tinyml.gbdt import from_xgboost_dumps  # noqa: E402
from tinyml.gbdt_compact import (  # noqa: E402
    MISSING_LEFT_BIT, NODE_BYTES, RIGHT_MASK, CompactBooster,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")


def _node(nid, fi, thr, yes, no, missing_yes=True):
    return {"nodeid": nid, "split": f"f{fi}", "split_condition": thr,
            "yes": yes["nodeid"], "no": no["nodeid"],
            "missing": (yes if missing_yes else no)["nodeid"],
            "children": [yes, no]}


def _leaf(nid, v):
    return {"nodeid": nid, "leaf": v}


def _rand_tree(rng, n_feat, depth, ctr):
    """随机生成一棵有深度的树（不只是决策桩——桩测不到右孩子偏移 > 2 的情况）。"""
    def rec(d):
        nid = next(ctr)
        if d == 0 or rng.random() < 0.3:
            return _leaf(nid, float(rng.normal()))
        y, n = rec(d - 1), rec(d - 1)
        return _node(nid, int(rng.integers(0, n_feat)), float(rng.normal()),
                     y, n, bool(rng.random() < 0.5))
    return rec(depth)


def _forest(n_feat=10, n_cls=5, rounds=6, depth=5, seed=0):
    rng = np.random.default_rng(seed)
    dumps = []
    for _ in range(rounds * n_cls):
        it = iter(range(100000))
        dumps.append(json.dumps(_rand_tree(rng, n_feat, depth, it)))
    return from_xgboost_dumps(dumps, n_feat, n_cls, base_score=0.35)


def test_紧凑表示跟原表示逐位相同():
    """这是"换存法不改判决"最直接的证明。"""
    b = _forest(seed=1)
    c = CompactBooster(b)
    rng = np.random.default_rng(9)
    xs = rng.normal(0, 1.5, size=(300, b.n_features)).astype(np.float32)
    # 掺一批正好落在阈值上的 + 一批 NaN：两条最容易在换布局时写错的路径
    thr = b.node_threshold[b.node_left != -1]
    feat = b.node_feature[b.node_left != -1]
    for k in range(min(20, len(thr))):
        x = rng.normal(0, 1, size=b.n_features).astype(np.float32)
        x[int(feat[k])] = np.float32(thr[k])
        xs = np.vstack([xs, x])
    x = np.full(b.n_features, np.nan, np.float32)
    xs = np.vstack([xs, x])

    bad = [i for i, x in enumerate(xs) if not np.array_equal(b.margins(x), c.margins(x))]
    assert not bad, f"{len(bad)}/{len(xs)} 条不一致，头几条 {bad[:5]}"


def test_节点确实是_6_字节且左孩子是下一个():
    b = _forest(seed=2)
    c = CompactBooster(b)
    assert len(pack_nodes(c)) == c.n_nodes * NODE_BYTES == c.n_nodes * 6
    internal = np.where(b.node_left != -1)[0]
    assert np.array_equal(b.node_left[internal], internal + 1), \
        "左孩子不恒等于 idx+1，紧凑布局的前提不成立"


def test_右偏移和_missing_位共用一个字节不打架():
    """missing 塞在右偏移的 bit7。要是哪天树大到偏移超过 127，
    这两件事就会串在一起——那时候必须报错，不能静默截断。"""
    b = _forest(seed=3)
    c = CompactBooster(b)
    internal = b.node_left != -1
    offs = (c.node_right[internal] & RIGHT_MASK).astype(int)
    assert offs.min() >= 2, f"内部节点的右偏移最小是 {offs.min()}，应该 ≥2"
    assert offs.max() <= RIGHT_MASK
    # 叶子必须是 0（否则会被当成内部节点走下去）
    assert np.all((c.node_right[~internal] & RIGHT_MASK) == 0)
    # missing 位确实被用上了（不然这条测试等于没测）
    assert np.any(c.node_right[internal] & MISSING_LEFT_BIT)


def test_特征超过_255_维要报错():
    b = _forest(n_feat=10, seed=4)
    b.n_features = 300
    with pytest.raises(ValueError, match="uint8"):
        CompactBooster(b)


def test_叶子值和阈值共用一个槽():
    """叶子把叶子分数放在阈值的位置。读错地方的话，叶子会返回一个阈值——
    数值上完全合理，但模型整个变了。"""
    b = _forest(seed=5)
    c = CompactBooster(b)
    leaves = np.where(b.node_left == -1)[0]
    want = b.leaf_value[b.node_feature[leaves]]
    assert np.array_equal(c.node_value[leaves], want)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    b = _forest(seed=7)
    c = CompactBooster(b)
    rng = np.random.default_rng(11)
    xs = rng.normal(0, 1.5, size=(40, b.n_features)).astype(np.float32)
    # 纯随机的 golden 有两个盲区，实测过：把 C 里的 `<` 改成 `<=`、
    # 把 NaN 方向改掉，**都不会被逮到**。因为随机浮点几乎撞不上阈值，
    # 也永远不是 NaN。这两条路径必须专门造输入。
    thr = b.node_threshold[b.node_left != -1]
    feat = b.node_feature[b.node_left != -1]
    for k in range(min(15, len(thr))):
        x = rng.normal(0, 1, size=b.n_features).astype(np.float32)
        x[int(feat[k])] = np.float32(thr[k])       # 正好等于阈值
        xs = np.vstack([xs, x])
    for k in range(5):                              # 单个通道是 NaN
        x = rng.normal(0, 1, size=b.n_features).astype(np.float32)
        x[int(feat[k]) if k < len(feat) else 0] = np.float32("nan")
        xs = np.vstack([xs, x])
    d = tmp_path_factory.mktemp("gc")
    for n, content in export(c, golden_x=xs).items():
        (d / n).write_text(content, encoding="utf-8")
    exe = d / "gc"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-fno-math-errno", "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", f"-I{d}", os.path.join(FW, "tm_gbdt_c.c"),
         str(d / "tm_gbdt_c_model.c"), os.path.join(ROOT, "tests", "host_gbdt_c.c"),
         "-lm", "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return b, c, xs, exe


def test_c_跟紧凑参考实现逐位一致(built):
    b, c, xs, exe = built
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rows = [ln.split() for ln in out.stdout.strip().splitlines()]
    assert len(rows) == len(xs)
    bad = []
    for i, (x, row) in enumerate(zip(xs, rows)):
        got = [int(v, 16) for v in row[1:]]
        m = c.margins(x)
        want = [struct.unpack("<I", struct.pack("<f", v))[0] for v in m]
        if got != want or int(row[0]) != int(np.argmax(m)):
            bad.append(i)
    assert not bad, f"{len(bad)}/{len(xs)} 条不一致，头几条 {bad[:5]}"


def test_c_跟原始表示也逐位一致(built):
    """串起来的那一条：C 的紧凑实现 ↔ 原来的 SoA 表示。
    中间隔了两层（换布局 + 换语言），所以这一条过了才算真的"不改判决"。"""
    b, c, xs, exe = built
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    rows = [ln.split() for ln in out.stdout.strip().splitlines()]
    for x, row in zip(xs, rows):
        got = [int(v, 16) for v in row[1:]]
        want = [struct.unpack("<I", struct.pack("<f", v))[0] for v in b.margins(x)]
        assert got == want
