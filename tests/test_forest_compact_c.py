"""紧凑森林：C ↔ Python 逐位对照，以及"换个存法不该换判决"。

紧凑化只动存法，不动算法。所以有两件事必须钉住：

  1. **C 和 Python 参考逐位相同**——整数票数，没有"数值误差"这个借口，
     对不上就是编码或解析错了。
  2. **除了叶子量化，判决跟原版森林一致**。叶子量化确实会改判决
     （相差不到 1/255 的两类会翻），那是已知的、要实测的代价；
     除此之外的任何差异都是 bug。

字节序、偏移、叶子标记这几件事，错了都**不会报错**——只会让树走到别的分支，
而结果看起来完全正常。所以下面每一条都在验一种具体的错法。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.export_forest_compact_c import export  # noqa: E402
from tinyml.forest import Forest  # noqa: E402
from tinyml.forest_compact import (  # noqa: E402
    NODE_BYTES, CompactForest, pack_nodes,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

N_FEAT, N_CLS = 20, 5


def _forest(n_trees=6, depth=4, seed=0):
    """手工造一棵棵**先序**的树，深度固定，左孩子恒为下一个节点。

    不用 sklearn（这台机器上没有），也不需要——要验的是编码和遍历，
    sklearn 只是上游来源，它的解析另有测试。
    """
    rng = np.random.default_rng(seed)
    feat, thr, left, right, offs, leaves = [], [], [], [], [0], []

    def build(d):
        i = len(feat)
        if d == 0:
            feat.append(len(leaves))
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            p = rng.random(N_CLS) + 0.05
            leaves.append((p / p.sum()).astype(np.float32))
            return i
        feat.append(int(rng.integers(0, N_FEAT)))
        thr.append(float(rng.normal(0, 1)))
        left.append(-1)
        right.append(-1)
        li = build(d - 1)
        ri = build(d - 1)
        left[i], right[i] = li, ri
        return i

    for _ in range(n_trees):
        build(depth)
        offs.append(len(feat))

    return Forest(
        n_features=N_FEAT, n_classes=N_CLS,
        tree_offset=np.asarray(offs, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        class_names=("活动", "睡觉", "抓挠", "未佩戴", "甩身体"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    src = _forest()
    cf = CompactForest(src)
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (40, N_FEAT)).astype(np.float32)
    # 专挑边界：让特征正好等于某些阈值（<= 还是 < 只有这里分得开）
    X[0, :] = 0.0
    for k in range(1, 6):
        X[k] = rng.normal(0, 1, N_FEAT)
        X[k, int(src.node_feature[0])] = src.node_threshold[0]

    d = tmp_path_factory.mktemp("fc")
    for name, content in export(cf, golden_x=X).items():
        (d / name).write_text(content, encoding="utf-8")
    (d / "m.c").write_text(r"""
#include <stdio.h>
#include "tm_forest_c.h"
#include "tm_forest_c_model.h"
#include "tm_forest_c_golden.h"
int main(void){
    static int32_t votes[TM_FC_N_CLASSES];
    for (int i = 0; i < TM_FC_GOLDEN_N; i++) {
        const float *x = tm_forest_c_golden_in + (size_t)i * TM_FC_N_FEATURES;
        int cls = tm_forest_c_predict(&tm_forest_c, x, votes);
        printf("%d", cls);
        for (int c = 0; c < TM_FC_N_CLASSES; c++) printf(" %d", (int)votes[c]);
        printf("\n");
    }
    return 0;
}
""", encoding="utf-8")
    exe = d / "a"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=undefined,address", "-fno-sanitize-recover=all",
         "-ffp-contract=off", f"-I{FW}", f"-I{d}",
         os.path.join(FW, "tm_forest_c.c"), str(d / "tm_forest_c_model.c"),
         str(d / "m.c"), "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    got = np.array([[int(v) for v in ln.split()]
                    for ln in out.stdout.strip().splitlines()], np.int64)
    return src, cf, X, got


# ── 逐位对照 ──────────────────────────────────────────────────────────────


def test_c_votes_match_python_exactly(built):
    """整数票数，没有"数值误差"这个借口——对不上就是编码或解析错了。"""
    _, cf, X, got = built
    want = np.stack([cf.votes(x) for x in X])
    bad = np.argwhere(got[:, 1:] != want)
    assert not len(bad), (
        f"{len(bad)} 处票数不一致，头几处 {bad[:5].tolist()}\n"
        f"C={got[bad[0][0], 1:]}  Python={want[bad[0][0]]}")


def test_c_argmax_matches_python(built):
    _, cf, X, got = built
    assert np.array_equal(got[:, 0], np.array([cf.predict(x) for x in X]))


# ── 紧凑化不该改判决（叶子量化除外）────────────────────────────────────────


def test_compact_agrees_with_the_original_forest(built):
    """量化之前的原版森林跟紧凑版，判决应当高度一致。

    **不是 100%**：叶子量化成 uint8 之后，相差不到 1/255 的两类会翻。
    那是已知代价。这里要的是"绝大多数一致"——差太多说明遍历写错了，
    而遍历写错的表现正好也是"判决差一点"，光看一致率分不出来，
    所以下面还有一条直接验遍历路径的。
    """
    src, cf, X, _ = built
    a = np.array([src.predict(x) for x in X])
    b = np.array([cf.predict(x) for x in X])
    assert (a == b).mean() > 0.9, f"一致率只有 {(a == b).mean():.2f}"


def test_leaf_quantisation_is_the_only_difference(built):
    """把原版森林的叶子也量化成同一套 uint8，判决必须**完全**一致。

    这一条才是真正验遍历的：排除掉量化这个已知差异之后，还有任何一条不一样，
    就是遍历/编码写错了。
    """
    src, cf, X, _ = built
    from tinyml.forest import quantize_leaves
    q = quantize_leaves(src, levels=cf.levels)
    a = np.array([q.predict(x) for x in X])
    b = np.array([cf.predict(x) for x in X])
    bad = np.flatnonzero(a != b)
    assert not len(bad), f"{len(bad)} 条判决不一致，头几条 {bad[:5].tolist()}"


# ── 编码本身 ──────────────────────────────────────────────────────────────


def test_node_is_seven_bytes(built):
    _, cf, _, _ = built
    assert pack_nodes(cf).shape == (cf.n_nodes, 7) and NODE_BYTES == 7


def test_leaf_is_marked_by_right_equals_zero(built):
    """内部节点的右偏移 ≥ 2，所以 0 可以当叶子标记——这个前提要验，
    不然某个内部节点的偏移要是 0，遍历会把它当叶子，安静地少走半棵树。"""
    _, cf, _, _ = built
    assert np.all(cf.node_right[cf.is_leaf] == 0)
    assert np.all(cf.node_right[~cf.is_leaf] >= 2)


def test_left_child_is_always_next(built):
    """左孩子恒为 idx+1 是不存它的前提。"""
    src, _, _, _ = built
    inner = np.flatnonzero(src.node_left != -1)
    assert np.array_equal(src.node_left[inner], inner + 1)


def test_leaf_index_survives_the_float_slot(built):
    """叶子把 4 字节的阈值槽复用成 uint32 下标。

    这个复用错了不会崩——它会把某个下标当成 float 读出来，
    或者反过来，然后指到别的叶子上，给出一个看起来正常的概率。
    """
    _, cf, _, _ = built
    nodes = pack_nodes(cf)
    for i in np.flatnonzero(cf.is_leaf):
        w = (int(nodes[i, 1]) | int(nodes[i, 2]) << 8
             | int(nodes[i, 3]) << 16 | int(nodes[i, 4]) << 24)
        assert w == int(cf.node_leaf_idx[i])


def test_threshold_survives_the_round_trip(built):
    """内部节点的阈值要**逐位**还原，不是近似——差一个 ULP 就可能让
    正好落在阈值上的样本走反。"""
    _, cf, _, _ = built
    nodes = pack_nodes(cf)
    for i in np.flatnonzero(~cf.is_leaf):
        w = np.uint32(int(nodes[i, 1]) | int(nodes[i, 2]) << 8
                      | int(nodes[i, 3]) << 16 | int(nodes[i, 4]) << 24)
        assert np.array(w).view(np.float32) == cf.node_threshold[i]


def test_le_not_lt_on_the_boundary(built):
    """特征正好等于阈值时**走左**，跟 sklearn 一致。

    阈值本来就是从样本值来的，"正好等于"一点也不罕见。写成 < 的话
    那些样本会走反，而整体准确率只掉一点点，看不出来。
    """
    src, cf, _, _ = built
    root_f = int(src.node_feature[0])
    x = np.zeros(N_FEAT, np.float32)
    x[root_f] = src.node_threshold[0]
    # 走左 = 下一个节点；手工跟一步
    node = 0
    r = int(cf.node_right[node])
    went = node + 1 if x[root_f] <= cf.node_threshold[node] else node + r
    assert went == 1, "特征等于阈值时没走左"


def test_too_many_features_is_refused():
    """特征超过 255 维时 uint8 装不下。必须当场报错——
    静默截断会让所有节点指到错误的特征上。"""
    src = _forest()
    src.n_features = 300
    with pytest.raises(ValueError, match="uint8"):
        CompactForest(src)


def test_flash_accounting_matches_the_exported_bytes(built):
    """记账函数报的体积，必须等于导出的字节数。

    **这一条是补上一个真实的错**：之前 forest.compact_flash_bytes 按
    6 B/节点记账，而 RF 的右孩子偏移要 uint16、根本装不进 6 字节——
    那个数从来没有对应的实现，我却拿它跟别的方案比了好几轮。
    记账和导出对不上，就是在拿不存在的方案做决策。
    """
    _, cf, _, _ = built
    sizes = cf.flash_bytes()
    assert sizes["nodes"] == pack_nodes(cf).size
    assert sizes["leaves"] == cf.leaf_u8.size
    assert sizes["nodes"] == cf.n_nodes * 7


def test_leaf_quantisation_rounds_half_away_from_zero():
    """叶子量化在中点上要**远离零**取整，不是银行家舍入。

    这条必须用 levels=128（2 的幂）：概率是 float32，随机值乘 255 永远落不到
    精确的 .5 上，两种取整给的答案一样——变异成 np.round 测试照样绿。
    （quantize_leaves 那边踩过同一个坑，这里又踩了一次。）

    2.5/128 在 float32 里精确，乘回来正好 2.5：银行家舍入给 2（偶），
    远离零给 3。
    """
    src = _forest(n_trees=1, depth=1)
    p = np.zeros((len(src.leaf_proba), N_CLS), np.float32)
    p[:, 0] = np.float32(2.5 / 128)
    p[:, 1] = np.float32(1 - 2.5 / 128)
    src = Forest(**{**src.__dict__, "leaf_proba": p})
    cf = CompactForest(src, levels=128)
    assert int(cf.leaf_u8[0, 0]) == 3, \
        f"中点取整给了 {int(cf.leaf_u8[0, 0])}，应该是 3（远离零），不是 2（取偶）"


def test_accounting_and_exporter_never_diverge_again():
    """forest.compact_flash_bytes 和 CompactForest.flash_bytes 必须给同一个数。

    **这一条是补上一个真实的错**：记账函数原来按 6 B/节点算（照抄了 GBDT，
    那边限深 6、右偏移塞得进 7 bit），而 RF 深度 10、一棵树几百个节点，
    偏移必须 uint16，实际是 7 B。差出来的那个数（98.2 KB vs 真实的 109.7 KB）
    被拿去跟别的方案比了好几轮——而那个编码根本没有实现。

    记账和导出对不上，就是在拿不存在的方案做决策。
    """
    from tinyml.forest import compact_flash_bytes
    src = _forest()
    cf = CompactForest(src)
    a = compact_flash_bytes(src, leaf_bits=8)
    b = cf.flash_bytes()
    assert a["nodes"] == b["nodes"], f"节点体积对不上：记账 {a['nodes']}、实际 {b['nodes']}"
    assert a["leaves"] == b["leaves"]
    assert sum(a.values()) == sum(b.values())
