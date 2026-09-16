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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

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


def _depth3():
    """深 2 的满树：根 → 两个内部 → 四个叶子。
    节点下标：0=根, 1=左内部, 2=左左叶, 3=左右叶, 4=右内部, 5=右左叶, 6=右右叶"""
    return _tree(
        children_left =[1,  2, -1, -1,  5, -1, -1],
        children_right=[4,  3, -1, -1,  6, -1, -1],
        feature       =[0,  1, -2, -2,  2, -2, -2],
        threshold     =[0., 0., -2., -2., 0., -2., -2.],
        value=[[[6, 2, 2]], [[4, 1, 0]], [[3, 0, 0]], [[1, 1, 0]],
               [[2, 1, 2]], [[2, 0, 0]], [[0, 1, 2]]],
    )


def test_按深度截断_节点数变少且下标不错位():
    m = SimpleNamespace(estimators_=[_depth3()], n_features_in_=3)
    full = from_sklearn(m)
    assert len(full.node_feature) == 7

    cut = from_sklearn(m, max_depth=1)
    # 根 + 两个被剪成叶子的孩子 = 3 个节点
    assert len(cut.node_feature) == 3
    assert list(cut.tree_offset) == [0, 3]
    # 孩子下标必须是**重新编号后**的，不能还指着原树的 1 和 4
    assert int(cut.node_left[0]) == 1 and int(cut.node_right[0]) == 2
    assert int(cut.node_left[1]) == -1 and int(cut.node_left[2]) == -1


def test_截断处的概率用的是那个节点自己的分布():
    """剪成叶子之后的概率必须是**那个内部节点上已经存好的** value，
    不是随便取一个子树的。取错的话模型行为会整个变掉，而且不报错。"""
    m = SimpleNamespace(estimators_=[_depth3()], n_features_in_=3)
    cut = from_sklearn(m, max_depth=1)
    left_leaf = int(cut.node_feature[1])
    # 节点 1 的 value 是 [4,1,0] → [0.8, 0.2, 0]
    assert np.allclose(cut.leaf_proba[left_leaf], [0.8, 0.2, 0.0])
    right_leaf = int(cut.node_feature[2])
    # 节点 4 的 value 是 [2,1,2] → [0.4, 0.2, 0.4]
    assert np.allclose(cut.leaf_proba[right_leaf], [0.4, 0.2, 0.4])


def test_截断到深度_0_就只剩根():
    m = SimpleNamespace(estimators_=[_depth3()], n_features_in_=3)
    cut = from_sklearn(m, max_depth=0)
    assert len(cut.node_feature) == 1
    assert int(cut.node_left[0]) == -1
    # 根的 value 是 [6,2,2] → 类别 0
    assert cut.predict(np.zeros(3, np.float32)) == 0


def test_按样本数截断():
    """min_samples_leaf 是最便宜的大杠杆：不限深的树尾部全是只覆盖一两个样本的
    分支，那部分是过拟合不是信息。"""
    t = _depth3()
    t.tree_.weighted_n_node_samples = np.array([10., 5., 4., 1., 5., 3., 2.])
    m = SimpleNamespace(estimators_=[t], n_features_in_=3)
    # 节点 1 有 5 个样本 → 保留；它的孩子一个 4 一个 1
    cut = from_sklearn(m, min_samples_leaf=5)
    # 根(10)、左(5)、右(5) 保留为内部/叶子，孙子层（4,1,3,2）都 < 5 被剪掉
    assert len(cut.node_feature) == 3


def test_不截断时跟原来一样():
    """加了截断参数之后，不传参数的行为必须一字不差——老的导出结果不能变。"""
    m = SimpleNamespace(estimators_=[_depth3(), _stump()], n_features_in_=3)
    f = from_sklearn(m)
    assert list(f.tree_offset) == [0, 7, 10]
    for x in ([0.0, 0.0, 0.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]):
        assert 0 <= f.predict(np.array(x, np.float32)) < 3


def _rand_tree(depth, rng):
    cl, cr, fe, th, va = [], [], [], [], []

    def rec(d):
        i = len(cl)
        cl.append(-1); cr.append(-1); fe.append(-2); th.append(-2.0)
        va.append([[int(rng.integers(1, 9)) for _ in range(3)]])
        if d == 0:
            return i
        fe[i] = int(rng.integers(0, 5))
        th[i] = float(rng.normal())
        cl[i] = rec(d - 1)
        cr[i] = rec(d - 1)
        return i

    rec(depth)
    return _tree(cl, cr, fe, th, va)


def test_体积公式_每节点约_20_字节():
    """把「约 20 B/节点」这个经验值钉住。整个 RF 能不能上端侧的判断就建立在它上面，
    改了结构（比如孩子下标从 int32 换成 int16）这个数会变，那时候文档里所有的
    预算估算都要跟着改——所以让它在这里红一下，比让人拿着旧数去做决策强。"""
    from tinyml.forest import flash_bytes
    rng = np.random.default_rng(0)
    m = SimpleNamespace(estimators_=[_rand_tree(8, rng) for _ in range(10)],
                        n_features_in_=5)
    f = from_sklearn(m)
    b = sum(flash_bytes(f).values())
    per = b / len(f.node_feature)
    assert 19.0 <= per <= 21.0, f"每节点 {per:.1f} B，跟文档里的 20 对不上了"


def test_深度每加一层体积翻倍():
    """这条指数关系是"为什么必须剪枝"的全部理由。写成测试是为了让它别停在直觉上。"""
    from tinyml.forest import flash_bytes
    rng = np.random.default_rng(1)
    m = SimpleNamespace(estimators_=[_rand_tree(8, rng) for _ in range(6)],
                        n_features_in_=5)
    sizes = [sum(flash_bytes(from_sklearn(m, max_depth=d)).values()) for d in (3, 4, 5, 6)]
    for a, b in zip(sizes, sizes[1:]):
        assert 1.9 < b / a < 2.2, f"{a} → {b} 不是翻倍关系"


def test_剪得越狠节点数只减不增():
    rng = np.random.default_rng(2)
    m = SimpleNamespace(estimators_=[_rand_tree(6, rng) for _ in range(4)],
                        n_features_in_=5)
    counts = [len(from_sklearn(m, max_depth=d).node_feature) for d in (1, 2, 3, 4, 5, 6)]
    assert counts == sorted(counts), counts
    assert len(from_sklearn(m).node_feature) == counts[-1], "不限深应该等于剪到最深"


def test_特征分组完整覆盖且不重叠():
    """砍特征只能按计算组砍。分组表要是漏了或者重叠了，"砍掉第 N 组"就会
    砍到不相干的维度上，而且不会报错——只会让模型输入悄悄变形。"""
    from tinyml.features import feature_groups, n_features
    for n_ch in (6, 8):
        groups = feature_groups(n_ch)
        covered = []
        for _, a, b, _ in groups:
            covered.extend(range(a, b))
        assert covered == sorted(covered), "分组的下标没有按顺序排"
        assert covered == list(range(n_features(n_ch))), (
            f"{n_ch} 通道：分组覆盖 {len(covered)} 维，总共 {n_features(n_ch)} 维")


def test_分组的边界跟实际拼接顺序一致():
    """分组表是手写的，extract_one 的拼接顺序是代码里的。两者分家的话，
    "acc_x 频域"那一组实际指向的可能是别的东西。这里用一个可辨识的构造去对：
    把某一组之外的输入全置零，看该组的特征是不是真的只跟那一段输入有关。"""
    import numpy as np
    from tinyml.features import extract_one, feature_groups
    n_ch, n_t, hz = 8, 32, 16.0
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=(n_t, n_ch)).astype(np.float32)
    f_base = extract_one(base, hz)

    groups = {nm: (a, b) for nm, a, b, _ in feature_groups(n_ch)}
    # 只改 gyr_x（通道 3），那么 gyr_x 的时域/频域组必须变，
    # 而 acc_x 的时域组必须一点都不变
    mod = base.copy()
    mod[:, 3] += np.float32(5.0)
    f_mod = extract_one(mod, hz)

    a, b = groups["gyr_x 时域"]
    assert not np.array_equal(f_base[a:b], f_mod[a:b]), "改了 gyr_x，它的时域组却没变"
    a, b = groups["gyr_x 频域"]
    assert not np.array_equal(f_base[a:b], f_mod[a:b]), "改了 gyr_x，它的频域组却没变"
    a, b = groups["acc_x 时域"]
    assert np.array_equal(f_base[a:b], f_mod[a:b]), "改了 gyr_x，acc_x 的时域组不该变"
    a, b = groups["acc 模长 时域"]
    assert np.array_equal(f_base[a:b], f_mod[a:b]), "改了 gyr_x，acc 模长不该变"
    a, b = groups["gyro 模长 时域"]
    assert not np.array_equal(f_base[a:b], f_mod[a:b]), "改了 gyr_x，gyro 模长该变"


def test_只取前_n_棵树():
    """RF 的树是 bagging 出来的、独立同分布，取前 n 棵在统计上没区别——
    所以「减树」是 RF 特有的、合法的杠杆。

    **GBDT 完全不同**：那边树是顺序的，第 k 棵拟合前 k-1 棵的残差，只能从头
    截断不能挑。两者混了的话，GBDT 挑几棵出来的模型是废的，而它照样给得出结果。
    """
    m = SimpleNamespace(estimators_=[_depth3(), _stump(), _depth3()], n_features_in_=3)
    assert from_sklearn(m).n_trees == 3
    assert from_sklearn(m, n_trees=2).n_trees == 2
    assert from_sklearn(m, n_trees=1).n_trees == 1
    with pytest.raises(ValueError, match="超出范围"):
        from_sklearn(m, n_trees=4)
    with pytest.raises(ValueError, match="超出范围"):
        from_sklearn(m, n_trees=0)


def test_减树和限深可以叠加():
    m = SimpleNamespace(estimators_=[_depth3(), _depth3()], n_features_in_=3)
    f = from_sklearn(m, n_trees=1, max_depth=1)
    assert f.n_trees == 1 and len(f.node_feature) == 3


def test_classes_是_numpy_数组时不崩():
    """sklearn 的 classes_ 是 ndarray。写成 `a or b` 会抛 truth value ambiguous——
    这种写法对 list 能跑，恰恰最容易漏。"""
    m = SimpleNamespace(estimators_=[_stump()], n_features_in_=3,
                        classes_=np.array(["a", "b", "c"]))
    assert len(from_sklearn(m).class_names) == 3
