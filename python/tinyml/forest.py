"""随机森林的端侧表示：从 sklearn 的 RandomForestClassifier 抽成扁平数组，
外加一份 Python 参考前向。对应固件 tm_forest.c，两边必须逐位一致。

**判决方式照抄 sklearn**：RandomForestClassifier.predict_proba 是把每棵树叶子上的
类别概率**求平均**，不是各棵树 argmax 之后投票。两者结果不一样——多数投票会把
"很多棵树都觉得有点像抓挠"这种信息抹掉。端上要跟平台给同一个结论，就得照抄。

代价是叶子要存 n_classes 个 float32，不是一个类别号。3 分类就是 12 B/叶子。
省不了：改成存 argmax 就不是同一个模型了，那时候"端侧和平台不一致"就成了设计
本身的产物，查都没法查。

节点用扁平数组存，跟 sklearn 的 tree_ 一个套路：
    node_left[i] == -1  →  这是叶子，node_feature[i] 是它在 leaf_proba 里的下标
    否则                →  x[node_feature[i]] <= node_threshold[i] 走左，否则走右
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Forest:
    n_features: int
    n_classes: int
    tree_offset: np.ndarray    # int32 [n_trees + 1]，第 t 棵树的节点是 [off[t], off[t+1])
    node_feature: np.ndarray   # int32 [n_nodes]：内部节点=特征下标；叶子=leaf_proba 的下标
    node_threshold: np.ndarray  # float32 [n_nodes]，叶子处无意义
    node_left: np.ndarray      # int32 [n_nodes]，-1 表示叶子
    node_right: np.ndarray     # int32 [n_nodes]
    leaf_proba: np.ndarray     # float32 [n_leaves, n_classes]
    class_names: tuple = ()

    @property
    def n_trees(self):
        return len(self.tree_offset) - 1

    def predict_proba(self, x):
        """x: float32 [n_features] → float32 [n_classes]。

        累加顺序**必须**跟 C 一样（按树的下标从小到大）：浮点加法不满足结合律，
        换个顺序结果末位就可能不同，而那正是"板上跟 PC 差一点点"的经典来源。
        """
        x = np.asarray(x, np.float32)
        acc = np.zeros(self.n_classes, dtype=np.float32)
        for t in range(self.n_trees):
            node = int(self.tree_offset[t])
            while self.node_left[node] != -1:
                f = int(self.node_feature[node])
                # sklearn 的判决是 <= 走左。写成 < 的话，特征值正好等于阈值的样本
                # 会走反——而训练集里"正好等于"一点也不罕见，阈值本来就是从样本值来的
                node = int(self.node_left[node] if x[f] <= self.node_threshold[node]
                           else self.node_right[node])
            acc += self.leaf_proba[int(self.node_feature[node])]
        return (acc / np.float32(self.n_trees)).astype(np.float32)

    def predict(self, x):
        return int(np.argmax(self.predict_proba(x)))


def _pick_names(explicit, model):
    if explicit is not None and len(explicit):
        return tuple(explicit)
    auto = getattr(model, "classes_", None)
    if auto is None:
        return ()
    return tuple(auto)


def _leaf_proba(t, i):
    """节点 i 的类别分布，归一成概率。

    sklearn 1.3 之后 tree_.value 对分类器存的是**比例**不是样本数（以前是样本数）。
    除以自己的和之后两种情况结果一样，所以这里不去判版本——判版本的代码迟早
    会在某个没见过的版本上悄悄给错数。
    """
    v = np.asarray(t.value[i], np.float64).reshape(-1)
    ssum = v.sum()
    # 全零在正常训练里不会出现，但 sample_weight 全零之类的边角会。给均匀分布，
    # 别让 0/0 = nan 一路传到 argmax（nan 比较永远是 False，argmax 会返回 0，
    # 于是"这棵树坏了"表现成"它总投第一类"）
    return (v / ssum if ssum > 0 else np.full_like(v, 1.0 / len(v))).astype(np.float32)


def from_sklearn(model, class_names=None, max_depth=None, min_samples_leaf=None,
                 n_trees=None) -> Forest:
    """从 sklearn 的 RandomForestClassifier 抽出来，可选**就地截断**。

    max_depth / min_samples_leaf 不是重训，是**把已经训好的树在某个深度剪掉**：
    该节点直接变成叶子，类别分布用 sklearn 在那个节点上已经存好的 value。
    这在数学上等价于"训练时就设了这个 max_depth"吗？**不等价**——训练时限深的话，
    分裂点的选择会不同。但它有一个大得多的好处：**不用重训就能拿到
    「深度 → 体积 → 准确率」这条曲线**，而重训一轮要等很久。先用它定个范围，
    真正上线前再按定下来的参数重训一次。

    只读 tree_ 的那几个扁平数组，不依赖 sklearn 的对象结构——那几个字段
    （children_left/right、feature、threshold、value、weighted_n_node_samples）
    十来年没变过。
    """
    ests = getattr(model, "estimators_", None)
    if ests is None:
        raise TypeError(f"不是随机森林（没有 estimators_），实际是 {type(model)}")

    # **只取前 n 棵是合法的**：RF 的树是 bagging 出来的、互相独立同分布，
    # 取哪几棵在统计上没有区别。
    # （GBDT 完全不同——那边树是顺序的，第 k 棵拟合前 k-1 棵的残差，
    #  所以只能从头截断，不能挑。这个区别不能混。）
    if n_trees is not None:
        if not (1 <= n_trees <= len(ests)):
            raise ValueError(f"n_trees={n_trees} 超出范围（总共 {len(ests)} 棵）")
        ests = ests[:n_trees]

    offsets = [0]
    feat, thr, left, right = [], [], [], []
    leaves = []

    for est in ests:
        t = est.tree_
        cl = np.asarray(t.children_left, np.int64)
        cr = np.asarray(t.children_right, np.int64)
        n_samples = np.asarray(getattr(t, "weighted_n_node_samples",
                                       np.full(t.node_count, np.inf)), np.float64)

        def emit(src, depth):
            """先序发出子树，返回它在扁平数组里的下标。

            孩子的下标要等孩子发完才知道，所以先占位再回填——递归里直接写
            base + children_left[i] 那种写法只在"不截断"时成立，一截断就全错位了。
            """
            idx = len(feat)
            # min_samples_leaf 照抄 sklearn 的含义：**两个孩子都**至少有这么多样本，
            # 这个分裂才保留。看本节点自己的样本数是错的——那样参数名就在骗人，
            # 剪出来的树跟"训练时设同一个值"完全不是一回事。
            split_too_small = (
                min_samples_leaf is not None and cl[src] != -1
                and (n_samples[cl[src]] < min_samples_leaf
                     or n_samples[cr[src]] < min_samples_leaf)
            )
            is_leaf = (
                cl[src] == -1
                or (max_depth is not None and depth >= max_depth)
                or split_too_small
            )
            if is_leaf:
                feat.append(len(leaves))
                leaves.append(_leaf_proba(t, src))
                thr.append(np.float32(0.0))
                left.append(-1)
                right.append(-1)
                return idx
            feat.append(int(t.feature[src]))
            thr.append(np.float32(t.threshold[src]))
            left.append(-1)
            right.append(-1)
            li = emit(int(cl[src]), depth + 1)
            ri = emit(int(cr[src]), depth + 1)
            left[idx] = li
            right[idx] = ri
            return idx

        emit(0, 0)
        offsets.append(len(feat))

    return Forest(
        n_features=int(getattr(model, "n_features_in_", 0)),
        n_classes=int(len(leaves[0])),
        tree_offset=np.asarray(offsets, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        # **不能写 `class_names or getattr(...) or ()`**：sklearn 的 classes_ 是
        # numpy 数组，对数组用 `or` 会抛 "truth value ambiguous"。这种写法在
        # 别处（list）能跑，恰恰是最容易漏掉的那种。
        class_names=_pick_names(class_names, model),
    )


def flash_bytes(forest: Forest) -> dict:
    """导出成 C 之后各部分占多少 flash。用来跟 rf_footprint.py 的估算对账。"""
    n_nodes = len(forest.node_feature)
    n_leaves = len(forest.leaf_proba)
    return {
        "node_feature": n_nodes * 2,     # 导出成 uint16
        "node_threshold": n_nodes * 4,
        "node_left": n_nodes * 4,
        "node_right": n_nodes * 4,
        "leaf_proba": n_leaves * forest.n_classes * 4,
        "tree_offset": (forest.n_trees + 1) * 4,
    }
