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


def from_sklearn(model, class_names=None) -> Forest:
    """从 sklearn 的 RandomForestClassifier 抽出来。

    只读 tree_ 的那几个扁平数组，不依赖 sklearn 的对象结构——这样换 sklearn 版本
    也不容易崩（tree_ 的这几个字段十来年没变过）。
    """
    ests = getattr(model, "estimators_", None)
    if ests is None:
        raise TypeError(f"不是随机森林（没有 estimators_），实际是 {type(model)}")

    offsets = [0]
    feat, thr, left, right = [], [], [], []
    leaves = []
    for est in ests:
        t = est.tree_
        base = offsets[-1]
        cl = np.asarray(t.children_left, np.int64)
        cr = np.asarray(t.children_right, np.int64)
        for i in range(int(t.node_count)):
            if cl[i] == -1:
                v = np.asarray(t.value[i], np.float64).reshape(-1)
                s = v.sum()
                # 叶子上存的是各类样本数，要归一成概率。全零在正常训练里不会出现，
                # 但 sample_weight 全零之类的边角情况会——那时候给均匀分布，
                # 而不是让它变成 nan 一路传到 argmax
                p = (v / s) if s > 0 else np.full_like(v, 1.0 / len(v))
                feat.append(len(leaves))
                leaves.append(p.astype(np.float32))
                thr.append(np.float32(0.0))
                left.append(-1)
                right.append(-1)
            else:
                feat.append(int(t.feature[i]))
                thr.append(np.float32(t.threshold[i]))
                left.append(base + int(cl[i]))
                right.append(base + int(cr[i]))
        offsets.append(base + int(t.node_count))

    return Forest(
        n_features=int(getattr(model, "n_features_in_", 0)),
        n_classes=int(len(leaves[0])),
        tree_offset=np.asarray(offsets, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        class_names=tuple(class_names or getattr(model, "classes_", []) or ()),
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
