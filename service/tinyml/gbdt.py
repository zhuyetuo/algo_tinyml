"""GBDT（先支持 XGBoost）的端侧表示 + 参考前向。

跟随机森林的三个关键区别，每一条写错都不会报错、只会让结果悄悄不对：

1. **判决符号是 `<`，不是 `<=`。** sklearn 的树是 `x <= threshold` 走左，
   XGBoost 是 `x < split_condition` 走 yes。特征值正好等于阈值的样本会走反，
   而阈值本来就是从样本值来的，"正好等于"一点也不罕见。
2. **叶子存一个分数（margin），不是一组概率。** 所有树的分数**相加**（不是平均），
   加上 base_score，最后过 softmax 才是概率。
3. **多分类每一轮给每个类别各训一棵树。** 5 分类 50 轮是 **250 棵**不是 50 棵。
   树 t 属于类别 `t % n_classes`。算预算时漏掉这条会差 n_classes 倍。

**端上不做 softmax**：softmax 是保序的（`softmax(z)_i > softmax(z)_j ⟺ z_i > z_j`），
所以 argmax 直接在 margin 上取就行，结果完全一样，还省掉一堆 expf——
expf 属于 libm，各实现末位不保证一致，不用它就少一处对不齐的来源。
真要概率（比如按置信度过滤）再单独调 softmax。
"""

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Booster:
    n_features: int
    n_classes: int
    base_score: float
    tree_offset: np.ndarray     # int32 [n_trees + 1]
    node_feature: np.ndarray    # int32 [n_nodes]：内部节点=特征下标；叶子=leaf_value 下标
    node_threshold: np.ndarray  # float32 [n_nodes]
    node_left: np.ndarray       # int32 [n_nodes]，-1 表示叶子
    node_right: np.ndarray      # int32 [n_nodes]
    node_missing_left: np.ndarray  # uint8 [n_nodes]：特征是 NaN 时走左边吗
    leaf_value: np.ndarray      # float32 [n_leaves]，一个分数
    class_names: tuple = ()

    @property
    def n_trees(self):
        return len(self.tree_offset) - 1

    def margins(self, x):
        """→ float32 [n_classes] 的原始分数（没过 softmax）。

        累加顺序按树下标从小到大，跟 C 一致——浮点加法不满足结合律。
        """
        x = np.asarray(x, np.float32)
        acc = np.full(self.n_classes, np.float32(self.base_score), dtype=np.float32)
        for t in range(self.n_trees):
            node = int(self.tree_offset[t])
            while self.node_left[node] != -1:
                f = int(self.node_feature[node])
                v = x[f]
                if v != v:      # NaN
                    go_left = bool(self.node_missing_left[node])
                else:
                    # **XGBoost 是 `<`**，不是 sklearn 的 `<=`
                    go_left = bool(v < self.node_threshold[node])
                node = int(self.node_left[node] if go_left else self.node_right[node])
            c = t % self.n_classes
            acc[c] = np.float32(acc[c] + self.leaf_value[int(self.node_feature[node])])
        return acc

    def predict(self, x):
        """softmax 保序，所以 argmax 直接在 margin 上取，跟先 softmax 再 argmax 一样。"""
        return int(np.argmax(self.margins(x)))

    def truncate(self, rounds):
        """只留前 `rounds` 轮。**这跟一开始就用 n_estimators=rounds 训出来的模型
        完全相同，不是近似**——boosting 是顺序累加的，第 k 棵树拟合的是前 k-1 棵
        之后的残差，所以前 K 棵树跟总共训多少轮无关。

        （对比：随机森林按深度截断**是**近似——原来那些分裂是冲着"后面还要再分
        好几层"选的，半路砍掉用的是一批并非为浅树优化的分裂点。GBDT 没这个问题。）

        所以「减到多少轮、掉多少点」这条曲线，用一个训好的模型就能精确算出来，
        一次重训都不用。
        """
        keep = int(rounds) * self.n_classes
        if keep <= 0 or keep > self.n_trees:
            raise ValueError(f"rounds={rounds} 超出范围（总共 {self.n_trees // self.n_classes} 轮）")
        n_nodes = int(self.tree_offset[keep])
        # 节点数组按树先序排列，所以前 keep 棵树的节点就是前 n_nodes 个，
        # 不用重新编号——孩子下标都指向本树内部，天然还在范围里
        used_leaves = self.node_feature[:n_nodes][self.node_left[:n_nodes] == -1]
        max_leaf = int(used_leaves.max()) + 1 if len(used_leaves) else 0
        return Booster(
            n_features=self.n_features,
            n_classes=self.n_classes,
            base_score=self.base_score,
            tree_offset=self.tree_offset[:keep + 1].copy(),
            node_feature=self.node_feature[:n_nodes].copy(),
            node_threshold=self.node_threshold[:n_nodes].copy(),
            node_left=self.node_left[:n_nodes].copy(),
            node_right=self.node_right[:n_nodes].copy(),
            node_missing_left=self.node_missing_left[:n_nodes].copy(),
            leaf_value=self.leaf_value[:max_leaf].copy(),
            class_names=self.class_names,
        )

    def predict_proba(self, x):
        m = self.margins(x).astype(np.float64)
        e = np.exp(m - m.max())     # 减最大值防溢出
        return (e / e.sum()).astype(np.float32)


def _parse_tree(node, feat_idx, thr, left, right, miss_left, leaves, feat_name_to_idx):
    """递归展开一棵 XGBoost JSON dump 的树，返回根在扁平数组里的下标。

    孩子的下标要等孩子发完才知道，所以先占位再回填。
    """
    idx = len(feat_idx)
    if "leaf" in node:
        feat_idx.append(len(leaves))
        leaves.append(np.float32(node["leaf"]))
        thr.append(np.float32(0.0))
        left.append(-1)
        right.append(-1)
        miss_left.append(0)
        return idx

    split = node["split"]
    if isinstance(split, str) and split.startswith("f") and split[1:].isdigit():
        fi = int(split[1:])
    elif split in feat_name_to_idx:
        fi = feat_name_to_idx[split]
    else:
        raise ValueError(
            f"看不懂的特征名 {split!r}。dump 的时候别传 feature_names，"
            "让它输出 f0/f1/... 这种下标形式；传了名字又对不上表，"
            "特征就会整体错位而模型照样给得出结果")

    feat_idx.append(fi)
    thr.append(np.float32(node["split_condition"]))
    left.append(-1)
    right.append(-1)
    # XGBoost 的 missing 指向 yes 还是 no。我们的特征不该有 NaN，但真出了
    # NaN（比如某个通道传感器坏了导致方差为 0、除出 nan），行为必须跟训练时一致，
    # 否则那条样本会走到一个谁也没预料的分支
    kids = {k["nodeid"]: k for k in node["children"]}
    miss_left.append(1 if node.get("missing", node["yes"]) == node["yes"] else 0)

    li = _parse_tree(kids[node["yes"]], feat_idx, thr, left, right, miss_left,
                     leaves, feat_name_to_idx)
    ri = _parse_tree(kids[node["no"]], feat_idx, thr, left, right, miss_left,
                     leaves, feat_name_to_idx)
    left[idx] = li
    right[idx] = ri
    return idx


def from_xgboost_dumps(dumps, n_features, n_classes, base_score=0.5,
                       class_names=None, feature_names=None) -> Booster:
    """从 `booster.get_dump(dump_format="json")` 的结果构建。

    dumps: 每棵树一个 JSON 字符串，顺序就是 XGBoost 的树顺序
           （多分类是 轮0类0, 轮0类1, ..., 轮1类0, ...）。
    """
    name_map = {n: i for i, n in enumerate(feature_names or [])}
    offsets = [0]
    feat_idx, thr, left, right, miss_left, leaves = [], [], [], [], [], []
    for d in dumps:
        _parse_tree(json.loads(d) if isinstance(d, str) else d,
                    feat_idx, thr, left, right, miss_left, leaves, name_map)
        offsets.append(len(feat_idx))

    if n_classes > 1 and (len(dumps) % n_classes) != 0:
        raise ValueError(
            f"{len(dumps)} 棵树除不尽 {n_classes} 个类别。多分类每轮给每个类别各训"
            "一棵，树数必须是类别数的整数倍——除不尽说明 n_classes 给错了，"
            "而给错了树会被摊到错误的类别上，模型照样给得出结果")

    return Booster(
        n_features=int(n_features),
        n_classes=int(n_classes),
        base_score=float(base_score),
        tree_offset=np.asarray(offsets, np.int32),
        node_feature=np.asarray(feat_idx, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        node_missing_left=np.asarray(miss_left, np.uint8),
        leaf_value=np.stack(leaves).astype(np.float32) if leaves else np.zeros(0, np.float32),
        class_names=tuple(class_names or ()),
    )


def from_xgboost(model, class_names=None) -> Booster:
    """从 sklearn 接口的 XGBClassifier 直接构建。要装 xgboost。"""
    booster = model.get_booster()
    dumps = booster.get_dump(dump_format="json")
    n_classes = int(getattr(model, "n_classes_", 1))
    n_features = int(getattr(model, "n_features_in_", 0))
    # base_score 在不同版本里的取法不一样，取不到就用 xgboost 的默认 0.5。
    # 取错会让所有类别的 margin 同时平移一个常数——**对 argmax 没有影响**
    # （softmax 保序，平移不改顺序），只影响概率值，所以这里容错是安全的。
    base = getattr(model, "base_score", None)
    if base is None:
        try:
            base = float(json.loads(booster.save_config())["learner"]
                         ["learner_model_param"]["base_score"])
        except Exception:
            base = 0.5
    return from_xgboost_dumps(dumps, n_features, n_classes, float(base),
                              class_names or list(getattr(model, "classes_", [])))


def flash_bytes(b: Booster) -> dict:
    n_nodes = len(b.node_feature)
    n_leaves = len(b.leaf_value)
    return {
        "node_feature": n_nodes * 2,       # uint16
        "node_threshold": n_nodes * 4,
        "node_left": n_nodes * 4,
        "node_right": n_nodes * 4,
        "node_missing": n_nodes * 1,
        "leaf_value": n_leaves * 4,        # 一个分数，不是 n_classes 个概率
        "tree_offset": (b.n_trees + 1) * 4,
    }


def softmax_ref(m):
    m = np.asarray(m, np.float64)
    e = np.exp(m - m.max())
    return (e / e.sum()).astype(np.float32)


__all__ = ["Booster", "from_xgboost", "from_xgboost_dumps", "flash_bytes",
           "softmax_ref", "math"]
