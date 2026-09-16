"""随机森林的紧凑表示：7 字节一个节点（AoS）+ uint8 叶子。

跟 gbdt_compact.py 是同一个思路，但**节点不是 6 字节**——这一点我一开始记错了，
对着一个不存在的编码报了好几轮体积。差别在右孩子偏移：

  · GBDT 限深 6，一棵树最多 127 个节点，偏移塞得进 7 bit，剩一位放 NaN 方向；
  · RF 深度 10、一棵树平均 590 个节点，偏移必须 uint16。

所以 RF 的节点是：

    uint8_t  feature     内部节点=特征下标（193 < 256）
    float    threshold   内部节点=阈值；**叶子=叶子表下标**（4 字节复用成 uint32）
    uint16_t right       右孩子的相对偏移；**0 表示这是叶子**

叶子单独一张表，每类 1 字节（概率 × 255）。叶子为什么不能像 GBDT 那样塞进
value 槽：那边叶子是一个标量分数，这边是 n_classes 个概率。

三个前提，都在测试里验了：
  · 左孩子恒等于 idx+1（先序发出的），不用存；
  · 内部节点的右偏移 ≥ 2，所以 0 可以当叶子标记；
  · 特征数 ≤ 255。

**叶子量化会改判决**：相差不到 1/255 的两类会翻。这不是"几乎不变"，
是要实测的——quantize_leaves 那边有专门的测试，这里也有一条对比。

7 字节不对齐，读 threshold 是从偏移 1 开始的非对齐 4 字节。
**Cortex-M4 硬件支持非对齐字访问**，编译出来就是一条 ldr.w，不会拆成字节。
（M0 不支持，换核要重新验。）
"""

import numpy as np

from .forest import Forest

NODE_BYTES = 7
LEAF_MARK = 0            # right == 0 → 叶子


class CompactForest:
    """打包好的森林。判决跟 Forest 一致（叶子量化带来的差异除外）。"""

    def __init__(self, src: Forest, levels: int = 255):
        n = len(src.node_feature)
        if src.n_features > 255:
            raise ValueError(
                f"{src.n_features} 维特征超出 uint8。要么砍到 255 以内，"
                "要么把 feature 换成 uint16（每节点多 1 字节）")

        self.n_features = int(src.n_features)
        self.n_classes = int(src.n_classes)
        self.class_names = src.class_names
        self.tree_offset = np.asarray(src.tree_offset, np.int32).copy()
        self.levels = int(levels)

        feat = np.zeros(n, np.uint8)
        thr = np.zeros(n, np.float32)
        leaf_idx = np.zeros(n, np.uint32)
        right = np.zeros(n, np.uint16)
        is_leaf = np.zeros(n, bool)

        for i in range(n):
            if src.node_left[i] == -1:
                is_leaf[i] = True
                leaf_idx[i] = int(src.node_feature[i])
                continue
            if int(src.node_left[i]) != i + 1:
                raise ValueError(
                    f"节点 {i} 的左孩子是 {src.node_left[i]}，不是 {i + 1}。"
                    "紧凑布局假定树是**先序**存的、左孩子恒为下一个节点。")
            off = int(src.node_right[i]) - i
            if not (2 <= off <= 0xFFFF):
                raise ValueError(
                    f"节点 {i} 的右孩子偏移 {off} 超出 uint16。"
                    "说明有棵树超过 65535 个节点——那得先限深。")
            feat[i] = int(src.node_feature[i])
            thr[i] = src.node_threshold[i]
            right[i] = off

        self.node_feature = feat
        self.node_threshold = thr
        self.node_leaf_idx = leaf_idx
        self.node_right = right
        self.is_leaf = is_leaf

        # 叶子量化成 uint8。四舍五入远离零，跟 quantize_leaves 同一套规则——
        # np.round 是银行家舍入，会在 .5 上往偶数走，跟 C 走反
        p = np.asarray(src.leaf_proba, np.float64)
        q = np.clip(np.floor(p * self.levels + 0.5), 0, self.levels)
        self.leaf_u8 = q.astype(np.uint8)

    @property
    def n_trees(self):
        return len(self.tree_offset) - 1

    @property
    def n_nodes(self):
        return len(self.node_feature)

    @property
    def n_leaves(self):
        return len(self.leaf_u8)

    def flash_bytes(self):
        return {
            "nodes": self.n_nodes * NODE_BYTES,
            "leaves": self.n_leaves * self.n_classes,
            "tree_offset": (self.n_trees + 1) * 4,
        }

    def votes(self, x):
        """各棵树叶子的 uint8 概率之和，**整数**。

        不除以棵数、不还原成概率：argmax 对正的常数缩放不变，所以端上直接对
        这个整数取 argmax 就行。整数累加没有结合律问题，板上和 PC 逐位一定一样——
        这是紧凑版相对原版的一个额外好处（原版是 float32 求和）。
        """
        x = np.asarray(x, np.float32)
        acc = np.zeros(self.n_classes, np.int64)
        for t in range(self.n_trees):
            node = int(self.tree_offset[t])
            while True:
                r = int(self.node_right[node])
                if r == LEAF_MARK:
                    break
                f = int(self.node_feature[node])
                # sklearn 是 <= 走左。写成 < 的话，特征值正好等于阈值的样本会走反——
                # 而阈值本来就是从样本值来的，"正好等于"一点也不罕见
                node = node + 1 if x[f] <= self.node_threshold[node] else node + r
            acc += self.leaf_u8[int(self.node_leaf_idx[node])].astype(np.int64)
        return acc

    def predict_proba(self, x):
        """还原成概率，只为了给服务端一个置信度。端上不做这一步。"""
        v = self.votes(x).astype(np.float64)
        s = v.sum()
        return (v / s) if s > 0 else np.full(self.n_classes, 1.0 / self.n_classes)

    def predict(self, x):
        return int(np.argmax(self.votes(x)))


def pack_nodes(cf: CompactForest) -> np.ndarray:
    """打成 [n_nodes, 7] 的 uint8，就是板上那块内存的字节序（小端）。

    导出和 C 都从这里拿，不各写一遍——写两遍就迟早有一遍错，
    而字节序错了不会报错，只会让判决莫名其妙。
    """
    n = cf.n_nodes
    out = np.zeros((n, NODE_BYTES), np.uint8)
    out[:, 0] = cf.node_feature
    # 内部节点存阈值，叶子那 4 个字节存叶子表下标（uint32）
    words = np.where(cf.is_leaf,
                     cf.node_leaf_idx.astype(np.uint32),
                     cf.node_threshold.view(np.uint32))
    out[:, 1] = words & 0xFF
    out[:, 2] = (words >> 8) & 0xFF
    out[:, 3] = (words >> 16) & 0xFF
    out[:, 4] = (words >> 24) & 0xFF
    out[:, 5] = cf.node_right & 0xFF
    out[:, 6] = (cf.node_right >> 8) & 0xFF
    return out
