"""GBDT 的紧凑表示：6 字节一个节点，AoS（一个节点的字段连续存）。

**这不改任何判决。** 只是把同样的树换个存法，Python 参考实现和 C 都会得到
跟原来逐位相同的 margin——测试里钉死了这一点。

节点布局（`__attribute__((packed))`，6 字节）：

    uint8_t feature     内部节点=特征下标（193 < 256 够用）
    float   value       内部节点=阈值；**叶子=叶子分数**（复用同一个槽）
    uint8_t right       bit0-6 = 右孩子的相对偏移；bit7 = NaN 时往左走
                        整个字节 == 0 表示这是叶子

三个前提（都在测试里验了）：

  · **左孩子恒等于 idx+1**，不用存——解析器本来就是先序发出的。
  · 右偏移 = 1 + 左子树大小 ≤ 树大小 - 1 ≤ 126 < 128，所以 bit7 是空的，
    正好放 missing 方向，不用再开一个数组（开了就变成两条 cache line）。
  · 内部节点的右偏移 ≥ 2（左子树至少一个节点），所以 0 可以当叶子标记。

为什么值得做这一趟：

  体积 17 → 6 B/节点（2.8 倍）。更重要的是**访存**——原来是 SoA
  （feat[]/thr[]/left[]/right[] 四个独立数组），访问一个节点要碰 4 个相距几百 KB
  的地址 = 4 条 cache line；现在一个节点 6 字节连续，一条 32 字节的 line 放得下
  5 个，而且**左孩子就是下一个节点**，沿左分支走常常直接命中。
  GR5513 只有 8KB cache，这一项比体积更值钱。
"""

import numpy as np

from .gbdt import Booster

NODE_BYTES = 6
MISSING_LEFT_BIT = 0x80
RIGHT_MASK = 0x7F


class CompactBooster:
    """打包好的 GBDT。字段跟 Booster 一一对应，只是节点换了存法。"""

    def __init__(self, src: Booster):
        n = len(src.node_feature)
        self.n_features = src.n_features
        self.n_classes = src.n_classes
        self.base_score = src.base_score
        self.class_names = src.class_names
        self.tree_offset = src.tree_offset.copy()

        if src.n_features > 255:
            raise ValueError(
                f"{src.n_features} 维特征超出 uint8。要么砍特征到 255 以内，"
                "要么把 feature 字段换成 uint16（每节点多 1 字节）")

        feat = np.zeros(n, np.uint8)
        val = np.zeros(n, np.float32)
        right = np.zeros(n, np.uint8)

        for i in range(n):
            if src.node_left[i] == -1:
                # 叶子：value 槽放叶子分数，right 必须是 0
                feat[i] = 0
                val[i] = src.leaf_value[int(src.node_feature[i])]
                right[i] = 0
                continue

            if int(src.node_left[i]) != i + 1:
                raise ValueError(
                    f"节点 {i} 的左孩子是 {src.node_left[i]}，不是 {i + 1}。"
                    "紧凑布局假定树是**先序**存的、左孩子恒为下一个节点。"
                    "换了解析器的话这个假设要重新验。")
            off = int(src.node_right[i]) - i
            if not (2 <= off <= RIGHT_MASK):
                raise ValueError(
                    f"节点 {i} 的右孩子偏移 {off} 超出 1 字节能表示的范围（2~127）。"
                    "说明有棵树超过 128 个节点——限深到 6 的话不该出现，"
                    "检查一下 max_depth。")
            feat[i] = int(src.node_feature[i])
            val[i] = src.node_threshold[i]
            right[i] = off | (MISSING_LEFT_BIT if src.node_missing_left[i] else 0)

        self.node_feature = feat
        self.node_value = val
        self.node_right = right

    @property
    def n_trees(self):
        return len(self.tree_offset) - 1

    @property
    def n_nodes(self):
        return len(self.node_feature)

    def flash_bytes(self):
        return {
            "nodes": self.n_nodes * NODE_BYTES,
            "tree_offset": (self.n_trees + 1) * 4,
        }

    def margins(self, x):
        """跟 Booster.margins 逐位相同——同样的算术，只是取数的地方换了。

        累加顺序、比较符号（`<`）、NaN 方向、类别摊派（t % n_classes）
        全部照旧，所以不可能有判决差异。
        """
        x = np.asarray(x, np.float32)
        acc = np.full(self.n_classes, np.float32(self.base_score), dtype=np.float32)
        for t in range(self.n_trees):
            node = int(self.tree_offset[t])
            while True:
                r = int(self.node_right[node])
                if (r & RIGHT_MASK) == 0:
                    break
                v = x[int(self.node_feature[node])]
                if v != v:
                    go_left = bool(r & MISSING_LEFT_BIT)
                else:
                    go_left = bool(v < self.node_value[node])
                node = node + 1 if go_left else node + (r & RIGHT_MASK)
            c = t % self.n_classes
            acc[c] = np.float32(acc[c] + self.node_value[node])
        return acc

    def predict(self, x):
        return int(np.argmax(self.margins(x)))
