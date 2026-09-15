"""让分析脚本同时吃 XGBoost 和 sklearn 随机森林。

两种模型在"怎么变小"这件事上的轴不一样，这个差别不能抹平：

  · GBDT 的轴是**轮数**，而且截断是**精确**的（boosting 顺序累加，前 K 轮
    跟总共训多少轮无关）。
  · RF 的轴是**深度 / 叶子最小样本数**，截断是**近似**的（原来的分裂是冲着
    "后面还要再分好几层"选的，半路砍掉用的是并非为浅树优化的分裂点）。

所以适配层不只统一接口，还要把"这个数准不准"一起带出来——
`exact` 为假的时候，脚本会在输出里提醒那张表是悲观下界、定下参数还要重训确认。

打分那一侧：GBDT 给 margin（原始分数），RF 给概率。两者**都是保序的**，
所以 argmax 和"加个偏置挪工作点"这两件事在两边含义一致。
"""

import numpy as np

from .forest import Forest, flash_bytes as forest_flash
from .gbdt import Booster, flash_bytes as gbdt_flash


class ModelAdapter:
    def __init__(self, model, class_names=None):
        self.class_names = list(class_names or [])
        if hasattr(model, "get_booster"):
            from .gbdt import from_xgboost
            self.kind = "gbdt"
            self.obj = from_xgboost(model, class_names=self.class_names)
            self.n_classes = self.obj.n_classes
            self.axis_name = "轮数"
            self.exact = True
            self.total = self.obj.n_trees // self.obj.n_classes
        elif hasattr(model, "estimators_"):
            from .forest import from_sklearn
            self.kind = "rf"
            self.obj = from_sklearn(model, class_names=self.class_names)
            self.n_classes = self.obj.n_classes
            self.axis_name = "max_depth"
            self.exact = False
            self._sk = model
            self.total = int(max(
                (e.tree_.max_depth for e in model.estimators_), default=0))
        else:
            raise TypeError(
                f"既不是 XGBoost（没有 get_booster）也不是 sklearn 集成树"
                f"（没有 estimators_），实际是 {type(model)}")

    # ── 沿着各自的轴变小 ────────────────────────────────────────────────
    def variant(self, v, n_trees=None, quantize_leaves=False):
        """v 对 GBDT 是轮数，对 RF 是 max_depth。返回一个能 scores() 的对象。

        n_trees / quantize_leaves 只对 RF 有意义，因为**端上真正要跑的那一格是
        「棵数 × 深度 × uint8 叶子」三件事一起生效之后的模型**。只按深度截断
        得到的事件级指标对应不上任何一个塞得进 flash 的配置——拿那个数
        去跟 CNN 比，是在比两个不同的东西。

        GBDT 传这两个参数**直接报错**，不静默忽略：减树对 GBDT 是非法的
        （树是顺序的，第 k 棵拟合前 k-1 棵的残差），静默忽略会让调用方
        以为自己评的是减过树的模型。
        """
        if self.kind == "gbdt":
            if n_trees is not None or quantize_leaves:
                raise ValueError(
                    "n_trees / quantize_leaves 只对 RF 有意义。GBDT 的树是顺序的，"
                    "只能从头按轮数截断、不能挑树；叶子存的也不是概率。")
            return _GbdtView(self.obj.truncate(int(v)))
        from .forest import from_sklearn
        from .forest import quantize_leaves as _ql
        f = from_sklearn(self._sk, class_names=self.class_names,
                         max_depth=int(v), n_trees=n_trees)
        return _RfView(_ql(f) if quantize_leaves else f)

    def full(self):
        return _GbdtView(self.obj) if self.kind == "gbdt" else _RfView(self.obj)

    def default_axis(self):
        if self.kind == "gbdt":
            return [v for v in (5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 150,
                                self.total) if 1 <= v <= self.total]
        return [v for v in (3, 4, 5, 6, 8, 10, 12, 16, 20, self.total)
                if 1 <= v <= self.total]

    def caveat(self):
        if self.exact:
            return ("**这张表是精确的**：GBDT 按轮数截断 == 一开始就用那个轮数训"
                    "（boosting 顺序累加，前 K 棵树跟总共训多少轮无关）。"
                    "选定之后不用再重训确认。")
        return ("⚠ **这张表是悲观的下界**：RF 按深度截断跟「训练时设 max_depth」"
                "不等价——原来那些分裂是冲着「后面还要再分好几层」选的。"
                "按选定的深度**重训一次**通常会比表上更好，所以定了参数还要重训确认。")


class _GbdtView:
    def __init__(self, b: Booster):
        self.b = b
        self.n_nodes = len(b.node_feature)
        self.n_classes = b.n_classes

    def scores(self, x):
        """原始 margin。保序，所以 argmax 和加偏置的含义跟概率一致。"""
        return self.b.margins(x)

    def flash(self, per_node=0):
        if per_node:
            return per_node * self.n_nodes
        return sum(gbdt_flash(self.b).values())

    def compact_flash(self):
        from .gbdt_compact import CompactBooster
        return sum(CompactBooster(self.b).flash_bytes().values())


class _RfView:
    def __init__(self, f: Forest):
        self.f = f
        self.n_nodes = len(f.node_feature)
        self.n_classes = f.n_classes

    def scores(self, x):
        """各树叶子概率的平均（照抄 sklearn）。跟 margin 一样保序。"""
        return self.f.predict_proba(x)

    def flash(self, per_node=0):
        if per_node:
            return per_node * self.n_nodes
        return sum(forest_flash(self.f).values())

    def compact_flash(self):
        """RF 的紧凑布局：节点还是 6 字节，但叶子要存 n_classes 个概率，
        不能像 GBDT 那样塞进阈值那个槽——所以叶子表另算。"""
        n_leaves = len(self.f.leaf_proba)
        return 6 * self.n_nodes + n_leaves * self.n_classes * 4
