"""扫一遍「剪到多深 → 占多少 flash → 准确率掉多少」，**不用重训**。

要在有 sklearn 的机器上跑。

为什么能不重训：sklearn 在**每个**节点上都存了那个节点的类别分布，所以把一棵
训好的树在深度 d 处剪掉、该节点直接当叶子用，是个精确操作。它跟"训练时就设
max_depth=d"不等价（训练时限深，分裂点的选择会不同），但它能在几秒内给出整条
曲线，而重训一轮要等很久。**先用它定范围，定下来再按那个参数重训一次上线。**

用法：
    python python/prune_rf.py --model xxx.pkl --features feats.npy --labels y.npy
    python python/prune_rf.py --model xxx.pkl --budget 131072      # 只看体积

--features / --labels 是留出集（别用训练集——训练集上剪枝永远显得"掉点很多"，
因为被剪掉的正是那些把训练样本背下来的分支）。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.forest import flash_bytes, from_sklearn  # noqa: E402


def _macro_f1(y, p, n):
    """看 macro-F1 不看准确率：睡觉占绝大多数，准确率被它主导——
    一个"永远猜睡觉"的模型准确率能很好看，而它在唯一有用的那一类上全错。"""
    out = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        out.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", help="[N, n_features] 留出集特征 .npy")
    ap.add_argument("--labels", help="[N] 留出集标签 .npy")
    ap.add_argument("--budget", type=int, default=128 * 1024, help="留给模型的 flash 字节")
    ap.add_argument("--depths", default="", help="要扫的深度，逗号分隔；默认自动")
    ap.add_argument("--min-samples-leaf", default="",
                    help="另外扫一遍 min_samples_leaf，逗号分隔（比如 1,5,10,20）")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn。这个脚本要在训练机上跑。")

    bundle = joblib.load(args.model)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle

    # 不限深的树可能很深，递归会爆栈
    sys.setrecursionlimit(100000)

    X = y = None
    if args.features and args.labels:
        X = np.load(args.features).astype(np.float32)
        y = np.load(args.labels).astype(np.int64)
        print(f"留出集 {len(X)} 条")
    else:
        print("没给 --features/--labels，只报体积不报准确率。"
              "**强烈建议给**：只看体积的话，你只知道剪到多小，不知道剪坏没有。")

    full = from_sklearn(model)
    n_cls = full.n_classes
    depths = [int(d) for d in args.depths.split(",") if d] or \
        [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]

    def report(tag, f):
        b = sum(flash_bytes(f).values())
        n = len(f.node_feature)
        line = f"  {tag:<18} {n:>8} 节点  {b:>9,} B ({b / 1024:>7.1f} KB)"
        line += "  塞得下" if b <= args.budget else f"  超 {b / args.budget:.1f}×"
        if X is not None:
            pred = np.array([f.predict(x) for x in X])
            line += f"   macro-F1 {_macro_f1(y, pred, n_cls):.4f}"
        print(line)

    print(f"\nflash 预算 {args.budget:,} B（{args.budget / 1024:.0f} KB）")
    print("\n不剪：")
    report("原样", full)

    print("\n按深度剪：")
    for d in depths:
        report(f"max_depth={d}", from_sklearn(model, max_depth=d))

    msl = [int(v) for v in args.min_samples_leaf.split(",") if v]
    if msl:
        print("\n按叶子最小样本数剪（通常比限深掉点更少，因为它只砍掉"
              "真正没统计量的那些分支）：")
        for m in msl:
            report(f"min_samples_leaf={m}", from_sklearn(model, min_samples_leaf=m))

    print("""
怎么读这张表：
  - 找**掉点最少且塞得下**的那一行，不是最小的那一行。
  - 曲线通常有个拐点：拐点之前掉点很少（剪掉的是过拟合分支），之后陡降
    （开始剪到真信息了）。停在拐点前一两档。
  - 这里是**就地截断**，跟"训练时设这个参数"不等价。定下参数之后要按它重训一次，
    重训的结果通常**比这张表还好一点**（训练时限深会挑更适合浅树的分裂点）。""")


if __name__ == "__main__":
    main()
