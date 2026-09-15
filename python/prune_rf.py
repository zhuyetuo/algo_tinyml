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



def _need(path, flag):
    """--features / --labels 的路径检查。跟 --model 一样，不存在就说清楚怎么来——
    这两个文件不是训练的产物，得先用 dump_holdout.py 导一次，而那件事
    不说的话没人知道。"""
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    sys.exit(
        f"{flag} 找不到：{p}\n"
        f"  （相对路径会解析成 {os.path.abspath(p)}）\n"
        "  这个文件要先导一次——在 imu_train 目录下跑：\n"
        "    python ~/algo_tinyml/python/dump_holdout.py \\\n"
        "        --processed-dir data/processed_<DATE>_missing_drop_window \\\n"
        "        --hz 16 --remap configs/remap_custom_3class.yaml")


def resolve_model(path):
    """把 --model 的路径解析清楚，不存在就给一条**能照着改**的报错。

    这些脚本在 algo_tinyml 目录下跑，而模型在 imu_train 里——相对路径会解析到
    algo_tinyml 下面去。直接交给 joblib 的话只会甩一个 FileNotFoundError 的
    traceback，看不出是"路径写错了"还是"模型没训出来"。
    """
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    hint = ""
    if "..." in path:
        # 我在说明里用 `.../` 当占位符，照抄过来就是这个样子。直接点出来
        hint = "\n  路径里有 `...`——那是占位符，要换成真实目录名。"
    if not os.path.isabs(p):
        hint = (f"\n  注意这是相对路径，会解析成 {os.path.abspath(p)}。"
                "\n  模型在 imu_train 里的话要写全：~/imu_train/results/...")
    guess = os.path.expanduser("~/imu_train/results")
    if os.path.isdir(guess):
        hint += f"\n  ~/imu_train/results/ 下现有：{', '.join(sorted(os.listdir(guess))[:5]) or '（空）'}"
    sys.exit(f"找不到模型文件：{p}{hint}")

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


def budget_table(budget, n_classes=3):
    """(棵数 × 深度) 的组合各占多少 flash。按满树算，是**上界**。

    真实的树不满（纯了就停），通常是上界的 40~70%，所以卡在边界上的组合要实测。
    这张表的用处是先把明显没戏的排除掉——省得在 200 棵 × 深 8 上白调半天。
    """
    per_node = 14 + 0.5 * n_classes * 4
    print(f"\n满树上界（{per_node:.0f} B/节点，{n_classes} 分类），预算 "
          f"{budget / 1024:.0f}KB：")
    print(f"{'':>7}" + "".join(f"{'d=' + str(d):>12}" for d in range(3, 8)))
    for n in (50, 100, 200, 300):
        row = f"{n:>4} 棵"
        for d in range(3, 8):
            kb = n * (2 ** (d + 1) - 1) * per_node / 1024
            mark = "✓" if kb * 1024 <= budget else ("~" if kb * 1024 <= budget * 1.5 else "✗")
            row += f"{kb:>10.0f}KB{mark}"
        print(row)
    print("  ✓=预算内  ~=树不满时可能进得去，要实测  ✗=没戏")
    print("""
  **但体积塞得下不等于该这么做。** RF 的准来自"深树低偏差 + bagging 降方差"，
  限深把低偏差那一半拿掉了，而 bagging 只降方差、不降偏差——同样的节点预算下，
  GBDT（本来就为浅树设计，用 boosting 补偏差）几乎一定更好。
  所以**限深的 RF ≈ 一个更差的 GBDT**。下面那条曲线要是掉得厉害，别硬调 RF，
  直接换 GBDT 试。""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", help="[N, n_features] 留出集特征 .npy")
    ap.add_argument("--labels", help="[N] 留出集标签 .npy")
    ap.add_argument("--budget", type=int, default=128 * 1024, help="留给模型的 flash 字节")
    ap.add_argument("--depths", default="", help="要扫的深度，逗号分隔；默认自动")
    ap.add_argument("--trees", default="",
                    help="另外扫一遍棵数（比如 20,50,100）。**RF 减树是合法的**——"
                         "树是 bagging 出来的、独立同分布，取前 n 棵在统计上没区别。"
                         "GBDT 不行，那边树是顺序的。")
    ap.add_argument("--depth-with-trees", type=int, default=0,
                    help="扫棵数时固定用这个 max_depth（配合 --trees）")
    ap.add_argument("--min-samples-leaf", default="",
                    help="另外扫一遍 min_samples_leaf，逗号分隔（比如 1,5,10,20）")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn。这个脚本要在训练机上跑。")

    bundle = joblib.load(resolve_model(args.model))
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle

    # 不限深的树可能很深，递归会爆栈
    sys.setrecursionlimit(100000)

    X = y = None
    if args.features and args.labels:
        X = np.load(_need(args.features, "--features")).astype(np.float32)
        y = np.load(_need(args.labels, "--labels")).astype(np.int64)
        if len(X) != len(y):
            sys.exit(f"特征 {len(X)} 行、标签 {len(y)} 行，对不上。错位的话每一个"
                     "指标都是错的，而且看起来完全正常——用 dump_holdout.py 一起导。")
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
    budget_table(args.budget, n_cls)
    print("\n不剪：")
    report("原样", full)

    print("\n按深度剪：")
    for d in depths:
        report(f"max_depth={d}", from_sklearn(model, max_depth=d))

    trees = [int(v) for v in args.trees.split(",") if v]
    if trees:
        d = args.depth_with_trees or None
        tag_d = f"、深度限到 {d}" if d else "、不限深"
        print(f"\n按棵数剪{tag_d}（RF 的树独立同分布，取前 n 棵是合法的）：")
        for n in trees:
            report(f"{n} 棵", from_sklearn(model, max_depth=d, n_trees=n))

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
