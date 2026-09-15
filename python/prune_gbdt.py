"""扫「减到多少轮 → 占多少 flash → 掉多少点」。**不用重训，而且是精确的。**

跟 RF 那个 prune_rf.py 有一个本质区别，值得说清楚：

  - RF 按深度截断是**近似**：原来那些分裂是冲着"后面还要再分好几层"选的，
    半路砍掉用的是一批并非为浅树优化的分裂点。所以那张表是悲观的下界。
  - **GBDT 按轮数截断是精确的**：boosting 顺序累加，第 k 棵树拟合的是前 k-1 棵
    之后的残差，所以前 K 棵树跟总共训多少轮完全无关。**取前 K 轮 == 用
    n_estimators=K 训出来的模型**，一个 bit 都不差。

所以这张表**就是最终答案**，不需要再按选定的轮数重训一次去确认。

用法：
    python python/prune_gbdt.py \
        --model ~/imu_train/results/.../xgb/ml_xgb.pkl \
        --features holdout_feats.npy --labels holdout_y.npy \
        --classes 活动,睡觉,抓挠,未佩戴,甩身体 \
        --focus 抓挠

--focus 指定一个"最关心的类别"，表里会单独列它的 P/R/F1——macro-F1 会被
多数类拉着走，而产品价值往往全在某一个少数类上。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.gbdt import flash_bytes, from_xgboost  # noqa: E402


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


def prf(y, p, c):
    tp = int(np.sum((p == c) & (y == c)))
    fp = int(np.sum((p == c) & (y != c)))
    fn = int(np.sum((p != c) & (y == c)))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def macro_f1(y, p, n):
    return float(np.mean([prf(y, p, c)[2] for c in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", help="[N, n_features] 留出集特征 .npy")
    ap.add_argument("--labels", help="[N] 留出集标签 .npy")
    ap.add_argument("--classes", default="", help="类别名，逗号分隔")
    ap.add_argument("--focus", default="", help="最关心的类别名（比如 抓挠）")
    ap.add_argument("--budget", type=int, default=128 * 1024)
    ap.add_argument("--rounds", default="", help="要扫的轮数，逗号分隔；默认自动")
    ap.add_argument("--per-node", type=float, default=0,
                    help="每节点字节数。默认按当前编码算；想看紧凑编码下的体积传 6.125")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib。这个脚本要在训练机上跑。")

    bundle = joblib.load(resolve_model(args.model))
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    if not hasattr(model, "get_booster"):
        sys.exit(f"不是 XGBoost 模型，实际是 {type(model)}。RF 请用 prune_rf.py。")

    names = [s for s in args.classes.split(",") if s] or None
    b = from_xgboost(model, class_names=names)
    total_rounds = b.n_trees // b.n_classes
    print(f"{b.n_trees} 棵树 = {total_rounds} 轮 × {b.n_classes} 类，"
          f"{len(b.node_feature):,} 个节点，特征 {b.n_features} 维")

    focus_idx = None
    if args.focus and names and args.focus in names:
        focus_idx = names.index(args.focus)
        print(f"重点看类别「{args.focus}」（下标 {focus_idx}）")

    X = y = None
    if args.features and args.labels:
        X = np.load(_need(args.features, "--features")).astype(np.float32)
        y = np.load(_need(args.labels, "--labels")).astype(np.int64)
        if len(X) != len(y):
            sys.exit(f"特征 {len(X)} 行、标签 {len(y)} 行，对不上。错位的话每一个"
                     "指标都是错的，而且看起来完全正常——用 dump_holdout.py 一起导。")
        print(f"留出集 {len(X)} 条")
    else:
        print("没给 --features/--labels，只报体积。**强烈建议给**——"
              "只看体积的话你只知道减到多小，不知道减坏没有。")

    def per_node_bytes(bb):
        if args.per_node:
            return args.per_node
        fb = flash_bytes(bb)
        return sum(fb.values()) / max(len(bb.node_feature), 1)

    rounds = [int(r) for r in args.rounds.split(",") if r]
    if not rounds:
        rounds = sorted({r for r in (5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100,
                                     150, total_rounds) if 1 <= r <= total_rounds})

    print(f"\nflash 预算 {args.budget / 1024:.0f} KB"
          + (f"（按 {args.per_node} B/节点算）" if args.per_node else "（按当前编码算）"))
    hdr = f"{'轮数':>6}{'节点数':>10}{'flash':>11}{'':>8}"
    if X is not None:
        hdr += f"{'macro-F1':>11}"
        if focus_idx is not None:
            hdr += f"{args.focus + ' P':>10}{args.focus + ' R':>10}{args.focus + ' F1':>11}"
    print(hdr)
    print("-" * len(hdr))

    best_fit = None
    for r in rounds:
        t = b.truncate(r)
        n = len(t.node_feature)
        size = per_node_bytes(t) * n
        fits = size <= args.budget
        line = (f"{r:>6}{n:>10,}{size / 1024:>9.0f}KB"
                f"{'  塞得下' if fits else f'  超{size / args.budget:>4.1f}×'}")
        if X is not None:
            pred = np.array([t.predict(x) for x in X])
            line += f"{macro_f1(y, pred, b.n_classes):>11.4f}"
            if focus_idx is not None:
                pr, rc, f1 = prf(y, pred, focus_idx)
                line += f"{pr:>10.3f}{rc:>10.3f}{f1:>11.3f}"
            if fits:
                best_fit = (r, macro_f1(y, pred, b.n_classes))
        print(line)

    print(f"""
**这张表是精确的，不是估算。** GBDT 按轮数截断 == 一开始就用那个轮数训——
boosting 顺序累加，前 K 棵树跟总共训多少轮无关。所以**选定轮数之后不用再重训确认**。
（RF 按深度截断就不是这样，那个是近似，见 prune_rf.py。）

怎么读：
  - 看**你最关心那一类**的 F1，别只看 macro-F1。macro-F1 会被多数类拉着走，
    而产品价值往往全在某一个少数类上。
  - 曲线通常在某个轮数之后就平了（后面的轮在修越来越小的残差）。
    找那个拐点，不是找最小的。
  - 「塞得下」那一列按 {'紧凑编码' if args.per_node else '当前编码'}算。
    当前编码是 17 B/节点；不改任何判决能压到约 6.1 B/节点
    （feature 换 uint8、左孩子隐式、右孩子树内相对、叶子值复用 threshold、
    missing 压成 bitmap），也就是能多放 2.8 倍的轮数。
    想看那个口径就传 --per-node 6.125。""")
    if best_fit:
        print(f"\n当前口径下塞得进预算的最大轮数：{best_fit[0]} 轮，"
              f"macro-F1 {best_fit[1]:.4f}")


if __name__ == "__main__":
    main()
