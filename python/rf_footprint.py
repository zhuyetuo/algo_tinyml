"""量一棵随机森林搬到 GR5513 上要占多少 flash，以及砍到多少才塞得下。

为什么要有这个脚本：「RF 能不能上端侧」这个问题不该靠估。sklearn 的 RandomForest
默认 max_depth=None，节点数完全由数据决定——同一份配置，几百条样本和几万条样本
差一个数量级。只有把 .pkl 打开数一遍才知道。

这个脚本要在**有 sklearn 的机器上**跑（训练机 / 标注平台那台），因为它要加载 .pkl。

用法：
    python python/rf_footprint.py --model ~/imu_train/results/.../rf/xxx.pkl
    python python/rf_footprint.py --model xxx.pkl --flash-budget 131072
"""

import argparse
import sys

# 端上一个决策节点的编码方式。三种给出来是因为它们的取舍不一样，
# 不是"选个最小的"就完事：
#
#   compact16 : feature uint8 + threshold int16 + 左右孩子 uint16 = 7 B（对齐到 8 B）
#               阈值要先把特征量化成 int16，**会改变判决结果**——量化误差正好落在
#               阈值附近的样本会翻。所以它必须跟 golden vector 一起验，不能想当然。
#   f32       : feature uint16 + threshold float32 + 左右孩子 uint16 = 10 B（对齐 12 B）
#               跟训练时逐位一致（前提是特征也用 float32 算），最稳，最费。
#   packed    : 理论下界，不考虑对齐。只拿来看"最好能到多少"，别当方案。
ENCODINGS = {
    "compact16": 8,
    "f32": 12,
    "packed": 7,
}


def tree_stats(est):
    t = est.tree_
    n_nodes = int(t.node_count)
    # children_left == -1 (TREE_LEAF) 的是叶子。叶子不存阈值，但要存类别，
    # 3 分类的话 1 字节够（存 argmax）；要存概率就是 n_classes 个字节
    n_leaves = int((t.children_left == -1).sum())
    return n_nodes, n_nodes - n_leaves, n_leaves, int(t.max_depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="训练产出的 .pkl")
    ap.add_argument("--flash-budget", type=int, default=128 * 1024,
                    help="留给模型的 flash 字节数，默认 128KB")
    ap.add_argument("--leaf-bytes", type=int, default=4,
                    help="一个叶子存多少字节。存 argmax 用 1，存 n_classes 个概率就按类别数给")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn。这个脚本要在训练机上跑。")

    bundle = joblib.load(args.model)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    ests = getattr(model, "estimators_", None)
    if ests is None:
        sys.exit(f"{args.model} 里不是随机森林（没有 estimators_），实际是 {type(model)}")

    stats = [tree_stats(e) for e in ests]
    n_trees = len(stats)
    total_nodes = sum(s[0] for s in stats)
    total_internal = sum(s[1] for s in stats)
    total_leaves = sum(s[2] for s in stats)
    depths = [s[3] for s in stats]

    print(f"树：{n_trees} 棵")
    print(f"节点：{total_nodes} 个（内部 {total_internal}，叶子 {total_leaves}）")
    print(f"深度：最大 {max(depths)}，中位 {sorted(depths)[n_trees // 2]}")
    n_feat = getattr(model, "n_features_in_", None)
    if n_feat:
        print(f"特征维度：{n_feat}")
        if n_feat > 255:
            print("  ⚠ 特征超过 255 维，compact16 编码里的 feature uint8 不够用，要换 uint16")
    print()

    print(f"flash 预算 {args.flash_budget} B（{args.flash_budget / 1024:.0f} KB）")
    for name, per_node in sorted(ENCODINGS.items(), key=lambda kv: kv[1]):
        size = total_internal * per_node + total_leaves * args.leaf_bytes
        ratio = size / args.flash_budget
        verdict = "塞得下" if ratio <= 1 else f"超 {ratio:.1f} 倍"
        print(f"  {name:10s} {per_node} B/节点 → {size:>9,} B "
              f"({size / 1024:>8.1f} KB)  {verdict}")

    # 塞不下的话，能留几棵？按"每棵树平均大小"估，比"随便砍一半"有依据
    best = min(ENCODINGS.values())
    per_tree = (total_internal * best + total_leaves * args.leaf_bytes) / n_trees
    fits = int(args.flash_budget // per_tree)
    print()
    if fits >= n_trees:
        print("整片森林都塞得下。")
    elif fits >= 1:
        print(f"按最省的编码，预算内大约只能放 {fits} 棵（现在 {n_trees} 棵）。")
        print("  砍树之前先看限深：max_depth 设小往往比减树省得多，而且掉点更少——"
              "不限深的树尾部全是只覆盖几个样本的分支，那部分是过拟合，不是信息。")
    else:
        print("一棵都放不下。要么大幅限深重训，要么这条路不通。")

    print()
    print("注意：这里只算了**模型本身**。RF 还要在端上算 193 维手工特征，"
          "其中频域那部分要做 FFT/Welch——那部分的 flash、RAM 和耗时另算，"
          "而且它是 float 的，两边做不到逐位一致，只能定一个容差。")


if __name__ == "__main__":
    main()
