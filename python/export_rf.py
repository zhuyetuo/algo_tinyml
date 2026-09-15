"""把平台在用的随机森林 .pkl 导成板上的 C 文件 + golden vector。

**要在有 sklearn 的机器上跑**（训练机 / 标注平台那台），因为要加载 .pkl。
导出来的 C 不依赖 sklearn，也不依赖 numpy。

用法：
    python python/export_rf.py \
        --model ~/imu_train/results/.../rf/xxx.pkl \
        --features feats.npy \
        --out firmware/generated

--features 是一批**真实特征向量**（[N, n_features] 的 .npy），用来生成 golden vector。
拿真实的、不是随机的：随机向量会走到树里几乎不可能走到的分支组合上，验的不是
实际会跑的那条路。imu_train 那边 `src/ml/features.py` 算出来的存成 .npy 就行。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.export_features_c import export as export_feat_cfg  # noqa: E402
from tinyml.export_forest_c import export  # noqa: E402
from tinyml.forest import flash_bytes, from_sklearn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", help="[N, n_features] 的 .npy，用来做 golden vector")
    ap.add_argument("--out", default="firmware/generated")
    ap.add_argument("--golden", type=int, default=32)
    ap.add_argument("--classes", default="")
    ap.add_argument("--window", type=int, default=32, help="窗口点数（16Hz×2s=32）")
    ap.add_argument("--channels", type=int, default=8, help="6=acc+gyr，8=再加 pitch/roll")
    ap.add_argument("--nperseg", type=int, default=32)
    ap.add_argument("--hz", type=float, default=16.0)
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn。这个脚本要在训练机上跑。")

    bundle = joblib.load(args.model)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    names = [s for s in args.classes.split(",") if s] or None
    forest = from_sklearn(model, class_names=names)

    fb = flash_bytes(forest)
    total = sum(fb.values())
    print(f"树 {forest.n_trees} 棵，节点 {len(forest.node_feature)} 个，"
          f"叶子 {len(forest.leaf_proba)} 个，特征 {forest.n_features} 维")
    for k, v in fb.items():
        print(f"  {k:16s} {v:>10,} B")
    print(f"  {'合计':16s} {total:>10,} B（{total / 1024:.1f} KB）")
    if total > 128 * 1024:
        print("  ⚠ 超过 128KB。先试 max_depth 限深重训，通常比砍树掉点少——"
              "不限深的树尾部都是只覆盖几个样本的过拟合分支。")

    golden = None
    if args.features:
        xs = np.load(args.features).astype(np.float32)
        if xs.ndim != 2 or xs.shape[1] != forest.n_features:
            sys.exit(f"{args.features} 是 {xs.shape}，预期 [N, {forest.n_features}]")
        # 按预测类别轮流取，保证 golden 覆盖到不同判决路径
        by_cls = {}
        for x in xs:
            by_cls.setdefault(forest.predict(x), []).append(x)
        picked, i = [], 0
        while len(picked) < args.golden and any(i < len(v) for v in by_cls.values()):
            for c in sorted(by_cls):
                if i < len(by_cls[c]) and len(picked) < args.golden:
                    picked.append(by_cls[c][i])
            i += 1
        golden = np.stack(picked)
        print(f"golden vector {len(golden)} 条，覆盖 {len(by_cls)} 个类别")
    else:
        print("没给 --features，不生成 golden vector。"
              "**强烈建议给**：没有它，板上跑出来对不对只能靠肉眼看准确率。")

    os.makedirs(args.out, exist_ok=True)
    # 特征配置表（Hann 窗 + FFT 旋转因子 + 位反序）跟模型一起导：它们必须成套，
    # 分开导迟早会出现"换了窗口长度但表没换"，而那只会表现成准确率莫名其妙地低
    files = dict(export(forest, golden_x=golden))
    files.update(export_feat_cfg(args.window, args.channels, args.nperseg, args.hz))
    n_feat_expected = None
    try:
        from tinyml.features import n_features
        n_feat_expected = n_features(args.channels)
    except Exception:
        pass
    if n_feat_expected is not None and n_feat_expected != forest.n_features:
        sys.exit(f"模型要 {forest.n_features} 维特征，但 {args.channels} 通道只算得出 "
                 f"{n_feat_expected} 维。--channels 给错了，或者这个模型不是这套特征训的。"
                 "不拦住的话，特征会整体错位而模型照样给得出结果。")
    for name, content in files.items():
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print("写出", p)

    print()
    print("编译时务必带 -ffp-contract=off。允许 FMA 合并的话，中间结果少一次舍入，"
          "跟 Python 算出来的末位就不同——而那足以在阈值附近把 argmax 翻过去。")


if __name__ == "__main__":
    main()
