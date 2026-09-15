"""把 imu_train 的留出集特征和标签导出成 .npy，给 prune_gbdt / prune_rf /
export_* 的 --features / --labels 用。

**直接复用 imu_train 自己的代码**（`load_all_splits` + `apply_remap`），不重写一遍。
重写的话两边迟早分家，而分家的表现是"特征对得上、标签错位"——模型照样给得出结果，
只是所有指标都是错的。

特征不用现算：`src/ml/train.py` 已经把它缓存在
`{processed_dir}/{hz}hz/ml_features.npz` 里了（键 X_tr / X_val / X_te），
而且那是 remap **之后**算的，所以行数跟 remap 后的标签对得上。

用法（在 imu_train 目录下跑，或者用 --imu-train 指路径）：
    python ~/algo_tinyml/python/dump_holdout.py \\
        --processed-dir data/processed_2026_8_11-2026_8_27_raw_missing_drop_window \\
        --hz 16 --remap configs/remap_custom_3class.yaml \\
        --out-prefix holdout

    → holdout_feats.npy, holdout_y.npy, holdout_classes.txt
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--remap", default="configs/remap_custom_3class.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"],
                    help="导哪个分割。默认 val——imu_train 常见配置下 test 是空的")
    ap.add_argument("--imu-train", default="", help="imu_train 仓库路径，默认当前目录")
    ap.add_argument("--out-prefix", default="holdout")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.imu_train or "."))
    if not os.path.isdir(os.path.join(repo, "src", "data")):
        sys.exit(f"{repo} 看起来不是 imu_train 仓库（没有 src/data/）。"
                 "在 imu_train 目录下跑，或者用 --imu-train 指路径。")
    sys.path.insert(0, os.path.join(repo, "src", "data"))

    try:
        from dataset import load_all_splits
        from remap_utils import apply_remap, load_remap_yaml
    except ImportError as e:
        sys.exit(f"import 不到 imu_train 的模块（{e}）。--imu-train 指对了吗？")

    pdir = os.path.join(repo, args.processed_dir) if not os.path.isabs(args.processed_dir) \
        else args.processed_dir
    if not os.path.isdir(pdir):
        sys.exit(f"{pdir} 不存在。--processed-dir 给错了？")

    splits = load_all_splits(args.hz, pdir)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _), meta = splits
    # 照 src/ml/train.py 的取法：meta["classes"] 存进 npz 再读出来可能是字符串
    raw = meta.get("classes")
    if raw is None:
        sys.exit("meta 里读不到 classes，没法做 remap")
    classes = list(eval(raw)) if isinstance(raw, str) else list(raw)
    print(f"原始类别：{classes}")

    ys = {"train": ytr, "val": yva, "test": yte}
    keeps = {}
    if args.remap:
        remap_path = args.remap if os.path.isabs(args.remap) else os.path.join(repo, args.remap)
        cfg = load_remap_yaml(remap_path)
        new_classes = None
        for k in ("train", "val", "test"):
            ys[k], nc, keeps[k] = apply_remap(ys[k], classes, cfg)
            new_classes = nc
        classes = new_classes
        print(f"remap 后类别：{classes}")
    else:
        for k in ys:
            keeps[k] = np.ones(len(ys[k]), bool)

    cache = os.path.join(pdir, f"{args.hz}hz", "ml_features.npz")
    if not os.path.exists(cache):
        sys.exit(f"找不到特征缓存 {cache}。\n"
                 "  先跑一次 src/ml/train.py（任何 --model 都行），它会把特征缓存下来。")
    d = np.load(cache)
    feats = {"train": d["X_tr"], "val": d["X_val"], "test": d["X_te"]}

    X = feats[args.split]
    y = ys[args.split]
    if len(X) != len(y):
        # 这个必须拦住：行数对不上的话，特征和标签是**错位**的，
        # 算出来的每一个指标都是错的，而且看起来完全正常
        sys.exit(
            f"特征 {len(X)} 行，标签 {len(y)} 行，对不上。可能的原因：\n"
            f"  - 缓存 {cache} 是用别的 remap / 别的数据训练时生成的，删掉它重跑一次 train.py；\n"
            f"  - 训练时注入了合成数据（--synthetic*），那会改变行数。\n"
            "  **别手动对齐**——错位的标签会让所有指标都错，而且看不出来。")

    if len(X) == 0:
        sys.exit(f"{args.split} 分割是空的。imu_train 常见配置下 test_ratio=0，"
                 "试试 --split val。")

    fx = f"{args.out_prefix}_feats.npy"
    fy = f"{args.out_prefix}_y.npy"
    np.save(fx, X.astype(np.float32))
    np.save(fy, y.astype(np.int64))
    with open(f"{args.out_prefix}_classes.txt", "w", encoding="utf-8") as f:
        f.write(",".join(classes) + "\n")

    cnt = np.bincount(y, minlength=len(classes))
    print(f"\n导出 {args.split} 分割：{len(X)} 条 × {X.shape[1]} 维")
    for i, c in enumerate(classes):
        print(f"  {c:<8} {cnt[i]:>7,} 条 ({100.0 * cnt[i] / len(y):5.1f}%)")
    print(f"\n写出 {fx} / {fy} / {args.out_prefix}_classes.txt")
    print(f"\n接着可以：\n"
          f"  python ~/algo_tinyml/python/prune_gbdt.py --model <xgb.pkl> \\\n"
          f"      --features {fx} --labels {fy} \\\n"
          f"      --classes {','.join(classes)} --focus 抓挠")


if __name__ == "__main__":
    main()
