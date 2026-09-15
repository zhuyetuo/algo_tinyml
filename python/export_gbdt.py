"""把训好的 XGBoost 模型导成板上的 C + 特征常量表 + golden vector。

要在有 xgboost + sklearn 的机器上跑。

用法：
    python python/export_gbdt.py \
        --model ~/imu_train/results_edge/.../xgb/ml_xgb.pkl \
        --features feats.npy \
        --window 16 --channels 8 --hz 16 \
        --classes 活动,睡觉,抓挠,未佩戴,甩身体 \
        --out firmware/generated
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.export_features_c import export as export_feat_cfg  # noqa: E402
from tinyml.export_gbdt_c import export  # noqa: E402
from tinyml.features import n_features  # noqa: E402
from tinyml.gbdt import flash_bytes, from_xgboost  # noqa: E402


<<<<<<< HEAD

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
    if not os.path.isabs(p):
        hint = (f"\n  注意这是相对路径，会解析成 {os.path.abspath(p)}。"
                "\n  模型在 imu_train 里的话要写全：~/imu_train/results/...")
    guess = os.path.expanduser("~/imu_train/results")
    if os.path.isdir(guess):
        hint += f"\n  ~/imu_train/results/ 下现有：{', '.join(sorted(os.listdir(guess))[:5]) or '（空）'}"
    sys.exit(f"找不到模型文件：{p}{hint}")

=======
>>>>>>> origin/main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", help="[N, n_features] 的 .npy，用来做 golden vector")
    ap.add_argument("--out", default="firmware/generated")
    ap.add_argument("--golden", type=int, default=8,
                    help="golden vector 条数。每条 = n_features×4 B，别给太多")
    ap.add_argument("--classes", default="")
    ap.add_argument("--window", type=int, default=16, help="窗口点数（1 秒 @16Hz = 16）")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--nperseg", type=int, default=0, help="默认 min(窗口, 32) 向下取 2 的幂")
    ap.add_argument("--hz", type=float, default=16.0)
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib。这个脚本要在训练机上跑。")

<<<<<<< HEAD
    bundle = joblib.load(resolve_model(args.model))
=======
    bundle = joblib.load(args.model)
>>>>>>> origin/main
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    if not hasattr(model, "get_booster"):
        sys.exit(f"不是 XGBoost 模型（没有 get_booster），实际是 {type(model)}。"
                 "sklearn 的随机森林请用 export_rf.py。")

    names = [s for s in args.classes.split(",") if s] or None
    b = from_xgboost(model, class_names=names)

    rounds = b.n_trees // max(b.n_classes, 1)
    print(f"树 {b.n_trees} 棵（{rounds} 轮 × {b.n_classes} 类），"
          f"节点 {len(b.node_feature)} 个，叶子 {len(b.leaf_value)} 个，"
          f"特征 {b.n_features} 维")
    fb = flash_bytes(b)
    total = sum(fb.values())
    for k, v in fb.items():
        print(f"  {k:16s} {v:>10,} B")
    print(f"  {'合计':16s} {total:>10,} B（{total / 1024:.1f} KB）")
    if total > 128 * 1024:
        print(f"  ⚠ 超过 128KB 预算 {total / (128 * 1024):.1f} 倍。"
              f"轮数从 {rounds} 往下减是最直接的杠杆——GBDT 减轮数比 RF 剪枝温和，"
              "因为后面的轮本来就是在修越来越小的残差。")

    exp_dim = n_features(args.channels)
    if exp_dim != b.n_features:
        sys.exit(f"模型要 {b.n_features} 维特征，但 {args.channels} 通道只算得出 "
                 f"{exp_dim} 维。--channels 给错了，或者这个模型不是这套特征训的。"
                 "不拦住的话特征会整体错位，而模型照样给得出结果。")

    nps = args.nperseg
    if not nps:
        nps = min(args.window, 32)
        while nps & (nps - 1):      # 向下取 2 的幂（基-2 FFT 的限制）
            nps -= 1
    print(f"特征配置：窗口 {args.window} 点 × {args.channels} 通道，nperseg={nps}")

    golden = None
    if args.features:
        xs = np.load(args.features).astype(np.float32)
        if xs.ndim != 2 or xs.shape[1] != b.n_features:
            sys.exit(f"{args.features} 是 {xs.shape}，预期 [N, {b.n_features}]")
        # 按预测类别轮流取，保证覆盖到不同判决路径
        by = {}
        for x in xs:
            by.setdefault(b.predict(x), []).append(x)
        picked, i = [], 0
        while len(picked) < args.golden and any(i < len(v) for v in by.values()):
            for c in sorted(by):
                if i < len(by[c]) and len(picked) < args.golden:
                    picked.append(by[c][i])
            i += 1
        golden = np.stack(picked)
        print(f"golden vector {len(golden)} 条，覆盖 {len(by)} 个类别")
    else:
        print("没给 --features，不生成 golden vector。**强烈建议给**——"
              "没有它，板上跑出来对不对只能靠肉眼看准确率。")

    os.makedirs(args.out, exist_ok=True)
    files = dict(export(b, golden_x=golden))
    files.update(export_feat_cfg(args.window, args.channels, nps, args.hz))
    for name, content in files.items():
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print("写出", p)

    print("\n编译要带 -ffp-contract=off。")
    print("端上不做 softmax：它保序，argmax 直接在 margin 上取结果一样，"
          "还省掉 expf（libm 末位不保证一致）。")


if __name__ == "__main__":
    main()
