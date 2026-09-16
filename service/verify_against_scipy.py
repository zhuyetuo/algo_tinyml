"""量「端侧特征」跟「平台特征」差多少，以及**最终判别翻了多少**。

要在有 scipy + sklearn 的机器上跑（训练机 / 标注平台那台）。

为什么需要这个脚本：板上那份特征跟 imu_train 的 scipy 版**做不到逐位一致**，
原因是硬的——scipy 全程 float64，M4F 只有单精度 FPU；FFT 算法也不同，而浮点
加法不满足结合律。所以"一致"这件事只能量，不能证。

**量什么**要选对。特征差第几位小数不重要，重要的是森林的判别翻没翻——
所以这个脚本的主输出是「判别不一致的比例」和「翻掉的那些样本长什么样」，
特征级的差异只是辅助诊断。

用法：
    python python/verify_against_scipy.py \
        --windows windows.npy \
        --model ~/imu_train/results/.../rf/xxx.pkl \
        --hz 16

--windows 是 [N, T, C] 的真实窗口（imu_train 预处理产物里的 X 直接存出来就行）。
拿真实数据，别拿随机数：随机信号的谱是平的，正好避开了"主频落在两个 bin 中间"
这类最容易翻的情形。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.features import extract_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="[N, T, C] 的 .npy")
    ap.add_argument("--model", help="RF 的 .pkl。给了才会比最终判别（**主要输出**）")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--nperseg", type=int, default=32)
    ap.add_argument("--imu-train", default=os.path.expanduser("~/imu_train"),
                    help="imu_train 仓库路径，用来 import 它的 features.py")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.imu_train, "src", "ml"))
    try:
        import features as ref  # imu_train/src/ml/features.py —— 基准
    except ImportError as e:
        sys.exit(f"import 不到 imu_train 的 features.py（{e}）。用 --imu-train 指路径，"
                 "并确认那台机器上装了 scipy。")

    X = np.load(args.windows)
    if X.ndim != 3:
        sys.exit(f"{args.windows} 是 {X.shape}，预期 [N, T, C]")
    X = X[:args.limit].astype(np.float32)
    print(f"{len(X)} 个窗口，每个 {X.shape[1]} 点 × {X.shape[2]} 通道，{args.hz}Hz")

    f_ref = np.stack([ref._extract_one(x, args.hz) for x in X]).astype(np.float64)
    f_dev = np.stack([extract_one(x, args.hz, args.nperseg) for x in X]).astype(np.float64)
    if f_ref.shape != f_dev.shape:
        sys.exit(f"维度对不上：scipy 版 {f_ref.shape}，端侧版 {f_dev.shape}。"
                 "多半是拼接顺序或通道数不一致，这个必须先解决——顺序错了每一维都会"
                 "对到别的特征上，而模型照样给得出结果。")

    # 相对差。分母加一个下限，否则接近 0 的特征会把相对差放大成天文数字
    denom = np.maximum(np.abs(f_ref), 1e-6)
    rel = np.abs(f_dev - f_ref) / denom
    print("\n特征级相对差（辅助诊断，不是判据）：")
    print(f"  中位 {np.median(rel):.3e}   95 分位 {np.percentile(rel, 95):.3e}   "
          f"最大 {rel.max():.3e}")

    names = ref.feature_names(X.shape[2])
    worst = np.argsort(-rel.max(axis=0))[:8]
    print("  差得最多的几维：")
    for i in worst:
        nm = names[i] if i < len(names) else f"feat{i}"
        print(f"    {nm:28s} 最大相对差 {rel[:, i].max():.3e}")

    if not args.model:
        print("\n没给 --model，跳过判别比对。**建议给**：特征差多少不是判据，"
              "判别翻没翻才是。")
        return

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn")
    bundle = joblib.load(args.model)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle

    p_ref = model.predict(f_ref)
    p_dev = model.predict(f_dev)
    flip = p_ref != p_dev
    n_flip = int(flip.sum())
    print(f"\n{'=' * 52}")
    print(f"判别不一致：{n_flip} / {len(X)}（{100.0 * n_flip / len(X):.3f}%）  ← 这才是判据")
    print(f"{'=' * 52}")

    if n_flip:
        # 翻掉的样本通常是"本来就在两类边界上"的。把它们的置信度打出来——
        # 如果翻的都是低置信度样本，说明是边界抖动，不是实现错了；
        # 如果有高置信度样本被翻，那是 bug，要查
        pr = model.predict_proba(f_ref)
        conf = pr.max(axis=1)[flip]
        print(f"  翻掉的样本，原本的置信度：中位 {np.median(conf):.3f}，"
              f"最高 {conf.max():.3f}")
        if conf.max() > 0.9:
            print("  ⚠ 有高置信度样本被翻过去了。这不是边界抖动，是实现有问题——"
                  "去看上面「差得最多的几维」，多半能定位到某一类特征。")
        else:
            print("  翻的都是低置信度样本，属于边界抖动。是否可接受看产品要求，"
                  "但这不是实现错误。")


if __name__ == "__main__":
    main()
