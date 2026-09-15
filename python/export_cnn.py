"""把 imu_train 的 cnn 量化成 int8、导出 C 数组，并**实测**量化掉了多少。

这是 CNN 这条路上一直缺的那一步。之前给你的 0.8787 是 PyTorch 那侧的 float 数，
124.9 KB 是按参数量算的——两个都不是端侧实测的数。这个脚本给的是实测的。

用法（要在装了 torch 的机器上跑）：
    # 先导原始窗口（在 imu_train 目录下）
    python ~/algo_tinyml/python/dump_holdout.py --raw \\
        --processed-dir data/processed_<DATE> --hz 16 \\
        --remap configs/remap_custom_3class.yaml

    # 再量化 + 评估（在 algo_tinyml 目录下）
    python python/export_cnn.py \\
        --pt ~/imu_train/results/.../dl_cnn_best.pt \\
        --raw ~/imu_train/holdout_raw.npy --labels ~/imu_train/holdout_y.npy
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml import export, forward_int, quantize  # noqa: E402
from tinyml.export_c import _arena_bytes  # noqa: E402
from tinyml.torch_import import load_cnn, normalize  # noqa: E402


def _need(path, flag):
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    sys.exit(f"{flag} 找不到：{p}\n"
             f"  （相对路径会解析成 {os.path.abspath(p)}）\n"
             "  原始窗口要先导一次——在 imu_train 目录下跑 dump_holdout.py --raw")


def macro_f1(y, p, n):
    out = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        out.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(out)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="imu_train 的 dl_cnn_best.pt")
    ap.add_argument("--json", default="", help="配套的 .json；默认同名")
    ap.add_argument("--raw", required=True, help="留出集原始窗口 [N,C,T] .npy")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="firmware/generated_cnn")
    ap.add_argument("--calib", type=int, default=256,
                    help="拿多少条做量化校准。**从留出集里分层抽**，不是取前 N 条")
    ap.add_argument("--golden", type=int, default=16)
    args = ap.parse_args()

    net, meta = load_cnn(_need(args.pt, "--pt"), args.json or None)
    classes = meta["classes"]
    X = np.load(_need(args.raw, "--raw")).astype(np.float32)
    y = np.load(_need(args.labels, "--labels")).astype(np.int64)
    if len(X) != len(y):
        sys.exit(f"窗口 {len(X)} 条、标签 {len(y)} 条，对不上。用 dump_holdout.py 一起导。")
    print(f"留出集 {len(X)} 条，窗口 {X.shape[1]}×{X.shape[2]}（通道×点数），"
          f"{len(classes)} 类：{','.join(classes)}")

    Xn = normalize(X, meta)

    # 校准集**按类别分层抽**。取前 N 条的话，数据通常是按时间排的，
    # 前 N 条可能整段都是睡觉——那样抓挠那一段会在板上整段饱和到 127，
    # 表现成"抓挠召回低"，而且不会有任何报错。
    rng = np.random.default_rng(0)
    per = max(1, args.calib // len(classes))
    idx = []
    for c in range(len(classes)):
        pool = np.flatnonzero(y == c)
        if len(pool) == 0:
            print(f"  ⚠ 类别「{classes[c]}」在留出集里一条都没有，校准集覆盖不到它")
            continue
        idx.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    calib = Xn[np.concatenate(idx)]
    print(f"校准集 {len(calib)} 条（每类最多 {per} 条）")

    qnet = quantize(net, calib, class_names=classes)

    # ── 实测：float vs int8，在**整个留出集**上 ──────────────────────────
    f_pred = np.array([int(np.argmax(net.forward(x))) for x in Xn])
    q_pred = np.array([int(np.argmax(forward_int(qnet, qnet.quantize_input(x))[0]))
                       for x in Xn])
    n_cls = len(classes)
    f_macro, f_per = macro_f1(y, f_pred, n_cls)
    q_macro, q_per = macro_f1(y, q_pred, n_cls)

    print(f"\n{'类别':<10}{'float F1':>10}{'int8 F1':>10}{'差':>9}")
    for c in range(n_cls):
        print(f"{classes[c]:<10}{f_per[c]:>10.4f}{q_per[c]:>10.4f}{q_per[c] - f_per[c]:>+9.4f}")
    print(f"{'macro':<10}{f_macro:>10.4f}{q_macro:>10.4f}{q_macro - f_macro:>+9.4f}")
    print(f"\nfloat 与 int8 判别一致率 {float(np.mean(f_pred == q_pred)):.4f}")

    sat = float(np.mean([np.mean(np.abs(qnet.quantize_input(x)) >= 127) for x in Xn]))
    print(f"输入饱和比例 {sat:.4f}"
          + ("   ← 偏高，校准集可能没覆盖到剧烈动作" if sat > 0.02 else ""))

    # ── 体积和 RAM ────────────────────────────────────────────────────────
    n_w = sum(int(l.w.size) for l in qnet.layers if hasattr(l, "w"))
    n_b = sum(int(l.bias.size) * 3 for l in qnet.layers if hasattr(l, "bias"))  # bias+mult+shift
    flash = n_w + n_b * 4
    arena = _arena_bytes(qnet) * 2
    print(f"\n模型 flash：{n_w:,} B 权重（int8） + {n_b * 4:,} B 偏置/乘子（int32）"
          f" = {flash:,} B（{flash / 1024:.1f} KB）")
    print(f"推理 RAM（乒乓缓冲）：{arena:,} B（{arena / 1024:.1f} KB）")
    print("  权重是 const，进 flash 不占 RAM。这里的 RAM 只有中间张量。")

    os.makedirs(args.out, exist_ok=True)
    picked = [qnet.quantize_input(Xn[i]) for i in
              rng.choice(len(Xn), size=min(args.golden, len(Xn)), replace=False)]
    if len({int(np.argmax(forward_int(qnet, x)[0])) for x in picked}) < 2:
        print("⚠ golden vector 全落在同一类上——逐位比对仍然有效，但验不到不同判决路径")
    for name, content in export(qnet, golden_x_i8=np.stack(picked)).items():
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print("写出", p)

    print(f"""
注意 **ch_mean / ch_std 没有导进 C**：端上要在量化之前做同一套逐通道 z-score，
否则输入分布跟训练时对不上——效果明显下降但不报错。这两个数组在
{os.path.splitext(args.pt)[0]}.json 里，下一步要把它和输入量化一起做成
一个定点算子。在那之前，这里导出的 C 只有网络本身，**不是完整的端侧管线**。""")


if __name__ == "__main__":
    main()
