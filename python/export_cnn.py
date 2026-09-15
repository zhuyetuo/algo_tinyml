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

from tinyml import export, forward_int, forward_int_batch, quantize  # noqa: E402
from tinyml.progress import bar, chunks  # noqa: E402
from tinyml.export_c import _arena_bytes  # noqa: E402
from tinyml.torch_import import load_cnn, normalize  # noqa: E402
from event_eval import check_ordered, match_events, prf, to_events  # noqa: E402


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
    ap.add_argument("--focus", default="抓挠", help="事件级指标盯哪一类")
    ap.add_argument("--min-windows", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
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
    # 分段跑而不是一个大矩阵乘：一次算完两万多条的话进度条不会动，
    # 而且中间张量会很大（256 通道 × 8 点 × N）。段大小按经验取，不是调出来的。
    f_pred = np.empty(len(Xn), np.int64)
    q_pred = np.empty(len(Xn), np.int64)
    CH = 512
    with bar((len(Xn) + CH - 1) // CH, "评估留出集") as pb:
        for s, e in chunks(len(Xn), CH):
            blk = Xn[s:e]
            f_pred[s:e] = [int(np.argmax(net.forward(x))) for x in blk]
            xq = np.stack([qnet.quantize_input(x) for x in blk])
            q_pred[s:e] = np.argmax(forward_int_batch(qnet, xq), axis=1)
            pb.update()
    n_cls = len(classes)
    f_macro, f_per = macro_f1(y, f_pred, n_cls)
    q_macro, q_per = macro_f1(y, q_pred, n_cls)

    print(f"\n{'类别':<10}{'float F1':>10}{'int8 F1':>10}{'差':>9}")
    for c in range(n_cls):
        print(f"{classes[c]:<10}{f_per[c]:>10.4f}{q_per[c]:>10.4f}{q_per[c] - f_per[c]:>+9.4f}")
    print(f"{'macro':<10}{f_macro:>10.4f}{q_macro:>10.4f}{q_macro - f_macro:>+9.4f}")
    print(f"\nfloat 与 int8 判别一致率 {float(np.mean(f_pred == q_pred)):.4f}")

    sat = float(np.mean([np.mean(np.abs(qnet.quantize_input(x)) >= 127)
                         for x in Xn[::max(1, len(Xn) // 2000)]]))
    print(f"输入饱和比例 {sat:.4f}"
          + ("   ← 偏高，校准集可能没覆盖到剧烈动作" if sat > 0.02 else ""))

    # ── 事件级 ────────────────────────────────────────────────────────────
    # 窗口级的 F1 跟产品关心的事情不是一回事。产品问的是"这次抓挠报到了吗"，
    # 而一次抓挠横跨好几个窗口——窗口级把它算成好几次，事件级算一次。
    # **RF 那边报的是事件级 0.788**，不换成同一口径就没法比。
    if args.focus in classes:
        fc = classes.index(args.focus)
        run, err = check_ordered(y, fc)
        if err:
            print(f"\n事件级跳过：{err}")
        elif run < 1.5:
            # 打乱过的数据做事件聚合毫无意义：每个窗口都是独立的一段，
            # 聚合出来的"事件"是伪造的。宁可不报，也不能报一个假的数
            print(f"\n事件级跳过：目标类别平均游程只有 {run:.2f} 个窗口，"
                  "留出集看起来不是按时间排的")
        else:
            print(f"\n事件级（min_windows={args.min_windows}, "
                  f"max_gap={args.max_gap}，目标「{args.focus}」）：")
            true_ev = to_events(y == fc, args.min_windows, args.max_gap)
            print(f"{'':<8}{'报':>5}{'真值':>6}{'对':>5}{'误报':>6}{'漏':>5}"
                  f"{'事件P':>8}{'事件R':>8}{'事件F1':>9}")
            for tag, pred in (("float", f_pred), ("int8", q_pred)):
                ev = to_events(pred == fc, args.min_windows, args.max_gap)
                tp, fp, fn = match_events(ev, true_ev)
                p, r, f1 = prf(tp, fp, fn)
                print(f"{tag:<8}{len(ev):>5}{len(true_ev):>6}{tp:>5}{fp:>6}{fn:>5}"
                      f"{p:>8.3f}{r:>8.3f}{f1:>9.3f}")
    else:
        print(f"\n事件级跳过：--focus「{args.focus}」不在类别里（{','.join(classes)}）")

    # ── 体积和 RAM ────────────────────────────────────────────────────────
    n_w = sum(int(l.w.size) for l in qnet.layers if hasattr(l, "w"))
    n_b = sum(int(l.bias.size) * 3 for l in qnet.layers if hasattr(l, "bias"))  # bias+mult+shift
    flash = n_w + n_b * 4
    arena = _arena_bytes(qnet) * 2
    print(f"\n模型 flash：{n_w:,} B 权重（int8） + {n_b * 4:,} B 偏置/乘子（int32）"
          f" = {flash:,} B（{flash / 1024:.1f} KB）")
    print(f"推理 RAM（乒乓缓冲）：{arena:,} B（{arena / 1024:.1f} KB）")
    print("  权重是 const，进 flash 不占 RAM。这里的 RAM 只有中间张量。")

    # 逐层拆开。**不拆的话"模型太大"只能靠砍整体宽度来解决**，而实际上
    # 一维卷积的权重是 out_ch × in_ch × k，最后一层通常一家独大——
    # 只动那一层能省掉大部分体积，前面几层的感受野和通道数都不用动。
    print(f"\n逐层：{'层':<16}{'权重 B':>12}{'占比':>8}")
    for i, l in enumerate(qnet.layers):
        if not hasattr(l, "w"):
            continue
        nm = f"dense {l.w.shape[1]}→{l.w.shape[0]}" if l.w.ndim == 2 else \
            f"conv {l.w.shape[1]}→{l.w.shape[0]} k{l.w.shape[2]}"
        print(f"      {nm:<16}{l.w.size:>12,}{100.0 * l.w.size / n_w:>7.1f}%")
    if flash > 131072:
        print(f"\n⚠ 超出 128 KB 预算 {flash - 131072:,} B。占比最大的那一层"
              "减半，体积大约也减半——但那要**重训**，不是导出时能做的。")

    os.makedirs(args.out, exist_ok=True)
    picked = [qnet.quantize_input(Xn[i]) for i in
              rng.choice(len(Xn), size=min(args.golden, len(Xn)), replace=False)]
    if len({int(np.argmax(forward_int(qnet, x)[0])) for x in picked}) < 2:
        print("⚠ golden vector 全落在同一类上——逐位比对仍然有效，但验不到不同判决路径")
    for name, content in export(qnet, golden_x_i8=np.stack(picked),
                            prep=meta).items():
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print("写出", p)

    print("""
端上的调用顺序（ch_mean/ch_std 已经导进 tm_model.c 了）：
    tm_prep(&tm_model_prep, window_float, x_i8);   /* 逐通道 z-score + 量化 */
    tm_invoke(&tm_model, x_i8, out, arena, TM_ARENA_BYTES);
    int cls = tm_argmax(out, TM_N_CLASSES);
tm_prep 跟 Python 侧逐位一致（tests/test_prep_c.py 钉着），少了它输入分布
会跟训练时对不上——效果掉一截而且不报错。""")


if __name__ == "__main__":
    main()
