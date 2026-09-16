"""训练侧：复用 imu_train 预处理好的窗口，训一个能塞进 GR5513 的小网络。

**这是全仓库唯一依赖 PyTorch 的文件**，而且它只负责一件事：最后吐出一个
`model.npz`（纯 numpy 权重）。量化、导出 C、跟板上逐位对照那一整条链都不碰它，
所以：
  - 换框架、换训练机器，板上那一侧一行都不用改；
  - 工具链可以用随机权重自测，不用先有一个训好的模型（tests/ 就是这么跑的）。

输入：imu_train 预处理产物 data/processed_*/{train,val,test}.npz，里面 X 是
[N, T, C]（时间在前），这里转成 [N, C, T]（通道在前）——C 那边卷积内层循环沿 T 走，
通道在前访存才连续。

用法：
    python python/train_torch.py \
        --data ~/imu_train/data/processed_custom \
        --channels 6 --window 64 --epochs 40 \
        --out model.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_split(data_dir, name, n_ch, window):
    path = os.path.join(data_dir, f"{name}.npz")
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], np.float32)
    y = np.asarray(d["y"]).astype(np.int64)
    if X.ndim != 3:
        raise ValueError(f"{path} 里 X 是 {X.shape}，预期 [N, T, C]")
    # imu_train 存的是 [N, T, C]
    X = np.transpose(X, (0, 2, 1))
    if X.shape[1] < n_ch:
        raise ValueError(f"数据只有 {X.shape[1]} 个通道，要 {n_ch} 个")
    # 只取前 n_ch 个通道：imu_train 那边是 acc3 + gyr3 + pitch/roll 共 8 通道，
    # 端侧第一版只用 acc+gyr 6 个。姿态角是算出来的，端上再算一遍不划算，
    # 而且它是缓变量，对"抓挠"这种高频动作贡献有限
    X = X[:, :n_ch, :]
    if X.shape[2] < window:
        raise ValueError(f"窗口只有 {X.shape[2]} 点，要 {window} 点")
    X = X[:, :, :window]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="imu_train 的 processed_* 目录")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="model.npz")
    ap.add_argument("--calib", type=int, default=256,
                    help="存进 npz 的校准样本条数，量化时用")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("没装 torch。只有训练需要它；量化和导出那一条链只要 numpy。")

    Xtr, ytr = load_split(args.data, "train", args.channels, args.window)
    Xva, yva = load_split(args.data, "val", args.channels, args.window)
    n_classes = int(max(ytr.max(), yva.max())) + 1
    print(f"train {Xtr.shape} val {Xva.shape} 类别数 {n_classes}")

    from tinyml.net import flat_size
    nf = flat_size(args.window)

    # 结构必须跟 tinyml/net.py 的 make_net 一模一样——那边是量化和导出认的那个结构。
    # 两处写两遍是明摆着的隐患，但在这里换成"共享定义"会把 torch 依赖带进工具链，
    # 得不偿失。改结构时两边都要改，tests/test_export_matches_torch 会拦。
    model = nn.Sequential(
        nn.Conv1d(args.channels, 8, 5), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(8, 16, 3), nn.ReLU(), nn.MaxPool1d(4),
        nn.Flatten(), nn.Linear(nf, n_classes),
    )

    # 类别不平衡是这份数据的常态（睡觉占绝大多数，抓挠很少）。不加权的话模型会
    # 学成"永远猜睡觉"——准确率很好看，而它恰好在唯一有用的那一类上全错
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / n_classes, dtype=torch.float32)
    print("类别分布", cnt.tolist(), "→ 权重", w.tolist())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1)
            # 看 macro-F1 不看准确率，理由同上：准确率被多数类主导
            f1 = _macro_f1(yva_t.numpy(), pred.numpy(), n_classes)
        print(f"epoch {ep + 1}/{args.epochs}  val macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            best_state = {k: v.detach().cpu().numpy().copy()
                          for k, v in model.state_dict().items()}

    print(f"最好的 val macro-F1 {best:.4f}")

    # 校准集从**训练集**里按类别均匀取，不是随便取前 N 条：量化的激活范围要覆盖到
    # 剧烈动作，全取睡觉片段的话抓挠那一段会整段饱和到 ±127，模型在最该判对的
    # 时候瞎掉，而且不报任何错。tests/test_quantize.py 里把这件事钉成了测试
    per = max(1, args.calib // n_classes)
    picks = np.concatenate([np.where(ytr == c)[0][:per] for c in range(n_classes)])
    np.savez_compressed(
        args.out,
        calib_x=Xtr[picks],
        n_classes=np.int64(n_classes),
        window=np.int64(args.window),
        channels=np.int64(args.channels),
        **best_state,
    )
    print(f"写出 {args.out}（含 {len(picks)} 条校准样本）")
    print("下一步：python python/quantize_and_export.py --model", args.out)


def _macro_f1(y, p, n):
    f1s = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


if __name__ == "__main__":
    main()
