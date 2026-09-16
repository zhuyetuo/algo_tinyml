"""在 Ubuntu 服务器上跑**板上那份一模一样的 C**，喂真实数据，出效果和耗时。

不需要板子，不需要交叉编译器——用系统自带的 gcc 把 `core/*.c` 编出来跑。
编译选项跟固件一致（尤其 `-ffp-contract=off`），所以**判决结果跟板上逐位相同**。

耗时那一栏是 x86 的数，**跟 Cortex-M4F 没有可比性**，只能用来横向比 RF 和 CNN。
板上的真实耗时要用 DWT 周期计数器测，那要有板子。

用法：
    # 1. 先导出模型（要 sklearn 的机器上做，或者把导出的 C 文件拷过来）
    python service/export_rf.py --model xxx.pkl --features feats.npy --out core/models/generated

    # 2. 跑
    python service/run_host_sim.py --gen core/models/generated \\
        --data ~/imu_train/data/processed_custom/test.npz

--data 支持两种：
  - imu_train 的 .npz（取里面的 X，形状 [N, T, C]，会拼成连续的样本流）
  - 每行 n_ch 个数的纯文本 / csv
给了 .npz 里的 y 的话，还会顺便报 macro-F1。
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")


def build(gen_dir, out_dir):
    src = [
        os.path.join(FW, "tm_features.c"),
        os.path.join(FW, "tm_forest.c"),
        os.path.join(FW, "tm_window.c"),
        os.path.join(gen_dir, "tm_feat_cfg.c"),
        os.path.join(gen_dir, "tm_forest_model.c"),
        os.path.join(ROOT, "tools", "host_sim.c"),
    ]
    missing = [p for p in src if not os.path.exists(p)]
    if missing:
        sys.exit("缺文件：" + "\n  ".join(missing) +
                 "\n先跑 service/export_rf.py 导出模型和特征表。")
    exe = os.path.join(out_dir, "host_sim")
    cmd = [# host 工具用 gnu99：c99 会把 clock_gettime 藏起来。固件那边仍然是严格 c99
           "gcc", "-std=gnu99", "-O2", "-Wall",
           # 跟固件同一套浮点选项。少了它，这里算出来的就不是板上会算出来的
           "-ffp-contract=off", "-fno-fast-math",
           f"-I{FW}", f"-I{gen_dir}", *src, "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"编译失败：\n{r.stderr}")
    return exe


def load_stream(path, n_ch):
    """→ [N, n_ch] 的连续样本流，外加（可能有的）逐窗口标签。"""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        X = np.asarray(d["X"], np.float32)       # [N, T, C]
        y = np.asarray(d["y"]).astype(np.int64) if "y" in d else None
        if X.ndim != 3:
            sys.exit(f"{path} 里 X 是 {X.shape}，预期 [N, T, C]")
        if X.shape[2] < n_ch:
            sys.exit(f"数据只有 {X.shape[2]} 通道，模型要 {n_ch} 通道")
        # 把窗口首尾相接成一条流。**注意这不是原始连续信号**——窗口之间本来可能有
        # 重叠或跳跃，接起来之后接缝处的那几个窗口是假的。报告里会把它们标出来。
        return X[:, :, :n_ch].reshape(-1, n_ch), y, X.shape[1]
    arr = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None,
                     dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < n_ch:
        sys.exit(f"{path} 是 {arr.shape}，预期 [N, >={n_ch}]")
    return arr[:, :n_ch], None, None


def macro_f1(y, p, n):
    out = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        d = 2 * tp + fp + fn
        out.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="core/models/generated", help="导出的 C 文件所在目录")
    ap.add_argument("--data", required=True)
    ap.add_argument("--hop", type=int, default=0, help="窗口步长，默认窗口的一半")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本")
    args = ap.parse_args()

    # 从导出的头文件里读配置，别让人再填一遍——填错了不会报错，只会算出一堆
    # 看起来正常的垃圾
    cfg_h = os.path.join(args.gen, "tm_feat_cfg.h")
    if not os.path.exists(cfg_h):
        sys.exit(f"{cfg_h} 不存在。先跑 service/export_rf.py。")
    txt = open(cfg_h, encoding="utf-8").read()

    def macro(name):
        for line in txt.splitlines():
            if line.startswith(f"#define {name} "):
                return int(line.split()[2])
        sys.exit(f"{cfg_h} 里找不到 {name}")

    n_t, n_ch, dim = macro("TM_FEAT_N_T"), macro("TM_FEAT_N_CH"), macro("TM_FEAT_DIM")
    print(f"模型配置：窗口 {n_t} 点 × {n_ch} 通道，{dim} 维特征")

    stream, y, win_len = load_stream(args.data, n_ch)
    if args.limit:
        stream = stream[:args.limit]
    print(f"数据：{len(stream)} 个采样点")
    if win_len is not None:
        if win_len != n_t:
            print(f"  ⚠ 数据里的窗口是 {win_len} 点，模型要 {n_t} 点。"
                  "拼成流之后还能跑，但窗口边界对不上，结果只能当冒烟测试看。")
        print("  注意：.npz 里的窗口被首尾相接成了一条流，**接缝处的窗口是假的**"
              "（跨了两个本来不相邻的片段）。要看真实效果请喂原始连续信号。")

    with tempfile.TemporaryDirectory() as tmp:
        exe = build(args.gen, tmp)
        hop = args.hop or n_t // 2
        text = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in stream)
        r = subprocess.run([exe, str(hop)], input=text + "\n",
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"跑失败：\n{r.stderr}")

    rows = [ln.split() for ln in r.stdout.strip().splitlines()]
    if not rows:
        sys.exit("一个窗口都没出来——数据点数比窗口长度还少？")
    idx = np.array([int(v[0]) for v in rows])
    pred = np.array([int(v[1]) for v in rows])
    proba = np.array([[float(x) for x in v[2:]] for v in rows], np.float64)

    print(f"\n{r.stderr.strip()}")
    print(f"\n出了 {len(pred)} 个窗口的判决")
    cnt = np.bincount(pred, minlength=proba.shape[1])
    for c, n in enumerate(cnt):
        print(f"  类别 {c}: {n:>6} 个窗口 ({100.0 * n / len(pred):5.1f}%)"
              f"  平均置信度 {proba[pred == c, c].mean() if n else 0:.3f}")

    if y is not None and win_len == n_t:
        # 每个原窗口的中心大致对应流里的哪个位置——只在窗口长度一致时才有意义
        src_win = idx // n_t
        ok = src_win < len(y)
        if ok.sum():
            print(f"\n跟 .npz 里的标签比（只取对得上的 {int(ok.sum())} 个）：")
            print(f"  macro-F1 {macro_f1(y[src_win[ok]], pred[ok], proba.shape[1]):.4f}")
            print("  ⚠ 这个数只能当冒烟测试：窗口是拼接出来的，边界跟训练时不一致。"
                  "真实效果请在服务端用 imu_train 那套评估。")


if __name__ == "__main__":
    main()
