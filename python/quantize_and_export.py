"""把 train_torch.py 训出来的 model.npz 量化成 int8，导出板上用的 C 文件。

只依赖 numpy。产物：
    tm_model.h / tm_model.c   权重和网络结构
    tm_golden.h               golden vector —— 板上跑出来必须逐位相同

用法：
    python python/quantize_and_export.py --model model.npz --out firmware/generated
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml import export, forward_int, quantize  # noqa: E402
from tinyml.net import Conv1D, Dense, FloatNet, MaxPool1D  # noqa: E402


def from_torch_npz(path):
    """把 torch nn.Sequential 的 state_dict（已经存成 numpy）还原成 FloatNet。

    键名是 Sequential 的下标：0=conv1, 3=conv2, 7=linear。写死下标不优雅，但
    train_torch.py 里的结构是固定的，而"优雅地自动识别"会在结构改动时静默认错层。
    """
    d = np.load(path, allow_pickle=True)
    need = ["0.weight", "0.bias", "3.weight", "3.bias", "7.weight", "7.bias"]
    missing = [k for k in need if k not in d]
    if missing:
        raise SystemExit(f"{path} 里缺这些键：{missing}；结构跟 train_torch.py 对不上")
    net = FloatNet([
        Conv1D(np.asarray(d["0.weight"], np.float32), np.asarray(d["0.bias"], np.float32), relu=True),
        MaxPool1D(4),
        Conv1D(np.asarray(d["3.weight"], np.float32), np.asarray(d["3.bias"], np.float32), relu=True),
        MaxPool1D(4),
        Dense(np.asarray(d["7.weight"], np.float32), np.asarray(d["7.bias"], np.float32), relu=False),
    ])
    return net, np.asarray(d["calib_x"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="firmware/generated")
    ap.add_argument("--classes", default="", help="类别名，逗号分隔，只用于生成注释和 TM_CLASS_NAMES")
    ap.add_argument("--golden", type=int, default=16)
    args = ap.parse_args()

    net, calib = from_torch_npz(args.model)
    names = [s for s in args.classes.split(",") if s] or None
    qnet = quantize(net, calib, class_names=names)

    # golden vector 从校准集里挑，按预测类别轮流取——全是同一类的话，
    # 一个"永远返回同一个向量"的板上实现也能通过逐位比对，测试就空转了
    by_cls = {}
    for x in calib:
        xi = qnet.quantize_input(x)
        by_cls.setdefault(int(np.argmax(forward_int(qnet, xi)[0])), []).append(xi)
    picked, i = [], 0
    while len(picked) < args.golden and any(i < len(v) for v in by_cls.values()):
        for c in sorted(by_cls):
            if i < len(by_cls[c]) and len(picked) < args.golden:
                picked.append(by_cls[c][i])
        i += 1
    if len({int(np.argmax(forward_int(qnet, x)[0])) for x in picked}) < 2:
        print("⚠ golden vector 全落在同一类上。逐位比对还是有效的，但它验不到"
              "不同判决路径——建议换一批更有代表性的校准样本。")

    os.makedirs(args.out, exist_ok=True)
    for name, content in export(qnet, golden_x_i8=np.stack(picked)).items():
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print("写出", p)

    # 跟 float 比一下掉了多少。只打印不拦——掉多少算可接受是产品决策，
    # 但**必须让人看见这个数**，不然量化就成了一步看不见的损耗
    f_pred = [int(np.argmax(net.forward(x))) for x in calib]
    q_pred = [int(np.argmax(forward_int(qnet, qnet.quantize_input(x))[0])) for x in calib]
    agree = float(np.mean([a == b for a, b in zip(f_pred, q_pred)]))
    sat = float(np.mean([np.mean(np.abs(qnet.quantize_input(x)) >= 127) for x in calib]))
    print(f"int8 与 float 判别一致率 {agree:.4f}（校准集上）")
    print(f"输入饱和比例 {sat:.4f}" + ("  ← 偏高，校准集可能没覆盖到剧烈动作" if sat > 0.02 else ""))


if __name__ == "__main__":
    main()
