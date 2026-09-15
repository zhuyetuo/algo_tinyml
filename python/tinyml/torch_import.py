"""把 imu_train 的 `dl_cnn_best.pt` + `dl_cnn_best.json` 读成本仓库的 FloatNet。

为什么要有这一层：imu_train 的 CNN 是
    Conv1d(padding=k//2) → BatchNorm1d → ReLU → MaxPool1d(2) → Dropout   ×3
    → flatten → Linear
而端上的运行时只有 conv / maxpool / dense 三个算子。差的两样这样处理：

  · **BatchNorm 折进卷积**。推理期的 BN 是逐通道的仿射变换，折叠是恒等变换
    （见 net.fold_batchnorm），端上不需要这个算子，也不会多一次重量化。
  · **Dropout 直接丢掉**。推理期它就是恒等，PyTorch 在 model.eval() 下也是这样。

还有一样不能丢：**逐通道 z-score**。训练时输入做了 (x - ch_mean) / ch_std，
端上不做同一件事的话，输入分布跟训练时对不上——模型效果明显下降但**不会报错**。
imu_train 自己的推理端也是靠 .json 里那两个数组，这里照做。

参数不折进第一层卷积（数学上可行）的原因：折进去之后端上的输入就是原始量纲，
而 8 个通道的量纲差两个数量级（加速度 ±40，角速度 ±2000），
共用一个 int8 输入 scale 会把加速度那几路压成几个格子。先归一化再量化，
所有通道落在同一个范围里，int8 的分辨率才用得满。
"""

import json
import os

import numpy as np

from .net import Conv1D, Dense, FloatNet, MaxPool1D, fold_batchnorm


def _as_numpy(v):
    """torch 张量 → numpy，不 import torch。

    state_dict 里是 torch.Tensor，但它实现了 __array__，np.asarray 直接就能转。
    这样这个模块在没装 torch 的机器上也能用（比如只做量化的那台）。
    """
    return np.asarray(v, dtype=np.float32)


# 两条路线要的字段不一样，**不能用同一套必填项**：
#
#   · CNN 吃原始窗口，训练时对输入做了逐通道 z-score，所以端上必须有
#     ch_mean / ch_std（tm_prep 用），少了就是输入分布跟训练时对不上。
#   · RF 吃 193 维手工特征，**不做任何归一化**——森林的阈值就是按原始量纲的
#     特征值训的。给它 ch_mean 反而是错的。
#
# 我一开始用同一个 load_meta 去读 RF 的 ml_rf.json，报"缺 ch_mean"。
# 那不是 json 的问题，是我把 CNN 的加载器套到了 RF 上。
_COMMON = ("classes", "window_size", "hz")
_CNN_ONLY = ("ch_mean", "ch_std", "n_channels")


def load_meta(json_path, kind="cnn"):
    """读 imu_train 的训练元数据。kind: "cnn"（dl_*.json）或 "rf"（ml_*.json）。"""
    if kind not in ("cnn", "rf"):
        raise ValueError(f"kind={kind}，只支持 cnn / rf")
    with open(json_path, encoding="utf-8") as f:
        m = json.load(f)
    need = _COMMON + (_CNN_ONLY if kind == "cnn" else ())
    for k in need:
        if k not in m:
            raise ValueError(
                f"{json_path} 里缺 {k}（按 {kind} 读的）。\n"
                "  CNN 要的是 imu_train 的 dl_*.json，RF 要的是 ml_*.json——"
                "两者字段不同，别指错。")
    if kind == "cnn":
        std = np.asarray(m["ch_std"], np.float64)
        if np.any(std <= 0):
            # std 为 0 意味着那一路通道在训练集上是常数。除下去会得到 inf/nan，
            # 而 nan 一路传到 argmax 会安静地变成"总是第 0 类"
            raise ValueError(
                f"ch_std 里有非正值：{std.tolist()}；那一路通道在训练集上是常数？")
    return m


def normalize(x, meta):
    """逐通道 z-score，跟训练时一致。x: [..., C, T]。

    走 float32：这一份是给**训练侧的 float 前向**用的，对应 PyTorch 里的
    同一个运算。端上那一份是 prep_quantize_ref（float64），跟 C 逐位一致——
    两者故意不是同一个函数，因为它们要对齐的对象不同。
    """
    mean = np.asarray(meta["ch_mean"], np.float32).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float32).reshape(-1, 1)
    return ((np.asarray(x, np.float32) - mean) / std).astype(np.float32)


def prep_quantize_ref(x, meta, in_scale, in_zp):
    """归一化 + 量化，一步到位，跟固件 tm_prep() **逐位一致**。

    x: float [C, T]，**原始量纲**（不是归一化过的）→ int8 [C, T]。

    两步走（先减均值除标准差，再除 in_scale），不合成 (x*a + b)：
    合成会改变舍入，两边就差几个 LSB——而那种差异看起来完全像"正常的数值误差"，
    是最不容易被怀疑到的一类不一致。

    全程 float64，因为 C 那边也是 double。M4F 没有双精度 FPU，这段走软件浮点，
    但只有 n_ch × n_t = 128 次，跟后面几十万次乘加的卷积比可以忽略。
    """
    mean = np.asarray(meta["ch_mean"], np.float64).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float64).reshape(-1, 1)
    v = (np.asarray(x, np.float64) - mean) * (1.0 / std)
    q = v / np.float64(in_scale)
    # 四舍五入远离零。np.round 是银行家舍入（.5 取偶），跟 C 的 round() 走反
    r = np.sign(q) * np.floor(np.abs(q) + 0.5) + in_zp
    return np.clip(r, -128, 127).astype(np.int8)


def load_cnn(pt_path, json_path=None, eps=1e-5):
    """读 imu_train 的 cnn checkpoint，返回 (FloatNet, meta)。

    只认 `src/dl/models/cnn.py` 那个结构。别的模型（cnn_lstm / filternet /
    collar_cnn）有 LSTM、有多分支，端上运行时没有对应算子——**宁可在这里
    明确报错，也不要"尽力解析"**：解析错了不会崩，只会让板上安静地算出别的东西。
    """
    if json_path is None:
        json_path = os.path.splitext(pt_path)[0] + ".json"
    meta = load_meta(json_path)
    if meta.get("model") not in (None, "cnn"):
        raise ValueError(
            f"这个 checkpoint 是 {meta['model']}，不是 cnn。"
            "端上运行时只有 conv/maxpool/dense 三个算子，"
            f"{meta['model']} 里的算子没有对应实现。")

    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"读 .pt 要装 torch（{e}）。在训练机上跑这一步。")
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)

    # 键名形如 conv.0.weight / conv.1.{weight,bias,running_mean,running_var} / fc.weight
    # 每段 5 层（conv, bn, relu, pool, dropout），所以卷积在 0, 5, 10, ...
    layers = []
    i = 0
    while f"conv.{i}.weight" in sd:
        w = _as_numpy(sd[f"conv.{i}.weight"])
        b = _as_numpy(sd[f"conv.{i}.bias"])
        k = w.shape[2]
        conv = Conv1D(w, b, relu=True, pad=k // 2)   # padding=k//2，跟 cnn.py 一致
        bn = i + 1
        if f"conv.{bn}.running_var" not in sd:
            raise ValueError(f"conv.{i} 后面没有 BatchNorm（找不到 conv.{bn}.running_var）；"
                             "结构跟 imu_train 的 cnn.py 对不上")
        conv = fold_batchnorm(
            conv,
            gamma=_as_numpy(sd[f"conv.{bn}.weight"]),
            beta=_as_numpy(sd[f"conv.{bn}.bias"]),
            mean=_as_numpy(sd[f"conv.{bn}.running_mean"]),
            var=_as_numpy(sd[f"conv.{bn}.running_var"]),
            eps=eps,
        )
        # Dropout 推理期是恒等，直接不生成任何东西
        layers += [conv, MaxPool1D(2)]
        i += 5

    if not layers:
        raise ValueError("一层卷积都没解析出来；state_dict 的键名跟预期不符")
    if "fc.weight" not in sd:
        raise ValueError("找不到 fc.weight")
    layers.append(Dense(_as_numpy(sd["fc.weight"]), _as_numpy(sd["fc.bias"]), relu=False))

    net = FloatNet(layers)

    # **形状自检**：走一遍，确认展平维度跟 fc 对得上。对不上的话，
    # 报错要发生在这里，而不是量化完导出到板上之后才发现 dense 输入维度不对
    probe = np.zeros((meta["n_channels"], meta["window_size"]), np.float32)
    out = net.forward(probe)
    if out.shape != (len(meta["classes"]),):
        raise ValueError(
            f"前向出来是 {out.shape}，但类别数是 {len(meta['classes'])}。"
            "window_size 或 n_channels 跟 checkpoint 不配套。")
    return net, meta
