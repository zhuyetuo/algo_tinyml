"""一个**很小**的一维卷积网络，以及它的 float / int8 两套前向。

网络故意写死成一条链（conv → relu → pool → conv → relu → pool → dense），不做成
通用解释器：GR5513 上真正要跑的就是这一条，写成通用的会把 flash 和调试难度一起抬上去，
而且通用解释器里每多一个没用到的算子，都是一份没被 golden vector 覆盖的代码。

输入布局：x[C][T]，C 在外层（channel-first）。IMU 是 6 通道（acc xyz + gyr xyz），
T 是窗口点数。channel-first 是因为卷积内层循环沿 T 走，这样访存连续。
"""

from dataclasses import dataclass, field

import numpy as np

from .fixedpoint import (
    multiply_by_quantized_multiplier,
    quantize_multiplier,
)


# ── float 侧：训练产物就长这样，跟用什么框架训的无关 ──────────────────────
#
# 训练脚本（train_torch.py）最后只吐出一个 {名字: numpy 数组} 的 npz，
# 量化这一整条链**不依赖任何 ML 框架**。这样换框架、换训练机器都不影响板上这一侧，
# 而且量化器可以用随机权重测试——不用先有一个训好的模型才能验证工具链。


@dataclass
class Conv1D:
    w: np.ndarray  # float32 [out_ch, in_ch, k]
    b: np.ndarray  # float32 [out_ch]
    relu: bool = True
    pad: int = 0   # 两端各补多少。pad=k//2 就是 PyTorch 的 padding='same'（k 为奇数时）

    def forward(self, x):  # x: [in_ch, T] -> [out_ch, T + 2*pad - k + 1]
        oc, ic, k = self.w.shape
        if self.pad:
            # **补的是 0**，不是边缘复制。量化那一侧对应的是补 zero_point，
            # 因为 int8 里的"实数 0"就是 zp——这一点靠 _affine 保证 0 可精确表示。
            x = np.pad(x, ((0, 0), (self.pad, self.pad)))
        t_out = x.shape[1] - k + 1
        assert t_out > 0, "窗口比卷积核还短"
        y = np.empty((oc, t_out), dtype=np.float32)
        for o in range(oc):
            acc = np.full(t_out, self.b[o], dtype=np.float32)
            for c in range(ic):
                for j in range(k):
                    acc += self.w[o, c, j] * x[c, j:j + t_out]
            y[o] = acc
        return np.maximum(y, 0.0) if self.relu else y


def fold_batchnorm(conv: Conv1D, gamma, beta, mean, var, eps=1e-5) -> Conv1D:
    """把 BatchNorm1d **折进**卷积的权重和偏置，端上就不需要 BN 这个算子了。

    推理期的 BN 是一个逐通道的仿射变换（训练期不是——那时用的是 batch 统计量，
    而且 running_mean/var 还在更新）。所以：

        y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
          = conv_folded(x)，其中
            w' = w * s[:, None, None]        s = gamma / sqrt(var + eps)
            b' = (b - mean) * s + beta

    这在浮点上是**恒等变换**（不是近似），所以折完之后 float 前向的结果不变。
    对量化反而是好事：BN 单独存在的话要么多一层重量化（多一次精度损失），
    要么在端上引入 float —— 折进去之后两样都没有。

    注意 eps 要跟训练时一致。PyTorch 的 BatchNorm1d 默认 1e-5；填错的话
    误差很小但确实存在，属于"板上跟训练差一点点"里最难查的那一类。
    """
    s = np.asarray(gamma, np.float64) / np.sqrt(np.asarray(var, np.float64) + eps)
    n_out = conv.w.shape[0]
    for name, v in (("gamma", gamma), ("beta", beta), ("mean", mean), ("var", var)):
        if len(np.asarray(v).reshape(-1)) != n_out:
            raise ValueError(
                f"BN 的 {name} 长度 {len(np.asarray(v).reshape(-1))} "
                f"对不上卷积的输出通道数 {n_out}")
    w = conv.w.astype(np.float64) * s[:, None, None]
    b = (conv.b.astype(np.float64) - np.asarray(mean, np.float64)) * s \
        + np.asarray(beta, np.float64)
    return Conv1D(w.astype(np.float32), b.astype(np.float32),
                  relu=conv.relu, pad=conv.pad)


@dataclass
class MaxPool1D:
    pool: int

    def forward(self, x):
        ch, t = x.shape
        # 尾部不够一格就**丢掉**，不补零：补零会在信号末尾凭空造一个 0，
        # 对 relu 之后恒非负的特征图来说那是个假的"最小值"，不是假的最大值，
        # 影响比补零到别处小，但仍然是凭空数据。丢掉更干净，代价只是窗口长度要选好。
        t_out = t // self.pool
        return x[:, :t_out * self.pool].reshape(ch, t_out, self.pool).max(axis=2)


@dataclass
class Dense:
    w: np.ndarray  # float32 [out, in]
    b: np.ndarray  # float32 [out]
    relu: bool = False

    def forward(self, x):  # x: [ch, T] 会被展平成 ch*T
        v = x.reshape(-1)
        y = self.w @ v + self.b
        return np.maximum(y, 0.0) if self.relu else y


@dataclass
class FloatNet:
    layers: list

    def forward(self, x):
        for lyr in self.layers:
            x = lyr.forward(x)
        return x


def flat_size(n_t, n_ch_out=16):
    """默认结构走完两层 conv + 两次 pool 之后，展平给 dense 的维度。

    写成函数而不是常量：窗口长度是要调的（2 秒 @25Hz = 50 点，4 秒 = 100 点），
    写死 48 的话改窗口就会在矩阵乘那里炸，而报错信息跟"窗口太短"毫无关系。
    """
    t = n_t - 5 + 1          # conv1 k=5, valid
    t = t // 4               # pool
    t = t - 3 + 1            # conv2 k=3, valid
    t = t // 4               # pool
    if t <= 0:
        raise ValueError(f"窗口 {n_t} 点太短：两层 conv + 两次 pool 之后时间维被压成 {t}")
    return n_ch_out * t


def make_net(n_ch=6, n_classes=3, seed=None, n_t=64):
    """默认结构。参数量算给你看（n_ch=6, n_classes=3, T=64）：

        conv1 6→8  k5 : 6*8*5   = 240 权重 + 8 bias
        conv2 8→16 k3 : 8*16*3  = 384 权重 + 16 bias
        dense 48→3    : 48*3    = 144 权重 + 3 bias
        合计 768 个 int8 权重 + 27 个 int32 bias ≈ 0.9 KB flash

    比 GR5513 的余量小两个数量级——瓶颈从来不是这里，是 BLE 协议栈占掉的那部分。
    结构小还有一层意思：几百条自采片段的量级上，大模型只会过拟合。
    """
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape).astype(np.float32)

    nf = flat_size(n_t)
    return FloatNet([
        Conv1D(he((8, n_ch, 5), n_ch * 5), np.zeros(8, np.float32), relu=True),
        MaxPool1D(4),
        Conv1D(he((16, 8, 3), 8 * 3), np.zeros(16, np.float32), relu=True),
        MaxPool1D(4),
        Dense(he((n_classes, nf), nf), np.zeros(n_classes, np.float32), relu=False),
    ])


# ── 量化 ──────────────────────────────────────────────────────────────────


@dataclass
class QConv:
    w: np.ndarray        # int8 [out, in, k]，对称量化（zero_point 恒为 0）
    bias: np.ndarray     # int32 [out]，scale = in_scale * w_scale[o]
    mult: np.ndarray     # int32 [out]
    shift: np.ndarray    # int32 [out]
    in_zp: int
    out_zp: int
    relu: bool
    pad: int = 0


@dataclass
class QPool:
    pool: int


@dataclass
class QDense(QConv):
    pass


@dataclass
class QNet:
    layers: list
    in_scale: float
    in_zp: int
    out_scale: float
    out_zp: int
    n_ch: int
    n_t: int
    n_classes: int
    class_names: list = field(default_factory=list)

    def quantize_input(self, x_float):
        return quantize_input_ref(x_float, self.in_scale, self.in_zp)


def quantize_input_ref(x_float, scale, zp):
    """实数 → int8，跟固件 tm_quantize() 逐位一致。

    **不能用 np.round**：numpy 是银行家舍入（.5 取偶），C 的 roundf 是 .5 远离零。
    两者在 2.5 上一个给 2、一个给 3。传感器数据落在正好 .5 的概率不高，但"不高"
    不是"没有"，而这种偶发一位偏差在板上根本查不出来。统一成哪一种不重要，
    两边一样才重要——这里跟 C 走，因为 C 那边用标准库的 roundf 最省事也最不容易写错。
    """
    v = np.asarray(x_float, np.float64) / scale
    q = np.sign(v) * np.floor(np.abs(v) + 0.5) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def _affine(lo, hi):
    """从实数范围推 (scale, zero_point)。0 必须能被精确表示——relu 之后的 0、
    padding 的 0 都靠这一点，否则"什么都没发生"会变成一个非零的偏置。"""
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    scale = (hi - lo) / 255.0
    zp = int(round(-128 - lo / scale))
    return scale, int(np.clip(zp, -128, 127))


def quantize(net: FloatNet, calib_x, class_names=None):
    """训练后量化（PTQ）。calib_x: [N, C, T] 的 float 样本，用来定每层激活的范围。

    校准集必须是**真实数据**，而且要覆盖到剧烈动作——只拿睡觉的片段校准，
    抓挠那一段会整段饱和到 127，模型在最该判对的时候瞎掉。这个错误不会报，
    只会表现成"抓挠召回低"。
    """
    calib_x = np.asarray(calib_x, np.float32)
    assert calib_x.ndim == 3, "calib_x 应该是 [N, C, T]"

    in_scale, in_zp = _affine(calib_x.min(), calib_x.max())

    # 逐层跑一遍 float，收集每层输出的实际范围
    acts = [calib_x]
    for lyr in net.layers:
        acts.append(np.stack([lyr.forward(a) for a in acts[-1]]))

    qlayers = []
    cur_scale, cur_zp = in_scale, in_zp
    for i, lyr in enumerate(net.layers):
        out = acts[i + 1]
        if isinstance(lyr, MaxPool1D):
            # 池化不改变值域，scale/zp 原样传下去——重新算一遍反而会引入一次多余的重量化
            qlayers.append(QPool(lyr.pool))
            continue

        out_scale, out_zp = _affine(out.min(), out.max())
        w = lyr.w
        n_out = w.shape[0]
        flat = w.reshape(n_out, -1)
        # 逐输出通道对称量化：不同卷积核的幅度能差一个数量级，共用一个 scale
        # 会把小的那些压成全 0（它们往往正是学到细节的那几个核）
        w_scale = np.maximum(np.abs(flat).max(axis=1), 1e-12) / 127.0
        qw = np.clip(np.round(flat / w_scale[:, None]), -127, 127).astype(np.int8)
        qw = qw.reshape(w.shape)

        bias_scale = cur_scale * w_scale
        qb = np.round(lyr.b / bias_scale).astype(np.int64)
        qb = np.clip(qb, -(1 << 31), (1 << 31) - 1).astype(np.int32)

        mult, shift = [], []
        for o in range(n_out):
            m, s = quantize_multiplier(float(bias_scale[o] / out_scale))
            mult.append(m)
            shift.append(s)

        cls = QDense if isinstance(lyr, Dense) else QConv
        qlayers.append(cls(
            w=qw, bias=qb,
            mult=np.array(mult, np.int32), shift=np.array(shift, np.int32),
            in_zp=cur_zp, out_zp=out_zp, relu=lyr.relu,
            pad=getattr(lyr, "pad", 0),
        ))
        cur_scale, cur_zp = out_scale, out_zp

    return QNet(
        layers=qlayers, in_scale=in_scale, in_zp=in_zp,
        out_scale=cur_scale, out_zp=cur_zp,
        n_ch=calib_x.shape[1], n_t=calib_x.shape[2],
        n_classes=acts[-1].shape[1],
        class_names=list(class_names or []),
    )


# ── int8 前向：这就是板上那份 C 的参考实现，必须逐位一致 ──────────────────


def _requant(acc, lyr):
    """逐输出通道重量化。按通道循环而不是向量化，是为了跟 C 那边的循环一一对上——
    mult/shift 是 per-channel 的，向量化写法要处理 shift 的逐元素移位，容易跟 C 写岔。"""
    # dense 的 acc 是一维 [out]，卷积是二维 [out, T]。这里**必须**按 [out, ...] 补齐，
    # 不能用 atleast_2d —— 它会把 [3] 变成 [1,3]，于是三个类别全都套用了通道 0 的
    # 乘子，而结果看着完全正常（就是几个 LSB 的偏差）。这个 bug 是被 C 对照测试
    # 逮出来的，反过来说：没有那个测试，它会一直躺在参考实现里。
    acc = np.asarray(acc)
    if acc.ndim == 1:
        acc = acc[:, None]
    lo = lyr.out_zp if lyr.relu else -128
    out = np.empty(acc.shape, dtype=np.int8)
    for o in range(acc.shape[0]):
        v = multiply_by_quantized_multiplier(acc[o], int(lyr.mult[o]), int(lyr.shift[o]))
        out[o] = np.clip(v + lyr.out_zp, lo, 127).astype(np.int8)
    return out


def forward_int(qnet: QNet, x_i8):
    """x_i8: int8 [C, T] -> (int8 输出, int32 累加器)。

    返回累加器是为了调试：两边对不上时，看是哪一层的 acc 先岔开，比只看最后
    argmax 一致不一致有用得多。
    """
    x = np.asarray(x_i8, np.int8)
    last_acc = None
    for lyr in qnet.layers:
        if isinstance(lyr, QPool):
            ch, t = x.shape
            t_out = t // lyr.pool
            x = x[:, :t_out * lyr.pool].reshape(ch, t_out, lyr.pool).max(axis=2).astype(np.int8)
            continue
        xi = x.astype(np.int64) - lyr.in_zp
        if isinstance(lyr, QDense):
            acc = lyr.w.astype(np.int64) @ xi.reshape(-1) + lyr.bias.astype(np.int64)
        else:
            oc, ic, k = lyr.w.shape
            if lyr.pad:
                # **在减掉 zero_point 之后补 0**，等价于在 int8 域补 in_zp。
                # 顺序反过来（先补 0 再减 zp）会让 padding 位置贡献 -zp，
                # 那是个凭空的常数偏置，而且只在边界上——表现成"边缘几个点不对"，
                # 中间全对，最像"实现没问题只是数值误差"的那种错。
                xi = np.pad(xi, ((0, 0), (lyr.pad, lyr.pad)))
            t_out = xi.shape[1] - k + 1
            acc = np.empty((oc, t_out), dtype=np.int64)
            for o in range(oc):
                a = np.full(t_out, int(lyr.bias[o]), dtype=np.int64)
                for c in range(ic):
                    for j in range(k):
                        a += int(lyr.w[o, c, j]) * xi[c, j:j + t_out]
                acc[o] = a
        # C 那边累加器是 int32。这里用 int64 算，所以**必须显式检查**没有溢出——
        # 不查的话溢出只会表现成"板上和 PC 对不上"，而两边代码看起来都对。
        if np.abs(acc).max() > (1 << 31) - 1:
            raise OverflowError("累加器超出 int32，C 侧会回绕；把通道数或窗口调小，或改用 int64 累加")
        last_acc = acc
        x = _requant(acc, lyr)
        if isinstance(lyr, QDense):
            x = x.reshape(-1)
    return x, last_acc
