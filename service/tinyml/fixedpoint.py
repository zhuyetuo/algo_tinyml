"""定点重量化的那几个算子。**这个文件和 core/tm_runtime.c 必须逐位一致。**

为什么要自己写一份、而不是"反正差不多"：
量化模型在板上跑偏，最常见的原因不是模型不好，是重量化这一步的舍入规则两边不一样。
差一个 LSB，argmax 就可能翻，而且**不报任何错**——你只会看到"板上准确率莫名其妙低几个点"，
查起来极其痛苦。所以这里照抄 gemmlowp / TFLite 的那套规则（TFLite 板上跑的就是它），
Python 和 C 各实现一遍，再用 golden vector 逼着两边逐位对上。

用 int64 中间值是必须的：Python 的 int 不会溢出，C 的 int32 会，两边要对上就得
在 Python 这边**显式**按 int32 截断/饱和。
"""

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _as_i64(x):
    return np.asarray(x, dtype=np.int64)


def wrap_i32(x):
    """按 32 位有符号回绕。C 那边就是这个语义（靠 uint32 显式写出来，避开 UB）。"""
    v = _as_i64(x) & 0xFFFFFFFF
    return np.where(v >= (1 << 31), v - (1 << 32), v).astype(np.int64)


def saturating_rounding_doubling_high_mul(a, b):
    """(a*b*2) 的高 32 位，带四舍五入。gemmlowp 的 SaturatingRoundingDoublingHighMul。

    a、b 都是 int32。a==b==INT32_MIN 时真值 2^31 超出 int32，饱和到 INT32_MAX——
    这个分支实际几乎碰不到，但两边都得有，否则就是一个永远不触发、一触发就对不上的坑。
    """
    a = _as_i64(a)
    b = _as_i64(b)
    ab = a * b
    # 四舍五入：正数加半个 LSB，负数减半个 LSB（不是统一 +，否则负数方向偏一位）
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30)).astype(np.int64)
    out = (ab + nudge) >> 31
    out = np.where((a == INT32_MIN) & (b == INT32_MIN), INT32_MAX, out)
    return out.astype(np.int64)


def rounding_divide_by_pot(x, exponent):
    """除以 2^exponent，四舍五入、且对负数**向偶数方向**取整（gemmlowp 的规则）。

    直接用 >> 是向下取整，负数会系统性偏一位——这正是"板上比 PC 低半个点"的典型来源。
    """
    x = _as_i64(x)
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + np.where(x < 0, 1, 0).astype(np.int64)
    return (x >> exponent) + np.where(remainder > threshold, 1, 0).astype(np.int64)


def multiply_by_quantized_multiplier(x, multiplier, shift):
    """定点乘一个 (multiplier, shift) 表示的实数 —— 重量化的核心。

    实数 M ≈ multiplier * 2^(shift-31)，multiplier 是 [2^30, 2^31) 的 int32。
    shift > 0 时先左移（放大），shift < 0 时后右移（缩小）。
    """
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    # 左移按 32 位回绕，跟 C 那边一致。Python 的整数不会溢出，不显式回绕的话
    # 两边在极端值上就会分家——而那正是最难在板上复现的一类不一致。
    v = wrap_i32(_as_i64(x) * (1 << left_shift))
    v = saturating_rounding_doubling_high_mul(v, multiplier)
    return rounding_divide_by_pot(v, right_shift)


def quantize_multiplier(real_multiplier):
    """把实数 M（必须在 (0,1) 附近，典型 1e-4 ~ 1e-1）拆成 (int32 multiplier, shift)。

    M == 0 时返回 (0, 0)：某个输出通道权重全零时会出现，不特判的话下面的 while 会死循环。
    """
    if real_multiplier == 0.0:
        return 0, 0
    shift = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        shift -= 1
    while real_multiplier >= 1.0:
        real_multiplier /= 2.0
        shift += 1
    q = int(round(real_multiplier * (1 << 31)))
    if q == (1 << 31):  # 四舍五入正好溢出，等价于 0.5 再多进一位
        q //= 2
        shift += 1
    assert q <= INT32_MAX
    return q, shift


def sat_i8(x):
    return np.clip(_as_i64(x), -128, 127).astype(np.int8)
