"""采样 → 窗口的 Python 参考实现。对应固件 tm_window.c，两边必须逐位一致。

这一层的 bug 特别隐蔽：窗口没对齐、hop 算错、量化用错 scale——模型再对也白搭，
而表现出来只是"板上准确率比训练时低"，看起来跟推理代码毫无关系。所以它和
tm_runtime 一样要有参考实现、要对答案。
"""

import numpy as np

from .net import quantize_input_ref


class Window:
    def __init__(self, n_ch, n_t, hop, in_scale, in_zp):
        self.n_ch = n_ch
        self.n_t = n_t
        self.hop = hop if hop > 0 else n_t
        self.in_scale = in_scale
        self.in_zp = in_zp
        self.buf = np.zeros((n_ch, n_t), dtype=np.int8)
        self.head = 0
        self.filled = 0
        self.since = 0

    def push(self, sample):
        """喂一个样本（长度 n_ch 的实数）。凑齐一个窗口就返回 int8 [n_ch, n_t]，否则 None。"""
        q = quantize_input_ref(np.asarray(sample, np.float64), self.in_scale, self.in_zp)
        self.buf[:, self.head] = q
        self.head = (self.head + 1) % self.n_t
        self.filled = min(self.filled + 1, self.n_t)
        self.since += 1
        # 攒满之前不出窗：用半截窗口（后面补零）去推理，等于喂给模型一段训练时
        # 从没见过的信号，它会给出一个看起来正常、实际毫无依据的类别
        if self.filled < self.n_t or self.since < self.hop:
            return None
        self.since = 0
        idx = (self.head + np.arange(self.n_t)) % self.n_t
        return self.buf[:, idx].copy()
