"""把固件那份 C 包装成 imu_train 认得的 `predict_proba` 对象。

这样 `imu_train/src/infer_csv_scratch.py` 的 `infer_file()` 就能原样驱动端侧模型——
CSV 读取、降采样、滑窗、重力对齐、姿态角、片段聚合**一行都不用重写**。

为什么这件事重要到要专门说：那条预处理链里有一处顺序是致命的——

    tilt = append_raw_tilt_batch(X)[:, :, 6:8]   # 必须在重力对齐**之前**算
    X_aligned = gravity_align(X)
    X_aligned = concat([X_aligned, tilt])

反过来先对齐再算倾角的话，重力对齐会把每个窗口的平均倾角归零，
**绝对姿态（躺着/坐着/站着）整个消失**。不报错，只是效果差一截。
自己抄一遍预处理，迟早会在这种地方跟训练侧分家，而分家的表现是
"平台上看着对、板上不对"——查都没法查。所以这里只提供模型，不碰管线。

置信度是**服务端算的，端上没有**：
argmax 对仿射变换保序，所以板子直接对 int8 分数取 argmax 就行，
不用还原成实数、更不用算 softmax。这里还原只是为了给标注平台一个分数，
不影响判决——判决在 C 里已经定了。
"""

import numpy as np


def softmax(z):
    """数值稳定的 softmax。减最大值不是可选的——logit 到 ±100 就会 overflow 成 inf/nan，
    而 nan 一路传到 argmax 会安静地变成"总是第 0 类"。"""
    z = np.asarray(z, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class EdgeCNN:
    """CNN 那条：原始窗口直接进 C（tm_prep 归一化 + tm_invoke 推理）。

    predict_proba 的入参是 imu_train 那边的 `X_aligned`，形状 [N, T, C]
    （时间在前），而 C 要的是 channel-first [C, T]。这里转一次。
    **转错方向不会报错**——T=16、C=8 不相等，形状检查会拦住；但如果哪天
    窗口点数正好等于通道数，就只能靠下面那条断言了。
    """

    def __init__(self, engine, classes):
        self.engine = engine
        self.classes = list(classes)
        if len(self.classes) != engine.n_classes:
            raise ValueError(
                f"类别数对不上：传进来 {len(self.classes)} 个，模型是 {engine.n_classes} 个。"
                "多半是 .json 和 .pt 不配套。")
        # imu_train 那边靠这个字段判断要不要走手工特征。端侧 CNN 吃原始窗口，
        # 所以是 True——写死在这里比让调用方记住可靠
        self.is_dl = True

    def predict_proba(self, X_aligned):
        """X_aligned: float [N, T, C] → float64 [N, n_classes] 概率。"""
        X = np.asarray(X_aligned, np.float32)
        if X.ndim != 3:
            raise ValueError(f"要 [N, T, C]，给的是 {X.shape}")
        n, t, c = X.shape
        if (c, t) != (self.engine.n_ch, self.engine.n_t):
            raise ValueError(
                f"窗口是 {t} 点 × {c} 通道，模型要 {self.engine.n_t} 点 × "
                f"{self.engine.n_ch} 通道。窗口长度或通道数跟训练时不一致。")
        if n == 0:
            return np.empty((0, len(self.classes)), np.float64)
        # [N, T, C] → [N, C, T]
        _, scores = self.engine.infer(np.ascontiguousarray(X.transpose(0, 2, 1)))
        logits = (scores.astype(np.float64) - self.engine.out_zp) * self.engine.out_scale
        return softmax(logits)

    def predict(self, X_aligned):
        return np.argmax(self.predict_proba(X_aligned), axis=1)
