"""把 imu_train 的 scikit-learn 模型（ml_*.pkl）挂成一个端侧服务能跑的模型。

跟 EdgeCNN / EdgeRF 的区别：**那两个跑的是板上那份 C，这个跑的是服务器上的
sklearn**。所以这一条路线回答的不是"板子会报什么"，而是"这个模型效果怎么样"。

为什么需要它：想在平台上并排比两个**服务端**模型时，现在没有别的地方可挂。
algo_service 是线上服务，只有一个模型、不能动；端侧服务本来只收导出成 C 的
模型，而"先看看效果好不好、值不值得导出成 C"恰恰发生在导出之前。
典型场景就是这个：训了一版只用加速计的模型，想先在平台上跑真实样本看看
效果掉多少，再决定要不要为它改固件。

**后处理跟别的端侧模型完全一样**（EdgeRunner 统一处理），也就是跟线上
「稳定版 v2」同一份代码。所以跟线上版本比，差的只有模型本身。

## 跟 EdgeRF 的一个关键差别：is_dl

EdgeCNN / EdgeRF 对 infer_file 声明 `is_dl=True`，因为它们要拿**原始窗口**，
特征在 C 里自己算。这里相反：sklearn 模型吃的就是 imu_train 那套手工特征，
所以要 `is_dl=False`，让 infer_file 先跑 extract_features。

写反了**不会报错**：is_dl=True 时传进来的是 [N, T, 8] 的窗口，
sklearn 会抱怨维度不对——这个还好，会当场炸。反过来才危险。
"""

from __future__ import annotations

import os


class SkModel:
    """薄薄一层：转发 predict_proba，外加一次**特征维度的前置检查**。

    为什么要自己查一遍维度，而不是等 sklearn 报错：sklearn 的原话是

        X has 193 features, but RandomForestClassifier is expecting 57 features

    单看这句话，人第一反应是去查特征提取哪里多算了，而真正的原因通常是
    "这个 pkl 不是这套特征训的"（比如真 3 通道训的 57 维模型被挂到了
    8 通道的推理链上）。所以这里把话说全。
    """

    def __init__(self, model, classes, path="", feature_select=None):
        self.model = model
        self.classes = list(classes)
        self.path = path
        # 训练时只用了 193 维里的一部分（比如 acc3 那条只用 113 维非陀螺仪的）。
        # 推理链照常算 193 维，这里按下标取列 —— 于是 5 通道的模型
        # **不需要一条新的预处理链**就能上线。
        #
        # 下标是训练时存进 ml_*.json 的，不在这里按特征名现筛：
        # 服务这边依赖的是它自己那份 imu_train，版本一旦跟训练机不同，
        # 筛出来的下标会整体错位，而错位**不报错**——每一维都对到别的特征上。
        # from_dim 就是为了让这种错位当场暴露。
        self.select = None
        self.select_from = None
        if feature_select:
            import numpy as np
            self.select = np.asarray(feature_select["indices"], np.int64)
            self.select_from = int(feature_select["from_dim"])
        # 给 EdgeRunner 看的：它靠这个决定传给 infer_file 的 is_dl
        self.is_dl = False
        n_out = getattr(model, "n_classes_", None)
        if n_out is not None and int(n_out) != len(self.classes):
            # 类别数对不上时**当场退出**。少一类的话 predict_proba 的列
            # 会跟 classes 错位，平台上看到的每一条都标着错的行为名，
            # 而数值本身完全正常——没有任何迹象
            raise ValueError(
                f"{path}: 模型有 {int(n_out)} 类，meta 里写的是 "
                f"{len(self.classes)} 类（{self.classes}）。"
                "pkl 和它旁边那个 json 不是一次训练出来的。")

    @property
    def n_features(self):
        return getattr(self.model, "n_features_in_", None)

    def _select(self, feats):
        if self.select is None:
            return feats
        got = int(getattr(feats, "shape", (0, 0))[-1])
        if got != self.select_from:
            raise ValueError(
                f"{self.path}: 这个模型训练时是从 {self.select_from} 维里取列的，"
                f"而推理链算出来 {got} 维。\n"
                "  两边的 imu_train 特征代码不是同一版——按老下标取列**不会报错**，"
                "只会让每一维都对到别的特征上，模型照样给得出结果。\n"
                "  把训练机那份 src/ml/features.py 同步过来，或者重训一次。")
        return feats[:, self.select]

    def predict_proba(self, feats):
        feats = self._select(feats)
        want = self.n_features
        got = getattr(feats, "shape", (None, None))[-1]
        if want is not None and got is not None and int(want) != int(got):
            raise ValueError(
                f"{self.path}: 这个模型要 {int(want)} 维特征，推理链算出来的是 "
                f"{int(got)} 维。\n"
                "  推理链固定按 8 通道算，出来是 193 维。想只用加速计的话，"
                "训练要走 imu_train 里那两条之一：\n"
                "    acc3/      5 通道（acc3 + pitch/roll）113 维，"
                "ml_*.json 里带 feature_select，服务按下标取列；\n"
                "    acc_only/  8 通道形状、陀螺仪置零，193 维，整份直接喂。\n"
                "  只留 acc 三列直接训出来的是 57 维，那条路这里跑不了。")
        return self.model.predict_proba(feats)

    def predict(self, feats):
        return self.model.predict(self._select(feats))


def load(pkl_path, classes, feature_select=None):
    """读 pkl。joblib 是 imu_train 存模型用的那个，这里必须用同一个。"""
    # **先查路径再 import**：没装 joblib 的机器上，给错路径的人会先看到
    # "No module named 'joblib'"，然后去装一个装完还是错的依赖
    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))
    if os.path.isdir(pkl_path):
        raise ValueError(f"{pkl_path} 是个目录，sk 这条路线的 gen 要指到 .pkl 文件")
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "kind: sk 要 joblib（imu_train 存模型用的就是它）："
            "pip install joblib scikit-learn") from e
    return SkModel(joblib.load(pkl_path), classes, path=pkl_path,
                   feature_select=feature_select)
