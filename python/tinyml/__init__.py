"""宠物项圈端侧推理（GR5513 / Cortex-M4F）的量化与导出工具链。

只依赖 numpy。训练用什么框架（imu_train 那边是 PyTorch）跟这里无关——
训练侧最后只交出一个 {名字: numpy 数组}，剩下的量化、导出、逐位对照都在这里做。
这样换框架不会波及板上那一侧，而且工具链可以用随机权重自测，不用等模型训好。
"""

from .net import (  # noqa: F401
    Conv1D, Dense, FloatNet, MaxPool1D, QNet,
    forward_int, make_net, quantize,
)
from .export_c import export  # noqa: F401
from .forest import Forest, from_sklearn  # noqa: F401
from .export_forest_c import export as export_forest  # noqa: F401
from .features import extract_one, n_features  # noqa: F401
from .export_features_c import export as export_feat_cfg  # noqa: F401
