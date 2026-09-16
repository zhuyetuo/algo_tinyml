"""整条链 golden vector 的导出。

它有两个作用，第二个容易被忽略：
  1. 验中间那道接缝（特征排列顺序跟模型对不对得上）；
  2. **让 tm_features 被链进镜像**——链接时 --gc-sections 会把没人调的代码整段丢掉。
     自检不走完整条链的话，实测 tm_features / tm_invoke 在最终镜像里根本不存在，
     量出来的固件体积是假的。
"""

import os
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))
sys.path.insert(0, os.path.dirname(__file__))

from tinyml.export_pipeline_c import export  # noqa: E402
from tinyml.features import extract_one  # noqa: E402
from test_features_c import FS, NPERSEG, _windows  # noqa: E402
from test_rf_pipeline_c import built  # noqa: E402,F401


def _floats(header, name):
    m = re.search(rf"{name}\[\] = \{{(.*?)\}};", header, re.S)
    assert m, f"头文件里找不到 {name}"
    return [float.fromhex(v.strip().rstrip("f")) for v in m.group(1).split(",")]


def test_导出的概率跟参考实现一致(built):
    forest, _ = built
    ws = _windows(4, seed=31)
    h = export(forest, ws, FS, NPERSEG)["tm_pipeline_golden.h"]
    got = _floats(h, "tm_pipeline_proba")
    want = np.concatenate([forest.predict_proba(extract_one(w, FS, NPERSEG)) for w in ws])
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert struct.pack("<f", np.float32(g)) == struct.pack("<f", w)


def test_输入按通道在前排布(built):
    """板上是 [n_ch][n_t]，参考实现吃的是 [n_t][n_ch]。转置写反了不会报错——
    特征全算错，而模型照样给得出类别。"""
    forest, _ = built
    ws = _windows(2, seed=32)
    h = export(forest, ws, FS, NPERSEG)["tm_pipeline_golden.h"]
    got = np.array(_floats(h, "tm_pipeline_in"), np.float32)
    want = np.concatenate([w.T.reshape(-1) for w in ws]).astype(np.float32)
    assert np.array_equal(got, want)


def test_特征维度对不上时导出要报错(built):
    forest, _ = built
    n_ch = 6   # 森林是按 8 通道（193 维）训的，喂 6 通道只能算出 171 维
    ws = np.zeros((2, 32, n_ch), np.float32)
    with pytest.raises(ValueError, match="维特征"):
        export(forest, ws, FS, NPERSEG)
