"""导出评估用的动作门：跟板上 tm_gate 同一个规矩（|a| 抖动小 → 未佩戴 / 静止，其余在剩下的类里取最大）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))
from export_train import _gated_argmax  # noqa: E402


def test_gate_overrides_low_motion_and_excludes_gated_classes():
    gate = {"channel": 2, "notworn_g": 0.001, "still_g": 0.005, "notworn": 0, "still": 1}
    T = 40
    X = np.zeros((3, T, 5))
    X[1, :, 2] = 1 + 0.003 * np.sin(np.arange(T))     # 抖动 ~0.002：静止
    X[2, :, 2] = 1 + 0.2 * np.sin(np.arange(T))       # 在动：交给 CNN
    scores = np.array([[0, 0, 5, 1, 0],               # 几乎不动 → 未佩戴，不管 CNN 说什么
                       [0, 0, 5, 1, 0],
                       [9, 9, 1, 3, 0]], float)       # 在动：未佩戴 / 静止不许选，挑剩下的最大 → 抓挠(3)
    assert _gated_argmax(scores, X, gate).tolist() == [0, 1, 3]


def test_no_gate_is_plain_argmax():
    s = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert _gated_argmax(s, np.zeros((2, 4, 3)), None).tolist() == [1, 0]
