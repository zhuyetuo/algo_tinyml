"""叶子量化的验证。

这件事值得单独测，因为它**改变判决结果**——不像减树那样是统计上等价的操作。
量化误差正好落在两个类概率相差 < 1/255 的样本上时，argmax 会翻。所以要测的
不是"它不变"，而是"它变得可控、可复现、而且量化本身是对的"。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.forest import Forest, compact_flash_bytes, quantize_leaves


def _forest(leaf_proba, n_classes):
    """一棵只有根一个叶子的树 × n 个，方便直接盯住叶子那一侧。"""
    n = len(leaf_proba)
    return Forest(
        n_features=4,
        n_classes=n_classes,
        tree_offset=np.arange(n + 1, dtype=np.int32),
        node_feature=np.arange(n, dtype=np.int32),
        node_threshold=np.zeros(n, np.float32),
        node_left=np.full(n, -1, np.int32),
        node_right=np.full(n, -1, np.int32),
        leaf_proba=np.asarray(leaf_proba, np.float32),
    )


def test_quantize_grid_is_exact_multiples_of_1_over_255():
    f = _forest([[0.1, 0.9], [0.5, 0.5], [0.0, 1.0]], 2)
    q = quantize_leaves(f)
    # 还原之后每个值都应该正好落在 k/255 上
    scaled = q.leaf_proba.astype(np.float64) * 255
    assert np.allclose(scaled, np.round(scaled), atol=1e-6)


def test_quantize_rounds_half_away_from_zero_not_bankers():
    """正好落在中点的值，np.round 往偶数走，我们要往上走。

    **这里必须用 levels=128 而不是默认的 255**，否则这条测试是空的：
    leaf_proba 是 float32，2.5/255 存进去就不是精确的 2.5/255，乘回 255 得到
    2.4999995，两种取整规则给的答案一样，变异成 np.round 也照样绿。
    128 是 2 的幂，2.5/128 在 float32 里精确，乘回来正好是 2.5 —— 这才分得开：
    banker's 给 2（偶），远离零给 3。
    """
    f = _forest([[2.5 / 128, 1 - 2.5 / 128]], 2)
    q = quantize_leaves(f, levels=128)
    assert float(q.leaf_proba[0, 0]) * 128 == pytest.approx(3.0)


def test_quantize_does_not_mutate_input():
    """调用方要能拿量化前后两份跑同一批样本对比，原地改就没法比了。"""
    f = _forest([[0.3, 0.7]], 2)
    before = f.leaf_proba.copy()
    quantize_leaves(f)
    assert np.array_equal(f.leaf_proba, before)


def test_quantize_error_is_bounded_by_half_a_step():
    rng = np.random.default_rng(0)
    p = rng.random((200, 5))
    p /= p.sum(1, keepdims=True)
    q = quantize_leaves(_forest(p, 5))
    assert np.max(np.abs(q.leaf_proba - p.astype(np.float32))) <= 0.5 / 255 + 1e-6


def test_quantize_clips_into_range():
    """越界的输入要被夹回 [0,1]，还原出来的值不能 > 1。

    **1.0000002 这种量级测不出东西**：乘 255 之后是 255.00005，floor(+0.5)
    还是 255，clip 是个空操作，删掉 clip 测试照样绿。要用真正会溢出一格的值。
    （sample_weight 不归一、或者上游传进来的是计数而不是概率时，就是这个量级。）
    """
    f = _forest([[1.01, 0.0]], 2)
    q = quantize_leaves(f)
    assert 0.0 <= float(q.leaf_proba[0, 0]) <= 1.0


@pytest.mark.parametrize("levels", [0, 256, -1])
def test_quantize_rejects_levels_uint8_cannot_hold(levels):
    with pytest.raises(ValueError):
        quantize_leaves(_forest([[0.5, 0.5]], 2), levels=levels)


def test_compact_flash_uint8_leaves_are_a_quarter_of_float32():
    f = _forest(np.full((10, 5), 0.2), 5)
    b32 = compact_flash_bytes(f, leaf_bits=32)
    b8 = compact_flash_bytes(f, leaf_bits=8)
    assert b32["leaves"] == 10 * 5 * 4
    assert b8["leaves"] == 10 * 5 * 1
    # 节点那一侧不该受叶子位宽影响
    assert b32["nodes"] == b8["nodes"] == 10 * 6


def test_compact_flash_rejects_unsupported_leaf_width():
    with pytest.raises(ValueError):
        compact_flash_bytes(_forest([[0.5, 0.5]], 2), leaf_bits=16)
