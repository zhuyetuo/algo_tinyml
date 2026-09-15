"""16 点窗口（1 秒 @16Hz）的 C ↔ Python 对照。

**为什么单独有这个文件**：原来的 test_features_c.py 只测 32 点窗口，而
imu_train 那边实际在用的是 `--window_s 1 --stride_s 0.5`，16Hz 下就是 **16 个点**。
只测 32 点等于没测到真实配置——FFT 的级数、Welch 的分段数、jerk 的长度都不一样。

实测下来非熵的 193 维**全部逐位相同**，熵那几维最多差 2 ULP（32 点窗口时是 1 ULP）。
窗口越短谱线越少、每个 bin 的占比越大，logf 的末位误差在求和里显得更明显，
2 ULP 仍在 libm 舍入能解释的范围内。容差按窗口长度分开写，不用一个统一的松容差
盖住两边——那样会把真 bug 一起放过去。
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.export_features_c import export as export_cfg  # noqa: E402
from tinyml.features import extract_one, n_features  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

N_T, N_CH, NPERSEG, FS = 16, 8, 16, 16.0
DIM = n_features(N_CH)

# 每个窗口长度自己的熵容差。分开写是有意的：统一用最松的那个，
# 就会把 32 点窗口上真正的 bug 放过去
ENTROPY_ULP = {16: 2, 32: 1}


def _entropy_idx():
    idx, p = [], 11 * N_CH
    for _ in range(min(6, N_CH)):
        idx.append(p + 3)
        p += 8
    p += 8
    for _ in range(2):
        p += 11
        idx.append(p + 3)
        p += 8
    return set(idx)


ENT = _entropy_idx()


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = rng.normal(0, 2.0, size=(N_T, N_CH)).astype(np.float32)
        w[:, 2] += np.float32(9.8)
        if i % 4 == 1:
            w[:, 1] = np.float32(0.5)                       # 常数通道
        if i % 4 == 2:
            w[:, 5] = np.resize(np.array([0, 1, 2, 2, 2, 1, 0, -1], np.float32),
                                N_T).astype(np.float32)     # 带平顶的峰
        if i % 4 == 3:
            col = np.resize(np.array([1.0, -1.0], np.float32), N_T).copy()
            col[[2, 3, 9]] = np.float32(0.0)
            col -= np.float32(col.mean())                   # 一部分样本正好落在均值上
            w[:, 4] = col
        out.append(w)
    return np.stack(out)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("feat16")
    for name, content in export_cfg(N_T, N_CH, NPERSEG, FS).items():
        (d / name).write_text(content, encoding="utf-8")
    binp = d / "feat16"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-fno-math-errno",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-DTM_FEAT_MAX_T={N_T}", f"-DTM_FEAT_MAX_NPERSEG={NPERSEG}",
         f"-I{FW}", f"-I{d}", os.path.join(FW, "tm_features.c"),
         str(d / "tm_feat_cfg.c"), os.path.join(ROOT, "tests", "host_features.c"),
         "-lm", "-o", str(binp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def _run(exe, ws):
    text = "\n".join(" ".join(f"{v:.9g}" for v in w.T.reshape(-1)) for w in ws)
    r = subprocess.run([str(exe)], input=text + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [[int(v, 16) for v in line.split()] for line in r.stdout.strip().splitlines()]


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def test_16点窗口维度仍是_193():
    """特征维度只由通道数决定，跟窗口长度无关。写成测试是因为"换个窗口长度
    维度会不会变"这个问题，靠想是想不清楚的。"""
    assert DIM == 193


def test_16点窗口_除熵外全部逐位相同(exe):
    ws = _windows(16, seed=5)
    got = _run(exe, ws)
    bad = []
    for i, (w, row) in enumerate(zip(ws, got)):
        ref = extract_one(w, FS, NPERSEG)
        assert len(row) == len(ref) == DIM
        for k, (a, b) in enumerate(zip(row, ref)):
            if k not in ENT and a != _bits(b):
                bad.append((i, k))
    assert not bad, f"{len(bad)} 处不一致：{bad[:5]}"


def test_16点窗口_熵在容差内(exe):
    ws = _windows(16, seed=5)
    got = _run(exe, ws)
    worst = 0
    for w, row in zip(ws, got):
        ref = extract_one(w, FS, NPERSEG)
        for k in ENT:
            worst = max(worst, abs(row[k] - _bits(ref[k])))
    assert worst <= ENTROPY_ULP[N_T], (
        f"熵差了 {worst} ULP，超过 16 点窗口的容差 {ENTROPY_ULP[N_T]}。"
        "libm 的舍入解释不了这么多，去查 logf 之外的地方")
