"""193 维特征：C ↔ Python 参考实现逐位对照。

**范围要说清楚**：这里验的是 `tinyml/features.py`（参考）跟 `tm_features.c`（板上）
一致，不是跟 imu_train 的 scipy 版一致。后者做不到逐位（scipy 是 float64、FFT 算法
不同），只能量——那是 `verify_against_scipy.py` 的事，要在有 scipy 的机器上跑。

除频谱熵外，所有特征要求**逐位相同**。熵用了 `logf`，属于 libm，各实现不保证
正确舍入，所以单独给 1 ULP 的容差。把它跟别的混在一起用统一容差是不对的——
那会顺带把别处真正的 bug 也放过去。
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

N_T, N_CH, NPERSEG, FS = 32, 8, 32, 16.0
DIM = n_features(N_CH)

# 频谱熵在特征向量里的下标：前 11*n_ch 是时域，然后每 8 个一组是一个通道的频域，
# 组内第 4 个（下标 3）是熵。模长那两组也各有一个。
def _entropy_idx():
    idx = []
    p = 11 * N_CH
    for _ in range(min(6, N_CH)):
        idx.append(p + 3)
        p += 8
    p += 8                      # 全局 8 维
    for _ in range(2):          # acc 模长、gyro 模长
        p += 11
        idx.append(p + 3)
        p += 8
    return idx


ENTROPY_IDX = set(_entropy_idx())


def _windows(n, seed=0):
    """合成窗口。混进几种会走到边角分支的形态，随机高斯是盖不到的：
    常数通道（相关系数除零）、带平台的方波（find_peaks 的平顶）、
    正好落在均值上的样本（np.sign 给 0）。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = rng.normal(0, 2.0, size=(N_T, N_CH)).astype(np.float32)
        w[:, 2] += np.float32(9.8)                       # 重力偏置
        if i % 5 == 1:
            w[:, 1] = np.float32(0.5)                    # 常数通道
        if i % 5 == 2:
            w[:, 0] = np.repeat([1.0, 1.0, -1.0, -1.0],
                                N_T // 4).astype(np.float32)   # 方波
            # 方波还不够：它的平顶不是"被两侧更小的值夹住"的局部极大，
            # find_peaks 根本不会数到它。要的是 0,1,2,2,2,1,0 这种**带平顶的峰**——
            # 平台合并写错（每个点算一个峰）只有在这种形状上才看得出来
            pat = np.array([0, 1, 2, 2, 2, 1, 0, -1], np.float32)
            w[:, 5] = np.resize(pat, N_T).astype(np.float32)
        if i % 5 == 3:
            w[:, 3] = np.float32(0.0)                    # 全零：均值也是 0，sign 全 0
            # 全零还不够——sign 全变成 0 还是全变成 +1，穿越次数都是 0，
            # 把 sign(0) 写错也测不出来。要的是**一部分**样本正好落在均值上：
            # ±1 交替（均值 0）中间插几个 0，这时 sign 给 0 会多算两次穿越
            col = np.resize(np.array([1.0, -1.0], np.float32), N_T).copy()
            col[[3, 4, 11, 20]] = np.float32(0.0)
            col -= np.float32(col.mean())                # 保证均值恰好是 0
            w[:, 4] = col
        out.append(w)
    return np.stack(out)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("feat")
    for name, content in export_cfg(N_T, N_CH, NPERSEG, FS).items():
        (d / name).write_text(content, encoding="utf-8")
    binp = d / "feat"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         # 没有它，编译器可以把 a*b+c 合成一次 FMA（少一次舍入），跟 Python
         # 分两步算的结果末位就不同。这是整条链里最容易被忽略的一个开关
         "-ffp-contract=off",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", f"-I{d}", os.path.join(FW, "tm_features.c"),
         str(d / "tm_feat_cfg.c"), os.path.join(ROOT, "tests", "host_features.c"),
         "-lm", "-o", str(binp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def _run(exe, ws):
    # C 那边是 [n_ch][n_t]（通道在前），参考实现吃的是 [n_t][n_ch]
    lines = []
    for w in ws:
        lines.append(" ".join(f"{v:.9g}" for v in w.T.reshape(-1)))
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [[int(v, 16) for v in line.split()] for line in r.stdout.strip().splitlines()]


def test_特征维度是_193(exe):
    assert DIM == 193, DIM   # 8 通道。6 通道是 171
    assert len(_run(exe, _windows(1))[0]) == DIM


def test_除熵以外全部逐位相同(exe):
    ws = _windows(25, seed=3)
    got = _run(exe, ws)
    assert len(got) == len(ws)
    bad = []
    for i, (w, row) in enumerate(zip(ws, got)):
        ref = extract_one(w, FS, NPERSEG)
        for k, (c_bits, r_val) in enumerate(zip(row, ref)):
            if k in ENTROPY_IDX:
                continue
            if c_bits != _bits(r_val):
                c_val = struct.unpack("<f", struct.pack("<I", c_bits))[0]
                bad.append((i, k, c_val, float(r_val)))
    assert not bad, f"{len(bad)} 处不一致，头几处 (窗口, 特征下标, C, Python)：{bad[:5]}"


def test_熵最多差一个_ulp(exe):
    """logf 属于 libm，各实现不保证正确舍入，所以只要求差 1 ULP。
    放宽的**只有这几个下标**，别的照样逐位——统一放宽会把真 bug 一起放过去。"""
    ws = _windows(25, seed=3)
    got = _run(exe, ws)
    worst = 0
    for w, row in zip(ws, got):
        ref = extract_one(w, FS, NPERSEG)
        for k in ENTROPY_IDX:
            worst = max(worst, abs(row[k] - _bits(ref[k])))
    assert worst <= 1, f"熵差了 {worst} ULP，超出 libm 舍入能解释的范围"


def test_边角分支真的被走到了(exe):
    """守住上面两个测试的有效性。合成数据要真的走到这些分支，
    否则"逐位一致"只证明了常规路径。"""
    ws = _windows(25, seed=3)
    seen_const = seen_plateau = seen_zero_sign = False
    for w in ws:
        for c in range(N_CH):
            col = w[:, c]
            if float(col.std()) <= 1e-8:
                seen_const = True
            d = col - col.mean()
            if np.any(np.sign(d) == 0):
                seen_zero_sign = True
            if np.any(np.diff(col) == 0):
                seen_plateau = True
    assert seen_const, "没造出常数通道，相关系数的除零分支没被走到"
    assert seen_plateau, "没造出平顶，find_peaks 的平台分支没被走到"
    assert seen_zero_sign, "没造出正好落在均值上的样本，np.sign 给 0 的分支没被走到"
