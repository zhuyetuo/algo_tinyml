"""3 轴（5 通道 = acc3 + pitch/roll，79 维）：C ↔ Python 参考实现逐位对照。

8 通道那份在 test_features_c.py。单独一份而不是参数化，是因为 5 通道的
边角用例（常数通道、平顶、落在均值上）要落在不同的列上——第 5 列在这里不存在。

顺带核对维度跟 imu_train 那边的 3 轴特征（11×5 时域 + 8×3 频域 = 79）一致：
频域只对 acc 三轴算，pitch/roll 不算，也没有 SMA/相关系数/模长/jerk 那几组。
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))
sys.path.insert(0, os.path.dirname(__file__))

from tinyml.export_features_c import export as export_cfg  # noqa: E402
from tinyml.features import extract_one, feature_groups, n_features, n_sensor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")

N_T, N_CH, NPERSEG, FS = 32, 5, 32, 16.0
DIM = n_features(N_CH)
ENTROPY_IDX = {11 * N_CH + 8 * k + 3 for k in range(3)}


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = rng.normal(0, 2.0, size=(N_T, N_CH)).astype(np.float32)
        w[:, 2] += np.float32(9.8)
        # pitch/roll 是慢变量：小幅漂移 + 偏置
        w[:, 3] = np.float32(0.3) + np.cumsum(rng.normal(0, 0.02, N_T)).astype(np.float32)
        w[:, 4] = np.float32(-0.8) + np.cumsum(rng.normal(0, 0.02, N_T)).astype(np.float32)
        if i % 5 == 1:
            w[:, 1] = np.float32(0.5)                          # 常数通道
        if i % 5 == 2:
            pat = np.array([0, 1, 2, 2, 2, 1, 0, -1], np.float32)
            w[:, 0] = np.resize(pat, N_T).astype(np.float32)   # 带平顶的峰
        if i % 5 == 3:
            col = np.resize(np.array([1.0, -1.0], np.float32), N_T).copy()
            col[[3, 4, 11, 20]] = np.float32(0.0)
            col -= np.float32(col.mean())
            w[:, 3] = col                                      # 正好落在均值上
        out.append(w)
    return np.stack(out)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("feat5")
    for name, content in export_cfg(N_T, N_CH, NPERSEG, FS).items():
        (d / name).write_text(content, encoding="utf-8")
    binp = d / "feat"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-fno-math-errno",
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
    lines = [" ".join(f"{v:.9g}" for v in w.T.reshape(-1)) for w in ws]
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [[int(v, 16) for v in line.split()] for line in r.stdout.strip().splitlines()]


def test_维度和通道划分():
    assert n_sensor(5) == 3 and n_sensor(8) == 6 and n_sensor(6) == 6
    assert DIM == 79
    assert n_features(6) == 171 and n_features(8) == 193
    g = feature_groups(5)
    assert [x[0] for x in g] == ["acc_x 时域", "acc_y 时域", "acc_z 时域", "pitch 时域", "roll 时域",
                                 "acc_x 频域", "acc_y 频域", "acc_z 频域"]


def test_维度跟imu_train一致():
    """训练那边的 3 轴特征名表就是这 79 个，顺序也一样。imu_train 不在的机器上跳过。"""
    imu = os.path.expanduser(os.environ.get("IMU_TRAIN", "~/imu_train"))
    fp = os.path.join(imu, "src", "ml", "features.py")
    if not os.path.exists(fp):
        pytest.skip("没有 imu_train")
    import importlib.util
    spec = importlib.util.spec_from_file_location("imu_feat", fp)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ImportError as e:   # scipy 之类不在
        pytest.skip(f"imu_train 的依赖不全：{e}")
    names = m.feature_names(5)
    assert len(names) == DIM
    assert names[33:44] == [f"pitch_{f}" for f in m.TIME_FEAT_NAMES]
    assert names[55:63] == [f"acc_x_{f}" for f in m.FREQ_FEAT_NAMES]
    assert m.n_sensor_channels(6) == 6 and m.n_sensor_channels(5) == 3


def test_C维度是79(exe):
    assert len(_run(exe, _windows(1))[0]) == DIM


def test_除熵以外全部逐位相同(exe):
    ws = _windows(25, seed=3)
    got = _run(exe, ws)
    bad = []
    for i, (w, row) in enumerate(zip(ws, got)):
        ref = extract_one(w, FS, NPERSEG)
        assert len(row) == len(ref) == DIM
        for k, (c_bits, r_val) in enumerate(zip(row, ref)):
            if k in ENTROPY_IDX:
                continue
            if c_bits != _bits(r_val):
                c_val = struct.unpack("<f", struct.pack("<I", c_bits))[0]
                bad.append((i, k, c_val, float(r_val)))
    assert not bad, f"{len(bad)} 处不一致，头几处：{bad[:5]}"


def test_熵最多差两个_ulp(exe):
    """熵 = Σ p·log p 十几项累加，每项 logf 各差 1 ULP 时和可以差到 2。
    8 通道那份实测正好 1，这份的第 23 号窗口实测 2——都是 libm 舍入能解释的，
    别的下标仍然逐位。"""
    ws = _windows(25, seed=3)
    got = _run(exe, ws)
    worst = 0
    for w, row in zip(ws, got):
        ref = extract_one(w, FS, NPERSEG)
        for k in ENTROPY_IDX:
            worst = max(worst, abs(row[k] - _bits(ref[k])))
    assert worst <= 2


def test_边角分支真的被走到了():
    ws = _windows(25, seed=3)
    seen_const = seen_plateau = seen_zero_sign = False
    for w in ws:
        for c in range(N_CH):
            col = w[:, c]
            if float(col.std()) <= 1e-8:
                seen_const = True
            if np.any(np.sign(col - col.mean()) == 0):
                seen_zero_sign = True
            if np.any(np.diff(col) == 0):
                seen_plateau = True
    assert seen_const and seen_plateau and seen_zero_sign
