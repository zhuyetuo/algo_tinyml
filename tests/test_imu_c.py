"""IMU 原始采样 → 模型输入通道（core/tm_imu.c）对照训练时的 Python。

参考实现照抄 imu_train/src/gravity_align.py（gravity_align + raw_tilt）——这个仓库
不依赖 imu_train，所以抄一份在这里；那边改了这里要跟着改，这个测试就是提醒。

比三件事：
  1. 出窗时机：第 n_t 个样本出第一个，之后每 hop 个（跟训练滑窗起点 0, hop, 2hop… 一致）
  2. 通道排布：对齐后的 acc[,gyr]，最后两路是对齐**前**算的 pitch/roll，通道在前
  3. 数值：Python 旋转是 float64，C 存 float32，差在末位
"""

import os
import subprocess

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE = os.path.join(ROOT, "core")


def gravity_align(window):
    acc = window[:, :3]
    gyr = window[:, 3:6] if window.shape[1] >= 6 else None
    g_est = acc.mean(axis=0)
    g_norm = np.linalg.norm(g_est)
    if g_norm < 1e-6:
        return window
    g_unit = g_est / g_norm
    ref = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(g_unit, ref), -1.0, 1.0))
    if dot > 0.9999:
        return window
    if dot < -0.9999:
        R = np.diag(np.array([1.0, -1.0, -1.0]))
    else:
        axis = np.cross(g_unit, ref)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    out = window.copy()
    out[:, :3] = (R @ acc.T).T
    if gyr is not None:
        out[:, 3:6] = (R @ gyr.T).T
    return out


def raw_tilt(acc):
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    roll = np.arctan2(ay, az)
    return np.stack([pitch, roll], axis=1).astype(np.float32)


def reference(data, n_t, hop):
    """跟 infer_csv_scratch.infer_file 一样：滑窗 → tilt（对齐前）→ 对齐 → 拼接。返回 [(出窗样本号, [n_ch][n_t])]"""
    out = []
    for st in range(0, len(data) - n_t + 1, hop):
        w = data[st:st + n_t].astype(np.float32)
        x = np.concatenate([gravity_align(w), raw_tilt(w[:, :3])], axis=1).astype(np.float32)
        out.append((st + n_t - 1, x.T))
    return out


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("imu")
    binp = d / "imu"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{CORE}", os.path.join(CORE, "tm_imu.c"), os.path.join(ROOT, "tests", "host_imu.c"),
         "-lm", "-o", str(binp)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def run(exe, data, n_t, hop):
    n_sensor = data.shape[1]
    inp = f"{n_sensor} {n_t} {hop} {len(data)}\n" + "\n".join(" ".join(f"{v:.9g}" for v in row) for row in data)
    r = subprocess.run([str(exe)], input=inp, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = []
    for line in r.stdout.splitlines():
        v = line.split()
        res.append((int(v[0]), np.array(v[1:], np.float32).reshape(n_sensor + 2, n_t)))
    return res


def _imu(n, n_sensor, rng, g=(0.3, -0.2, 0.93)):
    t = np.arange(n) / 16.0
    acc = np.array(g, np.float32) + 0.2 * np.sin(2 * np.pi * 2 * t)[:, None] + 0.05 * rng.standard_normal((n, 3))
    if n_sensor == 3:
        return acc.astype(np.float32)
    gyr = 30 * np.cos(2 * np.pi * 3 * t)[:, None] + 5 * rng.standard_normal((n, 3))
    return np.concatenate([acc, gyr], axis=1).astype(np.float32)


@pytest.mark.parametrize("n_sensor,n_t,hop", [(6, 16, 8), (3, 16, 8), (6, 32, 5), (6, 16, 16)])
def test_stream_matches_training(exe, n_sensor, n_t, hop):
    data = _imu(200, n_sensor, np.random.default_rng(n_sensor * 100 + n_t + hop))
    got, want = run(exe, data, n_t, hop), reference(data, n_t, hop)
    assert [i for i, _ in got] == [i for i, _ in want]
    for (_, a), (_, b) in zip(got, want):
        assert np.allclose(a, b, rtol=1e-5, atol=2e-5 * max(1.0, float(np.abs(b).max()))), np.abs(a - b).max()


@pytest.mark.parametrize("g", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)])
def test_edge_cases(exe, g):
    """不转 / 翻 180° / 没有重力 / 侧躺——Rodrigues 的几个分支都走一遍"""
    rng = np.random.default_rng(1)
    data = np.tile(np.array(list(g) + [1.0, 2.0, 3.0], np.float32), (16, 1))
    data[:, :3] += 1e-4 * rng.standard_normal((16, 3)).astype(np.float32) * (np.linalg.norm(g) > 0)
    (_, a), = run(exe, data, 16, 16)
    (_, b), = reference(data, 16, 16)
    assert np.allclose(a, b, atol=1e-4)
