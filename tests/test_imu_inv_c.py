"""方向无关通道（-DTM_IMU_INVARIANT=1）：core/tm_imu.c 对照训练时的 Python。

参考实现照抄 imu_train/src/data/gravity_align.py 的 invariant_channels（这个仓库不依赖
imu_train，那边改了这里要跟着改）。比三件事：
  1. 数值跟训练一致（Python float64、C 存 float32，差在末位）；
  2. 设备任意固定旋转（倒装、绕重力轴转、侧倾），C 输出不变——这是做它的目的；
  3. 出窗时机跟原来一样（流式接口不变）。
"""

import os
import subprocess

import numpy as np
import pytest

from test_imu_c import _imu, run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE = os.path.join(ROOT, "core")


def invariant_channels(window):
    acc = window[:, :3].astype(np.float64)
    g = acc.mean(axis=0)
    gn = np.linalg.norm(g)
    u = g / gn if gn >= 1e-6 else np.array([0.0, 0.0, 1.0])

    def split(x):
        vert = x @ u
        mag = np.sqrt(np.sum(x * x, axis=1))
        hor = np.sqrt(np.maximum(mag * mag - vert * vert, 0.0))
        return vert, hor, mag

    v, h, m = split(acc)
    cols = [v, h, m]
    if window.shape[1] >= 6:
        cols += list(split(window[:, 3:6].astype(np.float64)))
    jerk = np.zeros(len(acc))
    jerk[1:] = np.sqrt(np.sum(np.diff(acc, axis=0) ** 2, axis=1))
    cols += [np.arctan2(h, v), jerk]
    return np.stack(cols, axis=1).astype(np.float32)


def reference(data, n_t, hop):
    return [(st + n_t - 1, invariant_channels(data[st:st + n_t].astype(np.float32)).T)
            for st in range(0, len(data) - n_t + 1, hop)]


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    binp = tmp_path_factory.mktemp("imu_inv") / "imu"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-DTM_IMU_INVARIANT=1",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{CORE}", os.path.join(CORE, "tm_imu.c"), os.path.join(ROOT, "tests", "host_imu.c"),
         "-lm", "-o", str(binp)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def _rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("n_sensor,n_t,hop", [(3, 32, 16), (6, 16, 8), (3, 16, 8)])
def test_matches_training(exe, n_sensor, n_t, hop):
    data = _imu(200, n_sensor, np.random.default_rng(n_sensor + n_t))
    got, want = run(exe, data, n_t, hop), reference(data, n_t, hop)
    assert [i for i, _ in got] == [i for i, _ in want]
    for (_, a), (_, b) in zip(got, want):
        assert np.allclose(a, b, rtol=1e-5, atol=2e-5 * max(1.0, float(np.abs(b).max()))), np.abs(a - b).max()


@pytest.mark.parametrize("n_sensor", [3, 6])
def test_any_mounting_gives_same_input(exe, n_sensor):
    rng = np.random.default_rng(7)
    data = _imu(160, n_sensor, rng)
    base = run(exe, data, 32, 16)
    R_list = [np.diag([1.0, -1.0, -1.0]),                                   # 倒装
              np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]  # 绕重力轴转 90°
    R_list += [_rot(rng) for _ in range(4)]                                 # 任意方向
    for R in R_list:
        d = data.copy()
        d[:, :3] = data[:, :3] @ R.T
        if n_sensor == 6:
            d[:, 3:6] = data[:, 3:6] @ R.T
        for (_, a), (_, b) in zip(run(exe, d.astype(np.float32), 32, 16), base):
            assert np.allclose(a, b, atol=2e-4 * max(1.0, float(np.abs(b).max()))), np.abs(a - b).max()


def test_still_and_upside_down(exe):
    """静止：竖直 = |a|，水平 = 0，偏角 = 0，jerk = 0——不管正着还是倒着"""
    for g in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.2, 0.0, -0.98)):
        data = np.tile(np.array(g, np.float32), (16, 1))
        (_, a), = run(exe, data, 16, 16)
        n = float(np.linalg.norm(np.array(g, np.float32)))
        assert np.allclose(a[0], n, atol=1e-5) and np.allclose(a[1], 0, atol=2e-3)
        assert np.allclose(a[3], 0, atol=2e-3) and np.allclose(a[4], 0)
