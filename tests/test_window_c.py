"""采样→窗口这一层的 C ↔ Python 比对。

比的是三件事，缺一件都会让"模型没问题但板上准确率低"这种查不动的问题溜过去：
  1. 在**第几个样本**出窗（hop 对不对、攒满之前不出窗）
  2. 窗口里时间是从旧到新排的（环形展平别搞反）
  3. 每个点的量化值逐位相同（舍入方向、饱和）
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.window import Window  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

N_CH, N_T, HOP = 6, 16, 5
SCALE, ZP = 0.05, -7


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("win")
    binp = d / "win"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", os.path.join(FW, "tm_window.c"),
         os.path.join(ROOT, "tests", "host_window.c"), "-lm", "-o", str(binp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    # 幅度故意跨过量化上限：饱和路径也要被走到，不然 clip 写错没人知道
    return rng.normal(0, 3.0, size=(n, N_CH)).astype(np.float64)


def test_出窗时机和内容与_python_一致(exe):
    xs = _samples(97, seed=4)
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in xs) + "\n"
    r = subprocess.run([str(exe), str(SCALE), str(ZP)], input=stdin,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    c_rows = []
    for line in r.stdout.strip().splitlines():
        parts = [int(v) for v in line.split()]
        c_rows.append((parts[0], np.array(parts[1:], np.int8).reshape(N_CH, N_T)))

    w = Window(N_CH, N_T, HOP, SCALE, ZP)
    p_rows = []
    for i, s in enumerate(xs):
        out = w.push(s)
        if out is not None:
            p_rows.append((i, out))

    assert [i for i, _ in c_rows] == [i for i, _ in p_rows], "出窗时机对不上"
    assert c_rows, "一个窗口都没出，这个测试等于没测"
    for (i, a), (_, b) in zip(c_rows, p_rows):
        assert np.array_equal(a, b), f"第 {i} 个样本处出的窗内容不一致"


def test_攒满之前不出窗(exe):
    xs = _samples(N_T - 1, seed=5)
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in xs) + "\n"
    r = subprocess.run([str(exe), str(SCALE), str(ZP)], input=stdin,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", "缓冲还没满就出窗了——那等于拿补零的半截信号去推理"


def test_窗口里时间是从旧到新(exe):
    """用一个单调递增的信号来验方向。搞反了的话模型看到的是倒放的动作，
    而准确率只是"有点低"，不会报错。"""
    n = N_T + HOP
    xs = np.tile(np.arange(n, dtype=np.float64)[:, None] * 0.1, (1, N_CH))
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in xs) + "\n"
    r = subprocess.run([str(exe), str(SCALE), str(ZP)], input=stdin,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    first = np.array([int(v) for v in r.stdout.strip().splitlines()[0].split()][1:],
                     np.int8).reshape(N_CH, N_T)
    assert np.all(np.diff(first[0].astype(int)) > 0), f"时间方向反了：{first[0]}"
