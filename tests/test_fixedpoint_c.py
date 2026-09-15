"""定点原语的 C ↔ Python 扫描比对。

这个测试存在的直接原因：把 tm_runtime.c 里负数的舍入 nudge 故意改错之后，
整网的 golden vector **全部照过**——那一位偏差在后面的右移里被吃掉了。
端到端只能证明"这组输入下两边一样"，证不了算子本身对。算子必须单独扫，
而且要扫到负数、扫到边界。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.fixedpoint import (  # noqa: E402
    INT32_MAX, INT32_MIN,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")


def _cases():
    """(x, multiplier, shift) 的扫描集。

    shift > 0 时 C 里要先左移，x 大了会溢出 int32（未定义行为）——真实模型里
    shift 恒为负（重量化总是在缩小），所以正 shift 只配小 x 扫，不去制造一个
    实际跑不到、却会让测试变红的情况。
    """
    xs_small = [0, 1, -1, 2, -2, 3, -3, 5, -5, 127, -128, 1000, -1000]
    xs_big = [1 << 15, -(1 << 15), 1 << 20, -(1 << 20), 1 << 24, -(1 << 24),
              (1 << 30), -(1 << 30), INT32_MAX, INT32_MIN + 1, INT32_MIN]
    mults = [0, 1, -1, INT32_MAX, INT32_MIN]
    mults += [quantize_multiplier(r)[0] for r in (1e-6, 1e-4, 0.01, 0.3, 0.5, 0.75, 0.99)]
    out = []
    for m in mults:
        for s in (0, -1, -8, -15, -31):
            for x in xs_small + xs_big:
                out.append((x, m, s))
        for s in (1, 4):
            for x in xs_small:
                out.append((x, m, s))
    return out


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("fp")
    binp = d / "fp"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", os.path.join(FW, "tm_runtime.c"),
         os.path.join(ROOT, "tests", "host_fixedpoint.c"), "-o", str(binp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def test_定点原语_c_与_python_完全一致(exe):
    cases = _cases()
    stdin = "\n".join(f"{x} {m} {s}" for x, m, s in cases) + "\n"
    r = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == len(cases)

    bad = []
    for (x, m, s), line in zip(cases, lines):
        c_hi, c_pot, c_mbqm = (int(v) for v in line.split())
        p_hi = int(saturating_rounding_doubling_high_mul(np.int64(x), np.int64(m)))
        p_pot = int(rounding_divide_by_pot(np.int64(x), abs(s)))
        p_mbqm = int(multiply_by_quantized_multiplier(np.int64(x), m, s))
        if (c_hi, c_pot, c_mbqm) != (p_hi, p_pot, p_mbqm):
            bad.append((x, m, s, (c_hi, c_pot, c_mbqm), (p_hi, p_pot, p_mbqm)))
    assert not bad, f"{len(bad)}/{len(cases)} 处不一致，头几处：{bad[:3]}"


def test_扫描集确实覆盖到负数乘积(exe):
    """守住上面那个测试的有效性：如果扫描集里 a*b 全是非负的，
    负数那条舍入分支就没被走到——而它正是最容易写错的地方。"""
    neg = sum(1 for x, m, _ in _cases() if x * m < 0)
    assert neg > 100, f"负乘积只有 {neg} 例，扫描集覆盖不够"
