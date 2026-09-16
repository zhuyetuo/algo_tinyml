"""把板上那份 C 编出来，跟 Python 参考实现**逐位**对答案。

这是整个仓库最重要的一个测试。量化 + 定点最典型的失败不是崩溃，是"板上结果跟
训练时不一样，但不报错"——准确率掉几个点，你会去怀疑模型、数据、传感器，
就是不会怀疑某处舍入方向反了。所以要让这件事**编译期就有人盯着**。

对比的是 int8 输出向量本身，不是 argmax：只比 argmax 的话，一个已经算错、
只是恰好还没把类别翻过去的实现能一路混到量产。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml import export, forward_int, make_net, quantize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")

N_CH, N_T, N_CLASSES = 6, 64, 3


def _fake_windows(n, seed):
    """假的 IMU 窗口：一段有节律的抖动 + 噪声 + 重力偏置。

    这里**不是**在模拟真实行为，也不该当成数据来用——它的唯一作用是给量化器一个
    有动态范围的输入，让导出的 scale 不退化。真实校准集见 train_torch.py。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / 25.0  # 25Hz
    xs = []
    for _ in range(n):
        f = rng.uniform(2.0, 8.0)
        amp = rng.uniform(0.2, 3.0)
        w = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        x = np.stack([w * rng.uniform(0.3, 1.0) + rng.normal(0, 0.05, N_T)
                      for _ in range(N_CH)])
        x[2] += 9.8  # 重力压在某一轴上，制造一个明显的直流偏置
        xs.append(x.astype(np.float32))
    return np.stack(xs)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    net = make_net(N_CH, N_CLASSES, seed=1)
    calib = _fake_windows(48, seed=1)
    qnet = quantize(net, calib, class_names=["sleep", "active", "scratch"])

    # golden vector **挑着选**，不是随手取前 16 条：按预测类别轮流取，保证这组样例
    # 至少走到两条不同的判决路径。全是同一类的话，一个"永远返回同一个向量"的实现
    # 也能通过逐位比对，这个测试就空转了。
    cand = _fake_windows(96, seed=2)
    by_cls = {}
    for x in cand:
        xi = qnet.quantize_input(x)
        by_cls.setdefault(int(np.argmax(forward_int(qnet, xi)[0])), []).append(xi)
    picked, i = [], 0
    while len(picked) < 16:
        added = False
        for c in sorted(by_cls):
            if i < len(by_cls[c]):
                picked.append(by_cls[c][i])
                added = True
                if len(picked) == 16:
                    break
        if not added:
            break
        i += 1
    golden_i8 = np.stack(picked)

    out_dir = tmp_path_factory.mktemp("gen")
    for name, content in export(qnet, golden_x_i8=golden_i8).items():
        (out_dir / name).write_text(content, encoding="utf-8")

    exe = out_dir / "host"
    cmd = [
        "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        # 定点代码最容易踩的是移位和溢出这类未定义行为。开 UBSan 的意义在于：
        # 这类错在 x86 上"看起来能跑"，换到 Cortex-M 上才换一种错法暴露出来
        "-fsanitize=undefined", "-fno-sanitize-recover=all",
        f"-I{FW}", f"-I{out_dir}",
        os.path.join(FW, "tm_runtime.c"),
        str(out_dir / "tm_model.c"),
        os.path.join(ROOT, "tests", "host_main.c"),
        "-o", str(exe),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"编译失败：\n{r.stderr}"
    return qnet, golden_i8, exe


def test_c_与_python_参考实现逐位一致(built):
    qnet, golden_i8, exe = built
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, f"跑失败：\n{r.stderr}"
    got = np.array([[int(v) for v in line.split()]
                    for line in r.stdout.strip().splitlines()], dtype=np.int8)
    want = np.stack([forward_int(qnet, x)[0] for x in golden_i8])
    assert got.shape == want.shape
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"{len(bad)} 处不一致，头几处 (样本, 类别): {bad[:5].tolist()}\n"
        f"C={got[bad[0][0]]}  Python={want[bad[0][0]]}"
    )


def test_golden_不是一边倒(built):
    """守一个容易被忽略的退化：如果所有 golden vector 都判成同一类，
    上面那个测试就基本什么都没验——一个永远返回常量的实现也能通过。"""
    qnet, golden_i8, _ = built
    preds = [int(np.argmax(forward_int(qnet, x)[0])) for x in golden_i8]
    assert len(set(preds)) >= 2, f"golden vector 全判成了同一类：{preds}"
