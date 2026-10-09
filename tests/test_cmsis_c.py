"""可选加速路线（-DTM_USE_CMSIS）：开了开关结果还对不对。

两条标准：
  CNN（CMSIS-NN）  累加完全一样（整数），只有重量化那一步的**舍入平局**不同：gemmlowp（我们和
                    TFLite 参考）负数平局向远离零取整，CMSIS 的 doubling_high_mult 统一 +2^30 即向上。
                    乘子低位多是 0，平局不算罕见——所以标准是：每个输出最多差 1 LSB，差的元素
                    不超过 5%，argmax 一致。差 2 以上或大面积不同，就是布局 / offset 接错了。
  特征（CMSIS-DSP） 不可能逐位一致（基-8 FFT、两路并行累加），要求相对误差 < 1e-5，
                    并且在提交的 RF golden 窗口上判决跟朴素实现一致。
PC 上编的是 CMSIS 的通用 C 实现（没有 M4 的 SIMD），但算法和舍入跟板上同一份代码。
"""

import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml import cmsis, export, forward_int, make_net, quantize  # noqa: E402
from tinyml.export_features_c import export as export_cfg  # noqa: E402
from tinyml.features import extract_one  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")
RF_GEN = os.path.join(ROOT, "core", "models", "edge_rf_d10")

pytestmark = pytest.mark.skipif(not cmsis.available(), reason="third_party/cmsis 不在")


def _gcc(args):
    r = subprocess.run(["gcc", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def _cmsis_flags(kind):
    return [f"-D{k}={v}" for k, v in cmsis.defines(kind, host=True).items()] + \
        [f"-I{i}" for i in cmsis.include_dirs(kind, host=True)] + \
        ["-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]


# ── CNN：CMSIS-NN 逐位 ────────────────────────────────────────────────────────

def _fake_windows(n, n_ch, n_t, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / 25.0
    xs = []
    for _ in range(n):
        w = rng.uniform(0.2, 3.0) * np.sin(2 * np.pi * rng.uniform(2.0, 8.0) * t + rng.uniform(0, 6.28))
        x = np.stack([w * rng.uniform(0.3, 1.0) + rng.normal(0, 0.05, n_t) for _ in range(n_ch)])
        x[2] += 9.8
        xs.append(x.astype(np.float32))
    return np.stack(xs)


@pytest.fixture(scope="module", params=[(6, 64, 3), (8, 64, 5), (5, 32, 4)])
def cnn_built(request, tmp_path_factory):
    n_ch, n_t, n_cls = request.param
    net = make_net(n_ch, n_cls, seed=n_t, n_t=n_t)
    qnet = quantize(net, _fake_windows(48, n_ch, n_t, seed=1))
    golden = np.stack([qnet.quantize_input(x) for x in _fake_windows(24, n_ch, n_t, seed=2)])
    d = tmp_path_factory.mktemp(f"cnn{n_ch}x{n_t}")
    for name, content in export(qnet, golden_x_i8=golden).items():
        (d / name).write_text(content, encoding="utf-8")
    base = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined", "-fno-sanitize-recover=all",
            f"-I{FW}", f"-I{d}", os.path.join(FW, "tm_runtime.c"), str(d / "tm_model.c"),
            os.path.join(ROOT, "tests", "host_main.c")]
    _gcc([*base, "-o", str(d / "plain")])
    # CMSIS 自己的 .c 不用我们的 -Werror 管（-w），我们这几个文件照旧
    _gcc([*base, *_cmsis_flags("cnn"), "-w", *cmsis.sources("cnn"), "-o", str(d / "cmsis")])
    return qnet, golden, d


def _run_cnn(exe):
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.array([[int(v) for v in line.split()] for line in r.stdout.strip().splitlines()], dtype=np.int8)


def _close(got, want):
    diff = np.abs(got.astype(np.int32) - want.astype(np.int32))
    assert diff.max() <= 1, f"差 {diff.max()} LSB——不是舍入平局，是接错了：\nC={got[diff.argmax() // got.shape[1]]}  ref={want[diff.argmax() // got.shape[1]]}"
    frac = float((diff > 0).mean())
    assert frac <= 0.05, f"{frac:.1%} 的元素差 1 LSB，太多了，不像只是平局"
    assert np.array_equal(got.argmax(1), want.argmax(1))
    return frac


def test_cmsis_nn_跟python最多差一个lsb(cnn_built):
    qnet, golden, d = cnn_built
    got = _run_cnn(d / "cmsis")
    want = np.stack([forward_int(qnet, x)[0] for x in golden])
    _close(got, want)


def test_cmsis_nn_跟朴素实现最多差一个lsb(cnn_built):
    _, _, d = cnn_built
    _close(_run_cnn(d / "cmsis"), _run_cnn(d / "plain"))


def test_朴素实现本身逐位等于python(cnn_built):
    """对照组：没开开关的那份必须还是逐位的——改 CMSIS 那条不能把这条带歪。"""
    qnet, golden, d = cnn_built
    want = np.stack([forward_int(qnet, x)[0] for x in golden])
    assert np.array_equal(_run_cnn(d / "plain"), want)


def test_导出同时带两种权重排法(cnn_built):
    qnet, _, d = cnn_built
    src = (d / "tm_model.c").read_text(encoding="utf-8")
    assert src.count("#if TM_CMSIS_NN") == sum(1 for l in qnet.layers if hasattr(l, "w"))
    h = (d / "tm_model.h").read_text(encoding="utf-8")
    vals = [int(v) for v in __import__("re").findall(r"#define TM_ARENA_BYTES (\d+)", h)]
    assert len(vals) == 2 and vals[0] > vals[1], "CMSIS 那条的 arena 要多出 im2col 缓冲"


# ── 特征：CMSIS-DSP 容差 ─────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=[(8, 16, 16), (6, 64, 32), (5, 32, 32)])
def feat_built(request, tmp_path_factory):
    n_ch, n_t, nps = request.param
    d = tmp_path_factory.mktemp(f"feat{n_ch}x{n_t}")
    for name, content in export_cfg(n_t, n_ch, nps, 16.0).items():
        (d / name).write_text(content, encoding="utf-8")
    base = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-fno-math-errno",
            f"-DTM_FEAT_MAX_T={n_t}", f"-DTM_FEAT_MAX_NPERSEG={nps}",
            f"-I{FW}", f"-I{d}", os.path.join(FW, "tm_features.c"), str(d / "tm_feat_cfg.c"),
            os.path.join(ROOT, "tests", "host_features.c")]
    _gcc([*base, "-lm", "-o", str(d / "plain")])
    _gcc([*base, *_cmsis_flags("rf"), "-w", *cmsis.sources("rf"), "-lm", "-o", str(d / "cmsis")])
    rng = np.random.default_rng(7)
    ws = rng.normal(0, 2.0, size=(24, n_t, n_ch)).astype(np.float32)
    ws[:, :, 2] += np.float32(9.8)
    ws[3, :, 1] = np.float32(0.5)  # 常数通道
    return d, ws, n_ch, n_t, nps


def _run_feat(exe, ws):
    inp = "\n".join(" ".join(repr(float(v)) for v in w.T.reshape(-1)) for w in ws) + "\n"
    r = subprocess.run([str(exe)], input=inp, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.array([[struct.unpack("<f", struct.pack("<I", int(h, 16)))[0] for h in line.split()]
                     for line in r.stdout.strip().splitlines()], np.float32)


def test_cmsis_dsp_特征误差在十万分之一内(feat_built):
    d, ws, n_ch, n_t, nps = feat_built
    a = _run_feat(d / "plain", ws)
    b = _run_feat(d / "cmsis", ws)
    assert a.shape == b.shape and np.isfinite(b).all()
    rel = np.abs(a - b) / (np.abs(a) + 1e-3)
    assert rel.max() < 1e-5, f"最大相对误差 {rel.max():.2e}，位置 {np.unravel_index(rel.argmax(), rel.shape)}"


def test_cmsis_dsp_跟python参考一致(feat_built):
    d, ws, n_ch, n_t, nps = feat_built
    b = _run_feat(d / "cmsis", ws)
    py = np.stack([extract_one(w, 16.0, nps) for w in ws]).astype(np.float32)
    rel = np.abs(py - b) / (np.abs(py) + 1e-3)
    assert rel.max() < 1e-5


def test_cmsis_dsp_提交的rf_golden判决一致():
    """整条 RF 链在提交的导出上：朴素 vs CMSIS 的票数 argmax 一致（票数本身允许极少数差 1）。"""
    import re
    hdr = open(os.path.join(RF_GEN, "tm_forest_c_pipeline_golden.h"), encoding="utf-8").read()
    n = int(re.search(r"TM_FCP_GOLDEN_N (\d+)", hdr).group(1))
    vals = re.search(r"pipeline_in\[\] = \{([^}]*)\}", hdr).group(1).split(",")
    x = np.array([float.fromhex(v.strip().rstrip("f")) for v in vals], np.float32).reshape(n, -1)
    votes = np.array([int(v) for v in re.search(r"pipeline_votes\[\] = \{([^}]*)\}", hdr).group(1).split(",")]).reshape(n, -1)
    exe = {}
    for name, extra in (("plain", []), ("cmsis", [*_cmsis_flags("rf"), "-w", *cmsis.sources("rf")])):
        out = os.path.join(RF_GEN, "..", f"_t_{name}")
        _gcc(["-std=c99", "-O2", "-ffp-contract=off", "-fno-math-errno", "-DTM_FEAT_MAX_T=16", "-DTM_FEAT_MAX_NPERSEG=16",
              f"-I{FW}", f"-I{RF_GEN}", *extra, os.path.join(FW, "tm_features.c"), os.path.join(FW, "tm_forest_c.c"),
              os.path.join(RF_GEN, "tm_feat_cfg.c"), os.path.join(RF_GEN, "tm_forest_c_model.c"),
              os.path.join(ROOT, "tests", "host_rf_pipeline_votes.c"), "-lm", "-o", out])
        exe[name] = out
    try:
        inp = "\n".join(" ".join(repr(float(v)) for v in row) for row in x) + "\n"
        res = {}
        for name, path in exe.items():
            r = subprocess.run([path], input=inp, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            res[name] = np.array([[int(v) for v in line.split()] for line in r.stdout.strip().splitlines()])
    finally:
        for path in exe.values():
            if os.path.exists(path):
                os.remove(path)
    assert np.array_equal(res["plain"], votes), "朴素实现跟 golden 票数不一致——这不是 CMSIS 的问题"
    assert np.array_equal(res["plain"].argmax(1), res["cmsis"].argmax(1))
    assert np.abs(res["plain"] - res["cmsis"]).max() <= 1


# ── 交叉编译 + 打包 ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not shutil.which("arm-none-eabi-gcc"), reason="没有交叉编译器")
def test_footprint_报告cmsis变体并打出两个静态库(cnn_built):
    import json
    import tempfile
    import zipfile

    from footprint import build_static_lib, measure, write_bundle, bundle_readme
    fp = measure(RF_GEN, "rf", 16, 8, 5, {"trees": 20, "depth": 10, "nodes": 11810})
    a = fp["accel"]
    assert a and a["name"] == "CMSIS-DSP" and not a["errors"] and a["flash"]["linked"]
    assert 0 < a["flash"]["delta"] < 8 * 1024, a["flash"]
    meta = json.load(open(os.path.join(RF_GEN, "meta.json"), encoding="utf-8"))
    z = tempfile.mktemp(suffix=".zip")
    write_bundle(RF_GEN, "rf", z, bundle_readme("rf", meta, fp), window=16, n_classes=5)
    names = zipfile.ZipFile(z).namelist()
    assert "lib/libtinyml.a" in names and "lib/libtinyml_cmsis.a" in names
    assert "third_party/cmsis/dsp/Source/tm_cmsis_tables.c" in names and "third_party/cmsis/core/Include/cmsis_gcc.h" in names
    assert "CMSIS" in zipfile.ZipFile(z).read("README.md").decode()

    _, _, d = cnn_built
    with tempfile.TemporaryDirectory() as tmp:
        lib = build_static_lib(str(d), "cnn", 16, 3, tmp, use_cmsis=True)
        assert lib and "path" in lib, lib
    fp = measure(str(d), "cnn", 16, 6, 3, {"macs": 1, "arena_bytes": 960})
    a = fp["accel"]
    assert a and a["name"] == "CMSIS-NN" and not a["errors"] and a["ram"]["scratch"] > 0 and a["flash"]["delta"] > 0
