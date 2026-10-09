"""平台「下载端侧包」那个 zip 拿到 Linux 上能不能直接用：解压 → make test。

嵌入式拿到包的第一件事就是这个。包里漏一个头文件（之前就漏过 tm_accel.h）、
Makefile 选错源文件、golden 对不上，在这里都会红，而不是等对方来问「还少哪些文件」。
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "service"))

from footprint import bundle_readme, measure, write_bundle  # noqa: E402

MODELS = {"rf": os.path.join(ROOT, "core", "models", "edge_rf_d10"),
          "cnn": os.path.join(ROOT, "core", "models", "edge_cnn_i8")}


@pytest.mark.skipif(not shutil.which("make"), reason="没有 make")
@pytest.mark.parametrize("kind", ["rf", "cnn"])
def test_bundle_builds_and_selftests_on_linux(kind, tmp_path):
    gen = MODELS[kind]
    meta = json.load(open(os.path.join(gen, "meta.json"), encoding="utf-8"))
    fp = measure(gen, kind, 16, 8, 5, {})
    z = str(tmp_path / f"edge_{kind}.zip")
    write_bundle(gen, kind, z, bundle_readme(kind, meta, fp), window=16, n_classes=5)
    names = zipfile.ZipFile(z).namelist()
    for need in ("Makefile", "README.md", "pc/edge_cli.c", "pc/tm_edge_cfg.h", "pc/sample.csv",
                 "core/tm_imu.c", "core/tm_imu.h", "core/tm_accel.h"):
        assert need in names, need
    readme = zipfile.ZipFile(z).read("README.md").decode()
    assert "@" not in readme.replace("@ 16 Hz", "").replace("@16", ""), "README 里有没填的占位符"
    assert "make test" in readme and "pitch" in readme

    d = tmp_path / "x"
    zipfile.ZipFile(z).extractall(d)
    r = subprocess.run(["make", "test"], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest: 17/17 通过" in r.stdout
    rows = open(d / "build" / "sample_result.csv", encoding="utf-8").read().splitlines()
    assert rows[0].startswith("window,start_s,end_s,class_id,class,conf")
    assert len(rows) - 1 == (60 * 16 - 16) // 8 + 1

    # 原始计数 + 32Hz 也能喂：整数倍跟平台一样隔点抽，换算回 g 后结果跟直接喂 16Hz 的同一份数据一样
    lines = open(d / "pc" / "sample.csv", encoding="utf-8").read().splitlines()
    raw = [lines[0]]
    for ln in lines[1:]:
        v = ln.split(",")
        acc = [str(round(float(x) / (16 / 32768))) for x in v[1:4]]
        gyr = [str(round(float(x) / (2000 / 32768))) for x in v[4:7]]
        raw += [",".join([v[0]] + acc + gyr)] * 2   # 每个样本重复两次 = 32 Hz
    (d / "raw.csv").write_text("\n".join(raw) + "\n")
    r = subprocess.run(["./edge_cli", "raw.csv", "--in-hz", "32", "--acc-scale", str(16 / 32768),
                        "--gyr-scale", str(2000 / 32768)], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0 and "不像以 g 为单位" not in r.stderr, r.stderr
    got = [ln.split(",")[4] for ln in r.stdout.splitlines()[1:]]
    want = [ln.split(",")[4] for ln in rows[1:]]
    assert sum(a == b for a, b in zip(got, want)) >= 0.95 * len(want)

    # 50Hz（平台样本的采样率）：跟平台的 scipy resample_poly 逐窗口一致
    _check_50hz_matches_scipy(d)

    # 没换算的原始计数要报警
    r = subprocess.run(["./edge_cli", "raw.csv", "--in-hz", "32"], cwd=d, capture_output=True, text=True)
    assert "不像以 g 为单位" in r.stderr


def _check_50hz_matches_scipy(d):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    n_sensor = int(open(d / "pc" / "tm_edge_cfg.h").read().split("TM_EDGE_N_SENSOR")[1].split()[0])
    rng = np.random.default_rng(0)
    n = 50 * 90
    t = np.arange(n) / 50
    acc = np.stack([0.2 + 0.4 * np.sin(2 * np.pi * (0.5 + t / 60) * t),
                    -0.1 + 0.3 * np.cos(2 * np.pi * 3 * t * (t > 45)),
                    0.95 + 0.2 * np.sin(2 * np.pi * 1.3 * t)], 1) + 0.05 * rng.standard_normal((n, 3))
    gyr = 40 * np.sin(2 * np.pi * np.array([0.7, 2.1, 4.3]) * t[:, None]) + 5 * rng.standard_normal((n, 3))
    x = np.concatenate([acc, gyr], 1)[:, :n_sensor].astype(np.float32)
    x = np.array([[float(f"{v:.7g}") for v in row] for row in x], np.float32)  # 跟 CSV 里写的一样
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"][:n_sensor]
    (d / "in50.csv").write_text(",".join(cols) + "\n" + "\n".join(",".join(f"{v:.7g}" for v in r) for r in x) + "\n")
    y = signal.resample_poly(x, 8, 25, axis=0).astype(np.float32)   # 平台 downsample() 的那一步
    (d / "ref16.csv").write_text(",".join(cols) + "\n" + "\n".join(",".join(f"{v:.9g}" for v in r) for r in y) + "\n")
    a = subprocess.run(["./edge_cli", "in50.csv", "--in-hz", "50"], cwd=d, capture_output=True, text=True)
    b = subprocess.run(["./edge_cli", "ref16.csv"], cwd=d, capture_output=True, text=True)
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    ra, rb = a.stdout.splitlines()[1:], b.stdout.splitlines()[1:]
    assert len(ra) == len(rb) > 100
    assert [r.split(",")[3] for r in ra] == [r.split(",")[3] for r in rb]
    pa = np.array([[float(v) for v in r.split(",")[5:]] for r in ra])
    pb = np.array([[float(v) for v in r.split(",")[5:]] for r in rb])
    assert np.abs(pa - pb).max() < 1e-3
