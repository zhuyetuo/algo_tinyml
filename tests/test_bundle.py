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

import numpy as np
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


@pytest.mark.skipif(not shutil.which("arm-none-eabi-gcc"), reason="没有 arm-none-eabi-gcc")
@pytest.mark.parametrize("kind", ["rf", "cnn"])
def test_lib_bundle_has_no_source(kind, tmp_path):
    """编译库包（保密）：只有 .a + 头文件 + PC 验证工具，一个 .c 都不能有；验证工具能自检。"""
    from footprint import write_lib_bundle

    gen = MODELS[kind]
    meta = json.load(open(os.path.join(gen, "meta.json"), encoding="utf-8"))
    z = str(tmp_path / f"edge_{kind}_lib.zip")
    write_lib_bundle(gen, kind, z, meta, measure(gen, kind, 16, 8, 5, {}), window=16, n_classes=5)
    names = zipfile.ZipFile(z).namelist()
    assert not [n for n in names if n.endswith((".c", "Makefile")) and n != "example/board_selftest.c"], names
    assert "pc/sample_result.csv" in names and "example/board_selftest.c" in names
    for need in ("lib/libtinyml.a", "lib/BUILD_FLAGS.txt", "include/tm_imu.h", "README.md"):
        assert need in names, need
    readme = zipfile.ZipFile(z).read("README.md").decode()
    assert "core/" not in readme and "FFT" not in readme and "@ZIP" not in readme
    if "pc/edge_cli" in names:
        d = tmp_path / "x"
        zipfile.ZipFile(z).extractall(d)
        os.chmod(d / "pc" / "edge_cli", 0o755)
        r = subprocess.run(["./pc/edge_cli", "--selftest"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0 and "17/17" in r.stdout, r.stdout + r.stderr
        # README 里给的正确结果就是这个包实际跑出来的
        assert "selftest: 17/17 通过" in readme and "diff result.csv pc/sample_result.csv" in readme
        r = subprocess.run(["./pc/edge_cli", "pc/sample.csv"], cwd=d, capture_output=True, text=True)
        assert r.stdout == (d / "pc" / "sample_result.csv").read_text(encoding="utf-8")
        # 对方按 README 编板上自检：头文件 + 库就够
        exe = tmp_path / "st"
        flags = (d / "lib" / "BUILD_FLAGS.txt").read_text().split("\n")[0].split()
        defs = [f for f in flags if f.startswith("-D")]
        k = "-DTM_KIND_RF" if kind == "rf" else "-DTM_KIND_CNN"
        r = subprocess.run(["arm-none-eabi-gcc", "-c", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16",
                            "-mfloat-abi=softfp", "-ffp-contract=off", *defs, k, f"-I{d / 'include'}",
                            str(d / "example" / "board_selftest.c"), "-o", str(exe) + ".o"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not shutil.which("arm-none-eabi-gcc"), reason="没有 arm-none-eabi-gcc")
def test_keil_objects(tmp_path):
    """keil/ 下的目标文件：外部只依赖 __aeabi_* 和 memcpy/memset/strcmp（Keil 的 C 库都有），
    wchar_t 属性是「未使用」（不跟 ARMCC 的 2 冲突）；两种调用约定在 Cortex-M4 上板上自检都过。"""
    from footprint import KEIL_ALLOWED_UNDEF, write_lib_bundle

    gen = MODELS["rf"]
    meta = json.load(open(os.path.join(gen, "meta.json"), encoding="utf-8"))
    z = str(tmp_path / "lib.zip")
    write_lib_bundle(gen, "rf", z, meta, measure(gen, "rf", 16, 8, 5, {}), window=16, n_classes=5)
    d = tmp_path / "x"
    zipfile.ZipFile(z).extractall(d)
    readme = (d / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# 端侧模型包") and "Keil" in readme and "tinyml_hard.o" in readme
    assert "0x1." not in "".join(p.read_text(encoding="utf-8") for p in (d / "include").glob("*.h"))
    defs = [f for f in (d / "lib" / "BUILD_FLAGS.txt").read_text().split("\n")[0].split() if f.startswith("-D")]
    q = os.path.join(os.path.dirname(__file__), "pb_qemu")
    (tmp_path / "m.c").write_text("int tm_board_selftest(void); int main(void){return tm_board_selftest();}\n")
    for abi in ("softfp", "hard"):
        o = d / "keil" / f"tinyml_{abi}.o"
        undef = subprocess.run(["arm-none-eabi-nm", "-u", str(o)], capture_output=True, text=True).stdout.split()
        assert [s for s in undef if s != "U" and not KEIL_ALLOWED_UNDEF.match(s)] == []
        attrs = subprocess.run(["arm-none-eabi-readelf", "-A", str(o)], capture_output=True, text=True).stdout
        assert "Tag_ABI_PCS_wchar_t: None" in attrs
        assert ("VFP registers" in attrs) == (abi == "hard")
        if not shutil.which("qemu-system-arm"):
            continue
        elf = tmp_path / f"{abi}.elf"
        r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", f"-mfloat-abi={abi}",
                            "-ffp-contract=off", *defs, "-DTM_KIND_RF", f"-I{d / 'include'}", os.path.join(q, "boot.c"),
                            str(tmp_path / "m.c"), str(d / "example" / "board_selftest.c"), str(o), "-lm",
                            "--specs=rdimon.specs", "-T", os.path.join(q, "m4.ld"), "-o", str(elf)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run(["qemu-system-arm", "-M", "mps2-an386", "-nographic", "-semihosting-config",
                            "enable=on,target=native", "-kernel", str(elf)], capture_output=True, text=True, timeout=300)
        assert "selftest: 17/17 通过" in r.stdout, abi + r.stdout + r.stderr


@pytest.mark.skipif(not shutil.which("arm-none-eabi-gcc"), reason="没有 arm-none-eabi-gcc")
def test_invariant_model_bundle_ignores_mounting(tmp_path):
    """channel_mode=inv 的模型：库带 -DTM_IMU_INVARIANT=1 编；同一段数据换任意佩戴方向，
    包里的 edge_cli 逐窗判决完全一样（端到端的「怎么戴都一样」）。"""
    import csv

    from footprint import write_lib_bundle

    gen = tmp_path / "gen"
    shutil.copytree(MODELS["rf"], gen)
    meta = json.load(open(gen / "meta.json", encoding="utf-8"))
    meta["channel_mode"] = "inv"
    (gen / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    z = str(tmp_path / "lib.zip")
    write_lib_bundle(str(gen), "rf", z, meta, measure(str(gen), "rf", 16, 8, 5, {}), window=16, n_classes=5)
    d = tmp_path / "x"
    zipfile.ZipFile(z).extractall(d)
    assert "-DTM_IMU_INVARIANT=1" in (d / "lib" / "BUILD_FLAGS.txt").read_text()
    os.chmod(d / "pc" / "edge_cli", 0o755)

    rows = list(csv.reader(open(d / "pc" / "sample.csv", encoding="utf-8")))
    data = np.array(rows[1:], float)

    def classes_of(R):
        x = data.copy()
        x[:, 1:4] = data[:, 1:4] @ R.T
        x[:, 4:7] = data[:, 4:7] @ R.T
        p = tmp_path / "r.csv"
        p.write_text(",".join(rows[0]) + "\n" + "\n".join(",".join(f"{v:.6f}" for v in r) for r in x) + "\n")
        out = subprocess.run([str(d / "pc" / "edge_cli"), str(p)], capture_output=True, text=True, check=True).stdout
        return [ln.split(",")[3] for ln in out.splitlines()[1:]]

    base = classes_of(np.eye(3))
    c, s = np.cos(0.7), np.sin(0.7)
    for R in (np.diag([1.0, -1.0, -1.0]), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
              np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])):
        got = classes_of(R)
        same = sum(a == b for a, b in zip(base, got))
        assert same >= len(base) - 1, f"{same}/{len(base)}"   # 只允许输入精度（6 位小数）造成的个别边界窗口
