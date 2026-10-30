"""按接口协议交付的 pet_behavior 库：pet_behavior.h + libpetbehavior.a + example.c + 验收数据。

用一个 25 Hz 三轴的小森林（合成数据训的，只为走通整条链）打包，然后：
包里一个 .c 源码都没有（example.c 除外）、PC 上验收通过、ARM 上能链上、
头文件版本不一致 / 参数错 / 时间倒退各返回协议里的错误码。
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

from footprint import pb_type_of, write_pb_bundle  # noqa: E402

pytestmark = pytest.mark.skipif(not (shutil.which("arm-none-eabi-gcc") and shutil.which("gcc")),
                                reason="要 arm-none-eabi-gcc 和 gcc")

HZ, N_T, HOP, N_CH = 25, 25, 12, 5
CLASSES = ["睡觉", "活动", "抓挠", "未佩戴", "甩身体"]


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        c = i % len(CLASSES)
        sd = {0: 0.01, 1: 0.15, 2: 0.5, 3: 0.0, 4: 1.0}[c]
        w = rng.normal(0, sd, size=(N_T, N_CH)).astype(np.float32)
        w[:, 2] += np.float32(1.0)
        w[:, 3] = np.float32(0.1) + rng.normal(0, sd / 5, N_T).astype(np.float32)
        w[:, 4] = np.float32(-0.1) + rng.normal(0, sd / 5, N_T).astype(np.float32)
        X.append(w)
        y.append(c)
    return np.stack(X), np.array(y)


@pytest.fixture(scope="module")
def gen25(tmp_path_factory):
    from sklearn.ensemble import RandomForestClassifier

    from export_train import nperseg_for
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_compact_c import export as export_compact, pipeline_golden
    from tinyml.features import extract_one
    from tinyml.forest import from_sklearn
    from tinyml.forest_compact import CompactForest

    nps = nperseg_for(N_T)
    X, y = _windows(300)
    F = np.stack([extract_one(w, HZ, nps) for w in X])
    model = RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0).fit(F, y)
    cf = CompactForest(from_sklearn(model, class_names=CLASSES))
    files = dict(export_compact(cf, golden_x=F[:16]))
    files["tm_forest_c_pipeline_golden.h"] = pipeline_golden(cf, X[:16], HZ, nps)
    files.update(export_feat_cfg(N_T, N_CH, nps, float(HZ)))
    d = tmp_path_factory.mktemp("gen25")
    for name, content in files.items():
        (d / name).write_text(content, encoding="utf-8")
    meta = {"hz": HZ, "window_size": N_T, "stride": HOP, "n_channels": N_CH, "classes": CLASSES,
            "train": {"tag": "train321", "kind": "rf"}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return str(d), meta


@pytest.fixture(scope="module")
def pkg(gen25, tmp_path_factory):
    gen, meta = gen25
    d = tmp_path_factory.mktemp("pb")
    z = str(d / "pb.zip")
    write_pb_bundle(gen, "rf", z, meta, {})
    zipfile.ZipFile(z).extractall(d / "x")
    os.chmod(d / "x" / "pc" / "pb_example", 0o755)
    return d / "x", zipfile.ZipFile(z).namelist()


def test_type_mapping():
    assert [pb_type_of(c) for c in CLASSES] == [0, 6, 4, 7, 6]
    assert pb_type_of("饮水") == 2 and pb_type_of("舔大腿") == 3 and pb_type_of("吃饭") == 1


def test_package_contents(pkg):
    d, names = pkg
    assert sorted(names) == sorted(["pet_behavior.h", "libpetbehavior.a", "example.c", "test_input.bin",
                                    "test_expect.bin", "pc/libpetbehavior_host.a", "pc/pb_example", "README.md"])
    h = (d / "pet_behavior.h").read_text(encoding="utf-8")
    assert "@" not in h and "PB_WORK_SIZE" in h and "PB_SAMPLE_RATE_HZ   25" in h
    readme = (d / "README.md").read_text(encoding="utf-8")
    assert "25 Hz" in readme and "FFT" not in readme and "森林" not in readme
    assert (d / "test_expect.bin").stat().st_size % 12 == 0 and (d / "test_expect.bin").stat().st_size > 0


def test_pc_acceptance(pkg):
    d, _ = pkg
    r = subprocess.run(["./pc/pb_example", "test_input.bin", "test_expect.bin"], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0 and "集成验收：通过" in r.stdout, r.stdout + r.stderr
    assert "pb_version = 0x00010141" in r.stdout   # v1 / train321


def test_example_builds_from_package(pkg, tmp_path):
    """对方按 README 拿 example.c + 头文件 + host 库自己编，也要过。"""
    d, _ = pkg
    exe = tmp_path / "ex"
    subprocess.run(["gcc", "-DPB_EXAMPLE_MAIN", f"-I{d}", str(d / "example.c"), str(d / "pc" / "libpetbehavior_host.a"),
                    "-o", str(exe), "-lm"], check=True)
    r = subprocess.run([str(exe), "test_input.bin", "test_expect.bin"], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_arm_links(pkg, tmp_path):
    d, _ = pkg
    r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=softfp",
                        f"-I{d}", str(d / "example.c"), str(d / "libpetbehavior.a"), "-lm", "--specs=nosys.specs",
                        "-Wl,--gc-sections", "-Wl,-e,pb_acceptance", "-nostartfiles", "-o", str(tmp_path / "a.elf")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_error_codes(pkg, tmp_path):
    d, _ = pkg
    src = tmp_path / "err.c"
    src.write_text(r'''
#include <stdio.h>
#include "pet_behavior.h"
static uint8_t mem[PB_WORK_SIZE] __attribute__((aligned(4)));
int main(void) {
    pb_config_t c = {PB_SAMPLE_RATE_HZ, 8, 0};
    pb_segment_t s[8];
    pb_sample_t x[50] = {{0}};
    for (int i = 0; i < 50; i++) x[i].z = 4096;
    printf("%d ", pb_push(x, 50, 0, s, 8));                       /* -2 未初始化 */
    printf("%d ", pb_init_v(&c, mem, sizeof mem, PB_HEADER_VERSION + 1));  /* -5 */
    pb_config_t bad = {16, 8, 0};
    printf("%d ", pb_init(&bad, mem, sizeof mem));                 /* -1 采样率 */
    pb_config_t bad2 = {PB_SAMPLE_RATE_HZ, 3, 0};
    printf("%d ", pb_init(&bad2, mem, sizeof mem));                /* -1 量程 */
    printf("%d ", pb_init(&c, mem, 16));                           /* -1 内存不够 */
    printf("%d ", pb_init(&c, mem, sizeof mem));                   /* 0 */
    uint8_t t, q;
    printf("%d ", pb_current(&t, &q));                             /* 1 还没数据 */
    printf("%d ", pb_push(x, 50, 1000, s, 8) >= 0);                /* 1 */
    printf("%d ", pb_current(&t, &q));                             /* 0 */
    printf("%d ", pb_push(x, 50, 1000, s, 8));                     /* -3 倒退 */
    printf("%d ", pb_push(x, 50, 3000, s, 8) >= 0);                /* 1 连续 */
    printf("%d ", pb_flush(s, 8) >= 0);                            /* 1 */
    printf("%d\n", pb_selftest());                                 /* 0 */
    return 0;
}
''')
    exe = tmp_path / "err"
    subprocess.run(["gcc", f"-I{d}", str(src), str(d / "pc" / "libpetbehavior_host.a"), "-o", str(exe), "-lm"], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.split()
    assert out == ["-2", "-5", "-1", "-1", "-1", "0", "1", "1", "0", "-3", "1", "1", "0"], out


def test_batch_size_does_not_change_output(pkg, tmp_path):
    """每批多少个样本、out 给多大，都不影响输出（片段只取决于数据和时间）。"""
    d, _ = pkg
    data = (d / "test_input.bin").read_bytes()
    alt = data[:8] + (7).to_bytes(2, "little") + data[10:]
    (tmp_path / "in7.bin").write_bytes(alt)
    r = subprocess.run([str(d / "pc" / "pb_example"), str(tmp_path / "in7.bin"), str(d / "test_expect.bin")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_rejects_six_axis_model(tmp_path):
    gen = os.path.join(ROOT, "core", "models", "edge_rf_d10")
    meta = json.load(open(os.path.join(gen, "meta.json"), encoding="utf-8"))
    with pytest.raises(ValueError, match="三轴"):
        write_pb_bundle(gen, "rf", str(tmp_path / "x.zip"), meta, {})


@pytest.mark.skipif(not shutil.which("qemu-system-arm"), reason="没有 qemu-system-arm")
def test_cortex_m4_matches_pc(pkg, tmp_path):
    """板上（Cortex-M4 + newlib 的 libm）的输出跟 PC 生成的 test_expect.bin 逐字节一致。
    qemu 模拟 mps2-an386，semihosting 读文件。"""
    d, _ = pkg
    q = os.path.join(os.path.dirname(__file__), "pb_qemu")
    elf = tmp_path / "m4.elf"
    r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=softfp",
                        f"-I{d}", os.path.join(q, "boot.c"), os.path.join(q, "main.c"), str(d / "example.c"),
                        str(d / "libpetbehavior.a"), "-lm", "--specs=rdimon.specs", "-T", os.path.join(q, "m4.ld"),
                        "-o", str(elf)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["qemu-system-arm", "-M", "mps2-an386", "-nographic", "-semihosting-config",
                        "enable=on,target=native", "-kernel", str(elf)], cwd=d, capture_output=True, text=True, timeout=300)
    assert "ARM acceptance=0" in r.stdout, r.stdout + r.stderr
