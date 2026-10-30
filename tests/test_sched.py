"""调度库（pb_hal_t + pb_on_event）：PC 上用假 IMU / 假 Flash / 假 BLE 模拟（tests/sched_sim.c），
跑断连、补发、确认、断电重启、校时、FIFO 溢出、Flash 写满这些场景。

模型用一个 16 Hz、32 点、步长 16、4 类的三轴小森林（合成数据训的，跟 train8 同规格），
只为走通整条链，不看识别效果。
"""

import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "service"))

from footprint import build_sched, sample_csv  # noqa: E402

pytestmark = pytest.mark.skipif(not (shutil.which("arm-none-eabi-gcc") and shutil.which("gcc")),
                                reason="要 arm-none-eabi-gcc 和 gcc")

HZ, N_T, HOP, N_CH = 16, 32, 16, 5
CLASSES = ["睡觉", "未佩戴", "活动", "抓挠"]
UNIT = 4096


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        c = i % len(CLASSES)
        sd = {0: 0.01, 1: 0.0, 2: 0.15, 3: 0.6}[c]
        w = rng.normal(0, sd, size=(N_T, N_CH)).astype(np.float32)
        w[:, 2] += np.float32(1.0)
        w[:, 3] = np.float32(0.1) + rng.normal(0, sd / 5, N_T).astype(np.float32)
        w[:, 4] = np.float32(-0.1) + rng.normal(0, sd / 5, N_T).astype(np.float32)
        X.append(w)
        y.append(c)
    return np.stack(X), np.array(y)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from sklearn.ensemble import RandomForestClassifier

    from export_train import nperseg_for
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_compact_c import export as export_compact, pipeline_golden
    from tinyml.features import extract_one
    from tinyml.forest import from_sklearn
    from tinyml.forest_compact import CompactForest

    nps = nperseg_for(N_T)
    X, y = _windows(400)
    F = np.stack([extract_one(w, HZ, nps) for w in X])
    model = RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0).fit(F, y)
    cf = CompactForest(from_sklearn(model, class_names=CLASSES))
    files = dict(export_compact(cf, golden_x=F[:16]))
    files["tm_forest_c_pipeline_golden.h"] = pipeline_golden(cf, X[:16], HZ, nps)
    files.update(export_feat_cfg(N_T, N_CH, nps, float(HZ)))
    d = tmp_path_factory.mktemp("sched")
    gen = d / "gen"
    gen.mkdir()
    for name, content in files.items():
        (gen / name).write_text(content, encoding="utf-8")
    meta = {"hz": HZ, "window_size": N_T, "stride": HOP, "n_channels": N_CH, "classes": CLASSES,
            "train": {"tag": "train8", "kind": "rf"}}
    (gen / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    out = build_sched(str(gen), "rf", meta, str(d / "build"))

    exe = d / "sched_sim"
    r = subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", f"-I{out['inc']}", os.path.join(ROOT, "tests", "sched_sim.c"),
                        out["lib_host"], "-lm", "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # IMU：合成数据（静止 → 走动 → 抖动）来几轮，int16 原始计数（±8g）
    rows = []
    for seed in range(1, 9):
        rows += sample_csv(HZ, 3, seconds=90, seed=seed).splitlines()[1:]
    raw = b"".join(struct.pack("<hhh", *[max(-32768, min(32767, round(float(v) / (8 / 32768))))
                                          for v in ln.split(",")[1:4]]) for ln in rows)
    (d / "imu.bin").write_bytes(raw)
    return {"exe": str(exe), "imu": str(d / "imu.bin"), "out": out, "seconds": len(rows) / HZ}


def run(sim, script, flash=8 * UNIT, mtu=244, fifo=128, unit=UNIT):
    r = subprocess.run([sim["exe"], sim["imu"], str(flash), str(unit), str(mtu), str(fifo)],
                       input=script, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr
    lines = r.stdout.splitlines()
    assert not [ln for ln in lines if ln.startswith(("BAD_", "TOO_LONG", "NO_REWAKE"))], lines
    segs = [tuple(int(v) for v in ln.split()[1:]) for ln in lines if ln.startswith("SEG ")]
    stats = [dict(kv.split("=") for kv in ln.split()[1:]) for ln in lines if ln.startswith("STATS")]
    cur = [ln for ln in lines if ln.startswith("CUR ")]
    return lines, segs, [{k: int(v) for k, v in s.items()} for s in stats], cur


def test_package(sim):
    o = sim["out"]
    h = open(o["header"], encoding="utf-8").read()
    assert "@" not in h and f"PB_WORK_SIZE        {o['work']}u" in h
    assert set(o["keil"]) == {"softfp", "hard"} and o["flash"] and o["flash"] > 0


def test_runs_offline_then_uploads_everything_once(sim):
    """没连手机时一直在算、存 Flash；连上后全部补发，确认后 unacked 归零；不重不漏"""
    lines, segs, stats, cur = run(sim, "run 300\nstats\nack on\nconnect\nrun 60\nstats\n")
    assert lines[0] == "IMU 16 8" and "INIT 0" in lines
    s0, s1 = stats
    assert s0["windows"] > 250 and s0["segments"] > 0 and s0["unacked"] == s0["segments"]
    seqs = [s[0] for s in segs]
    assert seqs == sorted(set(seqs)), "seq 递增且不重复"
    assert s1["unacked"] == 0 and len(segs) == s1["segments"]
    assert cur, "连着的时候每个窗口推一条当前行为"
    # 片段首尾相接、时长 > 0
    assert all(s[2] > 0 for s in segs) and all(s[3] < len(CLASSES) and s[4] <= 100 for s in segs)


def test_no_ack_resends_after_reconnect(sim):
    _, segs, stats, _ = run(sim, "run 200\nconnect\nrun 1\ndisconnect\nrun 100\nack on\nconnect\nrun 1\nstats\n")
    first = {s[0] for s in segs}
    assert len(segs) > len(first), "第一次没确认，重连后重发"
    assert stats[-1]["unacked"] == 0


def test_reboot_keeps_unacked_and_flags_old_times(sim):
    """断电重启：没确认的片段还在、照样补发；上次开机、没校时的片段时间标「未知」"""
    _, segs, stats, _ = run(sim, "run 200\nstats\nreboot\nrun 60\nack on\nconnect\nrun 1\nstats\n")
    before = stats[0]["segments"]
    assert before > 0
    assert len(segs) >= before
    old = [s for s in segs if s[5] & 2]
    assert len(old) == before, "上次开机未校时的片段：时间未知"
    assert stats[-1]["unacked"] == 0


def test_time_sync(sim):
    """校时后存的片段带 Unix 秒；校时前同一次开机的，发送时换算成 Unix 秒"""
    T = 1_730_000_000
    _, segs, _, _ = run(sim, f"run 120\nsync {T}\nrun 120\nack on\nconnect\nrun 1\n")
    assert segs and all(s[5] & 1 for s in segs)
    starts = [s[1] for s in segs]
    assert T - 130 <= min(starts) and max(starts) <= T + 130
    assert starts == sorted(starts)


def test_unsynced_age(sim):
    """没校时：start 是「距发送时多少秒」"""
    _, segs, _, _ = run(sim, "run 200\nack on\nconnect\nrun 1\n")
    assert segs and all(s[5] == 0 for s in segs)
    assert all(0 <= s[1] <= 210 for s in segs)


def test_fifo_overflow_is_a_gap(sim):
    """取数停了 30 秒（FIFO 只装得下 8 秒）：记一次断档；正常 5 秒一取、偶尔晚一点都不算"""
    _, _, stats, _ = run(sim, "run 60\nstall 30\nrun 60\nstats\n")
    assert stats[0]["gaps"] == 1
    _, _, stats, _ = run(sim, "run 60\nstall 6\nrun 60\nstats\n")
    assert stats[0]["gaps"] == 0


def test_flash_wraps_and_counts_lost(sim):
    """Flash 只有 2 块、一直不确认：写满后覆盖最旧的、记丢失；之后照样能存能发"""
    _, segs, stats, _ = run(sim, f"run {int(sim['seconds']) - 5}\nstats\nack on\nconnect\nrun 1\nstats\n",
                            flash=2 * 64, unit=64)
    s0, s1 = stats
    assert s0["lost"] > 0 and s0["unacked"] + s0["lost"] <= s0["segments"]
    assert s1["unacked"] == 0 and len(segs) == s0["unacked"]


def test_small_mtu(sim):
    """MTU 只有 23 字节：一包一个片段，照样发完"""
    _, segs, stats, _ = run(sim, "run 200\nack on\nconnect\nrun 1\nstats\n", mtu=23)
    assert segs and stats[-1]["unacked"] == 0


def test_in_lib_download(sim, tmp_path):
    """「编译库」下载包里带 sched/（三轴模型）；example_hal.c 照 README 能编、能跟库链上（ARM）"""
    import zipfile

    from footprint import write_lib_bundle

    gen = os.path.dirname(sim["out"]["inc"]).replace("build", "gen")
    meta = json.load(open(os.path.join(gen, "meta.json"), encoding="utf-8"))
    from footprint import measure
    z = str(tmp_path / "lib.zip")
    write_lib_bundle(gen, "rf", z, meta, measure(gen, "rf", N_T, N_CH, len(CLASSES), {}), window=N_T, n_classes=len(CLASSES))
    names = zipfile.ZipFile(z).namelist()
    for need in ("sched/pet_behavior.h", "sched/libpetbehavior.a", "sched/keil/petbehavior_hard.o",
                 "sched/keil/petbehavior_softfp.o", "sched/example_hal.c", "sched/README.md"):
        assert need in names, need
    assert not [n for n in names if n.endswith(".c") and n not in ("sched/example_hal.c", "example/board_selftest.c")]
    d = tmp_path / "x"
    zipfile.ZipFile(z).extractall(d)
    readme = (d / "sched" / "README.md").read_text(encoding="utf-8")
    assert "@" not in readme and "0x0A" in readme and "睡觉" in readme
    (tmp_path / "main.c").write_text("int app_pb_start(void); int main(void){return app_pb_start();}\n")
    for lib, abi in ((d / "sched" / "libpetbehavior.a", "softfp"), (d / "sched" / "keil" / "petbehavior_hard.o", "hard")):
        r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", f"-mfloat-abi={abi}",
                            "-std=c99", "-Wall", "-Werror", f"-I{d / 'sched'}", str(d / "sched" / "example_hal.c"),
                            str(tmp_path / "main.c"), str(lib), "-lm", "--specs=nosys.specs", "-Wl,--gc-sections",
                            "-o", str(tmp_path / f"{abi}.elf")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
