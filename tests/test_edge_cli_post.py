"""edge_cli --post：板上后处理（tm_post，调度库同一份）合成的片段，对照服务端稳定版 v2
（label_service/postprocess.stabilize(algo="viterbi")）。

走的是交给嵌入式的那条路：源码包 → make → ./edge_cli data.csv --post seg.csv。
服务端那边吃 edge_cli 自己打出来的逐窗口概率——两边只差后处理，比的就是后处理。
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "service"))
sys.path.insert(0, os.path.dirname(__file__))

from footprint import bundle_readme, measure, sample_csv, write_bundle  # noqa: E402

GEN = os.path.join(ROOT, "core", "models", "edge_rf_d10")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if not shutil.which("make"):
        pytest.skip("没有 make")
    d = tmp_path_factory.mktemp("post")
    meta = json.load(open(os.path.join(GEN, "meta.json"), encoding="utf-8"))
    z = str(d / "src.zip")
    write_bundle(GEN, "rf", z, bundle_readme("rf", meta, measure(GEN, "rf", 16, 8, 5, {})), window=16, n_classes=5)
    zipfile.ZipFile(z).extractall(d / "x")
    r = subprocess.run(["make"], cwd=d / "x", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = []
    for seed in range(1, 6):   # 合成数据 5 段接起来，状态切换多一些
        rows += sample_csv(16, 6, seconds=120, seed=seed).splitlines()[1:]
    hdr = sample_csv(16, 6, seconds=1).splitlines()[0]
    data = d / "data.csv"
    data.write_text(hdr + "\n" + "\n".join(f"{i / 16:.4f}," + r.split(",", 1)[1] for i, r in enumerate(rows)) + "\n")
    r = subprocess.run(["./edge_cli", str(data), "--out", "win.csv", "--post", "seg.csv"], cwd=d / "x",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "行为片段" in r.stderr
    win = list(csv.DictReader(open(d / "x" / "win.csv", encoding="utf-8")))
    seg = list(csv.DictReader(open(d / "x" / "seg.csv", encoding="utf-8")))
    return win, seg, meta


def test_segments_cover_timeline(run):
    win, seg, _ = run
    assert len(seg) > 3
    assert [s["segment"] for s in seg] == [str(i) for i in range(len(seg))]
    for a, b in zip(seg, seg[1:]):
        assert float(a["end_s"]) <= float(b["start_s"]) + 1e-6, "片段不重叠"
    assert float(seg[0]["start_s"]) == 0.0
    assert abs(float(seg[-1]["end_s"]) - float(win[-1]["end_s"])) < 1.01


def test_matches_server_stable_v2(run):
    """跟服务端离线 viterbi 逐段一致（类别 + 起止）"""
    from test_post_c import _load_postprocess

    win, seg, meta = run
    classes = list(meta["classes"])
    import datetime as dt
    t0 = dt.datetime(2026, 9, 15, 10, 0, 0)
    windows = [{"ts": (t0 + dt.timedelta(seconds=float(w["start_s"]))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "probs": {c: float(w[f"p_{c}"]) for c in classes}, "label": w["class"], "spec": None} for w in win]
    _, post = _load_postprocess()
    params = post.StableParams(event_labels=("抓挠", "甩身体"), spectral_min=0.0)
    srv = post.stabilize(windows, classes, classes, 1.0, 0.5, "majority", params, algo="viterbi")
    want = sorted((round(_secs(s["start_ts"], t0), 3), round(_secs(s["end_ts"], t0), 3), lab)
                  for lab, ss in srv.items() for s in ss)
    got = sorted((round(float(s["start_s"]), 3), round(float(s["end_s"]), 3), s["class"]) for s in seg)
    assert got == want


def _secs(ts, t0):
    import datetime as dt
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        ts = dt.datetime.fromisoformat(ts)
    return (ts - t0).total_seconds()
