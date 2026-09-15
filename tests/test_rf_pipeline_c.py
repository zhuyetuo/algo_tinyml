"""整条 RF 链的端到端对照：原始窗口 → 193 维特征 → 森林 → 类别 + 概率。

单独测特征过了、单独测森林过了，**不等于接起来就对**。中间那道接缝——特征的
排列顺序跟模型训练时是不是同一个——恰恰最容易错又最不会报错：顺序错了每一维
都对到别的特征上，模型照样给得出一个类别，只是准确率莫名其妙地差。
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml.export_features_c import export as export_cfg  # noqa: E402
from tinyml.export_forest_c import export as export_forest  # noqa: E402
from tinyml.features import extract_one, n_features  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

sys.path.insert(0, os.path.dirname(__file__))
from test_features_c import N_T, N_CH, NPERSEG, FS, _windows  # noqa: E402
from test_forest_c import _make_forest  # noqa: E402

DIM = n_features(N_CH)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    # 森林的特征维度必须跟特征提取出来的一致。这里用 193 维的森林，
    # 对不上的话导出脚本会直接报错——那个检查也是这么来的
    import test_forest_c as tf
    tf.N_FEAT = DIM
    forest = _make_forest(n_trees=15, depth=5, seed=11)
    forest.n_features = DIM
    # 随机树的特征下标是按旧的 N_FEAT 取的，重映射到 193 维上
    internal = forest.node_left != -1
    rng = np.random.default_rng(5)
    forest.node_feature = forest.node_feature.copy()
    forest.node_feature[internal] = rng.integers(0, DIM, internal.sum()).astype(np.int32)

    # 阈值要落在**这些特征实际的取值范围**里。用 N(0,1) 的随机阈值不行——
    # 真实特征的量级差很远（均值带 9.8 的重力偏置、极差十几、峰数是整数），
    # 随机阈值会让所有样本走同一边，整片森林退化成一个常数。那样"逐位一致"
    # 就只验到了一条路径，等于没验。
    calib = np.stack([extract_one(w, FS, NPERSEG) for w in _windows(40, seed=2)])
    lo = calib.min(axis=0)
    hi = calib.max(axis=0)
    thr = forest.node_threshold.copy()
    for i in np.where(internal)[0]:
        f = int(forest.node_feature[i])
        thr[i] = np.float32(rng.uniform(float(lo[f]), float(hi[f]) + 1e-6))
    forest.node_threshold = thr

    d = tmp_path_factory.mktemp("pipe")
    files = dict(export_forest(forest, golden_x=None))
    files.update(export_cfg(N_T, N_CH, NPERSEG, FS))
    for n, content in files.items():
        (d / n).write_text(content, encoding="utf-8")

    exe = d / "pipe"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", f"-I{d}",
         os.path.join(FW, "tm_features.c"), os.path.join(FW, "tm_forest.c"),
         str(d / "tm_feat_cfg.c"), str(d / "tm_forest_model.c"),
         os.path.join(ROOT, "tests", "host_rf_pipeline.c"), "-lm", "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return forest, exe


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def test_整条链逐位一致(built):
    forest, exe = built
    ws = _windows(20, seed=9)
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in w.T.reshape(-1)) for w in ws) + "\n"
    r = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == len(ws)

    bad = []
    preds = []
    for i, (w, line) in enumerate(zip(ws, lines)):
        parts = line.split()
        c_cls, c_bits = int(parts[0]), [int(v, 16) for v in parts[1:]]
        feat = extract_one(w, FS, NPERSEG)
        p = forest.predict_proba(feat)
        preds.append(c_cls)
        if c_cls != int(np.argmax(p)) or c_bits != [_bits(v) for v in p]:
            bad.append((i, c_cls, int(np.argmax(p))))
    assert not bad, f"{len(bad)}/{len(ws)} 条不一致，头一条：{bad[0]}"
    assert len(set(preds)) >= 2, (
        f"整批都判成同一类（{preds[0]}），这个测试就只验到了一条路径")


def test_特征维度对不上时不会悄悄跑过去(built):
    """森林的 n_features 跟特征提取的维度必须一致。不一致时读到的就是越界或者
    错位的特征，而森林照样能给出一个类别——所以要在导出时就拦住。"""
    forest, _ = built
    assert forest.n_features == DIM == 193
