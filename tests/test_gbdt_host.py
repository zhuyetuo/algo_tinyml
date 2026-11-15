"""GBDT（XGBoost）整条路线在 PC 上：导出 C → 编 .so（serve.build_gbdt）→ 两份 golden 自检 →
跟 Python 参考的判决逐条一致。窗口用 80 点（20Hz × 4s），顺带验 TM_FEAT_MAX_T 自动抬上去。"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

xgb = pytest.importorskip("xgboost")

import serve  # noqa: E402
from tinyml.export_features_c import export as export_feat_cfg  # noqa: E402
from tinyml.export_gbdt_compact_c import export, pipeline_golden  # noqa: E402
from tinyml.features import extract_one, n_features  # noqa: E402
from tinyml.gbdt import from_xgboost  # noqa: E402
from tinyml.gbdt_compact import CompactBooster  # noqa: E402

N_T, N_CH, HZ = 80, 8, 20.0


def _windows(rng, n):
    t = np.arange(N_T) / HZ
    ws = []
    for i in range(n):
        f = [0.5, 3.0, 6.0][i % 3]
        w = rng.normal(0, 0.02, (N_T, N_CH)).astype(np.float32)
        w[:, 0] += 1.0 + 0.3 * (i % 3) * np.sin(2 * np.pi * f * t)
        w[:, 2] += 1.0
        ws.append(w)
    return np.stack(ws), np.arange(n) % 3


def test_gbdt_导出_编译_自检_判决一致(tmp_path):
    rng = np.random.default_rng(0)
    nps = 32
    W, y = _windows(rng, 90)
    F = np.stack([extract_one(w, HZ, nps) for w in W])
    assert F.shape[1] == n_features(N_CH)
    m = xgb.XGBClassifier(n_estimators=6, max_depth=4, learning_rate=0.3, tree_method="hist", n_jobs=1)
    m.fit(F, y)
    cb = CompactBooster(from_xgboost(m, class_names=["a", "b", "c"]))
    files = dict(export(cb, golden_x=F[:12]))
    files["tm_gbdt_c_pipeline_golden.h"] = pipeline_golden(cb, W[:9], HZ, nps)
    files.update(export_feat_cfg(N_T, N_CH, nps, HZ))
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    eng = serve.RfEngine(serve.build_gbdt(str(tmp_path)))
    assert eng.n_classes == 3 and eng.n_t == N_T and eng.classes == ["a", "b", "c"]
    res = eng.selftest()
    assert [n for _, n, _ in res] == [12, 9]
    assert all(bad == 0 for _, _, bad in res), res
    cls, proba = eng.infer(np.ascontiguousarray(W.transpose(0, 2, 1)))
    want = np.array([cb.predict(f) for f in F])
    assert (cls == want).mean() >= 0.98          # 特征 C/Python 末位差可能让个别贴着阈值的翻
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-5)
