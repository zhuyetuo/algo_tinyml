"""体积曲线：这个模型剪到多小、F1 掉多少——给平台「训练记录」那页点一下出表。

rf：扫 棵数 × 深度 的网格。**不重训**：sklearn 在每个节点都存了类别分布，把树在
深度 d 剪断、该节点当叶子，是精确操作（见 prune_rf.py 顶部）；取前 n 棵也是合法的
（bagging 出来的树独立同分布）。体积按板上的紧凑编码算（7 B/节点 + uint8 叶子），
F1 用板上那套特征（tinyml/features.py，跟 C 逐位一致）在留出集上算——所以这张表
里的数就是导出到板上之后会看到的数，不是 sklearn 的数。

cnn：filters 决定体积，而换 filters 必须重训，所以这里只按结构算 int8 体积
（权重 + 偏置/乘子），给几档预设；F1 只有当前这一档（训练时的）。平台上点
「按这个规格重训」才有别的档的 F1。

最后一行打 `CURVE_RESULT {json}`，调用方（label_service）解析这一行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_train import _load_holdout, per_class_report  # noqa: E402

CNN_PRESETS = ([64, 128, 256], [48, 96, 192], [32, 64, 128], [24, 48, 96], [16, 32, 64], [8, 16, 32])


def cnn_flash_bytes(n_ch: int, window: int, n_classes: int, filters: list[int], k: int = 3) -> dict:
    """按 imu_train cnn.py 的结构算 int8 体积：conv 权重 out×in×k，fc 权重 cls×(ch×T/2^层数)；
    偏置/乘子/shift 每个输出通道 3 个 int32。跟 export_cnn.py 里实测那一段同一个公式。"""
    n_w = n_b = 0
    in_ch, t = n_ch, window
    macs = 0
    for oc in filters:
        n_w += oc * in_ch * k
        n_b += oc
        macs += oc * in_ch * k * t
        in_ch, t = oc, t // 2
    fc_in = in_ch * max(t, 1)
    n_w += n_classes * fc_in
    n_b += n_classes
    macs += n_classes * fc_in
    return {"weights": n_w, "bias_mult_shift": n_b * 3 * 4, "flash_bytes": n_w + n_b * 12, "macs": macs}


def rf_curve(model_pkl: str, processed_dir: str, remap: str | None, imu_train: str,
             trees: list[int], depths: list[int]) -> dict:
    import joblib

    from tinyml.features import extract_one, n_features
    from tinyml.forest import from_sklearn
    from tinyml.forest_compact import CompactForest

    meta_json = os.path.splitext(model_pkl)[0] + ".json"
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    classes = list(meta["classes"])
    window, hz = int(meta["window_size"]), int(meta["hz"])
    n_ch = int(meta.get("n_channels") or 8)
    nps = 1
    while nps * 2 <= min(window, 32):
        nps *= 2
    bundle = joblib.load(model_pkl)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    full = from_sklearn(model, class_names=classes)
    if full.n_features != n_features(n_ch):
        raise SystemExit(f"模型 {full.n_features} 维，端侧 {n_ch} 通道特征 {n_features(n_ch)} 维，对不上")
    sys.setrecursionlimit(100000)

    X, y, hold_classes, split = _load_holdout(imu_train, processed_dir, hz, remap)
    if X.shape[1] != window and X.shape[2] == window:
        X = X.transpose(0, 2, 1)
    if hold_classes != classes:
        if set(hold_classes) != set(classes):
            raise SystemExit(f"留出集类别 {hold_classes} 跟模型类别 {classes} 对不上")
        y = np.array([classes.index(c) for c in hold_classes])[y]
    print(f"留出集 {len(y)} 窗（{split}），算板上那套特征…", flush=True)
    feats = np.stack([extract_one(w, hz, nps) for w in X])

    n_total = len(model.estimators_)
    cur_depth = max(int(e.tree_.max_depth) for e in model.estimators_)
    trees = sorted({t for t in trees if 1 <= t <= n_total} | {n_total})
    depths = sorted({d for d in depths if d >= 1} | {cur_depth})
    rows = []
    for t in trees:
        for d in depths:
            cf = CompactForest(from_sklearn(model, class_names=classes, max_depth=d, n_trees=t))
            pred = np.fromiter((cf.predict(f) for f in feats), dtype=np.int64, count=len(feats))
            rep = per_class_report(y, pred, classes)
            rows.append({"trees": t, "depth": d, "nodes": int(cf.n_nodes),
                         "flash_bytes": int(sum(cf.flash_bytes().values())),
                         "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
                         "per_class": {c: v["f1-score"] for c, v in rep["per_class"].items()},
                         "current": (t == n_total and d == cur_depth)})
            r = rows[-1]
            print(f"  {t:>3} 棵 × 深 {d:>2}：{r['nodes']:>6} 节点 {r['flash_bytes'] / 1024:>7.1f} KB  "
                  f"macro-F1 {r['macro_f1']:.3f}", flush=True)
    return {"kind": "rf", "rows": rows, "classes": classes, "holdout_n": int(len(y)), "split": split,
            "current": {"trees": n_total, "depth": cur_depth}, "n_channels": n_ch, "window": window}


def cnn_curve(model_pt: str) -> dict:
    stem = model_pt.replace("_best.pt", "")
    with open(f"{stem}_best.json", encoding="utf-8") as f:
        meta = json.load(f)
    metrics = {}
    if os.path.exists(f"{stem}.json"):
        with open(f"{stem}.json", encoding="utf-8") as f:
            metrics = json.load(f)
    n_ch, window, classes = int(meta["n_channels"]), int(meta["window_size"]), list(meta["classes"])
    cfg = meta.get("model_cfg") or {}
    cur = [int(x) for x in cfg.get("filters", [64, 128, 256])]
    k = int(cfg.get("kernel_size", 3))
    presets = [list(p) for p in CNN_PRESETS]
    if cur not in presets:
        presets.insert(0, cur)
    rows = []
    for p in presets:
        sz = cnn_flash_bytes(n_ch, window, len(classes), p, k)
        is_cur = p == cur
        rows.append({"filters": p, "flash_bytes": sz["flash_bytes"], "weights": sz["weights"], "macs": sz["macs"],
                     "macro_f1": metrics.get("macro_f1") if is_cur else None,
                     "per_class": {c: v.get("f1-score") for c, v in (metrics.get("per_class") or {}).items()} if is_cur else None,
                     "current": is_cur})
        print(f"  filters {p}：int8 {sz['flash_bytes'] / 1024:>6.1f} KB，{sz['macs']:,} MAC/窗"
              + ("  ← 当前" if is_cur else ""), flush=True)
    return {"kind": "cnn", "rows": rows, "classes": classes, "current": {"filters": cur, "kernel_size": k},
            "n_channels": n_ch, "window": window,
            "note": "换 filters 要重训才有 F1；体积按结构精确算（权重 int8 + 偏置/乘子 int32），不含运行时代码"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ml_rf.pkl 或 dl_cnn_best.pt")
    ap.add_argument("--processed-dir", default="")
    ap.add_argument("--remap", default="")
    ap.add_argument("--imu-train", default=os.path.expanduser("~/imu_train"))
    ap.add_argument("--trees", default="5,10,15,20")
    ap.add_argument("--depths", default="4,6,8,10")
    a = ap.parse_args()
    mp = os.path.abspath(os.path.expanduser(a.model))
    if mp.endswith(".pt"):
        out = cnn_curve(mp)
    else:
        if not a.processed_dir:
            ap.error("rf 要 --processed-dir（留出集）")
        out = rf_curve(mp, os.path.expanduser(a.processed_dir), os.path.expanduser(a.remap) or None,
                       os.path.abspath(os.path.expanduser(a.imu_train)),
                       [int(x) for x in a.trees.split(",") if x], [int(x) for x in a.depths.split(",") if x])
    print("CURVE_RESULT " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
