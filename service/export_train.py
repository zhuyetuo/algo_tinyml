"""把标注平台「训练记录」里训出来的随机森林一键导成端侧模型，并挂到端侧服务上。

平台上点「导出到端侧」→ imu_train 的 label_service 调这个脚本 → 端侧服务 reload。
人不用碰命令行。手工跑也行：

    python service/export_train.py \\
        --model ~/imu_train/results/processed_ds_x__job6_acc3/16hz_remap_ui_job6/rf/ml_rf.pkl \\
        --tag train6 \\
        --processed-dir ~/imu_train/data/processed_ds_x__job6_acc3 \\
        --remap ~/imu_train/configs/remap_ui_job6.yaml

    python service/export_train.py --remove --tag train6      # 撤掉

做的事，按顺序：
  1. 读模型旁边的 ml_rf.json：通道数、窗口、采样率、类别——**全部从训练产出读**，
     不让人再填一遍（export_rf.py 是靠 --channels/--window 手填的，填错不报错）。
  2. 森林 → 紧凑编码 C（7 B/节点 + uint8 叶子）+ 特征配置表，写到
     core/models/generated_<tag>/（gitignore 里的目录，不进仓库）。
  3. 从 imu_train 的留出集拿真实窗口，生成两份 golden（森林 / 整条链）——
     没有 golden 的导出，服务起来时自检是过不了的。
  4. **用编出来的那份 C 在留出集上跑一遍**，算每类 P/R/F1。这才是「端侧 F1」：
     特征是板子那套 float32 实现、叶子量化过，跟 sklearn 那个数不是一回事。
     两者的差距也一并报出来（判决翻了百分之几）。
  5. 写 meta.json（ml_rf.json 的内容 + n_channels + train/edge 两节），
     登记到 edge_models.local.json。端侧服务 reload 之后平台上就能选。

最后一行打 `EXPORT_RESULT {json}`，调用方（label_service）解析这一行。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

LOCAL_JSON = os.path.join(ROOT, "edge_models.local.json")
GEN_ROOT = os.path.join(ROOT, "core", "models")


# ── 纯函数（不依赖 sklearn，测试能覆盖） ──────────────────────────────────


def nperseg_for(window: int) -> int:
    """Welch 段长：不超过窗口、不超过 32 的最大 2 的幂。
    板上现役的 16 点窗口用 16，32 点用 32——跟以前手填的一致。"""
    n = 1
    while n * 2 <= min(int(window), 32):
        n *= 2
    return n


def per_class_report(y, p, classes) -> dict:
    """跟 imu_train 训练脚本写进 ml_rf.json 的 per_class 同一个格式。"""
    y = np.asarray(y)
    p = np.asarray(p)
    out = {}
    f1s = []
    for i, name in enumerate(classes):
        tp = int(np.sum((p == i) & (y == i)))
        fp = int(np.sum((p == i) & (y != i)))
        fn = int(np.sum((p != i) & (y == i)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1-score": round(f1, 4), "support": int(np.sum(y == i))}
        # macro 只算留出集里出现过的类，跟 sklearn classification_report(labels=present) 一致
        if np.any(y == i):
            f1s.append(f1)
    return {"per_class": out,
            "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
            "accuracy": round(float(np.mean(y == p)) if len(y) else 0.0, 4),
            "n_windows": int(len(y))}


def register_local(tag: str, gen: str, meta_path: str, local_json: str = LOCAL_JSON) -> dict:
    """登记/更新 edge_models.local.json 里的一条。同 tag 覆盖。"""
    cfg = _read_local(local_json)
    models = [m for m in cfg.get("models", []) if m.get("tag") != tag]
    models.append({"tag": tag, "kind": "rf", "gen": gen, "meta": meta_path})
    cfg["models"] = models
    _write_local(cfg, local_json)
    return cfg


def unregister_local(tag: str, local_json: str = LOCAL_JSON) -> dict:
    cfg = _read_local(local_json)
    cfg["models"] = [m for m in cfg.get("models", []) if m.get("tag") != tag]
    _write_local(cfg, local_json)
    return cfg


def _read_local(path: str) -> dict:
    if not os.path.exists(path):
        return {"_说明": ["训练记录里「导出到端侧」的模型。export_train.py 自动维护，不进仓库。"],
                "models": []}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f) or {}
    cfg.setdefault("models", [])
    return cfg


def _write_local(cfg: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def gen_dir_for(tag: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag)
    return os.path.join(GEN_ROOT, f"generated_{safe}")


# ── 主流程 ────────────────────────────────────────────────────────────────


def _load_holdout(imu_train: str, processed_dir: str, hz: int, remap: str | None):
    """留出集的真实窗口 [N, T, C] + 标签（remap 之后）。val 空了退到 test，再退到 train。"""
    for p in (os.path.join(imu_train, "src", "data"), os.path.join(imu_train, "src", "ml"),
              os.path.join(imu_train, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dataset import load_all_splits
    from remap_utils import apply_remap, load_remap_yaml

    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _), meta = load_all_splits(hz, processed_dir)
    raw = meta.get("classes")
    classes = list(eval(raw)) if isinstance(raw, str) else list(raw)
    cfg = load_remap_yaml(remap) if remap else None
    for name, X, y in (("val", Xva, yva), ("test", Xte, yte), ("train", Xtr, ytr)):
        if len(y) == 0:
            continue
        if cfg:
            y2, new_classes, keep = apply_remap(y, classes, cfg)
            X = X[keep] if len(X) == len(keep) else X
            y = y2
            cls = new_classes
        else:
            cls = classes
        if name != "val":
            print(f"⚠ 留出集 val 是空的，用 {name} 算端侧 F1（{name}=train 的话这个数偏乐观）")
        return np.asarray(X, np.float32), np.asarray(y, np.int64), list(cls), name
    raise SystemExit(f"{processed_dir} 里一个窗口都没有")


def export(model_pkl: str, tag: str, processed_dir: str, remap: str | None,
           imu_train: str, out: str | None = None, local_json: str = LOCAL_JSON) -> dict:
    try:
        import joblib
    except ImportError:
        raise SystemExit("没装 joblib/sklearn。这个脚本要在训练机上跑。")
    import serve
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_compact_c import export as export_compact, pipeline_golden
    from tinyml.features import extract_one, n_features
    from tinyml.forest import from_sklearn
    from tinyml.forest_compact import CompactForest

    model_pkl = os.path.abspath(os.path.expanduser(model_pkl))
    if not os.path.exists(model_pkl):
        raise SystemExit(f"找不到模型：{model_pkl}")
    meta_json = os.path.splitext(model_pkl)[0] + ".json"
    if not os.path.exists(meta_json):
        raise SystemExit(f"模型旁边没有 {os.path.basename(meta_json)}，通道数/窗口/类别无从读起")
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    classes = list(meta["classes"])
    window = int(meta["window_size"])
    hz = int(meta["hz"])
    n_ch = int(meta.get("n_channels") or 8)
    nps = nperseg_for(window)

    bundle = joblib.load(model_pkl)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    forest = from_sklearn(model, class_names=classes)
    want = n_features(n_ch)
    if forest.n_features != want:
        raise SystemExit(
            f"模型要 {forest.n_features} 维特征，但 {n_ch} 通道的端侧特征是 {want} 维。"
            f"这个模型不是端侧那套特征能对上的（ml_rf.json 里 n_channels={n_ch}）")

    # 留出集：golden + 端侧 F1
    X, y, hold_classes, split = _load_holdout(imu_train, processed_dir, hz, remap)
    if X.shape[1] != window and X.shape[2] == window:
        X = X.transpose(0, 2, 1)
    if X.shape[1:] != (window, n_ch):
        raise SystemExit(f"留出集窗口是 {X.shape[1:]}，模型要 ({window}, {n_ch})")
    if hold_classes != classes:
        # 顺序不同还能对，名字集合不同就没法算 F1
        if set(hold_classes) != set(classes):
            raise SystemExit(f"留出集类别 {hold_classes} 跟模型类别 {classes} 对不上")
        remap_idx = np.array([classes.index(c) for c in hold_classes])
        y = remap_idx[y]

    print(f"模型 {os.path.relpath(model_pkl, os.path.expanduser('~'))}：{n_ch} 通道 × {window} 点 @{hz}Hz，"
          f"{len(classes)} 类，{forest.n_trees} 棵树 {len(forest.node_feature)} 节点；留出集 {len(y)} 窗（{split}）")

    cf = CompactForest(forest)
    size = cf.flash_bytes()
    flash = int(sum(size.values()))
    print(f"紧凑编码 {flash:,} B（{flash / 1024:.1f} KB）" + ("" if flash <= 131072 else "  ⚠ 超过 128KB"))

    # golden：森林从特征进（按类别轮流挑），整条链从窗口进
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    picked, seen = [], {}
    for i in order:
        c = int(y[i])
        if seen.get(c, 0) < 8:
            picked.append(i)
            seen[c] = seen.get(c, 0) + 1
        if len(picked) >= 32:
            break
    gold_w = X[picked]
    gold_f = np.stack([extract_one(w, hz, nps) for w in gold_w])
    files = dict(export_compact(cf, golden_x=gold_f))
    files["tm_forest_c_pipeline_golden.h"] = pipeline_golden(cf, gold_w[:16], hz, nps)
    files.update(export_feat_cfg(window, n_ch, nps, float(hz)))

    out = os.path.abspath(out or gen_dir_for(tag))
    os.makedirs(out, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(content)

    # 用编出来的 C 跑留出集 → 端侧 F1；自检也顺手过一遍
    eng = serve.RfEngine(serve.build_rf(out))
    for name, n, bad in eng.selftest():
        if bad != 0:
            raise SystemExit(f"导出后的自检没过：{name} {bad}（共 {n} 条）")
    pred = np.empty(len(X), np.int64)
    step = 512
    for i in range(0, len(X), step):
        cls_i, _ = eng.infer(np.ascontiguousarray(X[i:i + step].transpose(0, 2, 1)))
        pred[i:i + step] = cls_i
    edge = per_class_report(y, pred, classes)
    edge["flash_bytes"] = flash
    edge["split"] = split
    # 跟 sklearn（scipy 特征、不量化）的判决差多少——差得多的话端侧 F1 掉在哪里就清楚了
    try:
        from features import extract_features as _imu_feats  # imu_train src/ml
        sk_pred = model.predict(_imu_feats(X, hz, show_progress=False))
        edge["agree_with_sklearn"] = round(float(np.mean(sk_pred == pred)), 4)
    except Exception as e:  # noqa: BLE001
        print(f"（跟 sklearn 对判决这步跳过：{e}）")
    print(f"端侧 F1（板上那份 C 在 {split} 上）：macro {edge['macro_f1']}，准确率 {edge['accuracy']}"
          + (f"，跟 sklearn 判决一致 {edge['agree_with_sklearn']:.1%}" if "agree_with_sklearn" in edge else ""))
    for c, m in edge["per_class"].items():
        print(f"  {c:<12} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1-score']:.2f}  (n={m['support']})")

    job_id = None
    if tag.startswith("train") and tag[5:].isdigit():
        job_id = int(tag[5:])
    meta_out = dict(meta)
    meta_out["n_channels"] = n_ch
    meta_out["train"] = {"job_id": job_id, "tag": tag, "axes": max(3, n_ch - 2),
                         "sk_macro_f1": meta.get("macro_f1"), "model_path": model_pkl,
                         "exported_at": int(time.time())}
    meta_out["edge"] = edge
    meta_path = os.path.join(out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    register_local(tag, out, meta_path, local_json)
    result = {"tag": tag, "spec": f"edge:{tag}", "gen": out, "meta": meta_path,
              "n_channels": n_ch, "window": window, "hz": hz, "classes": classes,
              "edge": edge}
    print(f"已登记到 {local_json}；端侧服务 reload 之后平台上就能选「edge:{tag}」")
    return result


def remove(tag: str, local_json: str = LOCAL_JSON) -> dict:
    cfg = unregister_local(tag, local_json)
    d = gen_dir_for(tag)
    removed = False
    if os.path.isdir(d) and os.path.basename(d).startswith("generated_"):
        shutil.rmtree(d, ignore_errors=True)
        removed = True
    return {"tag": tag, "removed_dir": removed, "remaining": [m["tag"] for m in cfg["models"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="端侧模型标签，训练记录用 train<任务号>")
    ap.add_argument("--model", help="imu_train 训出来的 ml_rf.pkl（旁边要有 ml_rf.json）")
    ap.add_argument("--processed-dir", help="这次训练的预处理目录（留出集在里面）")
    ap.add_argument("--remap", default="", help="训练时用的 remap yaml，没有就不传")
    ap.add_argument("--imu-train", default=os.path.expanduser("~/imu_train"))
    ap.add_argument("--out", default="", help="导出目录，默认 core/models/generated_<tag>")
    ap.add_argument("--local", default=LOCAL_JSON, help="登记到哪份清单")
    ap.add_argument("--remove", action="store_true", help="撤掉这个标签（清单 + 导出目录）")
    args = ap.parse_args()

    if args.remove:
        r = remove(args.tag, args.local)
    else:
        if not args.model or not args.processed_dir:
            ap.error("导出要 --model 和 --processed-dir")
        r = export(args.model, args.tag, os.path.expanduser(args.processed_dir),
                   os.path.expanduser(args.remap) or None,
                   os.path.abspath(os.path.expanduser(args.imu_train)),
                   args.out or None, args.local)
    print("EXPORT_RESULT " + json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
