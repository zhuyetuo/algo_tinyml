"""端侧推理服务：对外说 algo_service 那套契约，内里跑的是固件那份 C。

标注平台已经会调 algo_service 的 `/api/v1/label/infer`，而 **algo_service 是线上
服务，不能动**。所以端侧模型从这里单独起一个服务，平台加个 client 指过来就行，
不用改线上那套。

推理链路：
    CSV → imu_train 的预处理 → 固件的 C（tm_prep + tm_invoke）→ 片段聚合

预处理和片段聚合**整段复用 imu_train 的 `infer_csv_scratch.infer_file()`**，
一行都没重写。理由不是省事：那条链里有一处顺序是致命的——

    tilt = append_raw_tilt_batch(X)[:, :, 6:8]   # 必须在重力对齐**之前**算
    X_aligned = gravity_align(X)

反过来的话，重力对齐会把每个窗口的平均倾角归零，绝对姿态（躺着/坐着/站着）
整个消失。不报错，只是效果差一截。自己抄一遍，迟早在这种地方跟训练侧分家，
而分家的表现是"平台上看着对、板上不对"。

**这个服务只做端上真会做的事**：
  · mode 只支持 raw。algo_service 的 stable/viterbi 端上没有，
    假装支持会让对比表里两列看着可比、实际不是一回事。
  · 片段聚合用的 min_windows / max_gap 就是固件 tinyml_task 里那两个参数。

用法：
    python python/edge_service.py \\
        --gen firmware/generated_cnn_a \\
        --meta ~/imu_train/results_edge_a/.../dl_cnn_best.json \\
        --imu-train ~/imu_train --host 0.0.0.0 --port 8900
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve  # noqa: E402
from tinyml.edge_model import EdgeCNN, EdgeRF  # noqa: E402
from tinyml.torch_import import load_meta  # noqa: E402


def add_imu_train(repo):
    """把 imu_train 的模块路径挂上。

    它自己的模块之间是平铺 import 的（`from gravity_align import ...`），
    所以 src 和 src/data 都要进 sys.path，只加仓库根目录是不够的。
    """
    repo = os.path.abspath(os.path.expanduser(repo))
    need = os.path.join(repo, "src", "infer_csv_scratch.py")
    if not os.path.exists(need):
        sys.exit(f"{repo} 看起来不是 imu_train 仓库（没有 {need}）。\n"
                 "  用 --imu-train 指对路径。")
    for p in (os.path.join(repo, "src"), os.path.join(repo, "src", "data")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


class EdgeRunner:
    """一个端侧模型 + 一套推理参数。"""

    def __init__(self, tag, engine, meta, imu_train, resample="poly", kind="cnn"):
        self.tag = tag
        self.engine = engine
        self.meta = meta
        self.kind = kind
        # 两条路线的包装不同，但对 infer_file 来说都是一个有 predict_proba
        # 的对象——这正是复用整条预处理链的前提
        self.model = EdgeCNN(engine, meta["classes"]) if kind == "cnn" \
            else EdgeRF(engine, meta["classes"])
        self.classes = list(meta["classes"])
        self.window_size = int(meta["window_size"])
        self.model_hz = int(meta["hz"])
        self.stride = int(meta.get("stride") or max(self.window_size // 2, 1))
        self.gravity_aligned = bool(meta.get("gravity_aligned", True))
        self.label_mode = str(meta.get("label_mode") or "majority")
        self.resample = resample
        # infer_file 里有全局状态（打印、递归），而且我们这边 C 有静态缓冲，
        # 并发进来会互相踩。端上本来就是单线程，这里串行没有损失
        self.lock = threading.Lock()

    def infer(self, csv_path, device_hz, min_windows, max_gap, targets):
        from infer_csv_scratch import infer_file, load_csv

        # missing_seconds 要自己算：infer_file 不返回它，但平台要这个字段
        # （用来判断"这段数据是不是因为蓝牙断联而不可信"）。
        # 少了它平台会当成 0——"数据完好"，而那是最不该默认的方向。
        acc, _, _, _, null_ratio = load_csv(csv_path)
        missing_seconds = float(len(acc)) / max(device_hz, 1) * float(null_ratio)

        with self.lock, tempfile.TemporaryDirectory() as out_dir:
            res = infer_file(
                csv_path, self.model, self.classes,
                window_size=self.window_size, stride=self.stride,
                device_hz=device_hz, model_hz=self.model_hz,
                gravity_aligned=self.gravity_aligned,
                quiet=True, scratch_only=True,
                output_dir=out_dir, min_windows=min_windows,
                keep_isolated=(min_windows <= 1),
                label_mode=self.label_mode, resample_method=self.resample,
                target_labels=targets, is_dl=True,
            )
            if res is None:
                # 一个窗口都没有（整段缺失、或者文件比窗口还短）。
                # 返回空结果而不是报错——平台那边"这个样本没片段"是正常情况
                return {"n_windows": 0, "segments": {t: [] for t in targets},
                        "missing_seconds": missing_seconds}

            segments, n_windows = {}, 0
            for label in targets:
                hits = glob.glob(os.path.join(out_dir, label, "_infer", "*_infer.json"))
                if not hits:
                    segments[label] = []
                    continue
                with open(hits[0], encoding="utf-8") as f:
                    d = json.load(f)
                # 字段名沿用 imu_train 的历史叫法（scratch_segments），
                # 那边注释里写明了不改名是为了不牵动一串下游脚本
                segments[label] = d.get("scratch_segments") or []
                n_windows = int(d.get("n_windows") or 0)

        return {"n_windows": n_windows, "segments": segments,
                "missing_seconds": missing_seconds}


class Handler(BaseHTTPRequestHandler):
    runners = {}
    default_tag = ""
    nas_root = ""

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _resolve(self, path):
        """把平台给的 NAS 相对路径解析成本地路径。

        平台跟 algo_service 共享同一份 NAS 挂载，传的是相对路径。
        这里必须防住 `../` 穿出挂载点——服务是内网的，但一个能读任意文件的
        HTTP 接口不该因为"内网"就放过去。
        """
        p = os.path.normpath(os.path.join(self.nas_root, path.lstrip("/")))
        root = os.path.abspath(self.nas_root)
        if not os.path.abspath(p).startswith(root + os.sep) and os.path.abspath(p) != root:
            raise ValueError(f"路径穿出了 NAS 根目录：{path}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 {p}（NAS 根 ={root}）")
        return p

    def _runner(self, tag):
        tag = tag or self.default_tag
        if tag not in self.runners:
            raise ValueError(f"没有这个端侧模型：{tag}。"
                             f"现有：{', '.join(self.runners) or '（一个都没有）'}")
        return self.runners[tag]

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/health", "/api/v1/label/health"):
            return self._send({"ok": True, "models": [
                {"tag": t, "classes": r.classes, "window": r.window_size,
                 "hz": r.model_hz, "stride": r.stride}
                for t, r in self.runners.items()]})
        self._send({"error": "no such path"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._send({"error": f"请求体不是合法 JSON：{e}"}, 400)

        if u.path == "/api/v1/label/infer":
            return self._send(self._one(body))
        if u.path == "/api/v1/label/infer_batch":
            items = body.get("items") or []
            out = []
            for it in items:
                merged = dict(body)
                merged.pop("items", None)
                merged.update(it)
                try:
                    out.append({"sample_id": it.get("sample_id"),
                                "path": it.get("path"),
                                "ok": True, "error": None,
                                "result": self._one(merged, raise_on_error=True)})
                except Exception as e:  # noqa: BLE001
                    # **单个失败不能整批失败**——一个坏 CSV 让一整天的样本
                    # 全部白跑，那是最没必要的一种损失
                    out.append({"sample_id": it.get("sample_id"),
                                "path": it.get("path"), "ok": False,
                                "error": f"{type(e).__name__}: {e}", "result": None})
            return self._send(out)
        self._send({"error": "no such path"}, 404)

    def _one(self, body, raise_on_error=False):
        try:
            mode = str(body.get("mode") or "raw")
            if mode != "raw":
                # **不假装支持**。stable/viterbi 是 algo_service 的后处理，
                # 端上没有。返回一个"支持了"的结果，会让对比表里两列看着可比、
                # 实际不是一回事——那比报错糟得多
                raise ValueError(
                    f"端侧模型只有 raw，没有 {mode}。"
                    "stable/viterbi 是 algo_service 的后处理，板子上不存在——"
                    "在这里假装支持会让模型对比变成拿两个不同的东西相比。")
            r = self._runner(body.get("model"))
            path = self._resolve(str(body["path"]))
            device_hz = float(body.get("device_hz") or r.model_hz)
            targets = body.get("labels") or ["抓挠"]
            res = r.infer(path, device_hz,
                          int(body.get("min_windows") or 1),
                          int(body.get("max_gap") or 2),
                          list(targets))
            res.update({
                # model_path 平台用来算 model_tag（取文件名去后缀），
                # 所以这里给一个能一眼看出是端侧、且带模型标识的假路径
                "model_path": f"edge://{r.tag}.edge",
                "mode": mode,
                "candidates": [],
            })
            return res
        except Exception as e:  # noqa: BLE001
            if raise_on_error:
                raise
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", required=True, metavar="TAG=DIR",
                    help="端侧模型：标签=导出目录。可以给多次。"
                         "标签里含 rf 的走 RF 路线（tm_features+tm_forest），"
                         "否则走 CNN（tm_prep+tm_invoke）")
    ap.add_argument("--meta", action="append", required=True, metavar="TAG=JSON",
                    help="端侧模型：标签=imu_train 的元数据 json。可以给多次。"
                         "CNN 用 dl_*.json（带 ch_mean/ch_std），"
                         "RF 用 ml_*.json（不需要归一化，没那两项）")
    ap.add_argument("--imu-train", default=os.path.expanduser("~/imu_train"))
    ap.add_argument("--nas-root", default="/", help="平台传的相对路径相对于哪里")
    ap.add_argument("--resample", default="poly", choices=["poly", "training_match"],
                    help="device_hz != model_hz 时的重采样算法。"
                         "两种实测差 6~8%%，要跟训练数据用同一套就选 training_match")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--strict-port", action="store_true")
    args = ap.parse_args()

    add_imu_train(args.imu_train)

    def kv(items, what):
        out = {}
        for s in items:
            if "=" not in s:
                sys.exit(f"--{what} 要写成 标签=值 的形式，给的是：{s}")
            k, v = s.split("=", 1)
            out[k] = v
        return out

    gens, metas = kv(args.gen, "gen"), kv(args.meta, "meta")
    if set(gens) != set(metas):
        sys.exit(f"--gen 和 --meta 的标签对不上：{sorted(gens)} vs {sorted(metas)}")

    for tag in sorted(gens):
        # 按标签选路线。写死"含 rf 就是 RF"看着土，但比自动探测导出目录里
        # 有什么文件可靠——两条都导过的目录会让自动探测选错，而选错不报错
        kind = "rf" if "rf" in tag.lower() else "cnn"
        # 两条路线的元数据字段不同：CNN 要 ch_mean/ch_std（tm_prep 用），
        # RF 不做归一化所以没有那两项。按路线读，别用同一套必填项
        meta = load_meta(os.path.expanduser(metas[tag]), kind=kind)
        if kind == "rf":
            eng = serve.RfEngine(serve.build_rf(gens[tag]))
            print(f"  {tag:<16} [RF] {eng.n_ch}×{eng.n_t}，{eng.n_classes} 类，"
                  f"{eng.n_features} 维特征（在 C 里算）")
            fatal = False
            for name, n, bad in eng.selftest():
                if bad == -2:
                    # **没有 golden 不算通过。** 导出时忘了给 --features/--windows
                    # 就是这个结果，而"0 条全部通过"是这类自检最经典的失效方式
                    print(f"    {name:<18} ⚠ 没导 golden vector，验不了。"
                          f"重新 export_rf.py 时带上 --features / --windows")
                elif bad == -1:
                    print(f"    {name:<18} ✗ 推理直接失败了")
                    fatal = True
                elif bad == 0:
                    print(f"    {name:<18} ✓ {n} 条逐位一致")
                else:
                    print(f"    {name:<18} ✗ {bad} 个值对不上（共 {n} 条）")
                    fatal = True
            if fatal:
                sys.exit(
                    "RF 的 golden vector 自检没过。**先别怀疑模型**，按这个顺序查：\n"
                    "  ①编译选项漏了 -ffp-contract=off，或者别处塞了 -ffast-math；\n"
                    "  ②导出的 tm_forest_model.c 跟验过的不是同一份；\n"
                    "  ③导出时的窗口长度/通道数跟训练时不一致。")
        else:
            eng = serve.Engine(serve.build(gens[tag]))
            bad = eng.selftest()
            flag = "✓ 逐位一致" if bad == 0 else f"✗ {bad} 字节对不上"
            print(f"  {tag:<16} [CNN] {eng.n_ch}×{eng.n_t}，{eng.n_classes} 类，"
                  f"golden {eng.golden_n} 条 {flag}")
            if bad:
                sys.exit("golden vector 自检没过，导出和运行时不配套，"
                         "不要用这个服务的结果。")
        Handler.runners[tag] = EdgeRunner(tag, eng, meta, args.imu_train,
                                          args.resample, kind=kind)

    Handler.default_tag = sorted(gens)[0]
    Handler.nas_root = os.path.abspath(os.path.expanduser(args.nas_root))

    srv = serve.listen(args.host, args.port, Handler,
                       tries=1 if args.strict_port else 20)
    port = srv.server_address[1]
    if port != args.port:
        print(f"\n{args.port} 被占了，换到 {port}")
    print(f"\nNAS 根：{Handler.nas_root}")
    print(f"默认模型：{Handler.default_tag}")
    print("\n开着了，Ctrl-C 停：")
    for u in serve.urls(args.host, port, "api/v1/label/infer"):
        print(f"  {u}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停了")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
