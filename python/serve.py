"""把固件那份 C 编成 .so 起个 HTTP 服务，web 那边直接调，肉眼看效果。

**推理是真的那份 C**（tm_prep.c / tm_runtime.c / 导出的 tm_model.c），
不是 Python 重写。重写一份"服务端版"是最省事也最没用的做法——那样跑出来的
效果跟板上是什么关系谁也说不清，而"说不清"正是这个仓库要消灭的东西。

能验什么、不能验什么，说清楚：
  · 能验：模型导出对不对、整条链通不通、在真实数据上效果如何、
    golden vector 逐位对不对得上。
  · **不能**替代板上那次比对：tm_invoke 全程整数，逐位结果跟指令集无关，
    这部分 x86 上验过就等于板上验过；但 tm_prep 里有 double 和 round()，
    走的是各自平台的 libm，那一段只有真板子上跑过才算数。

只用标准库（http.server + ctypes），不装任何东西。

用法：
    python python/serve.py --gen firmware/generated_cnn_a
    python python/serve.py --gen firmware/generated_cnn_a \\
        --raw ~/imu_train/holdout_raw.npy --labels ~/imu_train/holdout_y.npy \\
        --port 8080 --host 0.0.0.0

然后浏览器打开 http://<服务器>:8080/
"""

import argparse
import ctypes
import errno
import json
import os
import subprocess
import sys
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")
HOST_C = os.path.join(ROOT, "host", "tm_host.c")


def build(gen_dir, out_so=None, cc="gcc"):
    """把固件的 C 编成共享库。

    -ffp-contract=off 不是可选项：FMA 收缩会少一次中间舍入，
    那样这个服务算出来的就不是板上会算出来的东西了。
    """
    gen_dir = os.path.abspath(os.path.expanduser(gen_dir))
    need = ["tm_model.c", "tm_model.h", "tm_golden.h"]
    missing = [f for f in need if not os.path.exists(os.path.join(gen_dir, f))]
    if missing:
        sys.exit(f"{gen_dir} 里缺 {missing}。\n"
                 "  先跑一次 export_cnn.py 生成，--out 指到这个目录。")
    out_so = out_so or os.path.join(tempfile.mkdtemp(), "tm_host.so")
    cmd = [cc, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", "-fno-math-errno", "-fPIC", "-shared",
           f"-I{FW}", f"-I{gen_dir}",
           os.path.join(FW, "tm_prep.c"), os.path.join(FW, "tm_runtime.c"),
           os.path.join(gen_dir, "tm_model.c"), HOST_C,
           "-lm", "-o", out_so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"编译失败：\n{r.stderr}")
    return out_so


class Engine:
    """ctypes 包一层。形状全从 C 里问，不在 Python 这边写死——
    写死的话换个模型就会静默错位，而错位的表现是"效果突然变差"。"""

    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        L = self.lib
        for name in ("th_n_ch", "th_n_t", "th_n_classes", "th_golden_n",
                     "th_selftest", "th_in_zp", "th_arena_bytes"):
            getattr(L, name).restype = ctypes.c_int
            getattr(L, name).argtypes = []
        L.th_class_name.restype = ctypes.c_char_p
        L.th_class_name.argtypes = [ctypes.c_int]
        for name in ("th_ch_mean", "th_ch_std"):
            getattr(L, name).restype = ctypes.c_double
            getattr(L, name).argtypes = [ctypes.c_int]
        for name in ("th_in_scale", "th_out_scale"):
            getattr(L, name).restype = ctypes.c_double
            getattr(L, name).argtypes = []
        L.th_out_zp.restype = ctypes.c_int
        L.th_out_zp.argtypes = []
        L.th_infer_batch.restype = ctypes.c_int
        L.th_infer_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8)]

        self.n_ch = L.th_n_ch()
        self.n_t = L.th_n_t()
        self.n_classes = L.th_n_classes()
        self.classes = [L.th_class_name(i).decode("utf-8")
                        for i in range(self.n_classes)]
        self.ch_mean = [L.th_ch_mean(i) for i in range(self.n_ch)]
        self.ch_std = [L.th_ch_std(i) for i in range(self.n_ch)]
        self.in_scale = L.th_in_scale()
        self.in_zp = L.th_in_zp()
        self.out_scale = L.th_out_scale()
        self.out_zp = L.th_out_zp()
        self.arena_bytes = L.th_arena_bytes()
        self.golden_n = L.th_golden_n()
        # C 里有静态缓冲，多线程同时进会互相踩。加锁比改 C 省事，
        # 而且端上本来就是单线程跑，这里并发也没有意义
        self.lock = threading.Lock()

    def selftest(self):
        with self.lock:
            return int(self.lib.th_selftest())

    def infer(self, wins):
        """wins: float32 [N, n_ch, n_t]（原始量纲）→ (类别 [N], 分数 [N, n_classes])。"""
        w = np.ascontiguousarray(np.asarray(wins, np.float32))
        if w.ndim == 2:
            w = w[None]
        if w.shape[1:] != (self.n_ch, self.n_t):
            raise ValueError(f"窗口形状要 [N, {self.n_ch}, {self.n_t}]，给的是 {w.shape}")
        n = w.shape[0]
        cls = np.empty(n, np.int8)
        sc = np.empty((n, self.n_classes), np.int8)
        with self.lock:
            rc = self.lib.th_infer_batch(
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
                cls.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                sc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        if rc != 0:
            raise RuntimeError("tm_invoke 返回错误（arena 不够？）")
        return cls.astype(np.int64), sc


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>端侧推理（固件 C）</title>
<style>
 body{font:14px/1.6 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;
      margin:0;padding:24px;background:#fafaf9;color:#1c1917}
 .wrap{max-width:1100px;margin:0 auto}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#78716c;margin:0 0 20px}
 .card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;
       padding:16px 18px;margin-bottom:16px}
 .row{display:flex;gap:24px;flex-wrap:wrap}
 .kv{font-size:13px} .kv b{color:#57534e;font-weight:500}
 .ok{color:#15803d;font-weight:600} .bad{color:#b91c1c;font-weight:600}
 button{font:inherit;padding:6px 14px;border:1px solid #d6d3d1;background:#fff;
        border-radius:7px;cursor:pointer} button:hover{background:#f5f5f4}
 input{font:inherit;padding:5px 8px;border:1px solid #d6d3d1;border-radius:7px;width:90px}
 #tl{width:100%;height:72px;border:1px solid #e7e5e4;border-radius:7px;display:block}
 .lg{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px}
 .sw{width:12px;height:12px;border-radius:3px;display:inline-block;
     vertical-align:-1px;margin-right:4px}
 table{border-collapse:collapse;font-size:13px;margin-top:8px}
 td,th{border-bottom:1px solid #f0efee;padding:4px 12px 4px 0;text-align:right}
 th:first-child,td:first-child{text-align:left}
 .note{font-size:12px;color:#78716c;margin-top:10px;line-height:1.7}
</style>
<div class="wrap">
<h1>端侧推理 · 跑的是固件那份 C</h1>
<p class="sub">tm_prep.c + tm_runtime.c + 导出的 tm_model.c，经 ctypes 调用。不是 Python 重写。</p>

<div class="card">
  <div class="row" id="meta"></div>
  <div class="note" id="selftest">自检中…</div>
</div>

<div class="card">
  <b>回放留出集</b>
  <div class="row" style="align-items:center;margin-top:10px">
    <label>起点 <input id="start" type="number" value="0" min="0"></label>
    <label>窗口数 <input id="n" type="number" value="600" min="1" max="4000"></label>
    <button onclick="go(0)">回放</button>
    <button onclick="go(-1)">← 上一段</button>
    <button onclick="go(1)">下一段 →</button>
    <button onclick="findFocus()">跳到下一次「<span id="fname">抓挠</span>」</button>
  </div>
  <canvas id="tl" style="margin-top:12px"></canvas>
  <div class="lg" id="lg"></div>
  <div id="stat"></div>
  <div class="note">
    上半条是<b>真值</b>，下半条是<b>端侧 C 的判决</b>。对齐的地方就是判对了。<br>
    这条时间轴上每一格是一个窗口（1 秒 @16Hz，步长按训练时的 stride）。
  </div>
</div>

<div class="card note">
  <b>这个页面能证明什么、不能证明什么</b><br>
  能：模型导出没错、整条链通、在真实数据上的效果、golden vector 逐位对得上。<br>
  不能：替代板上那次比对。tm_invoke 全程整数，逐位结果跟指令集无关，
  x86 上验过就等于板上验过；但 tm_prep 里有 double 和 round()，
  走各自平台的 libm，<b>那一段只有真板子上跑过才算数</b>。
</div>
</div>
<script>
let M=null, COL=["#0ea5e9","#a8a29e","#e11d48","#292524","#f59e0b","#8b5cf6","#10b981"];
async function boot(){
  M=await (await fetch('/meta')).json();
  document.getElementById('meta').innerHTML=
    `<div class="kv"><b>类别</b> ${M.classes.join(' / ')}</div>`+
    `<div class="kv"><b>窗口</b> ${M.n_ch}×${M.n_t}</div>`+
    `<div class="kv"><b>arena</b> ${M.arena_bytes} B</div>`+
    `<div class="kv"><b>留出集</b> ${M.n_windows} 条</div>`;
  const s=await (await fetch('/selftest')).json();
  document.getElementById('selftest').innerHTML = s.bad===0
    ? `<span class="ok">✓ golden vector 自检通过</span>：${s.n} 条样例，`+
      `这份 C 算出来跟导出时 Python 算的<b>逐位相同</b>。`
    : `<span class="bad">✗ 自检失败</span>：${s.bad} 个字节对不上。`+
      `导出和运行时不配套，下面的效果不用看了。`;
  document.getElementById('lg').innerHTML=M.classes.map((c,i)=>
    `<span><span class="sw" style="background:${COL[i%COL.length]}"></span>${c}</span>`).join('');
  document.getElementById('fname').textContent=M.focus;
  go(0);
}
async function go(d){
  const st=document.getElementById('start'), n=+document.getElementById('n').value;
  if(d) st.value=Math.max(0,+st.value+d*n);
  const r=await (await fetch(`/replay?start=${+st.value}&n=${n}`)).json();
  draw(r);
}
async function findFocus(){
  const st=document.getElementById('start');
  const r=await (await fetch(`/next_focus?after=${+st.value}`)).json();
  if(r.index<0){ alert('后面没有了，从头再找'); st.value=0; return go(0); }
  st.value=Math.max(0,r.index-30); go(0);
}
function draw(r){
  const c=document.getElementById('tl'), x=c.getContext('2d');
  c.width=c.clientWidth*2; c.height=144; x.scale(1,1);
  const w=c.width/r.truth.length;
  x.clearRect(0,0,c.width,c.height);
  for(let i=0;i<r.truth.length;i++){
    x.fillStyle=COL[r.truth[i]%COL.length]; x.fillRect(i*w,4,Math.max(w,1),56);
    x.fillStyle=COL[r.pred[i]%COL.length];  x.fillRect(i*w,84,Math.max(w,1),56);
  }
  let agree=0; for(let i=0;i<r.truth.length;i++) if(r.truth[i]===r.pred[i])agree++;
  let rows=r.per_class.map(p=>`<tr><td>${p.name}</td><td>${p.support}</td>`+
      `<td>${p.tp}</td><td>${p.fp}</td><td>${p.fn}</td>`+
      `<td>${p.support?p.recall.toFixed(3):'—'}</td></tr>`).join('');
  document.getElementById('stat').innerHTML=
    `<table><tr><th>类别</th><th>真值窗口</th><th>对</th><th>误报</th><th>漏</th><th>召回</th></tr>`+
    rows+`</table><p class="kv" style="margin-top:8px"><b>这一段一致率</b> `+
    `${(agree/r.truth.length*100).toFixed(1)}% · <b>推理耗时</b> ${r.ms.toFixed(1)} ms / `+
    `${r.truth.length} 窗口（${(r.ms/r.truth.length*1000).toFixed(0)} µs 每窗口，x86）</p>`;
}
boot();
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    X = None
    y = None
    classes = None
    focus = "抓挠"

    def log_message(self, *a):
        pass          # 默认会把每个请求打到 stderr，回放时刷屏

    def _send(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        E = self.engine
        try:
            if u.path == "/":
                b = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/meta":
                self._send({
                    "classes": E.classes, "n_ch": E.n_ch, "n_t": E.n_t,
                    "arena_bytes": E.arena_bytes, "in_scale": E.in_scale,
                    "in_zp": E.in_zp, "ch_mean": E.ch_mean, "ch_std": E.ch_std,
                    "n_windows": 0 if self.X is None else int(len(self.X)),
                    "focus": self.focus,
                })
            elif u.path == "/selftest":
                bad = E.selftest()
                self._send({"n": E.golden_n, "bad": bad})
            elif u.path == "/replay":
                self._replay(int(q.get("start", [0])[0]), int(q.get("n", [600])[0]))
            elif u.path == "/next_focus":
                self._next_focus(int(q.get("after", [0])[0]))
            else:
                self._send({"error": "no such path"}, 404)
        except Exception as e:                      # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        """POST /infer  {"window": [[...n_t...] × n_ch]}  → 类别 + 分数。

        给 web 那边接真实设备数据用。窗口是**原始量纲**，归一化在 C 里做
        （tm_prep），调用方不需要知道 ch_mean/ch_std——知道了反而容易做两遍。
        """
        u = urlparse(self.path)
        if u.path != "/infer":
            return self._send({"error": "no such path"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            w = np.asarray(body["window"], np.float32)
            cls, sc = self.engine.infer(w)
            self._send({"class": self.engine.classes[int(cls[0])],
                        "index": int(cls[0]),
                        "scores": [int(v) for v in sc[0]]})
        except Exception as e:                      # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 400)

    def _replay(self, start, n):
        if self.X is None:
            return self._send({"error": "启动时没给 --raw/--labels，没法回放"}, 400)
        start = max(0, min(start, len(self.X) - 1))
        n = max(1, min(n, 4000, len(self.X) - start))
        blk = self.X[start:start + n]
        t0 = time.perf_counter()
        pred, _ = self.engine.infer(blk)
        ms = (time.perf_counter() - t0) * 1000.0
        truth = self.y[start:start + n]
        per = []
        for i, name in enumerate(self.engine.classes):
            per.append({
                "name": name,
                "support": int(np.sum(truth == i)),
                "tp": int(np.sum((pred == i) & (truth == i))),
                "fp": int(np.sum((pred == i) & (truth != i))),
                "fn": int(np.sum((pred != i) & (truth == i))),
                "recall": float(np.sum((pred == i) & (truth == i))
                                / max(int(np.sum(truth == i)), 1)),
            })
        self._send({"start": start, "truth": truth.tolist(),
                    "pred": pred.tolist(), "per_class": per, "ms": ms})

    def _next_focus(self, after):
        """跳到下一段目标类别。没有这个的话，人得手动翻几千个窗口才能
        看到一次抓挠——留出集里抓挠只占 1.1%。"""
        if self.y is None:
            return self._send({"index": -1})
        try:
            fi = self.engine.classes.index(self.focus)
        except ValueError:
            return self._send({"index": -1})
        hits = np.flatnonzero(self.y[after + 1:] == fi)
        self._send({"index": int(after + 1 + hits[0]) if len(hits) else -1})


def listen(host, port, handler, tries=20):
    """绑端口，占用了就往后顺延。

    端口占用太常见了（上一个实例没停、别的服务占着 8080），而默认行为是甩一个
    OSError 的 traceback——那玩意儿看着像程序坏了，其实只要换个端口。

    `port=0` 交给内核挑。`tries=1` 就是"只试这一个，占了就报错"（--strict-port），
    因为有时候端口是写死在别处的配置里的，静默换掉反而更糟。

    HTTPServer 已经设了 SO_REUSEADDR，所以这里的 EADDRINUSE **是真的有人在听**，
    不是 TIME_WAIT 的残留——不用靠重试等它自己好。
    """
    last = None
    for i in range(max(tries, 1)):
        p = 0 if port == 0 else port + i
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            last = e
    sys.exit(f"{host}:{port}..{port + tries - 1} 都被占着（{last}）。\n"
             f"  看是谁占的：  ss -ltnp | grep :{port}\n"
             "  或者直接指一个空的：--port 9000，或 --port 0 让内核挑。")


def urls(host, port):
    """把能点的地址列出来。

    绑 0.0.0.0 时只打印 "<服务器 IP>" 是在给人出题——用户得自己去查 IP。
    这里直接把本机的地址找出来。
    """
    if host not in ("0.0.0.0", "::"):
        return [f"http://{host}:{port}/"]
    out = [f"http://127.0.0.1:{port}/   （本机）"]
    ips = set()
    try:
        # 不发包，只是让内核挑一条出口路由，从而拿到对外那张网卡的地址。
        # socket.gethostbyname(gethostname()) 在很多机器上只会给 127.0.1.1
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    for ip in sorted(ips):
        if not ip.startswith("127."):
            out.append(f"http://{ip}:{port}/   （局域网，web 那边用这个）")
    if len(out) == 1:
        out.append(f"http://<服务器 IP>:{port}/   （没探到对外地址，自己填）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True,
                    help="export_cnn.py 的输出目录（含 tm_model.c/h、tm_golden.h）")
    ap.add_argument("--raw", default="", help="留出集原始窗口 .npy，用来回放")
    ap.add_argument("--labels", default="")
    ap.add_argument("--focus", default="抓挠")
    ap.add_argument("--host", default="127.0.0.1",
                    help="要让别的机器访问就写 0.0.0.0")
    ap.add_argument("--port", type=int, default=8080,
                    help="被占了会自动往后顺延；0 = 让内核挑一个空的")
    ap.add_argument("--strict-port", action="store_true",
                    help="端口被占就报错，不要自动换（端口写死在别处配置里时用）")
    ap.add_argument("--cc", default="gcc")
    args = ap.parse_args()

    so = build(args.gen, cc=args.cc)
    eng = Engine(so)
    print(f"编好了：{so}")
    print(f"模型：{eng.n_ch}×{eng.n_t} 窗口，{eng.n_classes} 类 "
          f"（{', '.join(eng.classes)}），arena {eng.arena_bytes} B")

    bad = eng.selftest()
    if bad == 0:
        print(f"golden vector 自检：{eng.golden_n} 条，**逐位相同** ✓")
    else:
        # 不退出：让人能打开页面看到红字和原因，比一行 traceback 有用
        print(f"⚠ golden vector 自检失败：{bad} 个字节对不上。"
              "导出和运行时不配套，效果不用看了。")

    if args.raw and args.labels:
        Handler.X = np.load(os.path.expanduser(args.raw)).astype(np.float32)
        Handler.y = np.load(os.path.expanduser(args.labels)).astype(np.int64)
        if len(Handler.X) != len(Handler.y):
            sys.exit(f"窗口 {len(Handler.X)} 条、标签 {len(Handler.y)} 条，对不上")
        print(f"留出集 {len(Handler.X)} 条，可以回放")
    else:
        print("没给 --raw/--labels，只能用 POST /infer，页面上的回放会是空的")

    Handler.engine = eng
    Handler.focus = args.focus
    srv = listen(args.host, args.port, Handler,
                 tries=1 if args.strict_port else 20)
    port = srv.server_address[1]
    if port != args.port:
        print(f"\n{args.port} 被占了，换到 {port}")
    print(f"\n开着了，Ctrl-C 停：")
    for u in urls(args.host, port):
        print(f"  {u}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停了")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
