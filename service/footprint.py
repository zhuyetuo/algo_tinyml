"""端侧资源占用：这份导出烧到板上要吃多少 flash / RAM、推一个窗口要多少算力、
交给嵌入式的源码包有多大。给平台「训练记录」那页显示，也写进 meta.json。

量的是**真的编出来的数**：用 arm-none-eabi-gcc 按固件同一套选项（-Os、Cortex-M4F、
-ffp-contract=off）把 core/ 的运行时和这份模型各编一遍，arm-none-eabi-size 读
text / data / bss。没装交叉编译器就退回宿主机 gcc，并在结果里标明是 x86 估算
（代码体积会偏大三到五成，模型那一块是常量表、两边一样）。

分几块报，各管一件事：
  模型         导出的模型 .c（树的节点/叶子表，或 int8 权重）——const，只占 flash
  常量表       tm_feat_cfg：Hann 窗 / FFT 旋转因子 / 位反序（RF 那条才有）
  工程代码     tm_features / tm_forest_c / tm_window / tm_post…（跟模型无关的那份 C）
  自检 golden  导出时带的对照向量，量产固件可以只留几条或者放到产测固件里
  RAM          编译出来的 bss + data（特征缓冲、窗口缓冲、后处理延迟线；CNN 另加乒乓 arena）
  推理算力     每个窗口要做的事：RF = 特征提取（几路 FFT）+ 树遍历（最多 棵数×深度 次比较）；
               CNN = 乘加次数。另外给一个 x86 上实测的每窗耗时，**只能横向比**，跟 M4 没可比性
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(ROOT, "core")

# 浮点 ABI 跟 GR551x SDK 一致（softfp）：预编的 .a 要能直接链进 SDK 工程，ABI 不同链接器会拒绝
ARM_FLAGS = ["-Os", "-std=c99", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=softfp",
             "-ffp-contract=off", "-fno-math-errno", "-ffunction-sections", "-fdata-sections"]
HOST_FLAGS = ["-Os", "-std=c99", "-ffp-contract=off", "-fno-math-errno", "-ffunction-sections", "-fdata-sections"]

RUNTIME = {
    "rf": ["tm_features.c", "tm_forest_c.c", "tm_window.c", "tm_post.c", "tm_post_cfg.c"],
    "cnn": ["tm_prep.c", "tm_runtime.c", "tm_window.c", "tm_post.c", "tm_post_cfg.c"],
}
MODEL_FILES = {"rf": ["tm_forest_c_model.c"], "cnn": ["tm_model.c"]}
TABLE_FILES = {"rf": ["tm_feat_cfg.c"], "cnn": []}
GOLDEN_HEADERS = {"rf": ["tm_forest_c_golden.h", "tm_forest_c_pipeline_golden.h"], "cnn": ["tm_golden.h"]}

_CTYPE_BYTES = {"int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
                "float": 4, "int": 4, "double": 8}


def toolchain() -> tuple[str, str, bool]:
    """(cc, size, 是不是真的交叉编译)"""
    if shutil.which("arm-none-eabi-gcc") and shutil.which("arm-none-eabi-size"):
        return "arm-none-eabi-gcc", "arm-none-eabi-size", True
    return shutil.which("gcc") or "gcc", shutil.which("size") or "size", False


def _size_of(size_bin: str, obj: str) -> dict:
    out = subprocess.run([size_bin, obj], capture_output=True, text=True, check=True).stdout.splitlines()
    text, data, bss = (int(x) for x in out[-1].split()[:3])
    return {"text": text, "data": data, "bss": bss}


def _compile_all(files: list[str], include: list[str], defines: dict, arm: bool, cc: str, size_bin: str) -> dict:
    """{文件名: {text, data, bss}}。编不过的那个记 error，别的照常。"""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            obj = os.path.join(tmp, os.path.basename(f) + ".o")
            cmd = [cc, "-c", *(ARM_FLAGS if arm else HOST_FLAGS),
                   *[f"-D{k}={v}" for k, v in defines.items()], *[f"-I{i}" for i in include], f, "-o", obj]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                out[os.path.basename(f)] = {"error": r.stderr.strip()[-400:]}
                continue
            out[os.path.basename(f)] = _size_of(size_bin, obj)
    return out


def golden_bytes(gen_dir: str, kind: str) -> int:
    """golden 头文件里的常量数组占多少 flash：数元素个数 × 类型字节数。"""
    total = 0
    for h in GOLDEN_HEADERS[kind]:
        p = os.path.join(gen_dir, h)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        for m in re.finditer(r"static\s+const\s+(\w+)\s+\w+\s*\[\s*\]\s*=\s*\{([^}]*)\}", src, re.S):
            n = len([x for x in m.group(2).split(",") if x.strip()])
            total += n * _CTYPE_BYTES.get(m.group(1), 4)
    return total


def source_bundle_bytes(gen_dir: str, kind: str) -> dict:
    """交给嵌入式的源码包：core/ 里用得到的 .c/.h + 这份导出目录。按文件算。"""
    names = set(RUNTIME[kind] + MODEL_FILES[kind] + TABLE_FILES[kind])
    files = {}
    for c in RUNTIME[kind]:
        for ext in (".c", ".h"):
            p = os.path.join(CORE, c.replace(".c", ext))
            if os.path.exists(p):
                files[f"core/{os.path.basename(p)}"] = os.path.getsize(p)
    for h in ("tm_forest_c.h", "tm_runtime.h", "tm_prep.h", "tm_window.h", "tm_post.h", "tm_post_cfg.h", "tm_features.h"):
        p = os.path.join(CORE, h)
        if os.path.exists(p) and f"core/{h}" not in files and h.replace(".h", ".c") in names:
            files[f"core/{h}"] = os.path.getsize(p)
    for p in sorted(glob.glob(os.path.join(gen_dir, "*"))):
        if os.path.isfile(p):
            files[f"model/{os.path.basename(p)}"] = os.path.getsize(p)
    return {"files": files, "total": sum(files.values())}


def measure(gen_dir: str, kind: str, window: int, n_ch: int, n_classes: int,
            extra: dict | None = None) -> dict:
    """主入口。extra 里可以带 trees/depth/nodes（rf）或 macs/arena_bytes（cnn）来算推理那一栏。"""
    gen_dir = os.path.abspath(gen_dir)
    cc, size_bin, arm = toolchain()
    defines = {"TM_FEAT_MAX_T": max(int(window), 16), "TM_FEAT_MAX_NPERSEG": max(int(window), 16),
               "TM_POST_MAX_CLASSES": max(int(n_classes), 2)}
    include = [CORE, gen_dir]
    runtime = _compile_all([os.path.join(CORE, f) for f in RUNTIME[kind]], include, defines, arm, cc, size_bin)
    model = _compile_all([os.path.join(gen_dir, f) for f in MODEL_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))],
                         include, defines, arm, cc, size_bin)
    tables = _compile_all([os.path.join(gen_dir, f) for f in TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))],
                          include, defines, arm, cc, size_bin)

    def flash(d: dict) -> int:
        return sum(v.get("text", 0) + v.get("data", 0) for v in d.values())

    def ram(d: dict) -> int:
        return sum(v.get("data", 0) + v.get("bss", 0) for v in d.values())

    extra = extra or {}
    # 窗口缓冲：tm_window 攒 n_ch × n_t 个 float，由调用方分配（不在 bss 里），单独算上
    window_buf = int(n_ch) * int(window) * 4
    arena = int(extra.get("arena_bytes") or 0)          # CNN 乒乓缓冲，由调用方分配
    # 板上后处理（稳定版 v2 那套）的状态：延迟线 TM_POST_DELAY 个窗口 × (每类概率 + 时间 + 标记)，
    # tm_post.h 里的那笔账。不用板上后处理（只报逐窗口判决）就不占
    post_state = 192 * (int(n_classes) * 4 + 4 + 5)
    ram_total = ram(runtime) + ram(tables) + window_buf + arena + post_state
    gold = golden_bytes(gen_dir, kind)
    bundle = source_bundle_bytes(gen_dir, kind)

    infer: dict = {}
    if kind == "rf":
        n_sensor = n_ch - 2 if n_ch % 3 == 2 else n_ch
        ffts = n_sensor + (2 if n_ch >= 6 else 0)
        infer = {
            "fft_per_window": ffts,
            "fft_points": int(window),
            "tree_compares_max": int(extra.get("trees", 0)) * int(extra.get("depth", 0)),
            "trees": extra.get("trees"), "depth": extra.get("depth"), "nodes": extra.get("nodes"),
            "note": f"每窗：{n_ch} 路时域统计 + {ffts} 次 {window} 点 FFT + 最多 {extra.get('trees', '?')}×{extra.get('depth', '?')} 次树节点比较。"
                    "树遍历是随机访存，8KB cache 帮不上忙；特征提取是顺序的",
        }
    else:
        infer = {"macs_per_window": int(extra.get("macs") or 0),
                 "note": f"每窗 {int(extra.get('macs') or 0):,} 次 int8 乘加，权重顺序读、cache 友好；"
                         "上 CMSIS-NN 还能快 4 倍左右（见 docs/frameworks.md）"}
    if extra.get("host_us_per_window") is not None:
        infer["host_us_per_window"] = extra["host_us_per_window"]

    return {
        "toolchain": "arm-none-eabi-gcc（Cortex-M4F, -Os）" if arm else f"{cc}（x86 估算，代码体积偏大三到五成）",
        "cross_compiled": arm,
        "flash": {
            "model": flash(model), "tables": flash(tables), "runtime": flash(runtime), "golden": gold,
            "total_without_golden": flash(model) + flash(tables) + flash(runtime),
            "total": flash(model) + flash(tables) + flash(runtime) + gold,
        },
        "ram": {
            "runtime_bss": ram(runtime) + ram(tables), "window_buffer": window_buf, "arena": arena,
            "post_state": post_state, "total": ram_total,
            "total_without_post": ram_total - post_state,
            "note": "模型是 const，落在 flash 里 CPU 直接读，不占 RAM（GR5513 的 flash 是内存映射的）",
        },
        "inference": infer,
        "source_bundle": bundle,
        "per_file": {"runtime": runtime, "model": model, "tables": tables},
        "chip": {"name": "GR5513", "flash_total": 512 * 1024, "ram_available": 112 * 1024,
                 "ble_baseline_flash": 104124, "ble_baseline_ram": 22128,
                 "note": "基线 = BLE 协议栈 + 最小应用（实测 104 KB / 22 KB）；给模型留的预算约 128 KB"},
        "measured_at": int(time.time()),
    }


def bench_host(infer_fn, windows, repeat: int = 3) -> float:
    """x86 上每窗微秒数。infer_fn(windows) 跑整批；取最快的一次。只能横向比。"""
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        infer_fn(windows)
        dt = (time.perf_counter() - t0) / max(1, len(windows)) * 1e6
        best = dt if best is None else min(best, dt)
    return round(best or 0.0, 1)


def build_static_lib(gen_dir: str, kind: str, window: int, n_classes: int, out_dir: str) -> dict | None:
    """把运行时 + 模型预编成 libtinyml.a（Cortex-M4F，softfp，跟 GR551x SDK 一致）。
    没有交叉编译器就返回 None——x86 的 .a 对板子没用，宁可不给。"""
    cc, _, arm = toolchain()
    if not arm:
        return None
    ar = shutil.which("arm-none-eabi-ar")
    if not ar:
        return None
    srcs = [os.path.join(CORE, f) for f in RUNTIME[kind]] + \
           [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]
    defines = {"TM_FEAT_MAX_T": max(int(window), 16), "TM_FEAT_MAX_NPERSEG": max(int(window), 16),
               "TM_POST_MAX_CLASSES": max(int(n_classes), 2)}
    os.makedirs(out_dir, exist_ok=True)
    objs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in srcs:
            obj = os.path.join(tmp, os.path.basename(f) + ".o")
            r = subprocess.run([cc, "-c", *ARM_FLAGS, *[f"-D{k}={v}" for k, v in defines.items()],
                                f"-I{CORE}", f"-I{gen_dir}", f, "-o", obj], capture_output=True, text=True)
            if r.returncode != 0:
                return {"error": r.stderr.strip()[-400:]}
            objs.append(obj)
        lib = os.path.join(out_dir, "libtinyml.a")
        if os.path.exists(lib):
            os.remove(lib)
        subprocess.run([ar, "rcs", lib, *objs], check=True)
    return {"path": lib, "bytes": os.path.getsize(lib), "flags": " ".join(ARM_FLAGS),
            "defines": " ".join(f"-D{k}={v}" for k, v in defines.items())}


def write_bundle(gen_dir: str, kind: str, out_zip: str, readme: str,
                 window: int | None = None, n_classes: int | None = None) -> int:
    """交给嵌入式的包：源码（core/ + 导出目录）+ 预编的 lib/libtinyml.a + include/ + README。返回字节数。"""
    import zipfile

    files = source_bundle_bytes(gen_dir, kind)["files"]
    lib = None
    if window and n_classes:
        with tempfile.TemporaryDirectory() as tmp:
            lib = build_static_lib(gen_dir, kind, window, n_classes, tmp)
            lib_bytes = open(lib["path"], "rb").read() if lib and "path" in lib else None
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            src = os.path.join(CORE, rel[5:]) if rel.startswith("core/") else os.path.join(gen_dir, rel[6:])
            z.write(src, rel)
        if lib and "path" in lib:
            z.writestr("lib/libtinyml.a", lib_bytes)
            # 只链 .a 的话头文件单独放一份，不用在 core/ 和 model/ 里翻
            for rel in files:
                if rel.endswith(".h") and not rel.endswith("golden.h"):
                    z.write(os.path.join(CORE, rel[5:]) if rel.startswith("core/") else os.path.join(gen_dir, rel[6:]),
                            "include/" + os.path.basename(rel))
            z.writestr("lib/BUILD_FLAGS.txt",
                       f"arm-none-eabi-gcc {lib['flags']} {lib['defines']}\n"
                       "浮点 ABI 是 softfp，跟 GR551x SDK 的 libble_sdk.a 一致；工程用 hard 的话别用这个 .a，拿源码重编。\n")
        for extra in ("board/README.md", "docs/ram_and_cache.md"):
            p = os.path.join(ROOT, extra)
            if os.path.exists(p):
                z.write(p, extra)
        z.writestr("README.txt", readme)
    return os.path.getsize(out_zip)


def bundle_readme(kind: str, meta: dict, fp: dict) -> str:
    fl, rm, inf = fp["flash"], fp["ram"], fp["inference"]
    api = (
        "调用顺序（RF）：\n"
        "  tm_window_push(&win, sample_float, buf)  每来一个 IMU 样本喂一次，攒满一个窗口返回 1\n"
        "  tm_features(&tm_feat_cfg, buf, feats)   窗口 → 特征（float[TM_FEAT_DIM]）\n"
        "  tm_forest_c_predict(&tm_forest_c, feats, votes)  → 每类整数票数，argmax 即类别\n"
        "  tm_post_on_window(...)                  可选：板上后处理（稳定版 v2），把逐窗口判决聚成事件\n"
        if kind == "rf" else
        "调用顺序（1D-CNN）：\n"
        "  tm_window_push(&win, sample_float, buf)  每来一个 IMU 样本喂一次，攒满一个窗口返回 1\n"
        "  tm_prep(&tm_model_prep, buf, x_i8)      逐通道 z-score + 量化成 int8（均值/方差已导进 tm_model.c）\n"
        "  tm_invoke(&tm_model, x_i8, out, arena, TM_ARENA_BYTES)  → int8 分数，tm_argmax 即类别\n"
        "  tm_post_on_window(...)                  可选：板上后处理\n"
    )
    return (
        f"端侧模型源码包  {meta.get('train', {}).get('tag', '')}\n\n"
        "两种接法，二选一：\n"
        "  A. 源码：把 core/*.c 和 model/*.c 加进工程一起编（推荐，编译选项跟自己的 SDK 一定一致）\n"
        "  B. 静态库：链 lib/libtinyml.a，include/ 里是头文件。预编选项见 lib/BUILD_FLAGS.txt，\n"
        "     Cortex-M4F + softfp（跟 GR551x SDK 一致）；工程是 hard ABI 的话链不上，用 A\n\n"
        f"模型：{kind}，{meta.get('n_channels')} 通道 × {meta.get('window_size')} 点 @{meta.get('hz')}Hz，"
        f"类别 {','.join(meta.get('classes') or [])}\n\n"
        f"编译：务必带 -ffp-contract=off（否则浮点末位跟 PC 对不上，golden 自检会红）；\n"
        f"      -DTM_FEAT_MAX_T={meta.get('window_size')} 按真实窗口开缓冲，不给的话按 64 编、RAM 多占一倍。\n"
        f"      核心代码不依赖任何 OS 接口，裸机或 RTOS 任务里都能调；只用 libm。\n\n"
        f"占用（{fp['toolchain']}）：\n"
        f"  flash  模型 {fl['model']:,} B + 常量表 {fl['tables']:,} B + 工程代码 {fl['runtime']:,} B"
        f" = {fl['total_without_golden']:,} B（{fl['total_without_golden'] / 1024:.1f} KB）；"
        f"自检 golden 另 {fl['golden']:,} B，量产可只留几条\n"
        f"  RAM    运行时 {rm['runtime_bss']:,} B + 窗口缓冲 {rm['window_buffer']:,} B"
        f"{' + arena ' + format(rm['arena'], ',') + ' B' if rm.get('arena') else ''}"
        f" = {rm['total_without_post']:,} B；用板上后处理再加 {rm['post_state']:,} B\n"
        f"  推理   {inf.get('note', '')}\n\n" + api +
        "\n文件：core/ 是跟模型无关的运行时（两条路线共用一份 C），model/ 是这份模型的导出。\n"
        "golden 自检：host/ 目录没打进来，自检怎么接见 board/README.md。\n"
    )
