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
BUNDLE_SRC = os.path.join(ROOT, "bundle")  # 包里 pc/ 和 Makefile、README 的模板

import sys  # noqa: E402

sys.path.insert(0, HERE)
from tinyml import cmsis  # noqa: E402

# 浮点 ABI 跟 GR551x SDK 一致（softfp）：预编的 .a 要能直接链进 SDK 工程，ABI 不同链接器会拒绝
ARM_FLAGS = ["-Os", "-std=c99", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=softfp",
             "-ffp-contract=off", "-fno-math-errno", "-ffunction-sections", "-fdata-sections"]
HOST_FLAGS = ["-Os", "-std=c99", "-ffp-contract=off", "-fno-math-errno", "-ffunction-sections", "-fdata-sections"]

RUNTIME = {
    "rf": ["tm_imu.c", "tm_features.c", "tm_forest_c.c", "tm_window.c", "tm_post.c", "tm_post_cfg.c"],
    "cnn": ["tm_imu.c", "tm_prep.c", "tm_runtime.c", "tm_window.c", "tm_post.c", "tm_post_cfg.c"],
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


ROOTS = {
    "rf": ["tm_features", "tm_forest_c_predict", "tm_window_push", "tm_post_on_window"],
    "cnn": ["tm_prep", "tm_invoke", "tm_window_push", "tm_post_on_window"],
}


def _linked_flash(files: list[str], include: list[str], defines: dict, arm: bool, cc: str, size_bin: str,
                  roots: list[str]) -> int | None:
    """把这些 .c 真的链一次（--gc-sections，以对外接口为根），读 text+data。
    跟逐文件 size 的区别：链接器会把没引用到的函数和表扔掉——CMSIS 的表文件里 5 档只用 1 档，
    逐文件算会多报 5 KB。链不过（缺 libm 符号之类）就返回 None，调用方退回逐文件的数。"""
    with tempfile.TemporaryDirectory() as tmp:
        objs = []
        for f in files:
            obj = os.path.join(tmp, os.path.basename(f) + ".o")
            r = subprocess.run([cc, "-c", *(ARM_FLAGS if arm else HOST_FLAGS),
                                *[f"-D{k}={v}" for k, v in defines.items()], *[f"-I{i}" for i in include], f, "-o", obj],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return None
            objs.append(obj)
        elf = os.path.join(tmp, "linked.elf")
        r = subprocess.run([cc, *(ARM_FLAGS if arm else HOST_FLAGS), "-nostdlib", "-nostartfiles",
                            "-Wl,--gc-sections", "-Wl,--unresolved-symbols=ignore-all", f"-Wl,-e,{roots[0]}",
                            *[f"-Wl,--undefined={r_}" for r_ in roots], *objs, "-o", elf], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        sz = _size_of(size_bin, elf)
        return sz["text"] + sz["data"]


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
    files = {}
    for c in RUNTIME[kind]:
        for ext in (".c", ".h"):
            p = os.path.join(CORE, c.replace(".c", ext))
            if os.path.exists(p):
                files[f"core/{os.path.basename(p)}"] = os.path.getsize(p)
    # 只有头文件、没有 .c 的那几个（tm_features.h / tm_runtime.h 都 include 它）——
    # 漏了它包里的 C 一个都编不过
    for h in ("tm_accel.h",):
        p = os.path.join(CORE, h)
        if os.path.exists(p):
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

    # 可选加速路线（-DTM_USE_CMSIS）：同一套运行时换成 CMSIS 内核再编一遍，看 flash/RAM 差多少。
    # 模型和常量表那两块不变，只有工程代码会变，外加 CMSIS 自己那几个 .c。
    accel = None
    model_srcs = [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]
    if cmsis.available() and _export_supports_cmsis(gen_dir, kind):
        inc = [CORE, gen_dir, *cmsis.include_dirs(kind, host=not arm)]
        dfs = {**defines, **cmsis.defines(kind, host=not arm)}
        rt_c = _compile_all([os.path.join(CORE, f) for f in RUNTIME[kind]], inc, dfs, arm, cc, size_bin)
        lib_c = _compile_all(cmsis.sources(kind), inc, dfs, arm, cc, size_bin)
        rt_err = [k for k, v in rt_c.items() if "error" in v] + [k for k, v in lib_c.items() if "error" in v]
        # 真链一次算差值：逐文件 size 会把 CMSIS 表文件里没用到的几档也算进去
        rt_files = [os.path.join(CORE, f) for f in RUNTIME[kind]]
        linked_plain = _linked_flash(rt_files + model_srcs, [CORE, gen_dir], defines, arm, cc, size_bin, ROOTS[kind])
        linked_cmsis = _linked_flash(rt_files + model_srcs + cmsis.sources(kind), inc, dfs, arm, cc, size_bin, ROOTS[kind])
        linked = linked_plain is not None and linked_cmsis is not None
        # CMSIS-NN 的 im2col 缓冲：导出脚本按层算好写在 tm_model.h 里（TM_ARENA_BYTES 的差）
        scratch = 0
        if kind == "cnn":
            scratch = _cmsis_arena_delta(os.path.join(gen_dir, "tm_model.h"))
        accel = {
            "name": "CMSIS-DSP" if kind == "rf" else "CMSIS-NN",
            "define": "-DTM_CMSIS_DSP=1" if kind == "rf" else "-DTM_CMSIS_NN=1",
            "bit_exact": False,
            "flash": {"runtime": flash(rt_c), "cmsis": flash(lib_c), "linked": linked,
                      "total_without_golden": (flash(model) + flash(tables) + flash(runtime) + (linked_cmsis - linked_plain))
                      if linked else flash(model) + flash(tables) + flash(rt_c) + flash(lib_c)},
            "ram": {"runtime_bss": ram(rt_c) + ram(lib_c) + ram(tables), "scratch": scratch,
                    "total_without_post": ram(rt_c) + ram(lib_c) + ram(tables) + window_buf + arena + scratch},
            "per_file": {"runtime": rt_c, "cmsis": lib_c},
            "errors": rt_err,
            "note": ("FFT 换 arm_cfft_f32、均值/功率/极值/点积换 CMSIS 向量函数；特征不再逐位一致"
                     "（相对误差 1e-6 这一级），森林判决极少数窗口会翻"
                     if kind == "rf" else
                     "卷积/池化/全连接换 arm_convolve_s8 / arm_max_pool_s8；整数累加一样，"
                     "只有重量化平局的舍入方向不同，个别输出差 1 LSB、类别不变；arena 多一段 int16 的 im2col 缓冲"),
        }
        if "host_us_per_window_cmsis" in (extra or {}):
            accel["host_us_per_window"] = extra["host_us_per_window_cmsis"]
        if "cmsis_agree" in (extra or {}):
            accel["agree_with_plain"] = extra["cmsis_agree"]

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
    if accel is not None:
        accel["flash"]["delta"] = accel["flash"]["total_without_golden"] - (flash(model) + flash(tables) + flash(runtime))
        accel["ram"]["delta"] = accel["ram"]["total_without_post"] - (ram_total - post_state)

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
        "accel": accel,
        "source_bundle": bundle,
        "per_file": {"runtime": runtime, "model": model, "tables": tables},
        "chip": {"name": "GR5513", "flash_total": 512 * 1024, "ram_available": 112 * 1024,
                 "ble_baseline_flash": 104124, "ble_baseline_ram": 22128,
                 "note": "基线 = BLE 协议栈 + 最小应用（实测 104 KB / 22 KB）；给模型留的预算约 128 KB"},
        "measured_at": int(time.time()),
    }


def _export_supports_cmsis(gen_dir: str, kind: str) -> bool:
    """cnn 的 CMSIS 路要导出脚本给的第二套权重排法（tm_model.c 里有 #if TM_CMSIS_NN）；
    老导出没有，开了开关会算错——那就不报这条，让人重导。rf 不需要导出配合。"""
    if kind != "cnn":
        return True
    p = os.path.join(gen_dir, "tm_model.c")
    return os.path.exists(p) and "TM_CMSIS_NN" in open(p, encoding="utf-8").read()


def _cmsis_arena_delta(model_h: str) -> int:
    """tm_model.h 里 #if TM_CMSIS_NN 的 TM_ARENA_BYTES 减普通的那个 = im2col 缓冲。"""
    if not os.path.exists(model_h):
        return 0
    vals = re.findall(r"#define\s+TM_ARENA_BYTES\s+(\d+)", open(model_h, encoding="utf-8").read())
    if len(vals) >= 2:
        return max(0, int(vals[0]) - int(vals[1]))
    return 0


def bench_host(infer_fn, windows, repeat: int = 3) -> float:
    """x86 上每窗微秒数。infer_fn(windows) 跑整批；取最快的一次。只能横向比。"""
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        infer_fn(windows)
        dt = (time.perf_counter() - t0) / max(1, len(windows)) * 1e6
        best = dt if best is None else min(best, dt)
    return round(best or 0.0, 1)


def build_static_lib(gen_dir: str, kind: str, window: int, n_classes: int, out_dir: str,
                     use_cmsis: bool = False) -> dict | None:
    """把运行时 + 模型预编成 libtinyml.a（Cortex-M4F，softfp，跟 GR551x SDK 一致）。
    use_cmsis=True 编的是 CMSIS 加速那条（libtinyml_cmsis.a，CMSIS 的 .o 一起打进去）。
    没有交叉编译器就返回 None——x86 的 .a 对板子没用，宁可不给。"""
    cc, _, arm = toolchain()
    if not arm:
        return None
    ar = shutil.which("arm-none-eabi-ar")
    if not ar:
        return None
    if use_cmsis and not cmsis.available():
        return None
    srcs = [os.path.join(CORE, f) for f in RUNTIME[kind]] + \
           [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]
    defines = {"TM_FEAT_MAX_T": max(int(window), 16), "TM_FEAT_MAX_NPERSEG": max(int(window), 16),
               "TM_POST_MAX_CLASSES": max(int(n_classes), 2)}
    inc = [CORE, gen_dir]
    if use_cmsis:
        srcs += cmsis.sources(kind)
        defines.update(cmsis.defines(kind, host=False))
        inc += cmsis.include_dirs(kind, host=False)
    os.makedirs(out_dir, exist_ok=True)
    objs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in srcs:
            obj = os.path.join(tmp, os.path.basename(f) + ".o")
            r = subprocess.run([cc, "-c", *ARM_FLAGS, *[f"-D{k}={v}" for k, v in defines.items()],
                                *[f"-I{i}" for i in inc], f, "-o", obj], capture_output=True, text=True)
            if r.returncode != 0:
                return {"error": r.stderr.strip()[-400:]}
            objs.append(obj)
        lib = os.path.join(out_dir, "libtinyml_cmsis.a" if use_cmsis else "libtinyml.a")
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
    lib = lib_cmsis = None
    lib_bytes = lib_cmsis_bytes = None
    if window and n_classes:
        with tempfile.TemporaryDirectory() as tmp:
            lib = build_static_lib(gen_dir, kind, window, n_classes, tmp)
            lib_bytes = open(lib["path"], "rb").read() if lib and "path" in lib else None
            lib_cmsis = build_static_lib(gen_dir, kind, window, n_classes, tmp, use_cmsis=True)
            lib_cmsis_bytes = open(lib_cmsis["path"], "rb").read() if lib_cmsis and "path" in lib_cmsis else None
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            src = os.path.join(CORE, rel[5:]) if rel.startswith("core/") else os.path.join(gen_dir, rel[6:])
            z.write(src, rel)
        # 可选加速路线要的 CMSIS 子集（源码接法 A 想开 -DTM_USE_CMSIS 时用；不开就不用管这个目录）
        if cmsis.available():
            for rel, src in cmsis.bundle_files(kind).items():
                z.write(src, rel)
        if lib_cmsis and "path" in lib_cmsis:
            z.writestr("lib/libtinyml_cmsis.a", lib_cmsis_bytes)
        if lib and "path" in lib:
            z.writestr("lib/libtinyml.a", lib_bytes)
            # 只链 .a 的话头文件单独放一份，不用在 core/ 和 model/ 里翻
            for rel in files:
                if rel.endswith(".h") and not rel.endswith("golden.h"):
                    z.write(os.path.join(CORE, rel[5:]) if rel.startswith("core/") else os.path.join(gen_dir, rel[6:]),
                            "include/" + os.path.basename(rel))
            z.writestr("lib/BUILD_FLAGS.txt",
                       f"libtinyml.a:        arm-none-eabi-gcc {lib['flags']} {lib['defines']}\n"
                       + (f"libtinyml_cmsis.a:  arm-none-eabi-gcc {lib_cmsis['flags']} {lib_cmsis['defines']}\n"
                          if lib_cmsis and "path" in lib_cmsis else "")
                       + "浮点 ABI 是 softfp，跟 GR551x SDK 的 libble_sdk.a 一致；工程用 hard 的话别用这些 .a，拿源码重编。\n"
                       "两个 .a 二选一：libtinyml.a 是逐位一致的朴素实现；libtinyml_cmsis.a 把热点换成了 CMSIS 内核"
                       "（CMSIS 的 .o 已经打在里面，不用再链 CMSIS）。接口、头文件完全一样。\n")
        extras = []
        for extra in ("board/README.md", "docs/ram_and_cache.md"):
            p = os.path.join(ROOT, extra)
            if os.path.exists(p):
                z.write(p, extra)
                extras.append(extra)
        # PC 参考实现：Linux 上 make 一下就能命令行跑，嵌入式拿它对板上结果
        geo = bundle_geometry(gen_dir, kind)
        z.write(os.path.join(BUNDLE_SRC, "edge_cli.c"), "pc/edge_cli.c")
        z.writestr("pc/tm_edge_cfg.h", _edge_cfg_h(geo))
        z.writestr("pc/sample.csv", sample_csv(geo["hz"], geo["n_ch"] - 2))
        mk = open(os.path.join(BUNDLE_SRC, "Makefile"), encoding="utf-8").read()
        z.writestr("Makefile", mk.replace("@KIND@", kind).replace("@N_T@", str(max(geo["n_t"], 16)))
                   .replace("@N_CLASSES@", str(max(geo["n_classes"], 2))))
        z.writestr("README.md", _fill_file_lists(readme, files, extras, has_lib=bool(lib and "path" in lib),
                                                  zip_name=os.path.basename(out_zip),
                                                  has_cmsis=any(n.startswith("third_party/") for n in z.namelist())))
    return os.path.getsize(out_zip)


def _define(path: str, name: str) -> int | None:
    if not os.path.exists(path):
        return None
    m = re.search(rf"#define\s+{name}\s+(\d+)", open(path, encoding="utf-8").read())
    return int(m.group(1)) if m else None


def bundle_geometry(gen_dir: str, kind: str) -> dict:
    """窗口几何：以导出的头文件为准（编进板子的就是它），meta.json 补采样率和步长。"""
    import json

    meta = {}
    mp = os.path.join(gen_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    if kind == "rf":
        n_ch = _define(os.path.join(gen_dir, "tm_feat_cfg.h"), "TM_FEAT_N_CH")
        n_t = _define(os.path.join(gen_dir, "tm_feat_cfg.h"), "TM_FEAT_N_T")
        n_cls = _define(os.path.join(gen_dir, "tm_forest_c_model.h"), "TM_FC_N_CLASSES")
    else:
        n_ch = _define(os.path.join(gen_dir, "tm_model.h"), "TM_N_CH")
        n_t = _define(os.path.join(gen_dir, "tm_model.h"), "TM_N_T")
        n_cls = _define(os.path.join(gen_dir, "tm_model.h"), "TM_N_CLASSES")
    n_t = n_t or int(meta.get("window_size") or 16)
    return {
        "n_ch": n_ch or int(meta.get("n_channels") or 8),
        "n_t": n_t,
        "n_classes": n_cls or len(meta.get("classes") or []) or 2,
        "hz": int(meta.get("hz") or 16),
        "hop": int(meta.get("stride") or max(n_t // 2, 1)),
    }


def _edge_cfg_h(geo: dict) -> str:
    return ("/* 自动生成：这一版模型的采样几何，pc/edge_cli.c 用。 */\n"
            "#ifndef TM_EDGE_CFG_H\n#define TM_EDGE_CFG_H\n"
            f"#define TM_EDGE_HZ {geo['hz']}        /* 模型采样率 */\n"
            f"#define TM_EDGE_HOP {geo['hop']}       /* 窗口步长（点），跟训练一致 */\n"
            f"#define TM_EDGE_N_SENSOR {geo['n_ch'] - 2}  /* IMU 轴数（模型通道数 - pitch/roll 两路） */\n"
            "#endif\n")


def sample_csv(hz: int, n_sensor: int, seconds: int = 60, seed: int = 0) -> str:
    """合成的示例 CSV（g / °/s）：静止趴着 → 走动 → 高频抖动。**不是真数据**，只用来验证能跑通。"""
    import math
    import random

    rnd = random.Random(seed)
    cols = ["timestamp", "acc_x", "acc_y", "acc_z"] + (["gyro_x", "gyro_y", "gyro_z"] if n_sensor == 6 else [])
    rows = [",".join(cols)]
    n = seconds * hz
    for i in range(n):
        t = i / hz
        phase = 3 * i // n
        if phase == 0:      # 静止：项圈微微倾斜
            a = [0.17, -0.05, 0.98]
            g = [0.0, 0.0, 0.0]
            na, ng = 0.01, 0.5
        elif phase == 1:    # 走动：~2 Hz 步频
            w = 2 * math.pi * 2.0 * t
            a = [0.17 + 0.25 * math.sin(w), -0.05 + 0.1 * math.sin(w / 2), 0.98 + 0.3 * math.cos(w)]
            g = [20 * math.sin(w), 10 * math.cos(w), 15 * math.sin(w / 2)]
            na, ng = 0.05, 5.0
        else:               # 抖动：~6 Hz 大幅
            w = 2 * math.pi * 6.0 * t
            a = [0.17 + 0.8 * math.sin(w), -0.05 + 0.6 * math.cos(w), 0.98 + 0.5 * math.sin(2 * w)]
            g = [150 * math.sin(w), 120 * math.cos(w), 80 * math.sin(w)]
            na, ng = 0.1, 15.0
        v = [x + rnd.gauss(0, na) for x in a] + ([x + rnd.gauss(0, ng) for x in g] if n_sensor == 6 else [])
        rows.append(f"{t:.4f}," + ",".join(f"{x:.5f}" for x in v))
    return "\n".join(rows) + "\n"


_FILE_NOTES = {
    "tm_features.c": "193 维（3 轴 79 维）手工特征：FFT / Welch / 时域统计",
    "tm_forest_c.c": "紧凑随机森林推理（整数累加，板上和 PC 逐位一致）",
    "tm_prep.c": "CNN 输入：逐通道 z-score + int8 量化",
    "tm_runtime.c": "int8 推理（conv1d / maxpool / dense）",
    "tm_window.c": "CNN 用的 int8 环形窗口（RF 用 tm_imu 的流式接口即可）",
    "tm_post.c": "可选：板上后处理（逐窗口判决 → 事件）",
    "tm_post_cfg.c": "后处理参数",
    "tm_accel.h": "可选加速（CMSIS）开关，默认关",
    "tm_feat_cfg.c": "特征常量表（Hann 窗 / FFT 旋转因子 / 位反序）",
    "tm_forest_c_model.c": "森林：节点表 + uint8 叶子",
    "tm_model.c": "CNN：int8 权重 + 归一化参数",
    "tm_forest_c_pipeline_golden.h": "golden：映射好的输入窗口 → 整数票数（自检用，不进量产固件）",
    "tm_forest_c_golden.h": "golden：特征 → 票数（自检用）",
    "tm_golden.h": "golden：int8 输入 → int8 输出（自检用）",
    "meta.json": "训练信息：类别、窗口、各类 F1",
}


def _fill_file_lists(readme: str, files: dict, extras: list, has_lib: bool, zip_name: str,
                     has_cmsis: bool = False) -> str:
    def lines(prefix):
        out, seen = [], set()
        for rel in files:
            if not rel.startswith(prefix) or rel == "core/tm_imu.c" or rel == "core/tm_imu.h":
                continue
            base = os.path.basename(rel)
            stem = base[:-2] if base.endswith((".c", ".h")) and base != "tm_accel.h" and not base.endswith("golden.h") else base
            if stem in seen:
                continue
            seen.add(stem)
            both = base.endswith((".c", ".h")) and stem != base
            name = f"{stem}.c/h" if both and f"{prefix}{stem}.h" in files and f"{prefix}{stem}.c" in files else base
            note = _FILE_NOTES.get(stem + ".c", _FILE_NOTES.get(base, ""))
            out.append(f"  {name:<16} {note}")
        return "\n".join(out)

    extra = ""
    if has_lib:
        extra += "lib/           预编静态库（Cortex-M4F softfp），见 lib/BUILD_FLAGS.txt\ninclude/       只链 .a 时用的头文件\n"
    if has_cmsis:
        extra += "third_party/   可选加速用的 CMSIS 子集（默认不用，见「可选加速」）\n"
    extra += "".join(f"{e:<22} 参考\n" for e in extras)
    return (readme.replace("@CORE_LIST@", lines("core/")).replace("@MODEL_LIST@", lines("model/"))
            .replace("@EXTRA_LIST@", extra).replace("@ZIP_NAME@", zip_name)
            .replace("@ZIP_STEM@", os.path.splitext(zip_name)[0]))


def _accel_readme(kind: str, fp: dict) -> str:
    a = fp.get("accel")
    if not a:
        return ""
    sw = "-DTM_CMSIS_DSP=1" if kind == "rf" else "-DTM_CMSIS_NN=1"
    srcs = ", ".join(os.path.basename(p) for p in cmsis.sources(kind))
    sub = "dsp" if kind == "rf" else "nn"
    inc = (f"third_party/cmsis/{sub}/Include third_party/cmsis/{sub}/PrivateInclude（PC 上编加 -D__GNUC_PYTHON__）"
           if kind == "rf" else f"third_party/cmsis/{sub}/Include")
    return (
        f"可选加速（{a['name']}，默认关）：\n"
        f"  接法 A 加 {sw}，并把 third_party/cmsis/{sub}/Source/ 里的 {srcs} 一起编，-I {inc}；\n"
        f"  交叉编译还要 CMSIS-Core 的 cmsis_compiler.h（SDK 自带，third_party/cmsis/core/Include 也有一份）。\n"
        f"  接法 B 直接换链 lib/libtinyml_cmsis.a（CMSIS 的 .o 已打在里面），头文件不变。\n"
        f"  {a['note']}。\n"
        f"  占用变化：flash {a['flash']['delta']:+,} B，RAM {a['ram']['delta']:+,} B"
        + (f"；x86 上每窗 {a['host_us_per_window']} µs" if a.get("host_us_per_window") is not None else "")
        + "。\n"
        + ("  M4F 上 CMSIS-NN 用 SMLAD 一次算两对乘加，卷积大致快 2～4 倍；" if kind != "rf" else
           "  M4F 上 CMSIS 的 FFT 是基-8 + 循环展开，特征那一段大致快 2～3 倍；")
        + "具体快多少要板上量。\n\n"
    )


def bundle_readme(kind: str, meta: dict, fp: dict) -> str:
    """包里的 README.md：Linux 上怎么编、怎么命令行测、8 通道怎么来、板上怎么调、占多少。
    文件清单那几处（@CORE_LIST@ 等）由 write_bundle 按实际打进去的文件填。"""
    fl, rm, inf = fp["flash"], fp["ram"], fp["inference"]
    classes = list(meta.get("classes") or [])
    n_ch = int(meta.get("n_channels") or 8)
    n_sensor = n_ch - 2
    n_t = int(meta.get("window_size") or 16)
    hz = int(meta.get("hz") or 16)
    hop = int(meta.get("stride") or max(n_t // 2, 1))
    six = n_sensor == 6
    api = (
        "```c\n"
        "#include \"tm_imu.h\"\n#include \"tm_feat_cfg.h\"\n#include \"tm_forest_c_model.h\"\n\n"
        f"static float x[{n_ch} * {n_t}], feats[TM_FEAT_DIM];\nint32_t votes[TM_FC_N_CLASSES];\n\n"
        f"if (tm_imu_push(&s, sample, x)) {{                 /* 1. 样本 → {n_ch} 通道窗口（见上一节） */\n"
        "    tm_features(&tm_feat_cfg, x, feats);          /* 2. 窗口 → 特征 float[TM_FEAT_DIM] */\n"
        "    int cls = tm_forest_c_predict(&tm_forest_c, feats, votes);  /* 3. → 类别下标；votes 是每类整数票数 */\n"
        "    /* 4. 可选：tm_post_on_window(...) 板上后处理（稳定版 v2），逐窗口判决聚成事件 */\n"
        "}\n```"
        if kind == "rf" else
        "```c\n"
        "#include \"tm_imu.h\"\n#include \"tm_model.h\"\n\n"
        f"static float x[{n_ch} * {n_t}];\nstatic int8_t xi[{n_ch} * {n_t}], arena[TM_ARENA_BYTES];\nint8_t out[TM_N_CLASSES];\n\n"
        f"if (tm_imu_push(&s, sample, x)) {{                 /* 1. 样本 → {n_ch} 通道窗口（见上一节） */\n"
        "    tm_prep(&tm_model_prep, x, xi);               /* 2. 逐通道 z-score + 量化成 int8 */\n"
        "    tm_invoke(&tm_model, xi, out, arena, TM_ARENA_BYTES);  /* 3. → int8 分数 */\n"
        "    int cls = tm_argmax(out, TM_N_CLASSES);\n"
        "    /* 4. 可选：tm_post_on_window(...) 板上后处理 */\n"
        "}\n```"
    )
    ch_rows = ["| 0, 1, 2 | 重力对齐后的 acc_x, acc_y, acc_z（g） |"]
    if six:
        ch_rows.append("| 3, 4, 5 | 重力对齐后的 gyro_x, gyro_y, gyro_z（°/s，跟 acc 用同一个 R） |")
    ch_rows.append(f"| {n_ch - 2} | pitch（弧度，对齐**前**算） |")
    ch_rows.append(f"| {n_ch - 1} | roll（弧度，对齐**前**算） |")
    footprint = (
        f"| 项 | 字节 |\n|---|---|\n"
        f"| flash：模型 | {fl['model']:,} |\n| flash：常量表 | {fl['tables']:,} |\n"
        f"| flash：工程代码 | {fl['runtime']:,} |\n"
        f"| **flash 合计**（不含 golden） | **{fl['total_without_golden']:,}（{fl['total_without_golden'] / 1024:.1f} KB）** |\n"
        f"| flash：自检 golden（量产可只留几条） | {fl['golden']:,} |\n"
        f"| RAM：运行时 | {rm['runtime_bss']:,} |\n| RAM：窗口缓冲 | {rm['window_buffer']:,} |\n"
        + (f"| RAM：arena | {rm['arena']:,} |\n" if rm.get("arena") else "")
        + f"| **RAM 合计** | **{rm['total_without_post']:,}** |\n"
        f"| RAM：用板上后处理另加 | {rm['post_state']:,} |\n\n"
        f"推理：{inf.get('note', '')}\n"
    )
    score_note = ("RF 是各类票数占比（叶子 uint8 求和后归一化），判类别看 argmax。" if kind == "rf" else
                  "CNN 的 int8 分数反量化后做 softmax，只为给个 0~1 的数；判类别看 int8 的 argmax。")
    rep = {
        "@TAG@": str((meta.get("train") or {}).get("tag") or "@ZIP_STEM@"),
        "@KIND_NAME@": "随机森林（rf）" if kind == "rf" else "1D-CNN int8（cnn）",
        "@N_CH@": str(n_ch), "@N_T@": str(n_t), "@HZ@": str(hz), "@HOP@": str(hop),
        "@N_SENSOR@": str(n_sensor),
        "@WIN_S@": f"{n_t / hz:g}", "@HOP_S@": f"{hop / hz:g}",
        "@CLASSES@": "  ".join(f"{i}={c}" for i, c in enumerate(classes)),
        "@GYR_COLS@": ",`gyro_x,gyro_y,gyro_z`" if six else "",
        "@GYR_NOHDR@": ",gx,gy,gz" if six else "",
        "@GYR_SAMPLE@": " gyro_x gyro_y gyro_z" if six else "",
        "@GYR_UNIT@": "、角速度 **°/s**" if six else "",
        "@GYR_ROT@": " 和 gyro" if six else "",
        "@GYR_SAMPLE_C@": ", gx_dps, gy_dps, gz_dps" if six else "",
        "@CH_TABLE@": "\n".join(ch_rows),
        "@API@": api,
        "@SCORE_NOTE@": score_note,
        "@ACCEL@": _accel_readme(kind, fp).replace("可选加速（", "### 可选加速（", 1).replace("默认关）：\n", "默认关）\n\n", 1),
        "@TOOLCHAIN@": fp.get("toolchain", ""),
        "@FOOTPRINT@": footprint,
    }
    s = open(os.path.join(BUNDLE_SRC, "README.template.md"), encoding="utf-8").read()
    for k, v in rep.items():
        s = s.replace(k, v)
    return s
