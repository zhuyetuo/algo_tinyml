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


def _strip_lib(path: str) -> None:
    """去掉调试信息和用不到的本地符号（静态函数名、文件名之类），重定位要的留着，链接不受影响。"""
    strip = shutil.which("arm-none-eabi-strip")
    if strip:
        subprocess.run([strip, "--strip-debug", "--strip-unneeded", path], check=True, capture_output=True)


def _host_edge_cli(gen_dir: str, kind: str, geo: dict, out_path: str) -> bool:
    """PC 上的 edge_cli 可执行文件（x86_64 Linux，静态链接、去符号）：只给编译好的东西，
    对方照样能在 Linux 上命令行跑 / 自检，拿它对板上结果。编不出来就不给。"""
    cc = shutil.which("gcc")
    if not cc:
        return False
    n_t = max(geo["n_t"], 16)
    srcs = [os.path.join(CORE, f) for f in RUNTIME[kind]] + \
           [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "tm_edge_cfg.h"), "w", encoding="utf-8") as f:
            f.write(_edge_cfg_h(geo))
        base = [cc, "-O2", "-std=c99", "-ffp-contract=off", "-fno-math-errno", "-s",
                f"-DTM_FEAT_MAX_T={n_t}", f"-DTM_FEAT_MAX_NPERSEG={n_t}",
                f"-DTM_POST_MAX_CLASSES={max(geo['n_classes'], 2)}", "-DTM_KIND_RF" if kind == "rf" else "-DTM_KIND_CNN",
                f"-I{CORE}", f"-I{gen_dir}", f"-I{tmp}", os.path.join(BUNDLE_SRC, "edge_cli.c"), *srcs,
                "-o", out_path, "-lm"]
        for extra in (["-static"], []):   # 静态的哪台 Linux 都能跑；没有静态 libc 就退回动态
            if subprocess.run(base[:1] + extra + base[1:], capture_output=True).returncode == 0:
                return True
    return False


def _expected_outputs(cli: str, sample: str, tmp: str) -> dict | None:
    """PC 验证工具在打包时实际跑出来的结果：写进 README 给对方对照，sample 的逐窗口结果另存一份。"""
    p = os.path.join(tmp, "sample.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write(sample)
    st = subprocess.run([cli, "--selftest"], capture_output=True, text=True)
    run = subprocess.run([cli, p], capture_output=True, text=True)
    if st.returncode != 0 or run.returncode != 0:
        raise RuntimeError("打包时 PC 验证工具自己都没跑通：" + (st.stdout + st.stderr + run.stderr)[-400:])
    summary = [ln for ln in run.stderr.splitlines() if not ln.startswith("每窗")]   # 耗时每台机器不同
    return {"selftest": st.stdout.strip(), "csv": run.stdout, "summary": "\n".join(summary).strip()}


def _board_selftest_check(gen_dir: str, kind: str, geo: dict, lib: dict, tmp: str) -> str:
    """example/board_selftest.c 链上打出来的 libtinyml.a 能编过（ARM），并且在 PC 上跑出来全通过。
    返回它的正确输出（README 里给对方看）。"""
    src = os.path.join(BUNDLE_SRC, "board_selftest.c")
    k = "-DTM_KIND_RF" if kind == "rf" else "-DTM_KIND_CNN"
    defs = lib["defines"].split()
    cc, _, arm = toolchain()
    r = subprocess.run([cc, "-c", *ARM_FLAGS, *defs, k, f"-I{CORE}", f"-I{gen_dir}", src,
                        "-o", os.path.join(tmp, "board_selftest.o")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("board_selftest.c 用 ARM 编不过：" + r.stderr[-400:])
    host = shutil.which("gcc")
    srcs = [os.path.join(CORE, f) for f in RUNTIME[kind]] + \
           [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]
    exe = os.path.join(tmp, "board_selftest")
    r = subprocess.run([host, *HOST_FLAGS, *defs, k, "-DTM_SELFTEST_MAIN", f"-I{CORE}", f"-I{gen_dir}", src, *srcs,
                        "-o", exe, "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("board_selftest.c 在 PC 上编不过：" + r.stderr[-400:])
    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("board_selftest 在 PC 上没通过：" + out.stdout[-400:])
    return out.stdout.strip()


def lib_readme(kind: str, meta: dict, fp: dict, has_cli: bool, has_cmsis: bool, zip_name: str = "",
               expected: dict | None = None) -> str:
    """编译库包的 README：从源码包 README 里借「输入映射」「资源占用」两节，其余按只给库重写。"""
    full = bundle_readme(kind, meta, fp)
    head = full.split("这个包是", 1)[0].rstrip()
    parts = re.split(r"(?m)^## ", full)
    sec = {p[:2]: "## " + p for p in parts[1:]}
    m2 = sec.get("2.", "")
    imu_code = re.search(r"```c\n.*?```", m2, re.S)
    mapping = (m2.split("\n", 1)[0].replace("## 2.", "## 3.") + "\n\n"
               "原始 IMU 样本（单位 g" + ("、°/s" if "°/s" in m2 else "") + "）直接喂 `tm_imu_push`，"
               "库内完成坐标变换，凑满一个窗口吐出模型输入。**不要自己预处理**，否则模型照样出类别，只是准确率莫名其妙地掉。\n\n"
               + (imu_code.group(0) if imu_code else ""))
    api = re.search(r"```c\n.*?```", sec.get("3.", ""), re.S)
    n_t = int(meta.get("window_size") or 16)
    n_cls = max(len(meta.get("classes") or []), 2)
    defs = f"-DTM_FEAT_MAX_T={n_t} -DTM_FEAT_MAX_NPERSEG={n_t} -DTM_POST_MAX_CLASSES={n_cls}"
    files = ("```\n"
             "lib/libtinyml.a        静态库（运行时 + 这一版模型），Cortex-M4F softfp\n"
             + ("lib/libtinyml_cmsis.a  同上，热点换成 CMSIS 内核（二选一，接口一样）\n" if has_cmsis else "")
             + "lib/BUILD_FLAGS.txt    库的编译选项\n"
             "include/               头文件（*golden.h 只给板上自检用）\n"
             "example/board_selftest.c  板上自检示例\n"
             + ("pc/edge_cli            Linux x86_64 命令行工具（自检 + 跑 CSV）\n"
                "pc/sample.csv          合成示例数据，只用来验证能跑通\n"
                "pc/sample_result.csv   sample.csv 的正确结果\n" if has_cli else "")
             + "```\n")
    ex = expected or {}
    kind_def = "-DTM_KIND_RF" if kind == "rf" else "-DTM_KIND_CNN"
    verify = ""
    if has_cli:
        head = "\n".join(ex.get("csv", "").splitlines()[:6])
        verify = ("## 1. 验证\n\n"
                  "两步：PC 上确认工具和数据没问题，板上确认库在你的工程里算得对。结果和下面示例一致即为正确。\n\n"
                  "### 1.1 PC 上（Linux x86_64）\n\n"
                  "```bash\n./pc/edge_cli --selftest\n```\n\n正确结果：\n\n"
                  f"```\n{ex.get('selftest', 'selftest: N/N 通过')}\n```\n\n"
                  "```bash\n./pc/edge_cli pc/sample.csv --out result.csv\n```\n\n"
                  "正确结果：屏幕（stderr）上的汇总应为\n\n"
                  f"```\n{ex.get('summary', '')}\n```\n\n"
                  "`result.csv` 应与包里的 `pc/sample_result.csv` 完全相同（`diff result.csv pc/sample_result.csv` 无输出），开头几行：\n\n"
                  f"```\n{head}\n```\n\n"
                  "每行一个窗口：序号、起止时间（秒）、类别下标、类别、置信度、各类概率。"
                  "`sample.csv` 是合成数据，只用于确认能跑通，类别本身没有意义。\n\n"
                  "自己的采集数据：`./pc/edge_cli 采集.csv --in-hz 50`（单位 g / °/s，`--help` 看全部参数），"
                  "板上同一段数据的逐窗口判决应与之一致。\n\n")
    verify += ("### 1.2 板上\n\n"
               "把 `example/board_selftest.c` 加进工程（编译选项同第 2 节，另加 `" + kind_def + "`），"
               "printf 重定向到串口，开机调用一次：\n\n"
               "```c\nint tm_board_selftest(void);   /* 返回 0 = 通过 */\n\nif (tm_board_selftest() != 0) {\n"
               "    /* 不通过：多半是编译选项不一致（-ffp-contract=off、浮点 ABI、上面的 -D 宏） */\n}\n```\n\n"
               "正确结果（串口）：\n\n"
               f"```\n{ex.get('board', 'selftest: N/N 通过')}\n```\n\n"
               "不通过时会逐条打出 `golden #i 不一致：本板/期望`，把串口输出发给算法侧即可。"
               "自检用的 golden 数据约占几 KB flash，量产固件可不带 `board_selftest.c`。\n\n")
    link = ("## 2. 接入工程\n\n"
            "1. `include/` 加进头文件路径，链 `lib/libtinyml.a` 和 `-lm`。\n"
            f"2. 包含这些头文件的源文件编译时加：`{defs}`（与库一致，否则结构体大小对不上）。\n"
            "3. 库是 Cortex-M4F + `-mfloat-abi=softfp`（跟 GR551x SDK 一致），hard ABI 的工程链不上。\n"
            "4. 建议链接加 `-Wl,--gc-sections`，没用到的函数会丢掉。\n"
            "5. 无 malloc、不依赖 OS，裸机或 RTOS 任务里都能调；不可重入，多任务调用请自行加锁。\n\n"
            "调用顺序：\n\n@API@\n\n")
    # 推理那行写了特征怎么算，保密包不带
    foot = "\n".join(ln for ln in sec.get("4.", "").split("\n") if not ln.startswith(("推理：", "| flash：自检 golden")))
    head = head.replace("@ZIP_STEM@", os.path.splitext(zip_name)[0])
    api_txt = api.group(0).replace("（见上一节）", "（见第 3 节）") if api else ""
    link = link.replace("@API@", api_txt)
    return f"{head}\n\n本包为编译好的静态库，不含源码。\n\n{files}\n{verify}{link}{mapping}\n\n{foot}"


def write_lib_bundle(gen_dir: str, kind: str, out_zip: str, meta: dict, fp: dict,
                     window: int, n_classes: int) -> int:
    """编译库包（保密）：只给预编的 .a + 头文件 + PC 验证工具，不含任何 .c。返回字节数。
    没有交叉编译器编不出 .a 就抛错——这个包没有 .a 就没意义。"""
    import zipfile

    files = source_bundle_bytes(gen_dir, kind)["files"]
    geo = bundle_geometry(gen_dir, kind)
    with tempfile.TemporaryDirectory() as tmp:
        lib = build_static_lib(gen_dir, kind, window, n_classes, tmp)
        if not lib or "path" not in lib:
            raise RuntimeError("编不出 libtinyml.a：" + ((lib or {}).get("error") or "没装 arm-none-eabi-gcc"))
        lib_cmsis = build_static_lib(gen_dir, kind, window, n_classes, tmp, use_cmsis=True)
        if lib_cmsis and "path" not in lib_cmsis:
            lib_cmsis = None
        cli = os.path.join(tmp, "edge_cli")
        has_cli = _host_edge_cli(gen_dir, kind, geo, cli)
        sample = sample_csv(geo["hz"], geo["n_ch"] - 2)
        board = _board_selftest_check(gen_dir, kind, geo, lib, tmp)
        expected = _expected_outputs(cli, sample, tmp) if has_cli else None
        expected = dict(expected or {}, board=board)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for name, l in (("libtinyml.a", lib), ("libtinyml_cmsis.a", lib_cmsis)):
                if l:
                    _strip_lib(l["path"])
                    z.write(l["path"], "lib/" + name)
            z.writestr("lib/BUILD_FLAGS.txt",
                       f"arm-none-eabi-gcc {lib['flags']} {lib['defines']}\n"
                       + (f"libtinyml_cmsis.a 另加：{lib_cmsis['defines']}\n" if lib_cmsis else ""))
            for rel in files:
                # golden 头文件也给：只是测试输入和标准答案，板上自检要用
                if rel.endswith(".h") and (not rel.endswith("golden.h") or rel.endswith(GOLDEN_HEADERS[kind][-1])):
                    src = os.path.join(CORE, rel[5:]) if rel.startswith("core/") else os.path.join(gen_dir, rel[6:])
                    z.write(src, "include/" + os.path.basename(rel))
            z.write(os.path.join(BUNDLE_SRC, "board_selftest.c"), "example/board_selftest.c")
            if expected.get("csv"):
                z.writestr("pc/sample_result.csv", expected["csv"])
            if has_cli:
                info = zipfile.ZipInfo("pc/edge_cli")
                info.external_attr = 0o755 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, open(cli, "rb").read())
                z.writestr("pc/sample.csv", sample)
            z.writestr("README.md", lib_readme(kind, meta, fp, has_cli, bool(lib_cmsis), os.path.basename(out_zip),
                                               expected))
    return os.path.getsize(out_zip)


# ── pet_behavior：按接口协议交付的编译库（三轴加速度模型才有） ──────────────────────────
PB_SRC = os.path.join(BUNDLE_SRC, "pb")
PB_LIB_VERSION = 1
# 协议类别（pet_behavior.h 的 PB_TYPE_*）。模型类别名按关键字归过去，归不上的算「其他」
PB_TYPES = ["休息", "进食", "喝水", "舔舐", "抓挠", "吠叫", "其他", "未佩戴"]
_PB_RULES = [(7, ("未佩戴", "没戴", "摘下")), (4, ("抓", "挠")), (3, ("舔",)), (2, ("喝", "饮水")),
             (1, ("吃", "进食", "食")), (5, ("吠", "叫")), (0, ("休息", "睡", "静止", "趴", "卧", "躺"))]


def pb_type_of(name: str) -> int:
    for code, keys in _PB_RULES:
        if any(k in name for k in keys):
            return code
    return 6


def pb_model_version(meta: dict) -> int:
    tag = str((meta.get("train") or {}).get("tag") or "")
    m = re.search(r"(\d+)$", tag)
    return int(m.group(1)) & 0xFFFF if m else 0


def _pb_model_cfg_h(kind: str, classes: list, hz: int, hop: int, model_version: int) -> str:
    names = ", ".join('"' + c.replace('"', "") + '"' for c in classes)
    types = ", ".join(str(pb_type_of(c)) for c in classes)
    return ("/* 自动生成：pet_behavior 的模型参数（只参与编译，不随包发布） */\n"
            "#ifndef PB_MODEL_CFG_H\n#define PB_MODEL_CFG_H\n#include <stdint.h>\n"
            f"#define {'PB_KIND_RF' if kind == 'rf' else 'PB_KIND_CNN'} 1\n"
            f"#define PB_MODEL_VERSION {model_version}u\n#define PB_HZ {hz}\n#define PB_HOP {hop}\n"
            f"static const uint8_t PB_TYPE_OF_CLASS[] = {{{types}}};\n"
            f"static const char *const PB_CLASS_NAMES[] = {{{names}}};\n#endif\n")


def _pb_header(work_size: int, hz: int, header_version: int) -> str:
    h = open(os.path.join(PB_SRC, "pet_behavior.h"), encoding="utf-8").read()
    return (h.replace("@WORK_SIZE@", str(work_size)).replace("@HZ@", str(hz))
            .replace("@HEADER_VERSION@", f"0x{header_version:08X}"))


def pb_test_input(hz: int, acc_range_g: int = 8, seconds: int = 180, batch: int | None = None) -> bytes:
    """验收输入 test_input.bin（格式见 example.c）：合成数据，换成 int16 原始计数。"""
    import struct

    batch = batch or min(hz * 10, 256)
    rows = []
    for seed in range(1, 4):   # 静止 → 走动 → 抖动，换种子来三轮，片段边界多一些
        rows += sample_csv(hz, 3, seconds=seconds // 3, seed=seed).splitlines()[1:]
    lsb = acc_range_g / 32768.0
    samples = []
    for ln in rows:
        v = [float(x) for x in ln.split(",")[1:4]]
        samples.append([max(-32768, min(32767, int(round(a / lsb)))) for a in v])
    out = b"PBIN" + struct.pack("<HBBHHI", hz, acc_range_g, 0, batch, 0, len(samples))
    return out + b"".join(struct.pack("<hhh", *s) for s in samples)


def write_pb_bundle(gen_dir: str, kind: str, out_zip: str, meta: dict, fp: dict) -> int:
    """按《宠物行为识别库 · 接口协议》交付：pet_behavior.h + libpetbehavior.a + example.c + 验收数据。
    不含源码。只支持三轴加速度（5 通道）模型。返回字节数。"""
    import zipfile

    geo = bundle_geometry(gen_dir, kind)
    if geo["n_ch"] != 5:
        raise ValueError(f"这一版模型是 {geo['n_ch'] - 2} 轴输入，pet_behavior 只支持三轴加速度模型")
    cc, _, arm = toolchain()
    ar, nm, strip = (shutil.which(f"arm-none-eabi-{t}") for t in ("ar", "nm", "strip"))
    host_cc = shutil.which("gcc")
    if not (arm and ar and nm and host_cc):
        raise RuntimeError("打 pet_behavior 库要 arm-none-eabi-gcc/ar/nm 和 gcc")
    classes = list(meta.get("classes") or [])
    n_t, hz, hop, n_cls = geo["n_t"], geo["hz"], geo["hop"], max(geo["n_classes"], 2)
    mv = pb_model_version(meta)
    hver = (PB_LIB_VERSION << 16) | mv
    defines = [f"-DTM_FEAT_MAX_T={max(n_t, 16)}", f"-DTM_FEAT_MAX_NPERSEG={max(n_t, 16)}",
               f"-DTM_POST_MAX_CLASSES={n_cls}"]
    srcs = [os.path.join(CORE, f) for f in RUNTIME[kind]] + \
           [os.path.join(gen_dir, f) for f in MODEL_FILES[kind] + TABLE_FILES[kind] if os.path.exists(os.path.join(gen_dir, f))]

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(" ".join(os.path.basename(c) for c in cmd[:1]) + " 失败：" + r.stderr.strip()[-600:])
        return r.stdout

    with tempfile.TemporaryDirectory() as tmp:
        inc = os.path.join(tmp, "inc")
        os.makedirs(inc)
        with open(os.path.join(inc, "pb_model_cfg.h"), "w", encoding="utf-8") as f:
            f.write(_pb_model_cfg_h(kind, classes, hz, hop, mv))
        # .c 拷到生成的头文件旁边编：#include "pet_behavior.h" 先找同目录，原地编会拿到带占位符的模板
        pb_c = shutil.copy(os.path.join(PB_SRC, "pet_behavior.c"), inc)
        incs = [f"-I{inc}", f"-I{CORE}", f"-I{gen_dir}"]

        # 1. 工作内存多大：ARM 和 x86 各量一次（x86 指针 8 字节，结构体大一点），头文件取大的
        sizes = []
        for comp, flags in ((cc, ARM_FLAGS), (host_cc, HOST_FLAGS)):
            with open(os.path.join(inc, "pet_behavior.h"), "w", encoding="utf-8") as f:
                f.write(_pb_header(0, hz, hver))
            obj = os.path.join(tmp, "probe.o")
            run([comp, "-c", *flags, *defines, "-DPB_SIZE_PROBE", *incs, pb_c, "-o", obj])
            out = run([nm if comp == cc else "nm", "-S", obj])
            m = re.search(r"^\S+\s+([0-9a-fA-F]+)\s+\w\s+pb_state_size_probe$", out, re.M)
            if not m:
                raise RuntimeError("量不出 pb_state_t 的大小")
            sizes.append(int(m.group(1), 16))
        work = (max(sizes) + 7) // 8 * 8
        header = _pb_header(work, hz, hver)
        with open(os.path.join(inc, "pet_behavior.h"), "w", encoding="utf-8") as f:
            f.write(header)

        # 2. 板上的 libpetbehavior.a 和 PC 上的 libpetbehavior_host.a
        def build(comp, flags, arch, ar_tool, out_lib):
            objs = []
            for src in srcs + [pb_c]:
                o = os.path.join(tmp, f"{arch}_{os.path.basename(src)}.o")
                run([comp, "-c", *flags, *defines, *incs, src, "-o", o])
                objs.append(o)
            run([ar_tool, "rcs", out_lib, *objs])
            return out_lib

        lib_arm = build(cc, ARM_FLAGS, "arm", ar, os.path.join(tmp, "libpetbehavior.a"))
        if strip:
            run([strip, "--strip-debug", "--strip-unneeded", lib_arm])
        lib_host = build(host_cc, HOST_FLAGS, "x86", shutil.which("ar") or "ar", os.path.join(tmp, "libpetbehavior_host.a"))
        run(["strip", "--strip-debug", "--strip-unneeded", lib_host]) if shutil.which("strip") else None
        # 板上实际占用：链一个只调接口的小程序，-gc-sections 之后量
        size_tool = shutil.which("arm-none-eabi-size")
        flash = ram_static = None
        if size_tool:
            probe_c = os.path.join(tmp, "use.c")
            with open(probe_c, "w") as f:
                f.write('#include "pet_behavior.h"\nstatic pb_segment_t s[8];\n'
                        'int main(void){pb_config_t c={PB_SAMPLE_RATE_HZ,8,0};static unsigned char m[PB_WORK_SIZE];'
                        'pb_init(&c,m,sizeof m);uint8_t t,q;pb_current(&t,&q);return pb_push(0,0,0,s,8)+pb_flush(s,8)+(int)pb_version();}\n')
            elf = os.path.join(tmp, "use.elf")
            r = subprocess.run([cc, *ARM_FLAGS, f"-I{inc}", probe_c, lib_arm, "-lm", "--specs=nosys.specs",
                                "-Wl,--gc-sections", "-o", elf], capture_output=True, text=True)
            base = os.path.join(tmp, "base.elf")
            with open(os.path.join(tmp, "base.c"), "w") as f:
                f.write("int main(void){return 0;}\n")
            r2 = subprocess.run([cc, *ARM_FLAGS, os.path.join(tmp, "base.c"), "-lm", "--specs=nosys.specs",
                                 "-Wl,--gc-sections", "-o", base], capture_output=True, text=True)
            if r.returncode == 0 and r2.returncode == 0:
                def sz(e):
                    t, d, b = map(int, run([size_tool, e]).splitlines()[1].split()[:3])
                    return t + d, d + b
                (f1, r1), (f0, r0) = sz(elf), sz(base)
                flash, ram_static = f1 - f0, r1 - r0 - work   # 去掉 probe 里的 m[PB_WORK_SIZE]

        # 3. PC 上的示例程序 + 验收数据
        ex = shutil.copy(os.path.join(PB_SRC, "example.c"), inc)
        exe = os.path.join(tmp, "pb_example")
        built = False
        for extra in (["-static"], []):
            if subprocess.run([host_cc, "-O2", "-s", *extra, "-DPB_EXAMPLE_MAIN", f"-I{inc}", ex, lib_host, "-o", exe, "-lm"],
                              capture_output=True).returncode == 0:
                built = True
                break
        if not built:
            raise RuntimeError("PC 上的 example 编不过")
        tin = os.path.join(tmp, "test_input.bin")
        with open(tin, "wb") as f:
            f.write(pb_test_input(hz))
        texp = os.path.join(tmp, "test_expect.bin")
        run([exe, tin, "--write", texp])
        chk = run([exe, tin, texp])
        if "集成验收：通过" not in chk:
            raise RuntimeError("PC 上自验不过：" + chk[-400:])

        readme = pb_readme(meta, fp, geo, classes, work, flash, ram_static, mv, os.path.getsize(texp) // 12)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("pet_behavior.h", header)
            z.write(lib_arm, "libpetbehavior.a")
            z.write(ex, "example.c")
            z.write(tin, "test_input.bin")
            z.write(texp, "test_expect.bin")
            z.write(lib_host, "pc/libpetbehavior_host.a")
            info = zipfile.ZipInfo("pc/pb_example")
            info.external_attr = 0o755 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, open(exe, "rb").read())
            z.writestr("README.md", readme)
    return os.path.getsize(out_zip)


def pb_readme(meta: dict, fp: dict, geo: dict, classes: list, work: int, flash: int | None,
              ram_static: int | None, model_version: int, n_expect: int) -> str:
    hz, n_t, hop = geo["hz"], geo["n_t"], geo["hop"]
    maps = "\n".join(f"| {c} | {pb_type_of(c)} | {PB_TYPES[pb_type_of(c)]} |" for c in classes)
    kb = lambda b: "-" if b is None else f"{b:,} B（{b / 1024:.1f} KB）"  # noqa: E731
    return f"""# 宠物行为识别库 pet_behavior

库版本 v0.{PB_LIB_VERSION}，模型版本 {model_version}（`pb_version()` = `0x{(PB_LIB_VERSION << 16) | model_version:08X}`）。
以静态库形式交付，不含源码。

| 文件 | 说明 |
| --- | --- |
| `pet_behavior.h` | 头文件 |
| `libpetbehavior.a` | 静态库（含模型），Cortex-M4F，`-mfloat-abi=softfp` |
| `example.c` | 调用示例 + 集成验收函数 `pb_acceptance()` |
| `test_input.bin` / `test_expect.bin` | 集成验收数据（{n_expect} 个片段） |
| `pc/pb_example`、`pc/libpetbehavior_host.a` | Linux x86_64 上先跑一遍验收用 |

## 1. 使用方法

```c
#include "pet_behavior.h"

static uint8_t pb_mem[PB_WORK_SIZE] __attribute__((aligned(4)));

pb_config_t cfg = {{ .sample_rate_hz = PB_SAMPLE_RATE_HZ, .acc_range_g = 8 }};
int r = pb_init(&cfg, pb_mem, sizeof pb_mem);      /* 0 = 成功（内含自检） */

/* 每 5~10 秒从 IMU FIFO 读一批，喂给库 */
pb_segment_t segs[8];
int n = pb_push(samples, count, t0_ms, segs, 8);
while (n > 0) {{
    /* 存储或上报 segs[0..n) */
    n = (n == 8) ? pb_push(NULL, 0, 0, segs, 8) : 0;   /* 一次没取完，再取 */
}}

/* 休眠前、数据断档前、上报前：取出剩余片段（同样 n == 8 时再调一次） */
n = pb_flush(segs, 8);

/* App 实时显示（可选） */
uint8_t type, conf;
if (pb_current(&type, &conf) == 0) {{ /* ... */ }}
```

## 2. 输入要求

| 项 | 要求 |
| --- | --- |
| 传感器 | 加速度计三轴，陀螺仪不需要 |
| 采样率 | **{hz} Hz**（`PB_SAMPLE_RATE_HZ`），间隔均匀 |
| 数据 | int16 原始计数，按采样先后排列；量程在 `pb_init` 中告知 |
| 时间 | `t0_ms` 为该批第一个样本的时间，其余按采样间隔推算；uint32 回绕可正常处理 |
| 批次 | 每批样本数不限；相邻两批须连续，断档前调用 `pb_flush` |

坐标系：佩戴后静止时重力大致落在某一轴上即可，库内部会自适应方向。

## 3. 输出说明

| type | 类别 |
| --- | --- |
""" + "\n".join(f"| {i} | {n} |" for i, n in enumerate(PB_TYPES)) + f"""

本版模型实际输出的类别：

| 模型类别 | type | 协议类别 |
| --- | --- | --- |
{maps}

- 片段在行为结束后输出，最大延迟约 1.5 分钟；时间精度约 {hop / hz:g} 秒。
- 开机后至少 {n_t / hz:g} 秒数据才开始判断。
- `out` 放不下的片段留在库内（最多 64 个），下次 `pb_push` / `pb_flush` 时输出；溢出丢弃的数量见 `pb_lost()`。

## 4. 资源与约束

| 项 | 数值 |
| --- | --- |
| Flash（代码 + 模型，`--gc-sections` 后） | {kb(flash)} |
| 工作内存 `PB_WORK_SIZE`（调用方提供） | {kb(work)} |
| 库内静态 RAM（.bss/.data） | {kb(ram_static)} |
| 推理频率 | 每 {hop / hz:g} 秒数据推理一次（{hop} 个样本） |
| 调用位置 | 任务或主循环，不可在中断中调用 |
| 线程安全 | 单实例，不可重入 |
| 动态内存 | 不使用 malloc |

编译链接：

- 库为 Cortex-M4F + `-mfloat-abi=softfp`（与 GR551x SDK 一致）；hard ABI 工程无法链接，请联系算法侧另出。
- 链 `libpetbehavior.a` 和 `-lm`；建议 `-Wl,--gc-sections`。

## 5. 错误码

| 值 | 宏 | 含义 |
| --- | --- | --- |
| -1 | `PB_ERR_PARAM` | 参数错误（采样率 / 量程 / 内存大小或对齐） |
| -2 | `PB_ERR_NOT_INIT` | 未初始化 |
| -3 | `PB_ERR_TIME` | 时间戳倒退 |
| -4 | `PB_ERR_SELFTEST` | 自检未通过，拒绝推理 |
| -5 | `PB_ERR_VERSION` | 头文件与库版本不一致（换库时头文件要一起换） |

## 6. 集成验收

1. PC 上（可选）：`./pc/pb_example test_input.bin test_expect.bin`，应输出「集成验收：通过」。
2. 板上：`pb_init` 返回 0。
3. 板上：把两个 bin 编进验收固件（如 `xxd -i`），调用 `example.c` 里的
   `pb_acceptance(in, in_len, expect, expect_len)`，返回 0 即输出与 `test_expect.bin` 逐字节一致。
4. 记录实际 Flash、RAM、单次 `pb_push` 耗时，回填接口协议第 6 节。

验收数据为合成数据，只用于确认集成正确，不代表识别效果。
"""


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
