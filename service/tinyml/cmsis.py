"""CMSIS 那条可选路线的文件清单：编哪些 .c、加哪些 -I、带哪些 -D。

footprint（量占用 / 打包）、serve（PC 上编 .so 对答案）、tests 三处共用这一份，
免得三个地方各抄一遍文件名、改一处漏两处。

third_party/cmsis/ 是从 CMSIS-DSP / CMSIS-NN / CMSIS-Core 裁出来的最小子集，
来源 commit 和裁剪规则见那里的 README.md。
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMSIS = os.path.join(ROOT, "third_party", "cmsis")

# 特征提取用到的 CMSIS-DSP：cfft + 几个向量统计 + 我们抽出来的 16～256 点表
DSP_SOURCES = [
    "arm_cfft_f32.c", "arm_cfft_init_f32.c", "arm_cfft_radix8_f32.c", "arm_bitreversal2.c",
    "arm_mean_f32.c", "arm_power_f32.c", "arm_rms_f32.c", "arm_max_no_idx_f32.c", "arm_min_no_idx_f32.c",
    "arm_dot_prod_f32.c", "tm_cmsis_tables.c",
]
# int8 CNN 用到的 CMSIS-NN：s8 卷积（含 DSP 路径的 im2col 内核）+ s8 最大池化
NN_SOURCES = [
    "arm_convolve_s8.c", "arm_convolve_get_buffer_sizes_s8.c",
    "arm_nn_mat_mult_kernel_s8_s16.c", "arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    "arm_nn_mat_mult_nt_t_s8.c", "arm_q7_to_q15_with_offset.c", "arm_s8_to_s16_unordered_with_offset.c",
    "arm_max_pool_s8.c",
]


def available() -> bool:
    return os.path.isdir(os.path.join(CMSIS, "dsp", "Include")) and os.path.isdir(os.path.join(CMSIS, "nn", "Include"))


def sources(kind: str) -> list[str]:
    """rf 只要 DSP，cnn 只要 NN（cnn 的前处理是 z-score，没有 FFT）。"""
    if kind == "rf":
        return [os.path.join(CMSIS, "dsp", "Source", f) for f in DSP_SOURCES]
    return [os.path.join(CMSIS, "nn", "Source", f) for f in NN_SOURCES]


def include_dirs(kind: str, host: bool) -> list[str]:
    """host=True 是 PC 上编（x86）：CMSIS-DSP 用 __GNUC_PYTHON__ 绕开 CMSIS-Core；
    交叉编译要 CMSIS-Core 的 cmsis_compiler.h（third_party 里带了，GR551x SDK 也有）。"""
    inc = []
    if kind == "rf":
        inc += [os.path.join(CMSIS, "dsp", "Include"), os.path.join(CMSIS, "dsp", "PrivateInclude")]
    else:
        inc += [os.path.join(CMSIS, "nn", "Include")]
    if not host:
        inc.append(os.path.join(CMSIS, "core", "Include"))
    return inc


def defines(kind: str, host: bool) -> dict:
    d = {"TM_CMSIS_DSP": 1} if kind == "rf" else {"TM_CMSIS_NN": 1}
    if host and kind == "rf":
        d["__GNUC_PYTHON__"] = 1
    return d


def bundle_files(kind: str) -> dict:
    """打进源码包的 third_party 文件：{zip 内路径: 本地路径}。头文件整目录带（互相 include），.c 只带用到的。"""
    out = {}
    sub = "dsp" if kind == "rf" else "nn"
    for root, _, names in os.walk(os.path.join(CMSIS, sub, "Include")):
        for n in names:
            p = os.path.join(root, n)
            out[os.path.relpath(p, ROOT)] = p
    if kind == "rf":
        p = os.path.join(CMSIS, "dsp", "PrivateInclude", "arm_compiler_specific.h")
        out[os.path.relpath(p, ROOT)] = p
    for p in sources(kind):
        out[os.path.relpath(p, ROOT)] = p
    for root, _, names in os.walk(os.path.join(CMSIS, "core", "Include")):
        for n in names:
            p = os.path.join(root, n)
            out[os.path.relpath(p, ROOT)] = p
    for n in ("README.md", f"{sub}/LICENSE", "core/LICENSE"):
        p = os.path.join(CMSIS, n)
        if os.path.exists(p):
            out[os.path.relpath(p, ROOT)] = p
    return out
