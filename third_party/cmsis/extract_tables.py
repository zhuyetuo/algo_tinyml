#!/usr/bin/env python3
"""从 CMSIS-DSP 源码抽 16～256 点的 FFT 表，生成 dsp/Source/tm_cmsis_tables.c。

用法：python3 extract_tables.py /path/to/CMSIS-DSP
升级 CMSIS-DSP 时重跑一次；数值逐字照抄，不做任何转换。
"""

import os
import sys

LENS = (16, 32, 64, 128, 256)


def grab(text: str, head: str) -> str:
    i = text.index(head)
    j = text.index("};", i) + 2
    return text[i:j]


def main(dsp_root: str) -> None:
    src = open(os.path.join(dsp_root, "Source", "CommonTables", "arm_common_tables.c"), encoding="utf-8").read()
    cs = open(os.path.join(dsp_root, "Source", "CommonTables", "arm_const_structs.c"), encoding="utf-8").read()
    out = [
        "/* 从 CMSIS-DSP 的 arm_common_tables.c / arm_const_structs.c 抽出来的 16～256 点 FFT 表。\n",
        " * 原文件 7 MB（4096 点以内全有），这里只留 tm_features 用得到的几张，数值逐字照抄。\n",
        " * 来源 commit 见 third_party/cmsis/README.md。Apache-2.0。 */\n\n",
        '#include "arm_math_types.h"\n#include "arm_common_tables.h"\n#include "arm_const_structs.h"\n\n',
    ]
    for n in LENS:
        out.append(grab(src, f"const float32_t twiddleCoef_{n}[") + "\n\n")
        out.append(grab(src, f"const uint16_t armBitRevIndexTable{n}[") + "\n\n")
    for n in LENS:
        out.append(grab(cs, f"const arm_cfft_instance_f32 arm_cfft_sR_f32_len{n} ") + "\n\n")
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsp", "Source", "tm_cmsis_tables.c")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"写到 {dst}，{sum(len(x) for x in out):,} 字节")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
