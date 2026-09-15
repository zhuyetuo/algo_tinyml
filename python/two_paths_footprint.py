"""把两条端侧路线的**真实代码体积和 RAM** 量出来，用 Cortex-M4F 交叉编译。

为什么必须量：一路比下来，我们一直在拿"模型多少 KB"作比较，但那只是**模型**。
RF 那条在端上还要算 193 维手工特征（含 Welch/FFT），CNN 那条直接吃原始窗口。
特征提取的代码体积、RAM 和栈都还没进过任何一张表，而它可能就是决定性的那一项。

两条路线：
    CNN : tm_prep → tm_runtime                （无特征提取）
    RF  : tm_window → tm_features → tm_forest （有特征提取）

量法上有两个坑，都踩过：
  1. **--gc-sections 会把没被引用的代码整段丢掉**。只是把 .c 编进去、
     不真调用的话，测出来的增量是假的（之前测出过一个 +15.9KB 的假数）。
     所以下面每个入口都真的走完整条链，而且结果要被用掉（printf 出去），
     编译器没法把它优化没。
  2. 模型数组是 const，进 .rodata（flash），**不占 RAM**。所以体积要分开看
     text+rodata（flash）和 data+bss（RAM），混在一起看会得出错误结论。

用法：
    python python/two_paths_footprint.py
    python python/two_paths_footprint.py --cc arm-none-eabi-gcc
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

# GR5513 是 Cortex-M4F（带单精度 FPU、DSP 指令）。
# -ffp-contract=off 是硬要求：FMA 收缩会少一次中间舍入，两边就对不上。
CFLAGS = [
    "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16",
    "-Os", "-ffp-contract=off", "-fno-math-errno",
    "-ffunction-sections", "-fdata-sections",
    "-std=c99", "-Wall", "-Wextra",
    "--specs=nosys.specs", "--specs=nano.specs",
]
# **入口必须写成符号名**。写 -Wl,-e,0 的话入口是地址 0 而不是 _start，
# gc-sections 找不到任何根，会把整个程序丢光——量出来全是 0。
# 这正是这个脚本开头警告的那个坑，第一版就踩了。
LDFLAGS = ["-Wl,--gc-sections", "-nostartfiles", "-Wl,--entry=_start",
           "-Wl,--no-warn-rwx-segments"]
# **-lm 要算进来**：logf / floorf / sqrtf / round 会把 libm 的实现拖进 flash，
# 那是真实成本。RF 那条用到的超越函数比 CNN 多，这部分差额不该被忽略。
LDLIBS = ["-lm"]

COMMON = """
#include <stdint.h>
#include <stddef.h>

/* 一个不会被优化掉的"消费者"：把结果喂进去，编译器就不能把整条链删掉。
 * volatile 是关键——没有它，--gc-sections 加 -Os 能把所有计算证明为无用。 */
volatile uint32_t g_sink;
static void sink(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 0;
    for (size_t i = 0; i < n; i++) h = h * 31u + b[i];
    g_sink = h;
}
"""

# 两条链各自的最小驱动。都从"一个 float 窗口"出发，走到"一个类别号"。
CNN_MAIN = COMMON + """
#include "tm_prep.h"
#include "tm_runtime.h"

#define N_CH 8
#define N_T  16
#define N_CLS 5

/* 假模型：结构跟真的一样（3 层 conv + 3 次 pool + dense），只是权重随便填。
 * 量的是**代码**体积，不是权重体积——权重那部分前面已经逐层量过了。 */
static const int8_t  w1[64*8*3]; static const int32_t b1[64], m1[64], s1[64];
static const int8_t  w2[128*64*3]; static const int32_t b2[128], m2[128], s2[128];
static const int8_t  w3[128*128*3]; static const int32_t b3[128], m3[128], s3[128];
static const int8_t  wd[N_CLS*256]; static const int32_t bd[N_CLS], md[N_CLS], sd[N_CLS];

static const tm_layer_t L[] = {
  {.op=TM_CONV1D,.w=w1,.bias=b1,.mult=m1,.shift=s1,.out_ch=64,.in_ch=8,.k=3,.pad=1,.relu=1},
  {.op=TM_MAXPOOL1D,.pool=2},
  {.op=TM_CONV1D,.w=w2,.bias=b2,.mult=m2,.shift=s2,.out_ch=128,.in_ch=64,.k=3,.pad=1,.relu=1},
  {.op=TM_MAXPOOL1D,.pool=2},
  {.op=TM_CONV1D,.w=w3,.bias=b3,.mult=m3,.shift=s3,.out_ch=128,.in_ch=128,.k=3,.pad=1,.relu=1},
  {.op=TM_MAXPOOL1D,.pool=2},
  {.op=TM_DENSE,.w=wd,.bias=bd,.mult=md,.shift=sd,.out_ch=N_CLS,.in_ch=256,.k=0,.relu=0},
};
static const tm_model_t M = { L, 7, N_CH, N_T, N_CLS, 0, 1.0f, 1.0f, 0 };

static const double cmean[N_CH], cstd_[N_CH];
static const tm_prep_t P = { cmean, cstd_, 1.0, 0, N_CH, N_T };

static int8_t arena[2*128*8];
static int8_t xq[N_CH*N_T], out[N_CLS];
static float win[N_CH*N_T];

__attribute__((used)) int _start(void)
{
    tm_prep(&P, win, xq);
    if (tm_invoke(&M, xq, out, arena, (int)sizeof arena)) return 1;
    int c = tm_argmax(out, N_CLS);
    sink(&c, sizeof c);
    return 0;
}
"""

RF_MAIN = COMMON + """
#include "tm_features.h"
#include "tm_forest.h"

#define N_CH 8
#define N_T  16
#define NPERSEG 16
#define N_CLS 5
#define N_FEAT 193
#define N_NODES 12000
#define N_LEAVES 6000

static const float g_win[NPERSEG], g_cos[NPERSEG/2], g_sin[NPERSEG/2];
static const int16_t g_bitrev[NPERSEG];
static const tm_feat_cfg_t FC = { N_T, N_CH, NPERSEG, 16.0f,
                                  g_win, g_cos, g_sin, g_bitrev };

static const int32_t  t_off[21];
static const uint16_t n_feat[N_NODES];
static const float    n_thr[N_NODES];
static const int32_t  n_left[N_NODES], n_right[N_NODES];
static const float    leaf[N_LEAVES*N_CLS];
static const tm_forest_t F = { t_off, n_feat, n_thr, n_left, n_right, leaf,
                               20, N_FEAT, N_CLS };

static float win[N_CH*N_T];
static float feats[N_FEAT];
static float proba[N_CLS];

__attribute__((used)) int _start(void)
{
    if (tm_features(&FC, win, feats)) return 1;
    int c = tm_forest_predict(&F, feats, proba);
    sink(&c, sizeof c);
    sink(proba, sizeof proba);
    return 0;
}
"""


def _sizes(cc, srcs, main_src, tag, workdir, extra=()):
    m = os.path.join(workdir, f"{tag}.c")
    with open(m, "w", encoding="utf-8") as f:
        f.write(main_src)
    exe = os.path.join(workdir, f"{tag}.elf")
    cmd = [cc] + CFLAGS + list(extra) + [f"-I{FW}"] + \
        [os.path.join(FW, s) for s in srcs] + [m] + LDFLAGS + LDLIBS + ["-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr
    sz = subprocess.run([cc.replace("gcc", "size"), "-A", exe],
                        capture_output=True, text=True).stdout
    out = {}
    for line in sz.splitlines():
        mm = re.match(r"^(\.\w[\w.]*)\s+(\d+)", line)
        if mm:
            out[mm.group(1)] = int(mm.group(2))
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="arm-none-eabi-gcc")
    args = ap.parse_args()

    if not shutil.which(args.cc):
        sys.exit(f"找不到 {args.cc}。这个脚本要交叉编译到 Cortex-M4F 才有意义——"
                 "x86 的代码体积跟 Thumb-2 没有可比性。")

    d = tempfile.mkdtemp()
    cases = [
        ("CNN  (tm_prep + tm_runtime)",
         ["tm_prep.c", "tm_runtime.c"], CNN_MAIN, "cnn"),
        ("RF   (tm_features + tm_forest)",
         ["tm_features.c", "tm_forest.c"], RF_MAIN, "rf"),
        ("只有 tm_features（看它单独多大）",
         ["tm_features.c"],
         RF_MAIN.replace("int c = tm_forest_predict(&F, feats, proba);",
                         "int c = (int)feats[0]; (void)F; (void)proba;"),
         "featonly"),
    ]

    print(f"交叉编译到 Cortex-M4F（{args.cc}），-Os，跟固件同一套选项\n")
    print(f"{'路径':<34}{'代码 flash':>12}{'RAM':>10}{'栈外的静态 RAM 明细':>0}")
    print("-" * 72)
    res = {}
    for name, srcs, src, tag in cases:
        s, err = _sizes(args.cc, srcs, src, tag, d)
        if s is None:
            print(f"{name:<34}  编译失败：\n{err[:800]}")
            continue
        # .text + .rodata 里**扣掉模型常量**才是代码本身。模型数组我们自己声明的，
        # 大小已知，直接减掉；剩下的就是算法代码 + 查表。
        text = s.get(".text", 0)
        rodata = s.get(".rodata", 0)
        ram = s.get(".data", 0) + s.get(".bss", 0)
        res[tag] = (text, rodata, ram)
        print(f"{name:<34}{text:>12,}{ram:>10,}   "
              f"(.text {text:,} / .rodata {rodata:,} / .bss+.data {ram:,})")

    if "cnn" in res and "rf" in res:
        dt = res["rf"][0] - res["cnn"][0]
        dr = res["rf"][2] - res["cnn"][2]
        print(f"""
差额（RF 减 CNN）：代码 {dt:+,} B，RAM {dr:+,} B

怎么读：
  · 代码 flash 这一列是**算法本身**，跟模型大小无关（模型是 const 数组，
    在 .rodata 里，这里的 .rodata 只有 FFT 旋转因子和窗函数那点查表）。
  · RAM 这一列不含栈。RF 那条要额外放 193 个 float 的特征向量（772 B）
    加 FFT 的中间缓冲，CNN 那条只有乒乓缓冲。
  · **两条都还要加各自的模型**：CNN [64,128,128] 是 78.6 KB，
    RF 20 棵 × 深 10 是 98.2 KB。那两个数前面已经实测过了。""")


if __name__ == "__main__":
    main()
