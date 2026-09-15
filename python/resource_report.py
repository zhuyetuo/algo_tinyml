"""端侧资源报告：模型多大、固件多大、跑起来占多少内存、能存几天的结果。

**每一行都标了是实测还是估算**，因为这两类数的可信度差很远：

  实测 = 编译器/链接器报出来的，或者程序真跑出来的
  估算 = 按运算量推的，**只能当数量级**，真实耗时要板子上用 DWT 周期计数器测

用法：
    python python/resource_report.py --gen firmware/generated
    python python/resource_report.py --gen firmware/generated --elf .../tinyml_app.elf
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")

GR5513_FLASH = 512 * 1024
GR5513_RAM_APP = 112 * 1024          # 128KB 减掉链接脚本前面 16KB 的系统栈


def _macros(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"#define (\w+)\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def _arm_size(objs):
    """→ (text, data, bss)；没有交叉编译器就返回 None。"""
    try:
        r = subprocess.run(["arm-none-eabi-size", "-t", *objs],
                           capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if r.returncode != 0:
        return None
    last = r.stdout.strip().splitlines()[-1].split()
    return int(last[0]), int(last[1]), int(last[2])


def _cross_compile(gen, tmp, extra_defines):
    srcs = [os.path.join(FW, f) for f in
            ("tm_features.c", "tm_forest.c", "tm_runtime.c", "tm_window.c")]
    srcs += [os.path.join(gen, f) for f in os.listdir(gen) if f.endswith(".c")]
    objs = []
    flags = ["-std=c99", "-Os", "-ffunction-sections", "-fdata-sections",
             "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=softfp", "-mfpu=fpv4-sp-d16",
             "-ffp-contract=off", f"-I{FW}", f"-I{gen}", *extra_defines]
    for s in srcs:
        o = os.path.join(tmp, os.path.basename(s)[:-2] + ".o")
        r = subprocess.run(["arm-none-eabi-gcc", *flags, "-c", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr
        objs.append(o)
    return objs, None


def _feature_ops(n_t, n_ch, nperseg):
    """特征提取的运算量，按结构推。数量级用，不是精确指令数。"""
    n_sig_time = n_ch + 2 + 1          # 各通道 + acc/gyr 模长 + jerk 模长
    n_sig_freq = min(6, n_ch) + 2      # 频域只算前 6 通道 + 两个模长
    # 时域：每个信号约 8 遍长度 n 的循环 + 一次插入排序（平均 n²/4 次比较）
    time_ops = n_sig_time * (8 * n_t + n_t * n_t // 4)
    # 频域：一次基-2 复数 FFT 约 5·N·log2(N) 次浮点运算，外加加窗/去趋势/求谱
    import math
    fft = 5 * nperseg * int(math.log2(nperseg))
    freq_ops = n_sig_freq * (fft + 6 * nperseg)
    return time_ops, freq_ops


def _cnn_macs(n_ch, n_t):
    t1 = n_t - 4
    t2 = t1 // 4 - 2
    t3 = t2 // 4
    return 8 * n_ch * 5 * t1 + 16 * 8 * 3 * t2 + 3 * (16 * t3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="firmware/generated")
    ap.add_argument("--elf", help="链好的固件 .elf（有的话报整机体积）")
    ap.add_argument("--events-per-day", type=int, default=200)
    ap.add_argument("--event-bytes", type=int, default=16)
    args = ap.parse_args()

    gen = args.gen
    if not os.path.isdir(gen):
        sys.exit(f"{gen} 不存在。先跑 export_rf.py 或 quantize_and_export.py。")

    feat = _macros(os.path.join(gen, "tm_feat_cfg.h"))
    fmod = _macros(os.path.join(gen, "tm_forest_model.h"))
    cmod = _macros(os.path.join(gen, "tm_model.h"))

    n_t = feat.get("TM_FEAT_N_T", 32)
    n_ch = feat.get("TM_FEAT_N_CH", 8)
    nps = feat.get("TM_FEAT_NPERSEG", 32)
    dim = feat.get("TM_FEAT_DIM", 193)

    print("=" * 66)
    print("端侧资源报告   GR5513BENDU：512KB Flash / 128KB RAM（应用可用 112KB）")
    print("=" * 66)
    print(f"配置：窗口 {n_t} 点 × {n_ch} 通道，{dim} 维特征，nperseg={nps}")

    # ── 1. 模型本身多大（实测：数导出文件里的常量） ──────────────────────
    print("\n【模型体积】实测")
    total_model = 0
    for name, f in (("随机森林", "tm_forest_model.c"), ("int8 CNN", "tm_model.c"),
                    ("特征常量表", "tm_feat_cfg.c")):
        p = os.path.join(gen, f)
        if not os.path.exists(p):
            continue
        # 用交叉编译单独量这个文件的 .rodata，比数源码里的常量准
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            o = os.path.join(tmp, "m.o")
            r = subprocess.run(
                ["arm-none-eabi-gcc", "-std=c99", "-Os", "-mcpu=cortex-m4", "-mthumb",
                 "-mfloat-abi=softfp", "-mfpu=fpv4-sp-d16", f"-I{FW}", f"-I{gen}",
                 "-c", p, "-o", o], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {name:<12} 编不过：{r.stderr.splitlines()[-1] if r.stderr else '?'}")
                continue
            sz = _arm_size([o])
            if sz:
                print(f"  {name:<12} {sz[0]:>9,} B ({sz[0] / 1024:>6.1f} KB)")
                total_model += sz[0]
    print(f"  {'合计':<12} {total_model:>9,} B ({total_model / 1024:>6.1f} KB)")

    # ── 2. 代码多大 + 跑起来占多少 RAM（实测） ───────────────────────────
    print("\n【推理代码 + 运行内存】实测（arm-none-eabi-gcc -Os, cortex-m4）")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        objs, err = _cross_compile(gen, tmp, [f"-DTM_FEAT_MAX_T={n_t}",
                                              f"-DTM_FEAT_MAX_NPERSEG={nps}"])
        if objs is None:
            print("  没有 arm-none-eabi-gcc（或编译失败），跳过。")
            print(f"  {err.splitlines()[-1] if err else ''}")
            code_sz = None
        else:
            sz = _arm_size(objs)
            code_sz = sz
            print(f"  代码 (text)   {sz[0]:>9,} B ({sz[0] / 1024:>6.1f} KB)  含模型常量")
            print(f"  静态 RAM (bss){sz[2]:>9,} B ({sz[2] / 1024:>6.1f} KB)"
                  "  特征提取的临时缓冲")
            print(f"  栈（估算）    {'约 1-2 KB':>9}            递归只有 FFT 那层，很浅")

    if args.elf and os.path.exists(args.elf):
        r = subprocess.run(["arm-none-eabi-size", args.elf], capture_output=True, text=True)
        if r.returncode == 0:
            t, d, b = (int(v) for v in r.stdout.strip().splitlines()[-1].split()[:3])
            print("\n【整机固件】实测（含 BLE 协议栈）")
            print(f"  Flash  {t + d:>9,} B ({(t + d) / 1024:>6.1f} KB)"
                  f"   占 512KB 的 {100.0 * (t + d) / GR5513_FLASH:.1f}%")
            print(f"  RAM    {b:>9,} B ({b / 1024:>6.1f} KB)"
                  f"   占可用 112KB 的 {100.0 * b / GR5513_RAM_APP:.1f}%")
            free = GR5513_FLASH - (t + d)
            print(f"  剩余 Flash {free:>9,} B ({free / 1024:.1f} KB)")

            # ── 4. 能存几天 ──────────────────────────────────────────
            print("\n【能存多少天的结果】按剩余 Flash 估算")
            print(f"  假设：一天 {args.events_per_day} 条事件，每条 {args.event_bytes} B")
            per_day = args.events_per_day * args.event_bytes
            days = free // per_day
            print(f"  一天 {per_day:,} B → 理论上 {days:,} 天（{days / 365:.1f} 年）")
            print("  **但别按这个数设计**：")
            print("   - 剩余 Flash 要留 OTA 升级的空间（通常要留下一个镜像的大小）；")
            print("   - 片上 Flash 有擦写寿命，得做磨损均衡，实际可用容量要打折；")
            print("   - SDK 的 NVDS（配对信息等）也占一块。")
            print("   现实中一天几百条事件的话，**存不满是常态**——瓶颈在同步频率，"
                  "不在容量。")

    # ── 3. 推理耗时（估算 + 怎么测真的） ────────────────────────────────
    print("\n【推理耗时】估算，**不是实测**")
    t_ops, f_ops = _feature_ops(n_t, n_ch, nps)
    n_nodes = None
    macs = _cnn_macs(n_ch, n_t)
    print(f"  特征提取：时域约 {t_ops:,} 次运算 + 频域约 {f_ops:,} 次 "
          f"= 约 {t_ops + f_ops:,}")
    print(f"  int8 CNN：约 {macs:,} 次乘加")
    print(f"  随机森林：约 树数 × 平均深度 次比较（看导出的森林，通常几千次）")
    print()
    print("  **值得注意**：RF 的特征提取比整个 CNN 还贵。RF 推理本身只有比较、很便宜，"
          "但它\n  吃的是 193 维手工特征，算那些特征的代价超过跑完一个小 CNN。"
          "\n  「RF 比 CNN 省」这个直觉在**端到端**的口径下不成立。")
    print()
    print("  64MHz 的 M4F 上，这个量级的运算量是**毫秒级**，一两秒跑一次，"
          "占空比 <1%。\n  但上面是运算量不是周期数——真实耗时要在板上用 DWT 周期计数器测：")
    print("""
      CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
      DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
      uint32_t t0 = DWT->CYCCNT;
      tm_features(&tm_feat_cfg, win, feat);
      uint32_t cycles = DWT->CYCCNT - t0;      /* ÷64e6 = 秒 */""")

    print("\n" + "=" * 66)
    print("哪些是实测、哪些是估算，上面每一节都标了。做决策前请分清——")
    print("体积和内存可以照着用，耗时只能当数量级，功耗一个数都没有（要板子）。")
    print("=" * 66)


if __name__ == "__main__":
    main()
