#!/usr/bin/env python3
"""检查一份导出目录是不是**真模型**，能不能提交。

    python3 scripts/check_export.py core/models/edge_rf_d10
    python3 scripts/check_export.py --all          # 仓库里已提交的全查一遍

为什么需要这个：自测时会造随机权重的演示导出，文件名、结构、类别名跟真的
一模一样，**编得过、跑得通、自检也过**（golden vector 是拿同一份假权重
生成的，当然自己跟自己对得上）。提交进去之后没有任何办法分辨，
而板子烧进去只会得到一堆看着正常的错结论。

这个脚本不去"猜"真假，它比的是**导出结果 vs 训练产出的元数据**：
树的棵数、类别、窗口几何对不对得上。假模型是从随机权重造的，
对不上真实训练的那份 .json。

所以导出目录里必须有一份 meta.json（从训练产出的 .json 拷过来）。
没有的话这里直接拒绝——"无从核对"和"核对通过"必须分得开。

**这是筛子，不是证明。**它能挡住"手滑把自测的演示导出提交了"这一类，
挡不住"有人拿真 meta 配假 C"。真正的保证只有一条：导出和 meta 来自
同一次训练产出。哪天要更强的保证，就在导出时写一份来源指纹进去。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 一份真模型的 .c 至少得有这么大。
#
# 这两个数是**量出来的**，不是拍的：
#   RF  演示导出 4 棵树 / 60 节点 → tm_forest_c_model.c 2.5 KB
#       真的 20 棵 × 深度 10      → 100 KB 量级
#   CNN 演示导出 [16,32,32]       → tm_model.c 27 KB
#       真的 [64,128,128] int8    → 模型本身 78.6 KB，.c 还要更大
#
# CNN 第一版我定的 20 KB，**27 KB 的演示导出直接通过了**——
# 假阴性比没有检查更糟，因为它给了一个"查过了"的错觉。
_MIN_BYTES = {"cnn": 50_000, "rf": 20_000}


def _defines(path: str) -> dict[str, str]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*#define\s+(\w+)\s+(.+?)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _class_names(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        m = re.search(r"CLASS_NAMES\[\]\s*=\s*\{(.+?)\}", f.read(), re.S)
    if not m:
        return []
    return re.findall(r'"([^"]*)"', m.group(1))


def kind_of(d: str) -> str:
    if os.path.exists(os.path.join(d, "tm_forest_c_model.h")):
        return "rf"
    if os.path.exists(os.path.join(d, "tm_model.h")):
        return "cnn"
    return ""


def check(d: str) -> list[str]:
    bad: list[str] = []
    kind = kind_of(d)
    if not kind:
        return [f"{d} 看着不像导出目录（既没有 tm_model.h 也没有 tm_forest_c_model.h）"]

    meta_p = os.path.join(d, "meta.json")
    if not os.path.exists(meta_p):
        return [f"{d}/meta.json 不存在。\n"
                "    导出时要把训练产出的那份 .json 一起拷进来，否则**没法核对**\n"
                "    这是不是真模型——而假模型看起来跟真的一模一样。\n"
                "    cp <训练产出>/xxx.json {d}/meta.json".format(d=d)]
    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)

    if kind == "rf":
        h = _defines(os.path.join(d, "tm_forest_c_model.h"))
        names = _class_names(os.path.join(d, "tm_forest_c_model.h"))
        n_tree = int(h.get("TM_FC_N_TREES", 0))
        n_node = int(h.get("TM_FC_N_NODES", 0))
        want = meta.get("n_estimators") or meta.get("n_trees")
        if want and int(want) != n_tree:
            bad.append(f"树的棵数对不上：导出 {n_tree} 棵，meta.json 说 {want} 棵。\n"
                       f"    **这多半就是拿演示导出当真模型了。**")
        if not want:
            # meta 里没有棵数时退到经验值。说清楚这是弱检查，别让人以为核对过了
            if n_tree <= 8:
                bad.append(f"只有 {n_tree} 棵树，而 meta.json 里没有 n_estimators 可以核对。\n"
                           f"    自测的演示导出就是 4 棵。确认这是真模型的话，\n"
                           f"    在 meta.json 里补上 n_estimators 再来。")
        if n_node <= 200:
            bad.append(f"只有 {n_node} 个节点，太小了（演示导出是 60 个）")
        model_c = os.path.join(d, "tm_forest_c_model.c")
    else:
        h = _defines(os.path.join(d, "tm_model.h"))
        names = _class_names(os.path.join(d, "tm_model.h"))
        model_c = os.path.join(d, "tm_model.c")
        n_t, n_ch = int(h.get("TM_N_T", 0)), int(h.get("TM_N_CH", 0))
        if meta.get("window_size") and int(meta["window_size"]) != n_t:
            bad.append(f"窗口点数对不上：导出 {n_t}，meta.json 说 {meta['window_size']}")
        if meta.get("n_channels") and int(meta["n_channels"]) != n_ch:
            bad.append(f"通道数对不上：导出 {n_ch}，meta.json 说 {meta['n_channels']}")
        # ch_mean/ch_std 是从**真实训练数据**统计出来的，演示导出那条路
        # 根本不产生它们。缺了就是没拿真训练产出当 meta
        for k in ("ch_mean", "ch_std"):
            if not meta.get(k):
                bad.append(f"meta.json 里没有 {k}。CNN 的归一化参数是从真实训练\n"
                           f"    数据统计的，演示导出那条路不产生它们——\n"
                           f"    这份 meta 多半不是训练产出的那个 .json。")

    want_cls = meta.get("classes")
    if want_cls and names and list(want_cls) != names:
        bad.append(f"类别对不上：\n    导出 {names}\n    meta {list(want_cls)}\n"
                   f"    **类别顺序错了，概率会安到别的类别上**，不报错。")

    if os.path.exists(model_c):
        size = os.path.getsize(model_c)
        if size < _MIN_BYTES[kind]:
            bad.append(f"{os.path.basename(model_c)} 只有 {size} 字节，"
                       f"真模型应该在 {_MIN_BYTES[kind]} 字节以上。演示导出就这么小。")
    else:
        bad.append(f"缺 {model_c}")

    # golden vector 必须有：没有的话板上自检等于没做
    golden = [f for f in os.listdir(d) if "golden" in f]
    if not golden:
        bad.append("没有 golden vector 头文件——板上自检会变成「0 条全部通过」，"
                   "那是这类自检最经典的失效方式")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="导出目录")
    ap.add_argument("--all", action="store_true", help="检查仓库里已提交的全部导出")
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.all or not dirs:
        fw = os.path.join(ROOT, "core", "models")
        dirs = [os.path.join(fw, n) for n in sorted(os.listdir(fw))
                if not n.startswith(".") and os.path.isdir(os.path.join(fw, n))]
    if not dirs:
        print("没有导出目录可查")
        return 0

    rc = 0
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        problems = check(d)
        rel = os.path.relpath(d, ROOT)
        if problems:
            rc = 1
            print(f"✗ {rel}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {rel}")
    if rc:
        print("\n有导出没通过检查。**别提交**——假模型编得过也跑得通，"
              "提交之后分不出来。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
