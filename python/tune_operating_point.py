"""扫两个**零 flash 成本**的旋钮，看能不能用少轮数换回事件级质量。

背景：实测下来事件级 precision 并不比窗口级好（误报是**成片**出现的，不是孤立
单窗口），所以聚合本身救不了。但还有两个旋钮，它们都不占一个字节 flash：

  1. **min_windows** —— 连续命中几个窗口才算一次事件。
     真事件平均 12.67 个窗口长，而默认才 3。如果误报比真事件短，
     调大它就能优先杀掉误报。
  2. **类别偏置** —— 给目标类别的 margin 加一个常数再取 argmax。
     板上就是一次加法（或者直接折进 base_score），**零代价**。
     少数类上 argmax 往往不是最优工作点：模型倾向于过报或漏报，
     一个偏置就能把工作点挪到想要的地方。

这两个都不改模型，所以**可以在选定轮数之后单独调**，而且调完不用重训。

用法：
    python python/tune_operating_point.py \\
        --model <xgb.pkl> --features holdout_feats.npy --labels holdout_y.npy \\
        --classes 活动,睡觉,抓挠,未佩戴,甩身体 --focus 抓挠 \\
        --rounds 50,100,200 --min-windows 3,5,8,12 --bias -1.5,-1,-0.5,0,0.5

⚠ 事件级指标的样本量往往很小（这份数据里真值只有 17 次事件）。
   一次事件的变动就能让 P 动 0.05，所以**只看大势，别做精细比较**。
   脚本会把真值事件数打出来提醒。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.gbdt import from_xgboost  # noqa: E402

_ev = None


def _load_event_eval():
    """复用 event_eval.py 的聚合和配对逻辑，不重写一遍——
    重写的话两边的事件定义迟早不一致，而那会让两个脚本给出矛盾的数。"""
    global _ev
    if _ev is None:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_eval.py")
        spec = importlib.util.spec_from_file_location("_ev_mod", path)
        _ev = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _ev
        spec.loader.exec_module(_ev)
    return _ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--focus", required=True)
    ap.add_argument("--rounds", default="50,100,200")
    ap.add_argument("--min-windows", default="3,5,8,12")
    ap.add_argument("--bias", default="-1.5,-1,-0.5,0,0.5",
                    help="给目标类别 margin 加的常数。负数=更保守（少报）")
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    ev = _load_event_eval()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib。这个脚本要在训练机上跑。")

    names = [s for s in args.classes.split(",") if s]
    if args.focus not in names:
        sys.exit(f"--focus {args.focus} 不在 --classes 里：{names}")
    fi = names.index(args.focus)

    X = np.load(ev._need(args.features, "--features")).astype(np.float32)
    y = np.load(ev._need(args.labels, "--labels")).astype(np.int64)
    if len(X) != len(y):
        sys.exit(f"特征 {len(X)} 行、标签 {len(y)} 行，对不上")

    mean_run, err = ev.check_ordered(y, fi)
    if err:
        sys.exit(err)
    if mean_run < 1.5:
        sys.exit(f"游程 {mean_run:.2f}，数据看着打乱过，事件级评估没有意义")

    bundle = joblib.load(ev.resolve_model(args.model))
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    b = from_xgboost(model, class_names=names)

    rounds = [int(v) for v in args.rounds.split(",") if v]
    mws = [int(v) for v in args.min_windows.split(",") if v]
    biases = [float(v) for v in args.bias.split(",") if v]

    print(f"留出集 {len(X)} 条，真值「{args.focus}」平均游程 {mean_run:.2f} 个窗口")
    print(f"扫 {len(rounds)} 个轮数 × {len(mws)} 个 min_windows × {len(biases)} 个偏置"
          f" = {len(rounds) * len(mws) * len(biases)} 组\n")

    # margin 只跟轮数有关，先算好，别在三重循环里重复算
    margins = {}
    for r in rounds:
        t = b.truncate(r)
        margins[r] = np.stack([t.margins(x) for x in X])
        print(f"  {r} 轮的 margin 算完")

    results = []
    for r in rounds:
        M = margins[r]
        for bias in biases:
            adj = M.copy()
            adj[:, fi] += np.float32(bias)
            hits = adj.argmax(axis=1) == fi
            for mw in mws:
                true_ev = ev.to_events(y == fi, mw, args.max_gap)
                pred_ev = ev.to_events(hits, mw, args.max_gap)
                tp, fp, fn = ev.match_events(pred_ev, true_ev)
                p, rc, f1 = ev.prf(tp, fp, fn)
                results.append((f1, r, bias, mw, len(pred_ev), len(true_ev),
                                tp, fp, fn, p, rc))

    n_true_ref = results[0][5] if results else 0
    print(f"\n⚠ 真值只有 **{n_true_ref} 次事件**（min_windows 变了真值也会变，"
          "表里每行的真值数单列）。\n"
          "  一次事件的变动就能让 P 动几个点——**只看大势，别做精细比较**。\n")

    results.sort(reverse=True)
    hdr = (f"{'事件F1':>8}{'轮数':>6}{'偏置':>7}{'minW':>6}"
           f"{'报':>5}{'真值':>6}{'对':>4}{'误报':>6}{'漏':>4}{'事件P':>8}{'事件R':>8}")
    print(f"按事件 F1 排前 {args.top}：")
    print(hdr)
    print("-" * len(hdr))
    for row in results[:args.top]:
        f1, r, bias, mw, npred, ntrue, tp, fp, fn, p, rc = row
        print(f"{f1:>8.3f}{r:>6}{bias:>7.1f}{mw:>6}{npred:>5}{ntrue:>6}"
              f"{tp:>4}{fp:>6}{fn:>4}{p:>8.3f}{rc:>8.3f}")

    # 基线：默认设置下每个轮数各是多少，用来看旋钮到底买到了什么
    print(f"\n基线（偏置 0、min_windows 3）：")
    print(hdr)
    print("-" * len(hdr))
    for row in sorted([x for x in results if x[2] == 0.0 and x[3] == 3],
                      key=lambda z: z[1]):
        f1, r, bias, mw, npred, ntrue, tp, fp, fn, p, rc = row
        print(f"{f1:>8.3f}{r:>6}{bias:>7.1f}{mw:>6}{npred:>5}{ntrue:>6}"
              f"{tp:>4}{fp:>6}{fn:>4}{p:>8.3f}{rc:>8.3f}")

    print("""
怎么用这张表：

  **找"少轮数 + 调过旋钮"能不能追上"多轮数 + 默认旋钮"。** 追得上的话，
  省下来的 flash 是白赚的——这两个旋钮在板上都是零成本
  （min_windows 是聚合参数，偏置是一次加法或者直接折进 base_score）。

  但要小心两件事：
  - **事件样本太少**，排第一的那组很可能只是运气。看**一片**好的区域，
    别只看榜首。
  - 这是在**留出集上调的**。旋钮调多了就是在留出集上过拟合——
    真要上线，最好留一段完全没参与调参的数据再确认一次。""")


if __name__ == "__main__":
    main()
