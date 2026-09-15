"""窗口级指标 ≠ 事件级指标。这个脚本量后者。

为什么要分开看：端上不是每个窗口报一次，是 `tinyml_task.c` 那套聚合——
**连续命中 min_windows 个窗口才算一次事件**，中间允许空 max_gap 个。
所以孤立的假阳性会被直接吃掉，而窗口级 precision 把它们全算进去了。

产品关心的是"今天报了几次抓挠、其中几次是真的"，那是事件级的数。
窗口级 P=0.50 看着很糟，但如果误报都是孤立的单窗口，事件级可能好得多——
**这个差别会直接影响"能减到多少轮"的决定**，所以要量不要猜。

聚合逻辑跟固件 `tinyml_task.c` 是同一套规则（这里是 Python 复刻，参数含义一致）。

用法：
    python python/event_eval.py --model <xgb.pkl> \\
        --features holdout_feats.npy --labels holdout_y.npy \\
        --classes 活动,睡觉,抓挠,未佩戴,甩身体 --focus 抓挠 \\
        --rounds 15,30,50,100,200

**前提：窗口必须按时间顺序排**。打乱过的数据上做事件聚合毫无意义——
脚本会先检查真值标签的游程长度，看着像打乱过就直接拒绝跑。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.adapter import ModelAdapter  # noqa: E402


def resolve_model(path):
    """跟别的脚本一样的 --model 路径守卫。漏了这个的话（我就漏了），
    路径写错只会甩一个 joblib 的 traceback。"""
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    hint = ""
    if "..." in path:
        hint = "\n  路径里有 `...`——那是占位符，要换成真实目录名。"
    if not os.path.isabs(p):
        hint += f"\n  相对路径会解析成 {os.path.abspath(p)}。"
    guess = os.path.expanduser("~/imu_train/results")
    if os.path.isdir(guess):
        hint += f"\n  ~/imu_train/results/ 下现有：{', '.join(sorted(os.listdir(guess))[:5]) or '（空）'}"
    sys.exit(f"找不到模型文件：{p}{hint}")


def _need(path, flag):
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    sys.exit(f"{flag} 找不到：{p}\n  先用 dump_holdout.py 导一次。")


def to_events(hits, min_windows, max_gap):
    """一串 bool（每个窗口命中没有）→ 事件区间列表 [(start, end), ...]。

    跟固件 tinyml_task.c 同一套规则：累计命中够 min_windows 才算一次；
    中间空不超过 max_gap 个窗口还算同一次（抓挠中间会停顿，一停就切断会把
    一次连续抓挠拆成好几个）。
    """
    events = []
    in_ev = False
    n_hit = gap = 0
    start = last_hit = 0
    for i, h in enumerate(hits):
        if h:
            if not in_ev:
                in_ev, n_hit, start = True, 0, i
            n_hit += 1
            gap = 0
            last_hit = i
        elif in_ev:
            gap += 1
            if gap > max_gap:
                if n_hit >= min_windows:
                    events.append((start, last_hit))
                in_ev, n_hit, gap = False, 0, 0
    if in_ev and n_hit >= min_windows:
        events.append((start, last_hit))
    return events


def match_events(pred_ev, true_ev):
    """预测事件和真值事件按**是否有重叠**配对。

    不要求边界对齐：一次抓挠持续多久，人标的和模型判的本来就会差几个窗口，
    按 IoU 之类的严格匹配会把本来算对的判成错的。产品问的是"这次抓挠报到了吗"。
    """
    used = set()
    tp = 0
    for ps, pe in pred_ev:
        for j, (ts, te) in enumerate(true_ev):
            if j in used:
                continue
            if ps <= te and ts <= pe:       # 有重叠
                used.add(j)
                tp += 1
                break
    fp = len(pred_ev) - tp
    fn = len(true_ev) - len(used)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def check_ordered(y, focus):
    """窗口是不是按时间排的。打乱过的数据做事件聚合毫无意义。

    判据：真值里目标类别的平均游程长度。按时间排的话，一次抓挠会连着好几个窗口
    （stride 0.5 秒，一次抓挠好几秒）；打乱过的话游程长度会接近 1。
    """
    hits = (y == focus)
    if not hits.any():
        return None, "留出集里一条目标类别都没有"
    runs, cur = [], 0
    for h in hits:
        if h:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classes", default="")
    ap.add_argument("--focus", required=True)
    ap.add_argument("--rounds", default="",
                    help="要扫的点。GBDT 是轮数，RF 是 max_depth。留空=自动")
    ap.add_argument("--trees", type=int, default=0,
                    help="[只对 RF] 只取前 n 棵。**要评的是端上真跑的那一格**，"
                         "而那一格是棵数和深度一起生效的——只截深度得到的数"
                         "对应不上任何一个塞得进 flash 的配置")
    ap.add_argument("--quantize-leaves", action="store_true",
                    help="[只对 RF] 叶子概率量化成 uint8，跟端上一致。"
                         "这会**改变判决**（相差不到 1/255 的两类会翻），所以要实测")
    ap.add_argument("--min-windows", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--force", action="store_true",
                    help="即使看着像打乱过也照跑（结果没有意义，只用来调试）")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib。这个脚本要在训练机上跑。")

    names = [s for s in args.classes.split(",") if s]
    if args.focus not in names:
        sys.exit(f"--focus {args.focus} 不在 --classes 里：{names}")
    fi = names.index(args.focus)

    X = np.load(_need(args.features, "--features")).astype(np.float32)
    y = np.load(_need(args.labels, "--labels")).astype(np.int64)
    if len(X) != len(y):
        sys.exit(f"特征 {len(X)} 行、标签 {len(y)} 行，对不上")

    mean_run, err = check_ordered(y, fi)
    if err:
        sys.exit(err)
    print(f"留出集 {len(X)} 条，真值里「{args.focus}」的平均游程长度 {mean_run:.2f} 个窗口")
    if mean_run < 1.5 and not args.force:
        sys.exit(
            f"游程长度只有 {mean_run:.2f}，看着像**打乱过的**数据。\n"
            "  事件聚合要求窗口按时间排——打乱过的话，连续命中这个概念就不存在了，"
            "算出来的事件级指标没有任何意义。\n"
            "  imu_train 的 val 分割如果是随机切的，就得改成按片段/按时间切，"
            "或者拿一段原始连续数据来评。\n"
            "  真要看的话加 --force，但**别拿那个数做决定**。")

    bundle = joblib.load(resolve_model(args.model))
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    ad = ModelAdapter(model, class_names=names)
    print(f"模型类型：{ad.kind}，变小的轴是 **{ad.axis_name}**（最大 {ad.total}）")

    true_ev = to_events(y == fi, args.min_windows, args.max_gap)
    print(f"真值里「{args.focus}」聚合出 {len(true_ev)} 次事件"
          f"（min_windows={args.min_windows}, max_gap={args.max_gap}）\n")

    hdr = (f"{ad.axis_name:>8}{'窗口 P':>10}{'窗口 R':>10}{'窗口 F1':>10}"
           f"{'  │':>4}{'事件数':>8}{'事件 P':>10}{'事件 R':>10}{'事件 F1':>10}")
    print(hdr)
    print("-" * len(hdr))
    axis = [int(v) for v in args.rounds.split(",") if v] or ad.default_axis()
    n_trees = args.trees or None
    if (n_trees or args.quantize_leaves) and ad.kind != "rf":
        sys.exit("--trees / --quantize-leaves 只对随机森林有意义，"
                 f"这个模型是 {ad.kind}。")
    if n_trees or args.quantize_leaves:
        bits = []
        if n_trees:
            bits.append(f"只取前 {n_trees} 棵")
        if args.quantize_leaves:
            bits.append("叶子量化成 uint8")
        print(f"（{('、'.join(bits))}——跟端上一致）\n")
    for r in axis:
        t = ad.variant(r, n_trees=n_trees, quantize_leaves=args.quantize_leaves)
        pred = np.array([int(np.argmax(t.scores(x))) for x in X])
        hits = pred == fi
        tp = int(np.sum(hits & (y == fi)))
        fp = int(np.sum(hits & (y != fi)))
        fn = int(np.sum(~hits & (y == fi)))
        wp, wr, wf = prf(tp, fp, fn)

        pred_ev = to_events(hits, args.min_windows, args.max_gap)
        etp, efp, efn = match_events(pred_ev, true_ev)
        ep, er, ef = prf(etp, efp, efn)
        print(f"{r:>8}{wp:>10.3f}{wr:>10.3f}{wf:>10.3f}{'  │':>4}"
              f"{len(pred_ev):>8}{ep:>10.3f}{er:>10.3f}{ef:>10.3f}")

    print(f"""
{ad.caveat()}

窗口级和事件级差多少，决定了"能减到多少"。

  - 事件级明显好于窗口级 → 误报大多是**孤立的单窗口**，聚合能吃掉。
    那就可以放心往下减轮数，省下来的 flash 比那点窗口级 precision 值钱。
  - 两者差不多 → 误报是**成片**出现的（模型在某类信号上系统性搞错），
    聚合救不了。这时候减轮数就是真的在牺牲产品质量。

另外可以调 --min-windows：它是事件级 precision 和 recall 之间最直接的旋钮，
调大了误报少、漏报多。跟轮数一起扫一遍，可能用"少轮数 + 大 min_windows"
就能换回 precision，而那是**零 flash 成本**的。""")


if __name__ == "__main__":
    main()
