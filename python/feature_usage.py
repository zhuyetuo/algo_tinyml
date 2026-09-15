"""树到底用了哪些特征？哪些可以**零代价**砍掉？

要在有 sklearn 的机器上跑。

核心是一个精确的事实：**树里从来没被引用过的特征，砍掉不影响任何判决**。
不用重训、不用验证、没有精度风险——那个特征根本没参与过任何一次分裂。
这跟"按 importance 砍"不是一回事：importance 低的特征仍然在参与判决，砍了会变。

砍特征的收益**不在模型体积**（每个节点只用 2 字节存特征下标，砍一半省 1 字节/节点）。
收益在：
  1. **特征提取的算力和代码体积**——这才是大头，见下面的分组；
  2. 每个窗口要算的中间缓冲变少（RAM）。

而且只能**按计算组砍**：同一个通道的 11 个时域统计量共享一趟循环和一次排序，
砍掉其中几个几乎不省时间；要省就得整组砍。

用法：
    python python/feature_usage.py --model xxx.pkl
    python python/feature_usage.py --model xxx.pkl --channels 8
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyml.features import feature_groups, n_features  # noqa: E402

# tools/bench_features.c 在这台机器上实测的相对成本（x86，但各组之间的比例可参考）。
# 单位是"相当于几次时域统计"。绝对值没意义，比例有意义。
COST = {"time": 1.0, "freq": 1.3, "derive+time": 1.2, "cheap": 0.15}



def resolve_model(path):
    """把 --model 的路径解析清楚，不存在就给一条**能照着改**的报错。

    这些脚本在 algo_tinyml 目录下跑，而模型在 imu_train 里——相对路径会解析到
    algo_tinyml 下面去。直接交给 joblib 的话只会甩一个 FileNotFoundError 的
    traceback，看不出是"路径写错了"还是"模型没训出来"。
    """
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return p
    hint = ""
    if not os.path.isabs(p):
        hint = (f"\n  注意这是相对路径，会解析成 {os.path.abspath(p)}。"
                "\n  模型在 imu_train 里的话要写全：~/imu_train/results/...")
    guess = os.path.expanduser("~/imu_train/results")
    if os.path.isdir(guess):
        hint += f"\n  ~/imu_train/results/ 下现有：{', '.join(sorted(os.listdir(guess))[:5]) or '（空）'}"
    sys.exit(f"找不到模型文件：{p}{hint}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--channels", type=int, default=8)
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("没装 joblib/sklearn。这个脚本要在训练机上跑。")

    bundle = joblib.load(resolve_model(args.model))
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    ests = getattr(model, "estimators_", None)
    if ests is None:
        sys.exit(f"没有 estimators_，实际是 {type(model)}")
    # GBDT 的 estimators_ 是二维的（每轮每类一棵）
    flat = []
    for e in np.asarray(ests).reshape(-1):
        if hasattr(e, "tree_"):
            flat.append(e)

    n_feat = int(getattr(model, "n_features_in_", n_features(args.channels)))
    if n_feat != n_features(args.channels):
        print(f"⚠ 模型是 {n_feat} 维，{args.channels} 通道应该是 "
              f"{n_features(args.channels)} 维。--channels 给错了？分组会对不上。")

    counts = np.zeros(max(n_feat, n_features(args.channels)), np.int64)
    total_nodes = 0
    for est in flat:
        t = est.tree_
        cl = np.asarray(t.children_left)
        f = np.asarray(t.feature)
        used = f[cl != -1]          # 只有内部节点才引用特征
        total_nodes += len(cl)
        for i in used:
            if 0 <= i < len(counts):
                counts[i] += 1

    n_internal = int(counts.sum())
    print(f"{len(flat)} 棵树，{total_nodes:,} 个节点，其中内部节点 {n_internal:,}")
    print(f"特征 {n_feat} 维\n")

    never = int(np.sum(counts[:n_feat] == 0))
    print(f"**从来没被用过的特征：{never} / {n_feat}**"
          f"（砍掉它们不影响任何判决，零风险）\n")

    groups = feature_groups(args.channels)
    rows = []
    for name, a, b, kind in groups:
        c = int(counts[a:b].sum())
        rows.append((name, a, b, kind, c, 100.0 * c / max(n_internal, 1)))

    print(f"{'计算组':<22}{'维度':>6}{'被引用':>10}{'占比':>8}{'相对成本':>10}")
    print("-" * 60)
    for name, a, b, kind, c, pct in rows:
        print(f"{name:<22}{b - a:>6}{c:>10,}{pct:>7.1f}%{COST[kind]:>10.2f}")

    # 按「成本 / 贡献」排序，找出砍掉最划算的组
    print("\n砍掉哪些组最划算（成本高但几乎没被用到的排前面）：")
    print("-" * 60)
    ranked = sorted(rows, key=lambda r: (r[5] + 1e-9) / COST[r[3]])
    cum_cost = sum(COST[r[3]] for r in rows)
    saved = 0.0
    lost = 0.0
    for name, a, b, kind, c, pct in ranked[:8]:
        saved += COST[kind]
        lost += pct
        print(f"  砍 {name:<20} 省 {100 * COST[kind] / cum_cost:>4.1f}% 算力，"
              f"放弃 {pct:>4.1f}% 的分裂   （累计：省 {100 * saved / cum_cost:.0f}%，"
              f"放弃 {lost:.1f}%）")

    print(f"""
怎么用这张表：

  - **「从来没被用过」那 {never} 维直接砍**，零风险。注意：这是**针对这一个模型**
    的结论，重训之后可能就用上了。所以砍完要在服务端重训一遍确认。
  - 剩下的按上面的排序砍，砍到「放弃的分裂占比」开始明显上升就停。
  - 砍完一定要**在服务端重训 + 重新评估**。砍特征不是导出时的开关，它改变了
    模型的输入空间。
  - 砍到 **127 维以内**的话，还能用 emlearn 那套更紧凑的编码（见 frameworks.md）。

  相对成本那一列是 tools/bench_features.c 实测的（x86，比例可参考）：
  实测下来**时域和频域基本对半**（42% vs 41%），FFT 并不占主导——
  32 点的 FFT 很便宜，而时域那 11 个统计量里有一次排序。
  所以"砍频域省算力"这个直觉是错的，要砍就得整通道砍。""")


if __name__ == "__main__":
    main()
