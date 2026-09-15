#!/usr/bin/env bash
# 一条命令跑完端侧选型要的全部分析。
#
# 为什么要有这个脚本：前面那几步是多行命令 + 一堆路径，粘贴的时候很容易把
# 上一次的输出一起粘回去执行（已经发生过一次，终端直接刷屏报错）。
# 一条短命令就没这个问题。
#
# 用法：
#   bash analyze.sh                      # 用下面的默认路径
#   bash analyze.sh <模型.pkl>           # 换个模型
#   DATE=xxx bash analyze.sh             # 换个数据批次
#
# 要在装了 sklearn/xgboost 的机器上跑。

set -uo pipefail
cd "$(dirname "$0")"

IMU=${IMU:-$HOME/imu_train}
DATE=${DATE:-2026_8_11-2026_8_27_raw_missing_drop_window}
PD="$IMU/data/processed_$DATE"
RES="$IMU/results/processed_$DATE/16hz_remap_custom_3class"
RES_EDGE="$IMU/results_edge/processed_$DATE/16hz_remap_custom_3class"
MODEL=${1:-$RES/xgb/ml_xgb.pkl}
FEATS="$IMU/holdout_feats.npy"
LABELS="$IMU/holdout_y.npy"
CLASSES=${CLASSES:-活动,睡觉,抓挠,未佩戴,甩身体}
FOCUS=${FOCUS:-抓挠}

hr() { printf '\n%s\n%s\n' "════════ $1 ════════" ""; }

[ -f "$MODEL" ] || { echo "模型不存在：$MODEL"; echo "用法：bash analyze.sh <模型.pkl>"; exit 1; }

# 留出集只导一次。--force-dump 能强制重导（换了数据/remap 之后要）
if [ ! -f "$FEATS" ] || [ "${FORCE_DUMP:-0}" = "1" ]; then
    hr "导留出集"
    ( cd "$IMU" && python "$OLDPWD/python/dump_holdout.py" \
        --processed-dir "data/processed_$DATE" --hz 16 \
        --remap configs/remap_custom_3class.yaml ) || exit 1
else
    echo "留出集已存在（FORCE_DUMP=1 可强制重导）：$FEATS"
fi

hr "1. 体积 × 轮数（当前编码）"
python python/prune_gbdt.py --model "$MODEL" --features "$FEATS" --labels "$LABELS" \
    --classes "$CLASSES" --focus "$FOCUS"

hr "2. 体积 × 轮数（紧凑编码口径）"
python python/prune_gbdt.py --model "$MODEL" --features "$FEATS" --labels "$LABELS" \
    --classes "$CLASSES" --focus "$FOCUS" --per-node 6.125

hr "3. 事件级指标"
python python/event_eval.py --model "$MODEL" --features "$FEATS" --labels "$LABELS" \
    --classes "$CLASSES" --focus "$FOCUS" --rounds 15,30,50,100,200

hr "4. 两个零 flash 成本的旋钮"
python python/tune_operating_point.py --model "$MODEL" \
    --features "$FEATS" --labels "$LABELS" \
    --classes "$CLASSES" --focus "$FOCUS" \
    --rounds 50,100,200 --min-windows 3,5,8,12 --bias=-1.5,-1,-0.5,0,0.5

hr "5. 树用到了哪些特征"
python python/feature_usage.py --model "$MODEL" --channels 8 2>/dev/null \
    || echo "（跳过：feature_usage 目前只支持 sklearn 的树，XGBoost 的还没做）"

# ── 随机森林那条，同一套分析跑一遍 ────────────────────────────────────
# 注意 RF 的轴是 max_depth 不是轮数，而且**截断是近似的**（脚本会在输出里提醒）
RF_FULL="$RES/rf/ml_rf.pkl"
if [ -f "$RF_FULL" ]; then
    hr "6. RF（不限深那个）：体积 —— 这个数一直没量过"
    python python/rf_footprint.py --model "$RF_FULL"

    hr "7. RF：深度 × 叶子最小样本数"
    python python/prune_rf.py --model "$RF_FULL" \
        --features "$FEATS" --labels "$LABELS" --min-samples-leaf 1,5,10,20

    hr "8. RF：事件级"
    python python/event_eval.py --model "$RF_FULL" --features "$FEATS" --labels "$LABELS" \
        --classes "$CLASSES" --focus "$FOCUS"

    hr "9. RF：两个零成本旋钮（偏置范围按概率给，比 GBDT 小两个数量级）"
    python python/tune_operating_point.py --model "$RF_FULL" \
        --features "$FEATS" --labels "$LABELS" --classes "$CLASSES" --focus "$FOCUS" \
        --min-windows 3,5,8,12 --bias=-0.3,-0.2,-0.1,0,0.1

    hr "10. RF：树用了哪些特征"
    python python/feature_usage.py --model "$RF_FULL" --channels 8
else
    echo ""
    echo "（跳过 RF：$RF_FULL 不存在）"
fi

echo
echo "════════ 跑完了 ════════"
echo "1~5 是 xgb，6~10 是 RF。"
echo "重点：第 4/9 步（旋钮能不能补回来）、第 6 步（RF 的真实体积，一直没量过）。"
