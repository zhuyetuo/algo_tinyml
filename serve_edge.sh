#!/usr/bin/env bash
# 起端侧推理服务（给标注平台调）。
#
#   ./serve_edge.sh            拉代码 + 停旧的 + 前台起（能看到自检输出）
#   ./serve_edge.sh -d         同上，但后台起，日志落到 logs/edge_service.log
#   ./serve_edge.sh stop       停掉
#   ./serve_edge.sh status     看在不在跑
#
# 模型目录和 .json 路径**自动找**，不用每次粘贴一长串。
# 找不到或者找到多个时明确报错并列出候选，不猜。
#
# 覆盖默认值（一般用不到）：
#   PORT=8901 NAS_ROOT=/mnt/nas ./serve_edge.sh

set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

PORT=${PORT:-8900}
HOST=${HOST:-0.0.0.0}
NAS_ROOT=${NAS_ROOT:-/home/toky/ai_data}
IMU_TRAIN=${IMU_TRAIN:-$HOME/imu_train}
PIDFILE=$ROOT/.edge_service.pid
LOGDIR=$ROOT/logs
LOG=$LOGDIR/edge_service.log

# ── 停 ────────────────────────────────────────────────────────────────────
# 用 pidfile 而不是 pkill -f：pkill 的模式会**匹配到发出这条命令的 shell 自己**
# （命令行里就含那串字符），把自己杀掉。这个坑踩过两次。
stop_it() {
    if [ -f "$PIDFILE" ]; then
        local pid
        read -r pid _ < "$PIDFILE" 2>/dev/null || true
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            for _ in $(seq 1 20); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.25
            done
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
            echo "停掉了旧的（pid $pid）"
        fi
        rm -f "$PIDFILE"
    fi
}

case "${1:-}" in
    stop)
        stop_it
        exit 0
        ;;
    status)
        # 端口从 pidfile 里读，**不是读环境变量的默认值**——
        # 服务起在 8931 而 status 报 8900，比不报更糟：人会照那个端口去连
        if [ -f "$PIDFILE" ]; then
            read -r _pid _port < "$PIDFILE"
            if [ -n "${_pid:-}" ] && kill -0 "$_pid" 2>/dev/null; then
                echo "在跑：pid $_pid，端口 ${_port:-未记录}"
                exit 0
            fi
        fi
        echo "没在跑"
        exit 1
        ;;
esac

BG=no
[ "${1:-}" = "-d" ] && BG=yes

# ── 找模型 ────────────────────────────────────────────────────────────────
# 训练产出的目录名带日期批次，写死的话换一批数据就得改脚本。
# 自动找；找到多个就列出来让人选，**不挑一个**——挑错了不会报错，
# 只会让平台上的结果对应到另一份模型。
pick_one() {
    local what=$1 pattern=$2 n
    local -a hits
    mapfile -t hits < <(compgen -G "$pattern" 2>/dev/null || true)
    n=${#hits[@]}
    if [ "$n" -eq 0 ]; then
        echo "找不到 $what（找的是 $pattern）" >&2
        return 1
    fi
    if [ "$n" -gt 1 ]; then
        echo "$what 找到多个，不猜。用环境变量指定其中一个：" >&2
        printf '  %s\n' "${hits[@]}" >&2
        return 1
    fi
    echo "${hits[0]}"
}

CNN_META=${CNN_META:-$(pick_one "CNN 的 .json" \
    "$IMU_TRAIN/results_edge_a/*/16hz_remap_custom_3class/dl_cnn_best.json")} || exit 1
RF_META=${RF_META:-$(pick_one "RF 的 .json" \
    "$IMU_TRAIN/results_edge_rf/*/16hz_remap_custom_3class/rf/ml_rf.json")} || exit 1
CNN_GEN=${CNN_GEN:-$ROOT/firmware/generated_cnn_a}
RF_GEN=${RF_GEN:-$ROOT/firmware/generated_rf}

for d in "$CNN_GEN" "$RF_GEN"; do
    [ -d "$d" ] || { echo "导出目录不存在：$d"; echo "  先跑 export_cnn.py / export_rf.py --compact"; exit 1; }
done
[ -d "$NAS_ROOT" ] || { echo "NAS 根不存在：$NAS_ROOT（用 NAS_ROOT= 覆盖）"; exit 1; }

# ── 拉代码 ────────────────────────────────────────────────────────────────
if [ -d .git ] && [ "${SKIP_PULL:-0}" != "1" ]; then
    echo "▶ git pull"
    git pull --ff-only 2>&1 | tail -2
fi

stop_it

ARGS=(
    python "$ROOT/python/edge_service.py"
    --gen  "edge_cnn_i8=$CNN_GEN"  --meta "edge_cnn_i8=$CNN_META"
    --gen  "edge_rf_d10=$RF_GEN"   --meta "edge_rf_d10=$RF_META"
    --imu-train "$IMU_TRAIN"
    --nas-root "$NAS_ROOT"
    --host "$HOST" --port "$PORT"
)

echo "▶ CNN  $CNN_GEN"
echo "       $CNN_META"
echo "▶ RF   $RF_GEN"
echo "       $RF_META"
echo "▶ NAS  $NAS_ROOT"
echo ""

if [ "$BG" = yes ]; then
    mkdir -p "$LOGDIR"
    "${ARGS[@]}" >>"$LOG" 2>&1 &
    BGPID=$!
    echo "$BGPID $PORT" > "$PIDFILE"
    echo "后台起了（pid $BGPID，端口 $PORT），日志：$LOG"
    # **等自检结果再返回**。直接退出的话，自检失败了人也看不到，
    # 而自检失败意味着后面所有结果都不可信
    for _ in $(seq 1 40); do
        grep -q "开着了" "$LOG" 2>/dev/null && break
        kill -0 "$BGPID" 2>/dev/null || break
        sleep 0.5
    done
    tail -20 "$LOG"
    if ! kill -0 "$BGPID" 2>/dev/null; then
        rm -f "$PIDFILE"
        echo ""
        echo "⚠ 起失败了，看上面。自检没过的话别用这个服务的结果。"
        exit 1
    fi
else
    # 前台：Ctrl-C 直接停，不留 pidfile 残留
    echo "$$ $PORT" > "$PIDFILE"
    trap 'rm -f "$PIDFILE"' EXIT
    exec "${ARGS[@]}"
fi
