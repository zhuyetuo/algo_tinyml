#!/usr/bin/env bash
# 起端侧推理服务（给标注平台调）——**PC 那一侧的入口**。
#
# **正式部署不用这个脚本**：端侧服务由 imu_train 的 docker compose 起
# （label_service/docker-compose.yml 里的 edge-service，`cd ~/imu_train && ./up.sh deploy`
# 一起起停），这个仓库只是代码库。这里留着是给本机调试用的。
#
# 跟 ./board.sh 的关系：两边编的是同一份 core/ 里的 C。这边编成 .so 给
# Python 调，那边编进固件。各留一份的话迟早分家，而分家的表现是
# "平台上看着对、板上不对"——查不到。
#
#   ./serve.sh            拉代码 + 停旧的 + 前台起（能看到自检输出）
#   ./serve.sh -d         同上，但后台起，日志落到 logs/edge_service.log
#   ./serve.sh stop       停掉
#   ./serve.sh status     看在不在跑
#
# 挂哪些模型看 edge_models.json。**加模型改那个文件，不用改这个脚本。**
# 里面的路径支持 glob（训练产出目录带日期批次），但必须唯一匹配——
# 匹配到多个会当场报错并列出候选，不替你挑。挑错了不报错，
# 只会让平台上的结果对应到另一份模型。
#
# 覆盖默认值（一般用不到）：
#   PORT=8901 NAS_ROOT=/mnt/nas MODELS=别的.json ./serve.sh

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

# ── 模型清单 ──────────────────────────────────────────────────────────────
# **加模型改 edge_models.json，不改这个脚本。**
# 以前这里给每个模型写死一串 glob，加一个模型要动三处；漏掉一处的表现是
# 服务照常起来，只是少了一个模型——不报错，只是那个模型在平台上不存在。
MODELS=${MODELS:-$ROOT/edge_models.json}
[ -f "$MODELS" ] || { echo "模型清单不存在：$MODELS（用 MODELS= 指定）"; exit 1; }
[ -d "$NAS_ROOT" ] || { echo "NAS 根不存在：$NAS_ROOT（用 NAS_ROOT= 覆盖）"; exit 1; }

# ── 拉代码 ────────────────────────────────────────────────────────────────
if [ -d .git ] && [ "${SKIP_PULL:-0}" != "1" ]; then
    echo "▶ git pull"
    git pull --ff-only 2>&1 | tail -2
fi

stop_it

ARGS=(
    # **-u（不缓冲）**：不加的话 Python 发现 stdout 不是终端就会开缓冲，
    # 后台起的时候日志文件**一直是空的**，直到缓冲满或进程退出。
    # 服务是常驻的，那意味着实际上永远看不到日志。
    python -u "$ROOT/service/edge_service.py"
    --models "$MODELS"
    --imu-train "$IMU_TRAIN"
    --nas-root "$NAS_ROOT"
    --host "$HOST" --port "$PORT"
)

echo "▶ 模型清单  $MODELS"
sed -n 's/.*"tag" *: *"\([^"]*\)".*/    · \1/p' "$MODELS"
echo "▶ NAS       $NAS_ROOT"
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
