#!/usr/bin/env bash
# 上板那一侧的入口：交叉编译 / 出包 / 看占用。
#
#   ./board.sh build           交叉编译（要 SDK_ROOT 和 arm-none-eabi-gcc）
#   ./board.sh build rf        编 RF 那份模型（默认是 CNN）
#   ./board.sh size            只看代码和 RAM 占多少，不用 SDK
#   ./board.sh release         出一份可烧录的包（release.sh）
#   ./board.sh flash           打印烧录步骤（docs/flash.md）
#
# 跟 ./serve.sh 的关系：**两边编的是同一份 core/ 里的 C**。
# 服务那边把它编成 .so 用 Python 调，这边编进固件。各留一份的话迟早分家，
# 而分家的表现是"平台上看着对、板上不对"——查不到。

set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

MODEL=${MODEL:-edge_cnn_i8}
case "${2:-}" in
    rf)  MODEL=edge_rf_d10 ;;
    cnn) MODEL=edge_cnn_i8 ;;
esac
GEN="$ROOT/core/models/$MODEL"

need_model() {
    [ -d "$GEN" ] || { echo "模型目录不存在：$GEN"; exit 1; }
    # 上板之前必须确认这不是自测的演示导出——它编得过、跑得通、自检也过，
    # 烧进去只会得到一堆看着正常的错结论
    python3 scripts/check_export.py "$GEN" || {
        echo ""
        echo "模型没过检查，**别烧**。"
        exit 1
    }
}

case "${1:-}" in
    build)
        need_model
        command -v arm-none-eabi-gcc >/dev/null || { echo "没有 arm-none-eabi-gcc"; exit 1; }
        [ -n "${SDK_ROOT:-}" ] || { echo "要给 SDK_ROOT（GR551x SDK 路径）"; exit 1; }
        echo "▶ 模型 $MODEL"
        cd board/tinyml_app/GCC && make SDK_ROOT="$SDK_ROOT" GEN_DIR="$GEN"
        ;;
    size)
        # 不需要 SDK：只编我们自己这几个文件，看代码段多大
        need_model
        command -v arm-none-eabi-gcc >/dev/null || { echo "没有 arm-none-eabi-gcc"; exit 1; }
        tmp=$(mktemp -d)
        for f in core/tm_runtime.c core/tm_prep.c core/tm_post.c core/tm_post_cfg.c \
                 core/tm_features.c core/tm_forest_c.c; do
            [ -f "$f" ] || continue
            arm-none-eabi-gcc -c -Os -std=c99 -mcpu=cortex-m4 -mthumb \
                -mfpu=fpv4-sp-d16 -mfloat-abi=hard -ffp-contract=off -fno-math-errno \
                -ffunction-sections -fdata-sections -Icore -I"$GEN" \
                "$f" -o "$tmp/$(basename "$f" .c).o" 2>/dev/null || echo "  (跳过 $f)"
        done
        arm-none-eabi-size "$tmp"/*.o
        rm -rf "$tmp"
        ;;
    release)
        need_model
        MODEL="$MODEL" ./release.sh
        ;;
    flash)
        sed -n '1,60p' docs/flash.md
        ;;
    *)
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
