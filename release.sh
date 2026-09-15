#!/usr/bin/env bash
# 出一份可以直接烧录的固件产物，连同一份说明放进 release/ 下。
#
# 为什么要这个脚本而不是"自己 make 一下"：
#   · 烧到板子上的那个 .bin，必须能回答"它里面是哪个模型、哪次提交、多大"。
#     光有一个 tinyml_app.bin 躺在 build/ 里，过两周谁也说不清它是什么。
#   · 产物要跟**导出的模型**绑定。model 目录换一份重新 make，文件名一模一样，
#     覆盖掉了没人知道 —— 这是这类项目最常见的一种事故。
#
# 用法：
#   ./release.sh --sdk /path/to/GR551x_SDK --gen firmware/generated_cnn_a
#   ./release.sh --gen firmware/generated_cnn_a          # 没 SDK：只出体积报告
#
# 没有 SDK 也能跑：那时不链接，只编译到 .o 并报体积。体积是现在就能确定的，
# 完整固件要等有 SDK 的机器。**不会假装成功** —— 产出里会写明缺什么。

set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

SDK=""
GEN="firmware/generated_cnn_a"
NOTE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sdk)  SDK="$2"; shift 2 ;;
        --gen)  GEN="$2"; shift 2 ;;
        --note) NOTE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "不认识的参数：$1"; exit 1 ;;
    esac
done

[ -d "$GEN" ] || { echo "模型目录不存在：$GEN"; echo "先跑 export_cnn.py --out $GEN"; exit 1; }
[ -f "$GEN/tm_model.c" ] || { echo "$GEN 里没有 tm_model.c，不是导出目录"; exit 1; }

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="-dirty"
STAMP=$(date +%Y%m%d_%H%M)
OUT="release/${STAMP}_${SHA}${DIRTY}"
mkdir -p "$OUT"

echo "════════ 出包 → $OUT ════════"

# ── 模型这一份到底是什么 ───────────────────────────────────────────────────
# 直接从生成的 .h 里读，不靠人记。记错了不会报错，只会让产物说明是假的
NCH=$(grep -oP '#define TM_N_CH\s+\K\d+'      "$GEN/tm_model.h" 2>/dev/null || echo ?)
NT=$(grep -oP '#define TM_N_T\s+\K\d+'        "$GEN/tm_model.h" 2>/dev/null || echo ?)
NCLS=$(grep -oP '#define TM_N_CLASSES\s+\K\d+' "$GEN/tm_model.h" 2>/dev/null || echo ?)
ARENA=$(grep -oP '#define TM_ARENA_BYTES\s+\K\d+' "$GEN/tm_model.h" 2>/dev/null || echo ?)
CLASSES=$(grep -oP 'TM_CLASS_NAMES\[\] = \{\K[^}]*' "$GEN/tm_model.h" 2>/dev/null | tr -d '"' || echo ?)
MODEL_SHA=$(cat "$GEN"/tm_model.c "$GEN"/tm_model.h 2>/dev/null | sha256sum | cut -c1-16)

cp -r "$GEN" "$OUT/model"

# ── 编译 ──────────────────────────────────────────────────────────────────
BUILT=no
cd firmware/gr551x/tinyml_app/GCC
if [ -n "$SDK" ]; then
    if [ -f "$SDK/platform/soc/linker/gcc/libble_sdk.a" ]; then
        echo "▶ 完整固件（SDK: $SDK）"
        if make SDK_ROOT="$SDK" GEN_DIR="$ROOT/$GEN" firmware 2>&1 | tail -20; then
            BUILT=yes
        fi
    else
        echo "⚠ $SDK 里没有 libble_sdk.a，当成没有 SDK 处理"
        SDK=""
    fi
fi
if [ "$BUILT" != yes ]; then
    echo "▶ 只编译 + 报体积（没有 SDK，链接不了）"
    make GEN_DIR="$ROOT/$GEN" SDK_ROOT=/nonexistent objs 2>&1 | tail -25 \
        || echo "（objs 也需要 SDK 的头文件时会失败，见下）"
fi
cd "$ROOT"

# 体积**不依赖 SDK**：我们这部分的 .c 单独交叉编译一遍就能量。
# 走 Makefile 的 objs 要 SDK 的头文件，没有 SDK 时那条路是死的，
# 但"我们的代码占多少"这个问题现在就能回答，没有理由等。
SIZES=$(
  set -e
  T=$(mktemp -d)
  CF="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Os
      -ffp-contract=off -fno-math-errno -std=c99 -ffunction-sections -fdata-sections"
  for f in firmware/tinyml/tm_runtime.c firmware/tinyml/tm_prep.c \
           firmware/tinyml/tm_bench.c firmware/tinyml/tm_window.c \
           "$GEN"/tm_model.c; do
      [ -f "$f" ] || continue
      arm-none-eabi-gcc $CF -Ifirmware/tinyml -I"$GEN" -c "$f" \
          -o "$T/$(basename "$f" .c).o" 2>/dev/null || true
  done
  arm-none-eabi-size -t "$T"/*.o 2>/dev/null
  rm -rf "$T"
)

# **只有这一次真的链接成功了才拷贝产物。**
# 第一版是"build/ 里有什么就拷什么"，结果把上一次残留的 tinyml_app.bin
# 抄进了一个 README 写着"没有可烧录固件"的目录里——一个自相矛盾、
# 且来源不明的 .bin。那正是这个脚本存在的理由，却差点由它自己制造出来。
B=firmware/gr551x/tinyml_app/GCC/build
if [ "$BUILT" = yes ]; then
    for f in tinyml_app.bin tinyml_app.hex tinyml_app.elf tinyml_app.map; do
        [ -f "$B/$f" ] && cp "$B/$f" "$OUT/"
    done
fi

BINSZ="（没链接）"
[ -f "$OUT/tinyml_app.bin" ] && BINSZ="$(stat -c%s "$OUT/tinyml_app.bin") 字节"

# ── 说明 ──────────────────────────────────────────────────────────────────
cat > "$OUT/README.md" <<EOF
# 端侧固件产物 ${STAMP}

| | |
|---|---|
| 代码提交 | \`${SHA}${DIRTY}\` |
| 模型目录 | \`${GEN}\` |
| 模型指纹 | \`${MODEL_SHA}\`（tm_model.c+h 的 sha256 前 16 位） |
| 输入窗口 | ${NCH} 通道 × ${NT} 点 |
| 类别 | ${CLASSES} |
| 推理 arena | ${ARENA} 字节 |
| 固件 .bin | ${BINSZ} |
| 备注 | ${NOTE:-（无）} |

**模型指纹的用处**：板子上烧的是哪一份模型，只有这个数说得准。
换一份模型重新导出，文件名一模一样，覆盖掉没人知道——这是这类项目最常见的事故。

$([ "$BUILT" = yes ] && echo "固件已链接，可以直接烧。" || cat <<'NOSDK'
## ⚠ 这份产物里**没有可烧录的固件**

链接需要 Goodix 的 GR551x SDK（libble_sdk.a、启动文件、链接脚本），
这台机器上没有。上面的体积数字是编译出来的，是准的；缺的只是链接这一步。

在有 SDK 的机器上补出来：

    ./release.sh --sdk /path/to/GR551x_SDK --gen ${GEN}

SDK 下载：https://www.goodix.com/zh/software_tool/gr551x_sdk
（GitHub 镜像：https://github.com/goodix-ble/GR551x.SDK）
NOSDK
)

## 体积

\`\`\`
${SIZES}
\`\`\`

GR5513BENDU：512KB Flash / 128KB RAM（链接脚本留给应用 112KB）。

**链接脚本里 FLASH 区写的是 8MB**（那是内存映射窗口，不是真实容量），
所以**超出 512KB 链接不会报错**——必须自己看上面 text 那个数。

## 怎么烧

见 \`docs/flash.md\`。一句话版本：GProgrammer 选 \`tinyml_app.hex\`，
或者 J-Link 烧 \`tinyml_app.bin\` 到 0x01000000。

## 烧完第一件事：看串口

波特率 115200。上电应该打印两段：

\`\`\`
tinyml 自检通过：N 条 golden vector 逐位一致
── 推理耗时（实测，主频 64000000 Hz）──
  CNN   首次 xxxxx 周期 = xxx us（占空比 x/1000，按每秒一窗）
\`\`\`

**第一行没出现，或者说自检失败，就别看后面的任何数据。**
自检失败只有一种解释：工具链或编译选项的问题，跟模型无关。
串口上会直接打出按什么顺序排查。
EOF

echo ""
echo "════════ 出完了 ════════"
ls -1 "$OUT"
echo ""
echo "说明：$OUT/README.md"
[ "$BUILT" = yes ] || echo "⚠ 没有 SDK，这份里没有可烧录的固件——README 里写了怎么补。"
