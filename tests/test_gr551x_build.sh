#!/usr/bin/env bash
# 交叉编译检查：我们这部分在 arm-none-eabi-gcc 下编不编得过、占多少。
# 需要 arm-none-eabi-gcc 和一份 GR551x SDK。两样缺一就跳过——这不是核心测试，
# 核心一致性测试（tests/）在 PC 上用 gcc 就能跑完。
set -euo pipefail
cd "$(dirname "$0")/.."

command -v arm-none-eabi-gcc >/dev/null || { echo "跳过：没有 arm-none-eabi-gcc"; exit 0; }
[ -n "${SDK_ROOT:-}" ] || { echo "跳过：没给 SDK_ROOT"; exit 0; }

GEN="${GEN_DIR:-$(pwd)/firmware/generated}"
[ -d "$GEN" ] || { echo "跳过：$GEN 不存在，先跑 export_rf.py 或 quantize_and_export.py"; exit 0; }

cd firmware/gr551x/tinyml_app/GCC
make SDK_ROOT="$SDK_ROOT" GEN_DIR="$GEN" objs
