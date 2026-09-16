#!/usr/bin/env bash
# 跑全部测试。需要 python3 + numpy + pytest + gcc，不需要 torch，也不需要板子。
#
# 这些测试会现场把 core/*.c 用 gcc 编出来跟 Python 参考实现对答案，
# 所以"改了 C 忘了改 Python"（或者反过来）在这里就会红，不用等烧板子。
set -euo pipefail
cd "$(dirname "$0")"

command -v gcc >/dev/null || { echo "没有 gcc。C↔Python 对照测试要它。"; exit 1; }
python3 -c "import numpy, pytest" 2>/dev/null || {
    echo "缺依赖：pip install numpy pytest"; exit 1; }

exec python3 -m pytest tests -q "$@"
