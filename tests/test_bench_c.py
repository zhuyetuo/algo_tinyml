"""tm_bench：周期计数的算术。

这层很薄，但每一处写错都会给出一个**看起来很合理的错数**，而"推理要多久"
是拿来定占空比和功耗的，错了不会有人发现。三个真实的坑：

  · 32 位回绕。64MHz 下 67 秒一圈，判大小会给出负的/巨大的耗时。
  · cycles * 1000000 先乘后除在 32 位里直接溢出。
  · DWT 没解锁时计数器恒为 0 —— 表现成"推理 0 微秒"，比报错难发现得多。
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "firmware", "tinyml")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    (d / "m.c").write_text(r"""
#include <stdio.h>
#include "tm_bench.h"
int main(void){
    /* 回绕：t0 接近 32 位上限，t1 已经绕回去了 */
    printf("wrap %u\n", (unsigned)tm_bench_elapsed(0xFFFFFFF0u, 0x0000000Fu));
    printf("plain %u\n", (unsigned)tm_bench_elapsed(1000u, 4000u));
    printf("same %u\n", (unsigned)tm_bench_elapsed(777u, 777u));
    /* 溢出：2^32-1 周期 @64MHz，先乘后除会炸 */
    printf("us_big %u\n", (unsigned)tm_bench_us(0xFFFFFFFFu, 64000000u));
    printf("us_64k %u\n", (unsigned)tm_bench_us(64000u, 64000000u));
    printf("us_zero_hz %u\n", (unsigned)tm_bench_us(1000u, 0u));

    tm_bench_t b; tm_bench_reset(&b);
    printf("empty_mean %u\n", (unsigned)tm_bench_mean(&b));
    tm_bench_accum(&b, 500); tm_bench_accum(&b, 100); tm_bench_accum(&b, 300);
    printf("mean %u min %u max %u n %u\n",
           (unsigned)tm_bench_mean(&b), (unsigned)b.min, (unsigned)b.max, (unsigned)b.n);

    /* 大量累加：total 必须是 64 位，否则 2^32/4e9 就绕了 */
    tm_bench_t c; tm_bench_reset(&c);
    for (int i = 0; i < 2000; i++) tm_bench_accum(&c, 4000000u);
    printf("big_mean %u\n", (unsigned)tm_bench_mean(&c));
    return 0;
}
""", encoding="utf-8")
    exe = d / "a"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{FW}", os.path.join(FW, "tm_bench.c"), str(d / "m.c"), "-o", str(exe)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return dict(
        (line.split()[0], line.split()[1:]) for line in out.stdout.strip().splitlines())


def test_elapsed_handles_32bit_wraparound(run):
    """64MHz 下计数器 67 秒绕一圈。判大小的写法会给出一个巨大的假耗时。"""
    assert int(run["wrap"][0]) == 0x1F


def test_elapsed_plain_and_zero(run):
    assert int(run["plain"][0]) == 3000
    assert int(run["same"][0]) == 0


def test_us_does_not_overflow_on_full_counter(run):
    """cycles * 1000000 在 32 位里 4295 周期就满了。必须走 64 位中间量。
    2^32-1 周期 @64MHz = 67.1 秒 = 67108863 µs。"""
    assert int(run["us_big"][0]) == 67108863


def test_us_basic(run):
    assert int(run["us_64k"][0]) == 1000        # 64000 周期 @64MHz = 1ms


def test_us_with_zero_hz_returns_zero_not_crash(run):
    """cpu_hz 传 0 是配置错误，但除零是未定义行为——UBSan 编译下会直接崩。"""
    assert int(run["us_zero_hz"][0]) == 0


def test_mean_of_empty_is_zero_not_division_by_zero(run):
    assert int(run["empty_mean"][0]) == 0


def test_min_max_mean(run):
    m = run["mean"]
    assert (int(m[0]), int(m[2]), int(m[4]), int(m[6])) == (300, 100, 500, 3)


def test_total_is_64bit(run):
    """2000 次 × 4e6 周期 = 8e9，超过 2^32。total 是 32 位的话均值会绕成乱数。"""
    assert int(run["big_mean"][0]) == 4000000
