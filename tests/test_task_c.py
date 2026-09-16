"""事件聚合层的测试。

这一层决定射频开销——端侧推理省电全靠它。逻辑不复杂，但每条规则错了都不会报错，
只会让统计出来的"今天抓了几次"是错的，而没人会去怀疑它。

在 PC 上用 gcc 编同一份 C 来测（它没有 SDK 依赖，就是为了能这样测）。
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "board", "tinyml_app", "Src", "user")

HARNESS = r"""
#include <stdio.h>
#include "tinyml_task.h"

/* 从 stdin 读 "类别 时间ms" 一行一个窗口，产生事件就打出来 */
int main(int argc, char **argv)
{
    tm_task_cfg_t cfg = { 2, 3, 2 };
    if (argc > 3) { cfg.event_class = atoi(argv[1]); cfg.min_windows = atoi(argv[2]);
                    cfg.max_gap_windows = atoi(argv[3]); }
    tm_task_t t; tm_task_init(&t, &cfg);
    tm_event_t ev; int cls; unsigned long ms;
    while (scanf("%d %lu", &cls, &ms) == 2) {
        if (tm_task_on_window(&t, cls, (uint32_t)ms, &ev))
            printf("EV %d %u %u %d\n", ev.class_id, ev.start_ms, ev.end_ms, ev.n_windows);
    }
    if (tm_task_flush(&t, &ev))
        printf("EV %d %u %u %d\n", ev.class_id, ev.start_ms, ev.end_ms, ev.n_windows);
    return 0;
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("task")
    (d / "h.c").write_text("#include <stdlib.h>\n" + HARNESS, encoding="utf-8")
    binp = d / "task"
    r = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-fsanitize=undefined", "-fno-sanitize-recover=all",
         f"-I{APP}", os.path.join(APP, "tinyml_task.c"), str(d / "h.c"), "-o", str(binp)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return binp


def _run(exe, windows, cfg=(2, 3, 2)):
    stdin = "\n".join(f"{c} {ms}" for c, ms in windows) + "\n"
    r = subprocess.run([str(exe), *map(str, cfg)], input=stdin,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = []
    for line in r.stdout.strip().splitlines():
        _, cls, s, e, n = line.split()
        out.append((int(cls), int(s), int(e), int(n)))
    return out


def test_连续命中够数才算一次事件(exe):
    ws = [(0, 0), (2, 1000), (2, 2000), (2, 3000), (0, 4000), (0, 5000), (0, 6000)]
    evs = _run(exe, ws)
    assert evs == [(2, 1000, 3000, 3)]


def test_命中不够数的不报(exe):
    """一两个窗口的命中多半是别的动作蹭上来的（甩头、被抱起来）。
    报上去只会污染统计，而统计正是这整条链的产出。"""
    ws = [(0, 0), (2, 1000), (2, 2000), (0, 3000), (0, 4000), (0, 5000)]
    assert _run(exe, ws) == []


def test_中间短暂停顿不拆成两次(exe):
    """抓挠中间会停顿（换姿势、挠另一边）。一停就切断的话，一次连续的抓挠会被
    拆成好几个事件——上报次数变多（更费电），统计出来的"抓了几次"也是错的。"""
    ws = [(2, 0), (2, 1000), (0, 2000), (0, 3000), (2, 4000), (2, 5000), (0, 6000),
          (0, 7000), (0, 8000)]
    evs = _run(exe, ws)
    assert len(evs) == 1, evs
    assert evs[0][1] == 0 and evs[0][2] == 5000 and evs[0][3] == 4


def test_停顿超过上限就分成两次(exe):
    ws = [(2, 0), (2, 1000), (2, 2000)] + [(0, 3000 + i * 1000) for i in range(4)] \
        + [(2, 7000), (2, 8000), (2, 9000), (0, 10000), (0, 11000), (0, 12000)]
    evs = _run(exe, ws)
    assert len(evs) == 2, evs
    assert evs[0][3] == 3 and evs[1][3] == 3


def test_收尾不做的话最后一次事件会丢(exe):
    """数据在事件正中间结束（要睡了 / 要上报汇总了）。不 flush 的话，
    最后那次事件永远发不出去——而它恰好是刚刚发生、最值得报的一个。"""
    ws = [(0, 0), (2, 1000), (2, 2000), (2, 3000)]   # 结尾还在事件里
    evs = _run(exe, ws)
    assert evs == [(2, 1000, 3000, 3)], "flush 没把结尾那次交出来"


def test_事件时间戳用的是命中窗口不是停顿(exe):
    """结束时间要取**最后一次命中**，不是"发现停顿够久"的那一刻。
    取后者的话，每个事件都会凭空多出 max_gap 个窗口的长度。"""
    ws = [(2, 0), (2, 1000), (2, 2000), (0, 3000), (0, 4000), (0, 5000)]
    evs = _run(exe, ws)
    assert evs[0][2] == 2000, f"结束时间应该是 2000（最后一次命中），实际 {evs[0][2]}"
