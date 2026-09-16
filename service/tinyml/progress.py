"""进度条。有 tqdm 就用 tqdm，没有就用自带的。

为什么不直接依赖 tqdm：这些脚本要在训练机、标注平台、我这台开发机之间来回跑，
多一个依赖就多一次"在你那台跑不起来"。而"跑了十分钟不知道是在算还是卡死了"
这件事本身就是上一轮的问题——不能靠"装个包就好了"来解决。

自带那份只写 stderr，所以 `python xxx.py > out.txt` 之后 out.txt 里是干净的结果，
进度条不会混进去。不是终端（重定向、CI）时自动闭嘴，只在开始和结束各打一行。
"""

import sys
import time


class _Bar:
    def __init__(self, total, desc, stream, min_seconds):
        self.total = max(int(total), 0)
        self.desc = desc
        self.stream = stream
        self.min_seconds = min_seconds
        self.n = 0
        self.start = time.time()
        self.shown = False
        self.tty = hasattr(stream, "isatty") and stream.isatty()

    def _render(self):
        el = time.time() - self.start
        # **只有跑得久才显示**。短任务弹一下进度条又消失，除了刷屏没有别的作用
        if not self.shown and el < self.min_seconds:
            return
        if not self.tty:
            if not self.shown:
                self.stream.write(f"{self.desc}：{self.total} 条，进行中……\n")
                self.stream.flush()
                self.shown = True
            return
        self.shown = True
        frac = self.n / self.total if self.total else 1.0
        width = 28
        filled = int(width * frac)
        # 剩余时间按已用时间线性外推。前几个百分点会很不准，所以等到 2% 之后再报
        eta = ""
        if frac > 0.02:
            eta = f" 剩 {self._fmt((el / frac) * (1 - frac))}"
        self.stream.write(
            f"\r{self.desc} [{'█' * filled}{'·' * (width - filled)}] "
            f"{self.n}/{self.total} {frac * 100:5.1f}% 用时 {self._fmt(el)}{eta}   ")
        self.stream.flush()

    @staticmethod
    def _fmt(s):
        s = int(s)
        return f"{s}s" if s < 60 else (f"{s // 60}分{s % 60:02d}秒" if s < 3600
                                       else f"{s // 3600}时{s % 3600 // 60:02d}分")

    def update(self, k=1):
        self.n += k
        self._render()

    def close(self):
        if not self.shown:
            return
        el = time.time() - self.start
        if self.tty:
            self.stream.write("\r" + " " * 90 + "\r")
        self.stream.write(f"{self.desc}：{self.n} 条，用时 {self._fmt(el)}\n")
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def bar(total, desc="", stream=None, min_seconds=1.0):
    """返回一个有 update()/close() 的进度条，可以当上下文管理器用。"""
    return _Bar(total, desc, stream or sys.stderr, min_seconds)


def track(it, total=None, desc="", stream=None, min_seconds=1.0):
    """包一个可迭代对象。`for x in track(xs, desc="评估"):`"""
    if total is None:
        try:
            total = len(it)
        except TypeError:
            total = 0
    b = bar(total, desc, stream, min_seconds)
    try:
        for x in it:
            yield x
            b.update()
    finally:
        b.close()


def chunks(n, size):
    """把 range(n) 切成若干段 (start, stop)。批量推理按段走，
    这样进度条能动，而不是"一个大矩阵乘跑十分钟然后突然结束"。"""
    if size <= 0:
        raise ValueError(f"size={size} 要为正")
    for s in range(0, n, size):
        yield s, min(s + size, n)
