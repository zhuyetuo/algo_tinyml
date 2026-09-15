"""服务端那份 .so 必须跟 Python 参考实现**逐位相同**。

这条不成立的话，整个服务就只是"一个跑得挺快的东西"——你在网页上看到的效果
跟板上会发生的事没有可证的关系，而那正是这个仓库要消灭的。

注意验的是**整条链**：tm_prep（归一化+量化）+ tm_invoke（推理）。
各自都已经有单独的对照测试了，这里补的是"接起来之后还对"——
接线错误（比如把归一化做了两遍、或者窗口的通道/时间维转置了）在单元测试里
一个都抓不到，而它恰恰是最容易犯的。
"""

import errno
import os
import socket
import subprocess
import sys
from http.server import BaseHTTPRequestHandler

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from tinyml import export, forward_int, quantize  # noqa: E402
from tinyml.net import Conv1D, Dense, FloatNet, MaxPool1D  # noqa: E402
from tinyml.torch_import import prep_quantize_ref  # noqa: E402
import serve  # noqa: E402

N_CH, N_T, N_CLS = 8, 16, 5


def _windows(n, seed):
    """带**量纲差异**的假窗口：前 3 路当加速度（±几），后面当角速度（±几百）。

    量纲一样的话，"漏乘某个通道的 std"这类错会被掩盖——所有通道恰好都对。
    """
    rng = np.random.default_rng(seed)
    return np.stack([
        np.stack([rng.normal(0, 3, N_T) if c < 3 else rng.normal(0, 200, N_T)
                  for c in range(N_CH)])
        for _ in range(n)]).astype(np.float32)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    rng = np.random.default_rng(0)
    layers, ic, t = [], N_CH, N_T
    for oc in (16, 32, 32):          # 结构跟真模型一致，宽度收窄只为跑得快
        layers += [Conv1D(rng.normal(0, np.sqrt(2 / (ic * 3)),
                                     (oc, ic, 3)).astype(np.float32),
                          rng.normal(0, .05, oc).astype(np.float32),
                          relu=True, pad=1),
                   MaxPool1D(2)]
        ic, t = oc, t // 2
    layers.append(Dense(rng.normal(0, np.sqrt(2 / (ic * t)),
                                   (N_CLS, ic * t)).astype(np.float32),
                        np.zeros(N_CLS, np.float32)))
    net = FloatNet(layers)

    X = _windows(200, 1)
    meta = {"ch_mean": [float(X[:, c].mean()) for c in range(N_CH)],
            "ch_std": [float(X[:, c].std()) for c in range(N_CH)]}
    mean = np.asarray(meta["ch_mean"], np.float32).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float32).reshape(-1, 1)
    Xn = ((X - mean) / std).astype(np.float32)

    q = quantize(net, Xn[:64], class_names=["活动", "睡觉", "抓挠", "未佩戴", "甩身体"])
    gen = tmp_path_factory.mktemp("gen")
    golden = np.stack([q.quantize_input(x) for x in Xn[:8]])
    for name, content in export(q, golden_x_i8=golden, prep=meta).items():
        (gen / name).write_text(content, encoding="utf-8")

    so = serve.build(str(gen), out_so=str(gen / "tm_host.so"))
    return serve.Engine(so), q, meta, X


def test_golden_selftest_passes(built):
    """导出的 golden vector 用这份 C 重算必须逐位相同。
    这是服务启动时那行 ✓ 背后的东西。"""
    eng = built[0]
    assert eng.selftest() == 0


def test_full_chain_matches_python_bitwise(built):
    """**整条链**：原始量纲窗口 → C 的类别和分数，跟 Python 参考逐位相同。"""
    eng, q, meta, _ = built
    X = _windows(120, seed=7)
    cls, sc = eng.infer(X)

    want_sc, want_cls = [], []
    for x in X:
        xi = prep_quantize_ref(x, meta, q.in_scale, q.in_zp)
        o = forward_int(q, xi)[0]
        want_sc.append(o)
        want_cls.append(int(np.argmax(o)))
    want_sc = np.stack(want_sc)

    bad = np.argwhere(sc != want_sc)
    assert not len(bad), (
        f"{len(bad)} 处分数不一致，头几处 {bad[:5].tolist()}\n"
        f"C={sc[bad[0][0]]}  Python={want_sc[bad[0][0]]}")
    assert np.array_equal(cls, np.array(want_cls))


def test_service_reads_shapes_from_c_not_hardcoded(built):
    """形状全从 C 里问。Python 这边写死的话，换个模型会**静默错位**，
    而错位的表现是"效果突然变差"，不是报错。"""
    eng, q, meta, _ = built
    assert (eng.n_ch, eng.n_t, eng.n_classes) == (q.n_ch, q.n_t, q.n_classes)
    assert eng.classes == list(q.class_names)
    assert eng.ch_mean == pytest.approx(meta["ch_mean"])
    assert eng.ch_std == pytest.approx(meta["ch_std"])
    assert eng.in_scale == pytest.approx(q.in_scale)
    assert eng.in_zp == q.in_zp


def test_batch_matches_single(built):
    """批量接口只是为了少跨几次 ctypes 边界，不能改变结果。"""
    eng = built[0]
    X = _windows(17, seed=11)
    many_c, many_s = eng.infer(X)
    for i, x in enumerate(X):
        c, s = eng.infer(x[None])
        assert int(c[0]) == int(many_c[i])
        assert np.array_equal(s[0], many_s[i])


def test_wrong_shape_is_rejected(built):
    """形状不对要当场报错。numpy 会很乐意广播出一个没有意义的结果。"""
    eng = built[0]
    with pytest.raises(ValueError, match="窗口形状"):
        eng.infer(np.zeros((3, N_CH, N_T + 1), np.float32))
    with pytest.raises(ValueError, match="窗口形状"):
        eng.infer(np.zeros((3, N_T, N_CH), np.float32))   # 通道/时间维转置了


def test_build_refuses_when_export_is_missing(tmp_path):
    """指错目录要给一条**能照着改**的报错，不是 gcc 的一堆 include 错误。"""
    with pytest.raises(SystemExit) as e:
        serve.build(str(tmp_path))
    assert "export_cnn.py" in str(e.value)


def test_build_command_contains_required_flags(monkeypatch, tmp_path):
    """把编译命令截下来看 flag，不真编。

    **-ffp-contract=off 不是可选项**：FMA 收缩会少一次中间舍入，那样服务
    算出来的就不是板上会算出来的东西。断言 flag 在命令里，比"结果碰巧一样"
    可靠——小模型上 FMA 的差异常常表现不出来。
    """
    for f in ("tm_model.c", "tm_model.h", "tm_golden.h"):
        (tmp_path / f).write_text("", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    serve.build(str(tmp_path), out_so=str(tmp_path / "x.so"))
    cmd = seen["cmd"]
    assert "-ffp-contract=off" in cmd, "FMA 收缩没关掉，结果跟板上对不上"
    assert "-fno-math-errno" in cmd
    # 固件的源文件必须是**原样**编进来的，不能是某个拷贝
    assert any(c.endswith("firmware/tinyml/tm_runtime.c") for c in cmd)
    assert any(c.endswith("firmware/tinyml/tm_prep.c") for c in cmd)


# ── 端口 ──────────────────────────────────────────────────────────────────


def _hold(port=0):
    """占住一个端口，返回 (socket, 端口号)。"""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s, s.getsockname()[1]


def test_busy_port_falls_through_to_the_next_one():
    """端口占用太常见了（上一个实例没停、别的服务占着 8080）。
    默认甩一个 OSError 的 traceback——那玩意儿看着像程序坏了，其实换个端口就行。"""
    held, p = _hold()
    try:
        srv = serve.listen("127.0.0.1", p, BaseHTTPRequestHandler)
        try:
            assert srv.server_address[1] != p
            assert srv.server_address[1] == p + 1
        finally:
            srv.server_close()
    finally:
        held.close()


def test_strict_port_refuses_instead_of_moving():
    """端口有时写死在别处的配置里，静默换掉比报错更糟——
    web 那边还连着旧端口，而服务显示"开着了"。"""
    held, p = _hold()
    try:
        with pytest.raises(SystemExit) as e:
            serve.listen("127.0.0.1", p, BaseHTTPRequestHandler, tries=1)
        assert "ss -ltnp" in str(e.value), "报错要说清楚怎么查是谁占的"
    finally:
        held.close()


def test_port_zero_lets_the_kernel_pick():
    srv = serve.listen("127.0.0.1", 0, BaseHTTPRequestHandler)
    try:
        assert srv.server_address[1] > 0
    finally:
        srv.server_close()


def test_non_address_in_use_errors_are_not_swallowed():
    """只有 EADDRINUSE 才该顺延。权限不够（绑 80）之类的错误顺延 20 次
    只会刷 20 遍同样的失败，最后给出一条误导的"都被占着"。"""
    with pytest.raises(OSError) as e:
        serve.listen("203.0.113.1", 9999, BaseHTTPRequestHandler)   # 绑不上的地址
    assert e.value.errno != errno.EADDRINUSE


def test_urls_resolves_a_reachable_address_for_wildcard_bind():
    """绑 0.0.0.0 时只打印 "<服务器 IP>" 是在给人出题。"""
    got = serve.urls("0.0.0.0", 8080)
    assert any("127.0.0.1:8080" in u for u in got)
    assert len(got) >= 2, f"没给出对外地址：{got}"
    assert all(":8080/" in u for u in got)


def test_urls_for_explicit_host_is_just_that_host():
    assert serve.urls("127.0.0.1", 9000) == ["http://127.0.0.1:9000/"]
