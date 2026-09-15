"""端侧推理服务：CSV 进，片段出，整条链走一遍。

这是**唯一一条覆盖"接缝"的测试**。各段自己都有对照测试了：
tm_prep、tm_invoke、EdgeCNN 的轴序、imu_train 的预处理。
但接起来之后才会出现的错——通道顺序、窗口长度、采样率、类别名对不上下标——
单元测试一个都抓不到，而它们全都**不报错，只是效果差一截**。

用真实的 `imu_train/src/infer_csv_scratch.infer_file()`，不是仿造的。
这台机器上没装 joblib（离线装不上），而 infer_csv_scratch 在模块顶层 import 它——
但只有 main() 里加载 .pkl 和并行会用到，infer_file 这条路上不碰。
所以打个桩，验的仍然是真代码。
"""

import csv
import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

IMU_TRAIN = os.path.expanduser(os.environ.get("IMU_TRAIN", "~/imu_train"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(IMU_TRAIN, "src", "infer_csv_scratch.py")),
    reason=f"找不到 imu_train（{IMU_TRAIN}），跳过端到端测试")

HZ, N_T, N_CH, N_CLS = 16, 16, 8, 5
CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]


@pytest.fixture(scope="module", autouse=True)
def _stub_joblib():
    """infer_csv_scratch 顶层 import joblib，但 infer_file 用不到它。"""
    if "joblib" not in sys.modules:
        try:
            import joblib  # noqa: F401
        except ImportError:
            sys.modules["joblib"] = types.SimpleNamespace(
                load=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("infer_file 不该用到 joblib")),
                Parallel=None, delayed=None)
    yield


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    import edge_service
    import serve
    from tinyml import export, quantize
    from tinyml.net import Conv1D, Dense, FloatNet, MaxPool1D

    edge_service.add_imu_train(IMU_TRAIN)

    rng = np.random.default_rng(0)
    layers, ic, t = [], N_CH, N_T
    for oc in (16, 32, 32):
        layers += [Conv1D(rng.normal(0, np.sqrt(2 / (ic * 3)),
                                     (oc, ic, 3)).astype(np.float32),
                          rng.normal(0, .05, oc).astype(np.float32),
                          relu=True, pad=1), MaxPool1D(2)]
        ic, t = oc, t // 2
    layers.append(Dense(rng.normal(0, np.sqrt(2 / (ic * t)),
                                   (N_CLS, ic * t)).astype(np.float32),
                        np.zeros(N_CLS, np.float32)))
    calib = rng.normal(0, 1.5, (64, N_CH, N_T)).astype(np.float32)
    meta = {"classes": CLASSES, "window_size": N_T, "hz": HZ, "stride": N_T // 2,
            "n_channels": N_CH, "gravity_aligned": True, "label_mode": "majority",
            "ch_mean": [0.0] * N_CH, "ch_std": [1.0] * N_CH}
    q = quantize(FloatNet(layers), calib, class_names=CLASSES)
    gen = tmp_path_factory.mktemp("gen_svc")
    for name, content in export(q, golden_x_i8=np.stack(
            [q.quantize_input(x) for x in calib[:4]]), prep=meta).items():
        (gen / name).write_text(content, encoding="utf-8")
    eng = serve.Engine(serve.build(str(gen), out_so=str(gen / "s.so")))
    return edge_service.EdgeRunner("edge_cnn_i8", eng, meta, IMU_TRAIN)


@pytest.fixture(scope="module")
def hot_label(runner, tmp_path_factory):
    """这个随机权重的模型在随机数据上实际会报哪一类。

    直接拿「抓挠」当目标是不行的：随机模型在随机数据上很可能一段都不报，
    于是所有"段数应该变少"之类的断言都在比空集，恒真。
    先问一遍模型，用它真会报的那一类，测试才验得到东西。
    """
    d = tmp_path_factory.mktemp("hot")
    p = d / "probe.csv"
    _write_csv(str(p), n_rows=HZ * 300, seed=5)
    out = runner.infer(str(p), HZ, min_windows=1, max_gap=2, targets=CLASSES)
    best = max(CLASSES, key=lambda c: len(out["segments"].get(c) or []))
    if not out["segments"].get(best):
        pytest.skip("这个模型在测试数据上一段都不报，段数相关的断言验不到东西")
    return best


def _write_csv(path, n_rows, seed=0, null_from=None):
    """造一段带时间戳的 IMU CSV，列名用 load_csv 认得的那套。"""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2026-08-11T09:00:00")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 列名必须是 load_csv 的 ACC_CANDIDATES / GYRO_CANDIDATES 认得的那几套。
        # 我第一版写的是 WitMotion 导出的原始表头（AccX(g) / AsX(°/s)），
        # load_csv 找不到列直接抛——**这本身就说明这条测试在验真东西**：
        # 列名不对时它不会安静地给出一堆零，而是当场炸。
        w.writerow(["timestamp", "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z"])
        for i in range(n_rows):
            ts = (t0 + np.timedelta64(int(i * 1000 / HZ), "ms")).astype(str)
            if null_from is not None and i >= null_from:
                w.writerow([ts, "", "", "", "", "", ""])
                continue
            a = rng.normal(0, 0.3, 3) + np.array([0, 0, 1.0])
            g = rng.normal(0, 20, 3)
            w.writerow([ts] + [f"{v:.5f}" for v in list(a) + list(g)])


# ── 整条链 ────────────────────────────────────────────────────────────────


def test_csv_to_segments_end_to_end(runner, tmp_path):
    """CSV 进，片段出。这一条过了，说明通道数、窗口长度、采样率、
    类别顺序这几件事在整条链上是一致的。"""
    p = tmp_path / "a.csv"
    _write_csv(str(p), n_rows=HZ * 60, seed=1)
    out = runner.infer(str(p), device_hz=HZ, min_windows=1, max_gap=2,
                       targets=["抓挠"])
    assert out["n_windows"] > 0, "一个窗口都没有——多半是窗口/步长算错了"
    assert "抓挠" in out["segments"]
    for seg in out["segments"]["抓挠"]:
        # 平台的 flatten_segments 要这四个字段，缺一个那段就被静默丢掉
        for k in ("start_ts", "end_ts", "conf_max", "conf_mean"):
            assert k in seg, f"片段缺字段 {k}，平台会把它当成坏数据丢掉"
        assert 0.0 <= seg["conf_max"] <= 1.0


def test_window_count_matches_the_sliding_window_formula(runner, tmp_path):
    """窗口数要能对上 (N - window)/stride + 1。对不上说明降采样或步长错了，
    而那会让时间轴整体偏移——片段时间全错，但每一段看起来都正常。"""
    n_rows = HZ * 30
    p = tmp_path / "b.csv"
    _write_csv(str(p), n_rows=n_rows, seed=2)
    out = runner.infer(str(p), device_hz=HZ, min_windows=1, max_gap=2, targets=["抓挠"])
    # **用 meta 里的值算期望，不是用 runner.stride**——后者是循环论证：
    # stride 取错了，期望值跟着一起错，测试照样绿（变异测试发现的）。
    win, stride = runner.meta["window_size"], runner.meta["stride"]
    expect = (n_rows - win) // stride + 1
    assert out["n_windows"] == expect, \
        f"窗口数 {out['n_windows']}，按 meta（窗口 {win}、步长 {stride}）应该是 {expect}"


def test_stride_comes_from_meta_not_a_guess(runner):
    """步长取错会让时间轴整体偏移——每一段片段看起来都正常，时间全是错的。"""
    assert runner.stride == runner.meta["stride"]
    assert runner.window_size == runner.meta["window_size"]


def test_missing_data_is_reported_not_swallowed(runner, tmp_path):
    """后半段全是空值。missing_seconds 必须报出来——

    平台拿它判断"这段数据是不是因为蓝牙断联而不可信"。
    不报的话平台按 0 处理，也就是"数据完好"，而那是最不该默认的方向。
    """
    n_rows = HZ * 60
    p = tmp_path / "c.csv"
    _write_csv(str(p), n_rows=n_rows, seed=3, null_from=n_rows // 2)
    out = runner.infer(str(p), device_hz=HZ, min_windows=1, max_gap=2, targets=["抓挠"])
    assert out["missing_seconds"] > 25, \
        f"一半数据是空的，missing_seconds 只报了 {out['missing_seconds']:.1f} 秒"


def test_short_file_returns_empty_not_crash(runner, tmp_path):
    """文件比一个窗口还短。返回空结果，不能崩——
    平台批量跑的时候一个坏文件不该让整批失败。"""
    p = tmp_path / "d.csv"
    _write_csv(str(p), n_rows=5, seed=4)
    out = runner.infer(str(p), device_hz=HZ, min_windows=1, max_gap=2, targets=["抓挠"])
    assert out["n_windows"] == 0
    assert out["segments"]["抓挠"] == []


def test_min_windows_filters_short_segments(runner, tmp_path, hot_label):
    """min_windows 是端上 tinyml_task 的那个参数，必须真的传到 infer_file。

    **第一版写的是"tight 的段数 <= loose 的段数"，那条是废的**：
    两边都是 0 时恒真，而随机权重的模型在随机数据上很可能一段都不报。
    变异测试里把 min_windows 写死成 1，测试照样绿。
    改成直接看每一段的窗口数——这是个确定性断言，不依赖模型报不报。
    """
    p = tmp_path / "e.csv"
    _write_csv(str(p), n_rows=HZ * 300, seed=5)
    loose = runner.infer(str(p), HZ, min_windows=1, max_gap=2, targets=[hot_label])
    tight = runner.infer(str(p), HZ, min_windows=5, max_gap=2, targets=[hot_label])
    assert loose["segments"][hot_label], "宽松设置下一段都没有，这条测试验不到东西"
    for seg in tight["segments"][hot_label]:
        assert seg["n_windows"] >= 5, \
            f"min_windows=5 却返回了只有 {seg['n_windows']} 个窗口的段"
    assert len(tight["segments"][hot_label]) < len(loose["segments"][hot_label]), \
        "收紧 min_windows 之后段数一点没少——参数多半没传进去"


def test_gravity_align_flag_actually_reaches_the_pipeline(runner, tmp_path, hot_label):
    """重力对齐开关必须真的起作用。

    关掉它不会报错，只会让绝对姿态信息还留在 acc 里、而模型是按对齐过的
    数据训的——效果差一截，查不出来。所以直接验"开和关结果不同"，
    而不是验某个具体数值。
    """
    import copy

    import edge_service
    off_meta = copy.deepcopy(runner.meta)
    off_meta["gravity_aligned"] = False
    off = edge_service.EdgeRunner("off", runner.engine, off_meta, IMU_TRAIN)
    assert off.gravity_aligned is False and runner.gravity_aligned is True

    p = tmp_path / "grav.csv"
    _write_csv(str(p), n_rows=HZ * 120, seed=8)
    a = runner.infer(str(p), HZ, min_windows=1, max_gap=2, targets=[hot_label])
    b = off.infer(str(p), HZ, min_windows=1, max_gap=2, targets=[hot_label])
    assert a["segments"] != b["segments"], \
        "开不开重力对齐结果一模一样——这个开关多半没传到 infer_file"


def test_multiple_targets_do_not_leak_into_each_other(runner, tmp_path):
    """两个目标类别各自一份片段，不能混。
    混了的话平台上"甩身体"那一栏会出现抓挠的段，而时间是对的、看不出来。"""
    p = tmp_path / "f.csv"
    _write_csv(str(p), n_rows=HZ * 90, seed=6)
    out = runner.infer(str(p), HZ, min_windows=1, max_gap=2,
                       targets=["抓挠", "甩身体"])
    assert set(out["segments"]) == {"抓挠", "甩身体"}


# ── 服务层 ────────────────────────────────────────────────────────────────


def test_nas_path_cannot_escape_the_root(tmp_path):
    """`../` 穿出挂载点要被拦住。服务是内网的，但一个能读任意文件的
    HTTP 接口不该因为"内网"就放过去。"""
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.nas_root = str(tmp_path)
    (tmp_path / "ok.csv").write_text("x", encoding="utf-8")
    assert h._resolve("ok.csv").endswith("ok.csv")
    with pytest.raises(ValueError, match="穿出"):
        h._resolve("../../etc/passwd")


def test_unsupported_mode_is_refused_not_faked(runner):
    """stable/viterbi 是 algo_service 的后处理，端上没有。

    **假装支持比报错糟得多**：对比表里两列看着可比，实际是拿两个不同的
    东西在比，而且没有任何迹象。
    """
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.runners = {"edge_cnn_i8": runner}
    h.default_tag = "edge_cnn_i8"
    h.nas_root = "/"
    r = h._one({"path": "/x", "mode": "viterbi"})
    assert "error" in r and "viterbi" in r["error"]


def test_unknown_model_tag_lists_what_exists(runner):
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.runners = {"edge_cnn_i8": runner}
    h.default_tag = "edge_cnn_i8"
    h.nas_root = "/"
    r = h._one({"path": "/x", "model": "不存在的"})
    assert "error" in r and "edge_cnn_i8" in r["error"]


def test_model_path_encodes_the_tag(runner, tmp_path):
    """平台用 model_path 的文件名（去后缀）当 model_tag。
    这个字段错了，两个端侧模型的结果会被记成同一个，互相覆盖。"""
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.runners = {"edge_cnn_i8": runner}
    h.default_tag = "edge_cnn_i8"
    h.nas_root = str(tmp_path)
    p = tmp_path / "g.csv"
    _write_csv(str(p), n_rows=HZ * 20, seed=7)
    r = h._one({"path": "g.csv", "mode": "raw"})
    assert r.get("model_path") == "edge://edge_cnn_i8.edge", r
    assert os.path.splitext(os.path.basename(r["model_path"]))[0] == "edge_cnn_i8"


# ── RF 那条路线 ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rf_runner(tmp_path_factory):
    """一个小森林，走 tm_features + tm_forest 的真实 C 路径。

    树是手工造的（不需要 sklearn，这台机器上也没有）：验的是**导出 + C 推理 +
    跟 infer_file 的接缝**，sklearn 只是上游来源，它的解析另有测试。
    """
    import edge_service
    import serve
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_c import export as export_forest
    from tinyml.features import n_features
    from tinyml.forest import Forest

    edge_service.add_imu_train(IMU_TRAIN)
    nfeat = n_features(N_CH)
    rng = np.random.default_rng(3)

    # 每棵树：根按某个特征分，两个叶子。简单但走的是真实的遍历代码
    n_trees = 6
    feat, thr, left, right, offs, leaves = [], [], [], [], [0], []
    for t in range(n_trees):
        base = len(feat)
        feat += [int(rng.integers(0, nfeat)), len(leaves), len(leaves) + 1]
        thr += [float(rng.normal(0, 1)), 0.0, 0.0]
        left += [base + 1, -1, -1]
        right += [base + 2, -1, -1]
        for _ in range(2):
            # **叶子要有峰**，像真森林那样。第一版用的是均匀随机再归一化，
            # 6 棵树一平均就趋近 1/5——于是"没有被 softmax 压平"那条断言
            # 测的不是 softmax，是我的假数据本来就接近均匀。
            p = np.full(N_CLS, 0.02)
            p[int(rng.integers(0, N_CLS))] = 0.92
            leaves.append((p / p.sum()).astype(np.float32))
        offs.append(len(feat))

    forest = Forest(
        n_features=nfeat, n_classes=N_CLS,
        tree_offset=np.asarray(offs, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        class_names=tuple(CLASSES),
    )
    gen = tmp_path_factory.mktemp("gen_rf")
    files = export_forest(forest)
    files.update(export_feat_cfg(N_T, N_CH, N_T, float(HZ)))
    for name, content in files.items():
        (gen / name).write_text(content, encoding="utf-8")

    eng = serve.RfEngine(serve.build_rf(str(gen), out_so=str(gen / "rf.so")))
    meta = {"classes": CLASSES, "window_size": N_T, "hz": HZ, "stride": N_T // 2,
            "n_channels": N_CH, "gravity_aligned": True, "label_mode": "majority",
            "ch_mean": [0.0] * N_CH, "ch_std": [1.0] * N_CH}
    return edge_service.EdgeRunner("edge_rf_d10", eng, meta, IMU_TRAIN, kind="rf")


def test_rf_feature_dim_matches_the_forest(rf_runner):
    """tm_features 产出的维度必须等于森林训练时的维度。

    对不上**不会崩**——森林按下标去读越界的特征，读到相邻内存，
    给出一个看起来完全正常的概率。所以这一条在构造时就拦。
    """
    e = rf_runner.engine
    assert e.feat_dim == e.n_features == 193


def test_rf_csv_to_segments_end_to_end(rf_runner, tmp_path):
    """RF 也走同一条 infer_file，出来的片段结构必须跟 CNN 那条一样——
    平台那边是同一套解析代码。"""
    p = tmp_path / "rf.csv"
    _write_csv(str(p), n_rows=HZ * 90, seed=11)
    out = rf_runner.infer(str(p), HZ, min_windows=1, max_gap=2, targets=CLASSES)
    assert out["n_windows"] > 0
    for label, segs in out["segments"].items():
        for seg in segs:
            for k in ("start_ts", "end_ts", "conf_max", "conf_mean"):
                assert k in seg, f"{label} 的片段缺字段 {k}"


def test_rf_probabilities_are_not_softmaxed_again(rf_runner, tmp_path):
    """森林给的已经是概率（各树叶子概率的平均）。再过一次 softmax 会把分布
    压平，置信度全错——而那不会报错，只是所有片段的 conf 都往 0.2 靠。"""
    from tinyml.edge_model import EdgeRF
    m = EdgeRF(rf_runner.engine, CLASSES)
    X = np.random.default_rng(7).normal(0, 1, (12, N_T, N_CH)).astype(np.float32)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    # 均匀分布是 1/5=0.2。softmax 过一遍会把最大值压到 0.2 附近；
    # 没过的话，随机叶子概率的最大值应当明显高于均匀
    # **直接跟"再过一次 softmax 会变成什么"比**，不拍阈值。
    # 前两版我拍了 0.25 和 0.5，一个太松一个太紧——而"多少算有峰"取决于
    # 几棵树投给几个类别，是个跟被测行为无关的量。自校准的比较没这个问题。
    from tinyml.edge_model import softmax
    flattened = softmax(p)
    assert p.max() > flattened.max() * 1.5, (
        f"森林给的最大概率 {p.max():.3f}，跟再过一次 softmax 的结果 "
        f"{flattened.max():.3f} 差不多——多半是真的又过了一次")


def test_rf_is_dl_flag_is_true(rf_runner):
    """infer_file 靠这个字段决定要不要在 Python 里算手工特征。

    RF 这条**特征也在 C 里算**，所以要拿到的是原始窗口，必须是 True。
    写成 False 的话 infer_file 会先算一遍 scipy 特征再喂进来——
    形状还正好是 193，不会报错，但算的是另一套。
    """
    assert rf_runner.model.is_dl is True


def test_rf_features_come_from_c_not_scipy(rf_runner):
    """C 的特征跟 Python 参考实现对得上（同一份算法的两个实现）。

    注意这里比的**不是** imu_train 的 scipy 版——那两者做不到逐位一致
    （float64 vs float32、FFT 实现不同），tm_features.h 顶部写明了。
    比的是 python/tinyml/features.py，它是 C 那份的参考实现。
    """
    from tinyml.features import extract_one
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (N_CH, N_T)).astype(np.float32)
    got = rf_runner.engine.features(x)                      # C 吃 [C, T]
    # **extract_one 吃的是 [T, C]，跟 C 相反。**第一版我直接把 [C,T] 传进去，
    # 它当成 T=8、C=16 算出了 281 维——形状检查当场拦下了。
    # 这正是这类接缝测试要抓的错，只不过这次是在测试里犯的。
    want = np.asarray(extract_one(x.T, float(HZ), nperseg=N_T), np.float32)
    assert got.shape == want.shape
    bad = np.flatnonzero(~np.isclose(got, want, rtol=1e-4, atol=1e-5))
    assert not len(bad), (
        f"{len(bad)} 维特征对不上，头几个下标 {bad[:5].tolist()}\n"
        f"C={got[bad[0]]}  Python={want[bad[0]]}")


def test_feature_dim_mismatch_is_refused_at_load_time(tmp_path):
    """森林按 N 维训的，但 tm_features 产出 M 维——**必须在加载时就炸**。

    这不是假想的错法：export_rf.py 的 --channels 填成 6（而模型是 8 通道训的）
    就会得到这个结果。而它**不会崩也不会报错**——森林按下标去读越界的特征，
    读到相邻内存，给出一个看起来完全正常的概率。
    等到推理时才发现是不可能的，因为那时候没有任何异常可看。
    """
    import serve
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_c import export as export_forest
    from tinyml.forest import Forest

    # 森林按 6 通道（171 维）训，特征配置按 8 通道（193 维）导 —— 对不上
    forest = Forest(
        n_features=171, n_classes=N_CLS,
        tree_offset=np.asarray([0, 1], np.int32),
        node_feature=np.asarray([0], np.int32),
        node_threshold=np.asarray([0.0], np.float32),
        node_left=np.asarray([-1], np.int32),
        node_right=np.asarray([-1], np.int32),
        leaf_proba=np.full((1, N_CLS), 1.0 / N_CLS, np.float32),
        class_names=tuple(CLASSES),
    )
    gen = tmp_path / "mismatch"
    gen.mkdir()
    files = export_forest(forest)
    files.update(export_feat_cfg(N_T, 8, N_T, float(HZ)))   # 8 通道 → 193 维
    for name, content in files.items():
        (gen / name).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="特征维度对不上"):
        serve.RfEngine(serve.build_rf(str(gen), out_so=str(gen / "bad.so")))
