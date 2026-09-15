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
    return _write_csv_at(path, n_rows, HZ, seed, null_from)


def _write_csv_at(path, n_rows, hz, seed=0, null_from=None):
    """指定采样率。**重采样那条路只有 hz != 16 时才走得到**，
    而真实数据是 50Hz 的 _raw.csv——线上每个样本都走那条。"""
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
            ts = (t0 + np.timedelta64(int(i * 1000 / hz), "ms")).astype(str)
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


def _rf_export(tmp_dir, with_forest_golden=True, with_pipeline_golden=True, seed=3):
    """导一份带（或不带）golden 的 RF。返回 (目录, forest)。"""
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_c import export as export_forest
    from tinyml.export_pipeline_c import export as export_pipeline
    from tinyml.features import n_features
    from tinyml.forest import Forest

    nfeat = n_features(N_CH)
    rng = np.random.default_rng(seed)
    feat, thr, left, right, offs, leaves = [], [], [], [], [0], []
    for _ in range(4):
        base = len(feat)
        feat += [int(rng.integers(0, nfeat)), len(leaves), len(leaves) + 1]
        thr += [float(rng.normal(0, 1)), 0.0, 0.0]
        left += [base + 1, -1, -1]
        right += [base + 2, -1, -1]
        for _ in range(2):
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
        class_names=tuple(CLASSES))

    gx = rng.normal(0, 1, (6, nfeat)).astype(np.float32) if with_forest_golden else None
    files = export_forest(forest, golden_x=gx)
    files.update(export_feat_cfg(N_T, N_CH, N_T, float(HZ)))
    if with_pipeline_golden:
        wins = rng.normal(0, 1, (4, N_T, N_CH)).astype(np.float32)
        files.update(export_pipeline(forest, wins, float(HZ), nperseg=N_T))
    for name, content in files.items():
        (tmp_dir / name).write_text(content, encoding="utf-8")
    return tmp_dir, forest


def test_rf_golden_selftest_passes(tmp_path):
    """导了 golden 的话，C 算出来必须跟 Python 参考**逐位**相同。

    比位模式不是比差值：用容差的话，"编译器开了 -ffast-math" 这种问题会被
    放过去——它造成的差异往往正好在容差里面，但会随输入放大。
    """
    import serve
    d = tmp_path / "g"
    d.mkdir()
    _rf_export(d)
    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "g.so")))
    report = eng.selftest()
    assert len(report) == 2
    for name, n, bad in report:
        assert n > 0, f"{name} 一条 golden 都没有"
        assert bad == 0, f"{name} 有 {bad} 个值对不上"


def test_missing_golden_is_not_reported_as_pass(tmp_path):
    """**没有 golden 不算通过。**

    导出时忘了给 --features/--windows 就是这个结果，而"0 条全部通过"
    是这类自检最经典的失效方式——它永远是绿的，而且完全没有验证任何东西。
    """
    import serve
    d = tmp_path / "nog"
    d.mkdir()
    _rf_export(d, with_forest_golden=False, with_pipeline_golden=False)
    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "n.so")))
    for name, n, bad in eng.selftest():
        assert bad == -2, f"{name} 没有 golden 却报了 {bad}"
        assert n == 0


def test_golden_selftest_catches_a_tampered_model(tmp_path):
    """把导出的森林改一个阈值，自检必须红。

    这一条钉的是"自检真的在比"——上面两条只能说明它跑通了，
    说明不了它有没有在比对。改一个阈值是最小的、最像"不小心"的改动。
    """
    import re

    import serve
    d = tmp_path / "tamper"
    d.mkdir()
    _rf_export(d)
    src = (d / "tm_forest_model.c").read_text(encoding="utf-8")
    # 改**叶子**，不是改阈值。
    #
    # 第一版改的是根节点阈值，结果自检是绿的——原阈值是 -2.56，而特征是
    # N(0,1)，本来就几乎全走右边，改成更负的仍然全走右边，一条都没翻。
    # 改叶子是确定性的：每个输入必定落到某个叶子上。
    #
    # 顺带：字面量是**十六进制浮点**（0x1.8p-2f）不是十进制——导出器故意用它，
    # 因为十六进制能精确往返。我第一版按十进制写正则，一个都匹配不上。
    line = next(ln for ln in src.splitlines() if "tm_forest_leaf" in ln)
    tampered = re.sub(r"-?0x[0-9a-f.]+p[+-]\d+f", "0x1.0p-1f", line)
    assert tampered != line, f"没改动任何叶子：{line[:120]}"
    (d / "tm_forest_model.c").write_text(src.replace(line, tampered, 1),
                                         encoding="utf-8")

    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "t.so")))
    bad_total = sum(bad for _, _, bad in eng.selftest() if bad > 0)
    assert bad_total > 0, "改了模型自检却没红——那它什么都没在比"


def test_golden_selftest_catches_a_one_ulp_difference(tmp_path):
    """叶子只改 **1 个 ULP**，自检也必须红。

    这一条钉的是"比的是 float 的位模式，不是差值小于某个阈值"。
    上面那条改叶子改得很狠，用 1e-3 的容差照样能发现——它证明不了这一点
    （变异测试里把位比较换成容差比较，那条照样绿）。

    为什么在意 1 个 ULP：位模式比较真正要抓的是 `-ffast-math`、
    "被优化成乘倒数"这类编译选项问题。它们造成的差异往往就是几个 ULP，
    正好落在任何合理的容差里面——**但它会随输入放大**。
    """
    import re
    import struct

    import serve
    d = tmp_path / "ulp"
    d.mkdir()
    _rf_export(d)
    src = (d / "tm_forest_model.c").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if "tm_forest_leaf" in ln)
    def nudge(mo):
        """按 float32 的位表示加 1，再打回十六进制浮点。

        直接在十六进制字面量上改一位是不行的——float32 尾数 24 位，
        而字面量写了 13 位十六进制（float64 的宽度），改错位置会一步跨过好几个 ULP。
        """
        v = np.float32(float.fromhex(mo.group(0).rstrip("f")))
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        return f"{struct.unpack('<f', struct.pack('<I', bits + 1))[0].hex()}f"

    # **所有**叶子各推 1 ULP，不是只推第一个。
    # 只推第一个时自检是绿的：那个叶子是树 0 的左叶，而根阈值是 -2.56、
    # 特征是 N(0,1)，几乎没有输入会走到它——跟上面改阈值那次是同一个陷阱。
    tampered = re.sub(r"-?0x[0-9a-f.]+p[+-]\d+f", nudge, line)
    assert tampered != line, "没改动任何叶子"
    (d / "tm_forest_model.c").write_text(src.replace(line, tampered, 1),
                                         encoding="utf-8")

    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "u.so")))
    bad_total = sum(bad for _, _, bad in eng.selftest() if bad > 0)
    assert bad_total > 0, \
        "差了 1 个 ULP 自检却没红——那它比的是容差，不是位模式"


# ── 紧凑编码那条（--compact 导出的） ──────────────────────────────────────


def _rf_export_compact(tmp_dir, seed=3, golden=True):
    """导一份**紧凑编码**的 RF（7 B/节点 + uint8 叶子）。"""
    from tinyml.export_features_c import export as export_feat_cfg
    from tinyml.export_forest_compact_c import export as export_compact
    from tinyml.features import n_features
    from tinyml.forest import Forest
    from tinyml.forest_compact import CompactForest

    nfeat = n_features(N_CH)
    rng = np.random.default_rng(seed)
    feat, thr, left, right, offs, leaves = [], [], [], [], [0], []

    def build(d):
        i = len(feat)
        if d == 0:
            feat.append(len(leaves))
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            p = np.full(N_CLS, 0.02)
            p[int(rng.integers(0, N_CLS))] = 0.92
            leaves.append((p / p.sum()).astype(np.float32))
            return i
        feat.append(int(rng.integers(0, nfeat)))
        thr.append(float(rng.normal(0, 1)))
        left.append(-1)
        right.append(-1)
        li, ri = build(d - 1), build(d - 1)
        left[i], right[i] = li, ri
        return i

    for _ in range(4):
        build(3)
        offs.append(len(feat))

    forest = Forest(
        n_features=nfeat, n_classes=N_CLS,
        tree_offset=np.asarray(offs, np.int32),
        node_feature=np.asarray(feat, np.int32),
        node_threshold=np.asarray(thr, np.float32),
        node_left=np.asarray(left, np.int32),
        node_right=np.asarray(right, np.int32),
        leaf_proba=np.stack(leaves).astype(np.float32),
        class_names=tuple(CLASSES))
    cf = CompactForest(forest)
    gx = rng.normal(0, 1, (8, nfeat)).astype(np.float32) if golden else None
    files = export_compact(cf, golden_x=gx)
    files.update(export_feat_cfg(N_T, N_CH, N_T, float(HZ)))
    for name, content in files.items():
        (tmp_dir / name).write_text(content, encoding="utf-8")
    return cf


@pytest.fixture(scope="module")
def rfc_runner(tmp_path_factory):
    import edge_service
    import serve
    edge_service.add_imu_train(IMU_TRAIN)
    d = tmp_path_factory.mktemp("gen_rfc")
    _rf_export_compact(d)
    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "rfc.so")))
    meta = {"classes": CLASSES, "window_size": N_T, "hz": HZ, "stride": N_T // 2,
            "gravity_aligned": True, "label_mode": "majority"}
    return edge_service.EdgeRunner("edge_rf_d10", eng, meta, IMU_TRAIN, kind="rf")


def test_build_rf_picks_compact_automatically(rfc_runner):
    """导出目录里有 tm_forest_c_model.c 就走紧凑那条，不用传开关。

    传开关的话，导的是紧凑版而开关忘了改，会以一堆 include 错误的形式炸出来，
    而不是一句话说清缺什么。
    """
    e = rfc_runner.engine
    assert e.n_features == 193 and e.feat_dim == 193
    assert e.classes == CLASSES


def test_compact_golden_selftest_passes(rfc_runner):
    """紧凑版的 golden 存的是**整数票数**——整数累加跟指令集无关，
    所以对不上一定是编码或解析错了，不可能是"数值误差"。"""
    report = rfc_runner.engine.selftest()
    forest = [r for r in report if "森林" in r[0]][0]
    assert forest[1] > 0, "一条 golden 都没有"
    assert forest[2] == 0, f"{forest[2]} 个票数对不上"


def test_compact_has_no_pipeline_golden_and_says_so(rfc_runner):
    """紧凑版暂时没有"整条链"的 golden。必须报 -2（没有），**不能报 0**——
    "0 条全部通过"永远是绿的，而且什么都没验。"""
    report = rfc_runner.engine.selftest()
    pipe = [r for r in report if "整条链" in r[0]][0]
    assert pipe[2] == -2 and pipe[1] == 0


def test_compact_csv_to_segments_end_to_end(rfc_runner, tmp_path):
    """紧凑 RF 也走同一条 infer_file，片段结构跟 CNN 那条一样。"""
    p = tmp_path / "rfc.csv"
    _write_csv(str(p), n_rows=HZ * 90, seed=13)
    out = rfc_runner.infer(str(p), HZ, min_windows=1, max_gap=2, targets=CLASSES)
    assert out["n_windows"] > 0
    for label, segs in out["segments"].items():
        for seg in segs:
            for k in ("start_ts", "end_ts", "conf_max", "conf_mean"):
                assert k in seg, f"{label} 的片段缺字段 {k}"
            assert 0.0 <= seg["conf_max"] <= 1.0


def test_compact_probabilities_sum_to_one(rfc_runner):
    """C 里把整数票数还原成概率给平台用。和必须是 1——
    除以的是票数总和，不是棵数（叶子量化之后每棵树贡献的总和不再是 255）。"""
    from tinyml.edge_model import EdgeRF
    m = EdgeRF(rfc_runner.engine, CLASSES)
    X = np.random.default_rng(17).normal(0, 1, (10, N_T, N_CH)).astype(np.float32)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_compact_pipeline_golden_covers_feature_order(tmp_path):
    """整条链的 golden：窗口 → 特征 → 森林，**整数票数**。

    比只验森林多覆盖一样东西：**特征的排列顺序**。那一维错位不会崩、
    不会报错，只会让每个阈值都对到别的特征上，而模型照样给得出结果——
    这是 RF 这条路上唯一没被别的自检覆盖的接缝。
    """
    import serve
    from tinyml.export_forest_compact_c import pipeline_golden

    d = tmp_path / "pg"
    d.mkdir()
    cf = _rf_export_compact(d)
    W = np.random.default_rng(23).normal(0, 1, (6, N_T, N_CH)).astype(np.float32)
    (d / "tm_forest_c_pipeline_golden.h").write_text(
        pipeline_golden(cf, W, float(HZ), N_T), encoding="utf-8")

    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "pg.so")))
    pipe = [r for r in eng.selftest() if "整条链" in r[0]][0]
    assert pipe[1] == 6, f"应该有 6 条 golden，实际 {pipe[1]}"
    assert pipe[2] == 0, f"{pipe[2]} 个票数对不上——多半是特征顺序或窗口尺寸错了"


def test_pipeline_golden_refuses_wrong_feature_dim(tmp_path):
    """窗口的通道数/长度跟森林训练时不一致，必须**在生成 golden 时就炸**。

    不拦的话会生成一份"自洽但错误"的 golden：板上跑出来跟它一致，
    于是自检通过，而整条链算的是另一套特征。
    """
    from tinyml.export_forest_compact_c import pipeline_golden
    d = tmp_path / "bad"
    d.mkdir()
    cf = _rf_export_compact(d)
    W = np.zeros((2, N_T, 6), np.float32)       # 6 通道，森林是 8 通道训的
    with pytest.raises(ValueError, match="维特征"):
        pipeline_golden(cf, W, float(HZ), N_T)


def test_pipeline_selftest_actually_compares(tmp_path):
    """把 golden 里的票数改一个，整条链自检必须红。

    **这一条是补一个真实的漏**：上面那条只断言"没有不一致"，
    而一个根本不做比较的实现也满足它——变异测试里把 C 侧的比较整段删掉，
    测试照样绿。要证明它真的在比，只能改数据看它红。
    """
    import re

    import serve
    from tinyml.export_forest_compact_c import pipeline_golden

    d = tmp_path / "tamperp"
    d.mkdir()
    cf = _rf_export_compact(d)
    W = np.random.default_rng(29).normal(0, 1, (4, N_T, N_CH)).astype(np.float32)
    g = pipeline_golden(cf, W, float(HZ), N_T)

    # 改第一个票数。加 1 就够——整数比较没有容差这回事
    m = re.search(r"(_pipeline_votes\[\] = \{)(-?\d+)", g)
    assert m, "没找到票数数组"
    g = g[:m.start(2)] + str(int(m.group(2)) + 1) + g[m.end(2):]
    (d / "tm_forest_c_pipeline_golden.h").write_text(g, encoding="utf-8")

    eng = serve.RfEngine(serve.build_rf(str(d), out_so=str(d / "tp.so")))
    pipe = [r for r in eng.selftest() if "整条链" in r[0]][0]
    assert pipe[2] > 0, "改了 golden 自检却没红——那它根本没在比"


# ── 采样率：重采样那条路 ──────────────────────────────────────────────────


def test_resampling_path_works_when_device_hz_differs(runner, tmp_path):
    """device_hz != model_hz 时要走重采样，而**我之前的端到端测试全传的
    device_hz == 16**，正好绕开了这条路。

    真实数据是 50Hz 的 `_raw.csv`，所以线上每一个样本都走这里。
    第一次跑批 303 个全失败就是这一段炸的。
    """
    p = tmp_path / "hz50.csv"
    _write_csv_at(str(p), n_rows=50 * 60, hz=50, seed=31)
    out = runner.infer(str(p), device_hz=50, min_windows=1, max_gap=2,
                       targets=["抓挠"])
    # 50Hz 一分钟 → 16Hz 约 960 点 → (960-16)/8+1 = 119 窗口左右
    assert out["n_windows"] > 100, f"只出了 {out['n_windows']} 个窗口，重采样没生效？"


def test_float_sample_rate_is_accepted_when_integral():
    """**这是那个真实 bug 的回归测试。**

    平台传过来的 sample_hz 经过 JSON 会变成 50（int）或 50.0（float），
    而我在服务里写了 `float(...)`——imu_train 的 downsample 用
    math.gcd(device_hz, model_hz)，gcd 只吃整数，50.0 直接抛
    "TypeError: 'float' object cannot be interpreted as an integer"。

    303 个样本全失败，而且每个只用 0.4 秒——快得根本来不及读完一个
    18 万行的 CSV。那个"快"本该早点提醒我是前置步骤炸了。
    """
    import edge_service
    assert edge_service._as_hz(50.0) == 50
    assert edge_service._as_hz(50) == 50
    assert edge_service._as_hz("16") == 16
    assert isinstance(edge_service._as_hz(50.0), int)


def test_fractional_sample_rate_is_refused_not_rounded():
    """真正的小数率要**报错，不是四舍五入**。

    重采样比是按整数比算的（gcd）。把 49.8 当成 50，整条时间轴会慢慢漂，
    而每一段片段的起止时间看起来一直是正常的——最难发现的那种错。
    """
    import edge_service
    with pytest.raises(ValueError, match="不是整数"):
        edge_service._as_hz(49.8)
    with pytest.raises(ValueError, match="不合法"):
        edge_service._as_hz(0)


def test_gcd_really_rejects_floats():
    """钉住这个前提本身：math.gcd 不吃 float。

    哪天 Python 放宽了这个限制，上面那条防护就成了多余的——
    但在那之前，它是必须的。
    """
    from math import gcd
    with pytest.raises(TypeError):
        gcd(50.0, 16)


def test_http_layer_handles_a_50hz_file(runner, tmp_path):
    """**走 Handler._one，不是直接调 runner.infer。**

    这一条是补一个真实的漏：上面那些端到端测试全是直接调 runner.infer()，
    而 `device_hz` 的类型转换在 `_one()` 里——HTTP 那一层从来没被测到。
    于是我在 `_one()` 里写的 `float(...)` 一路绿着上了线，
    线上 303 个样本全失败。

    变异测试也证实了：把 `_as_hz` 换回 `float`，上面那些测试**全是绿的**。
    """
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.runners = {"edge_cnn_i8": runner}
    h.default_tag = "edge_cnn_i8"
    h.nas_root = str(tmp_path)

    _write_csv_at(str(tmp_path / "raw50.csv"), n_rows=50 * 60, hz=50, seed=41)
    # device_hz 按 JSON 过来的样子给：**float**，跟平台实际发的一致
    r = h._one({"path": "raw50.csv", "mode": "raw", "device_hz": 50.0})
    assert "error" not in r, f"HTTP 层失败了：{r.get('error')}"
    assert r["n_windows"] > 100
    assert r["model_path"] == "edge://edge_cnn_i8.edge"


def test_http_layer_rejects_fractional_hz_with_a_clear_message(runner, tmp_path):
    import edge_service
    h = edge_service.Handler.__new__(edge_service.Handler)
    h.runners = {"edge_cnn_i8": runner}
    h.default_tag = "edge_cnn_i8"
    h.nas_root = str(tmp_path)
    _write_csv_at(str(tmp_path / "x.csv"), n_rows=200, hz=50, seed=1)
    r = h._one({"path": "x.csv", "mode": "raw", "device_hz": 49.8})
    assert "不是整数" in (r.get("error") or ""), r
