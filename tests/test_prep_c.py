"""归一化 + 输入量化：C ↔ Python 逐位对照。

这一段是端侧管线里**唯一跟训练脚本的超参直接绑死**的地方（ch_mean/ch_std 来自
训练集统计）。写错的后果不是崩溃，是"输入分布跟训练时对不上"——效果掉一截，
而模型、推理、数据看起来全都对。所以它必须跟 tm_runtime 一样有逐位对照。

专门构造的用例比随机输入重要得多：
  · 正好落在 .5 的值 —— 银行家舍入 vs 远离零，只有这里分得开
  · 超出 int8 范围的值 —— 验饱和
  · 负值 —— 舍入方向对负数最容易写反
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

from tinyml.torch_import import prep_quantize_ref  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FW = os.path.join(ROOT, "core")

N_CH, N_T = 8, 16


def _cases(meta, scale, zp):
    """既有随机的，也有专挑舍入/饱和边界的。

    只用随机浮点的话，"正好落在 .5"的概率约等于 0，把 round 换成 rint
    （银行家舍入）测试照样绿——那正是最想防住的一种写法。
    """
    rng = np.random.default_rng(0)
    mean = np.asarray(meta["ch_mean"], np.float64).reshape(-1, 1)
    std = np.asarray(meta["ch_std"], np.float64).reshape(-1, 1)

    xs = [rng.normal(0, 3, (N_CH, N_T)), rng.normal(0, 30, (N_CH, N_T))]

    # 反解出让 q 正好等于 k + 0.5 的原始值。
    # **zp 是取整之后才加的**，所以反解式里不能带 zp——带上就变成让
    # q = k+0.5-zp，落不到中点上，这条用例就白写了。
    half = np.empty((N_CH, N_T))
    for c in range(N_CH):
        for t in range(N_T):
            k = (t % 9) - 4                       # 有正有负，含 0
            half[c, t] = (k + 0.5) * scale * std[c, 0] + mean[c, 0]
    xs.append(half)

    # 饱和：远超 int8 两端
    xs.append(np.full((N_CH, N_T), 1e4))
    xs.append(np.full((N_CH, N_T), -1e4))
    # 恰好是均值 → 归一化后为 0 → 量化后应当正好是 zp
    xs.append(np.repeat(mean, N_T, axis=1))
    return np.stack(xs).astype(np.float32)


def _carr(v, suffix="f"):
    """生成 C 数组字面量。

    **用 %.17e 而不是 %g**：%g 会把 -10000.0 打成 "-10000"，加上 f 后缀就成了
    "invalid suffix on integer constant"；而且位数不够的话字面量回读出来
    跟 Python 里的值不是同一个数，逐位对照就名存实亡了。
    17 位十进制能精确往返 float64。
    """
    return "{" + ", ".join(f"{float(x):.17e}{suffix}"
                           for x in np.asarray(v).reshape(-1)) + "}"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    # **mean / std / scale 全取二进制精确的值**，不是随机的。
    # 随机值下 x = (k+0.5)*scale*std + mean 存成 float32 之后落不到精确的中点上，
    # 于是"银行家舍入 vs 远离零"这条分不开——变异成 rint 测试照样绿。
    # （叶子量化那边踩过同一个坑。）
    # 用 2 的幂仍然保留了逐通道差异，所以"漏乘某个通道的 std"照样会被逮到；
    # mean 取非零值，"忘了减均值"也跑不掉。
    meta = {"ch_mean": [0.5, -1.0, 2.0, -0.25, 4.0, -8.0, 0.125, -3.0],
            "ch_std": [1.0, 2.0, 4.0, 0.5, 8.0, 16.0, 0.25, 32.0]}
    scale, zp = 2.0 ** -5, -7
    X = _cases(meta, scale, zp)

    d = tmp_path_factory.mktemp("prep")
    rows = ",\n".join(_carr(x) for x in X)
    (d / "prep_case.h").write_text(
        f"#define TP_N_CH {N_CH}\n#define TP_N_T {N_T}\n"
        f"#define TP_MEAN {_carr(meta['ch_mean'], suffix='')}\n"
        f"#define TP_STD {_carr(meta['ch_std'], suffix='')}\n"
        f"#define TP_SCALE {scale:.17e}\n#define TP_ZP {zp}\n"
        f"#define TP_N_CASES {len(X)}\n"
        f"#define TP_INPUT {{\\\n{rows.replace(chr(10), chr(92) + chr(10))}\\\n}}\n",
        encoding="utf-8")

    exe = d / "host"
    r = subprocess.run([
        "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fsanitize=undefined,address", "-fno-sanitize-recover=all",
        # **必须关掉 FMA 收缩**：a*b+c 被合成一条 fma 指令会少一次中间舍入，
        # 结果跟分开算不同。这不是"更准"的问题，是两边对不上
        "-ffp-contract=off", "-fno-math-errno",
        f"-I{FW}", f"-I{d}",
        os.path.join(FW, "tm_prep.c"), os.path.join(ROOT, "tests", "host_prep.c"),
        "-lm", "-o", str(exe),
    ], capture_output=True, text=True)
    assert r.returncode == 0, f"编译失败：\n{r.stderr}"
    return meta, scale, zp, X, exe


def test_prep_c_matches_python_bitwise(built):
    meta, scale, zp, X, exe = built
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, f"跑失败：\n{r.stderr}"
    got = np.array([[int(v) for v in line.split()]
                    for line in r.stdout.strip().splitlines()], dtype=np.int8)
    want = np.stack([prep_quantize_ref(x, meta, scale, zp).reshape(-1) for x in X])
    assert got.shape == want.shape
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"{len(bad)} 处不一致，头几处 (用例, 下标): {bad[:5].tolist()}\n"
        f"C={got[bad[0][0], bad[0][1]]}  Python={want[bad[0][0], bad[0][1]]}")


def test_mean_input_maps_exactly_to_zero_point(built):
    """输入正好等于均值 → 归一化为 0 → 量化后必须正好是 zero_point。

    这一条钉的是"0 能被精确表示"。做不到的话，"什么都没发生"会变成一个
    非零的偏置，而 padding 补的正是这个 0。
    """
    meta, scale, zp, _, _ = built
    x = np.repeat(np.asarray(meta["ch_mean"], np.float64).reshape(-1, 1), N_T, axis=1)
    assert np.all(prep_quantize_ref(x, meta, scale, zp) == zp)


def test_saturates_both_ends(built):
    meta, scale, zp, _, _ = built
    assert np.all(prep_quantize_ref(np.full((N_CH, N_T), 1e4), meta, scale, zp) == 127)
    assert np.all(prep_quantize_ref(np.full((N_CH, N_T), -1e4), meta, scale, zp) == -128)


# ── 真实参数：这一组专门验 double 不能换成 float ──────────────────────────

REAL_META = {
    # 十进制、非 2 的幂 —— imu_train 的 .json 里就是这样的数。
    # 上面那组用 2 的幂是为了让中点精确（分得开舍入规则），代价是
    # float32 和 float64 在那组参数下结果完全一样，验不到精度这一维。
    "ch_mean": [0.3172839456, -1.0483920184, 2.7391028456, -0.0194837261,
                4.1827364519, -8.3910284756, 0.1249183726, -3.8471920384],
    "ch_std": [17.41592653589, 2.7182818284, 4.6692016091, 0.5772156649,
               8.3144626181, 16.1803398874, 0.2718281828, 32.0000000001],
}
REAL_SCALE, REAL_ZP = 0.0371829, -7


def _float_sensitive_inputs(meta, scale, zp, want=64):
    """搜出一批**用 float 算和用 double 算结果不同**的输入。

    这些点就落在量化的中点附近：float32 少几位尾数，在那里会翻到另一侧。
    实测约 0.35% 的近中点样本会翻——概率不高，但端上一天几万个窗口，
    "偶尔跟 PC 差一个格子"正是最难查的那类不一致。
    """
    mean = np.asarray(meta["ch_mean"], np.float64)
    std = np.asarray(meta["ch_std"], np.float64)

    def q64(x, c):
        q = ((np.float64(x) - mean[c]) * (1.0 / std[c])) / np.float64(scale)
        return np.sign(q) * np.floor(np.abs(q) + 0.5) + zp

    def q32(x, c):
        v = np.float32((np.float32(x) - np.float32(mean[c]))
                       * np.float32(1.0 / np.float32(std[c])))
        q = np.float32(v / np.float32(scale))
        return np.sign(q) * np.floor(np.abs(q) + 0.5) + zp

    found = [[] for _ in range(len(mean))]
    for c in range(len(mean)):
        for k in range(-110, 110):
            xc = (k + 0.5) * scale * std[c] + mean[c]
            for d in range(-300, 300):
                x = np.float32(xc) + np.float32(d) * np.float32(1e-7)
                if q64(x, c) != q32(x, c):
                    found[c].append(float(x))
                    break
            if len(found[c]) >= want:
                break
    n = min(want, min(len(f) for f in found)) if all(found) else 0
    if n == 0:
        pytest.skip("这组参数下搜不到 float/double 会分歧的点")
    return np.array([[f[i] for i in range(n)] for f in found], np.float32)


@pytest.fixture(scope="module")
def built_real(tmp_path_factory):
    X1 = _float_sensitive_inputs(REAL_META, REAL_SCALE, REAL_ZP, want=N_T)
    rng = np.random.default_rng(3)
    X = np.stack([X1, rng.normal(0, 20, (N_CH, N_T)).astype(np.float32)])

    d = tmp_path_factory.mktemp("prep_real")
    rows = ",\n".join(_carr(x) for x in X)
    (d / "prep_case.h").write_text(
        f"#define TP_N_CH {N_CH}\n#define TP_N_T {N_T}\n"
        f"#define TP_MEAN {_carr(REAL_META['ch_mean'], suffix='')}\n"
        f"#define TP_STD {_carr(REAL_META['ch_std'], suffix='')}\n"
        f"#define TP_SCALE {REAL_SCALE:.17e}\n#define TP_ZP {REAL_ZP}\n"
        f"#define TP_N_CASES {len(X)}\n"
        f"#define TP_INPUT {{\\\n{rows.replace(chr(10), chr(92) + chr(10))}\\\n}}\n",
        encoding="utf-8")

    exe = d / "host"
    r = subprocess.run([
        "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fsanitize=undefined,address", "-fno-sanitize-recover=all",
        "-ffp-contract=off", "-fno-math-errno",
        f"-I{FW}", f"-I{d}",
        os.path.join(FW, "tm_prep.c"), os.path.join(ROOT, "tests", "host_prep.c"),
        "-lm", "-o", str(exe),
    ], capture_output=True, text=True)
    assert r.returncode == 0, f"编译失败：\n{r.stderr}"
    return X, exe


def test_prep_stays_double_precision(built_real):
    """C 侧把 double 换成 float 必须让这条红。

    上面那组 2 的幂参数**验不到这一点**——变异成 float 之后测试照样绿
    （变异测试发现的）。真实的 .json 参数下才分得开。
    """
    X, exe = built_real
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, f"跑失败：\n{r.stderr}"
    got = np.array([[int(v) for v in line.split()]
                    for line in r.stdout.strip().splitlines()], dtype=np.int8)
    want = np.stack([prep_quantize_ref(x, REAL_META, REAL_SCALE, REAL_ZP).reshape(-1)
                     for x in X])
    bad = np.argwhere(got != want)
    assert not len(bad), (
        f"{len(bad)} 处不一致，头几处 {bad[:5].tolist()}\n"
        f"C={got[bad[0][0], bad[0][1]]}  Python={want[bad[0][0], bad[0][1]]}\n"
        "这组用例专挑 float/double 会分歧的点，红了多半是精度退化了")


def test_rounds_half_away_from_zero_on_negatives(built):
    """负数侧的 .5：远离零应该给 -1（更负），银行家舍入给 0。

    负数的舍入方向是这类代码最经典的写错处，而随机输入永远踩不到。
    """
    meta, scale, zp = built[0], built[1], built[2]
    std = np.asarray(meta["ch_std"], np.float64).reshape(-1, 1)
    mean = np.asarray(meta["ch_mean"], np.float64).reshape(-1, 1)
    # 让 q == -0.5：x = -0.5*scale*std + mean。
    # **反解式里不带 zp**——zp 是取整之后才加的
    x = -0.5 * scale * std + mean
    x = np.repeat(x, N_T, axis=1)
    got = prep_quantize_ref(x, meta, scale, zp)
    # q == -0.5 → 远离零取整给 -1（银行家舍入会给 0），再加 zero_point
    assert np.all(got == -1 + zp), f"实际 {np.unique(got)}，期望 {-1 + zp}"
