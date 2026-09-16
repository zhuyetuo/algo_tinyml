"""已提交的导出必须是**真模型**，而且服务要能从一份干净 clone 直接起。

这两件事都是"错了不报错"型：
  · 假模型编得过、跑得通、golden 自检也过（拿同一份假权重生成的当然对得上），
    烧到板上只会得到一堆看着正常的错结论。
  · meta 指向 ~/imu_train 的话，换台机器 clone 下来服务起不来，
    而模型明明就在仓库里。
"""

import importlib.util
import json
import os
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLESSED = ["core/models/edge_cnn_i8", "core/models/edge_rf_d10"]


def _check_mod():
    p = os.path.join(ROOT, "scripts", "check_export.py")
    spec = importlib.util.spec_from_file_location("_check_export", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tracked(path: str) -> list[str]:
    r = subprocess.run(["git", "ls-files", path], cwd=ROOT,
                       capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


@pytest.mark.parametrize("d", BLESSED)
def test_committed_export_is_a_real_model(d):
    """提交进来的导出要过 check_export。

    **没提交时跳过**，不是当成通过——现在还没提交，等提交了这条自动生效。
    跳过和通过分不开的话，这条检查等于不存在。
    """
    files = _tracked(d)
    if not files:
        pytest.skip(f"{d} 还没提交（提交之后这条自动开始检查）")
    problems = _check_mod().check(os.path.join(ROOT, d))
    assert not problems, f"{d} 没过导出检查：\n" + "\n".join(problems)


@pytest.mark.parametrize("d", BLESSED)
def test_committed_export_ships_its_meta(d):
    """导出目录里必须有 meta.json，而且跟着一起提交。

    没有它就没法核对这是不是真模型，服务也读不到类别/几何。
    """
    files = _tracked(d)
    if not files:
        pytest.skip(f"{d} 还没提交")
    assert f"{d}/meta.json" in files, \
        f"{d}/meta.json 没提交——服务读不到类别和窗口几何，也没法核对真伪"


def test_check_export_catches_a_demo_export(tmp_path):
    """**这条验的是检查本身有用。**

    造一份"演示导出"（4 棵树、2.5 KB）配一份真 meta（20 棵树），
    检查必须拦下来。拦不住的话前面那两条全是摆设。
    （第一版 CNN 的门槛我定在 20 KB，而演示导出是 27 KB，直接就通过了——
    假阴性比没有检查更糟，因为它给了"查过了"的错觉。）
    """
    d = tmp_path / "generated_fake"
    d.mkdir()
    (d / "tm_forest_c_model.h").write_text(
        "#define TM_FC_N_TREES 4\n#define TM_FC_N_NODES 60\n"
        '#define TM_FC_N_CLASSES 3\n'
        'static const char *const TM_FC_CLASS_NAMES[] = {"活动", "睡觉", "抓挠"};\n',
        encoding="utf-8")
    (d / "tm_forest_c_model.c").write_text("x" * 2515, encoding="utf-8")
    (d / "tm_forest_c_golden.h").write_text("#define TM_FC_GOLDEN_N 8\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(
        {"classes": ["活动", "睡觉", "抓挠"], "n_estimators": 20}), encoding="utf-8")

    problems = _check_mod().check(str(d))
    assert problems, "演示导出没被拦下来"
    assert any("棵数对不上" in p for p in problems), problems
    assert any("字节" in p for p in problems), problems


def test_check_export_catches_a_small_cnn(tmp_path):
    """CNN 那一路也要拦得住。这条是补上面那个假阴性的。"""
    d = tmp_path / "generated_fake_cnn"
    d.mkdir()
    (d / "tm_model.h").write_text(
        "#define TM_N_T 16\n#define TM_N_CH 8\n#define TM_N_CLASSES 3\n"
        'static const char *const TM_CLASS_NAMES[] = {"活动", "睡觉", "抓挠"};\n',
        encoding="utf-8")
    (d / "tm_model.c").write_text("x" * 27288, encoding="utf-8")   # 演示导出的真实大小
    (d / "tm_golden.h").write_text("#define TM_GOLDEN_N 4\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(
        {"classes": ["活动", "睡觉", "抓挠"], "window_size": 16, "n_channels": 8}),
        encoding="utf-8")
    problems = _check_mod().check(str(d))
    assert problems, "27 KB 的演示 CNN 导出没被拦下来"


def test_check_export_refuses_without_meta(tmp_path):
    """没有 meta 就**拒绝**，不是"那就当它是真的"。

    「无从核对」和「核对通过」必须分得开。
    """
    d = tmp_path / "generated_nometa"
    d.mkdir()
    (d / "tm_model.h").write_text("#define TM_N_T 16\n", encoding="utf-8")
    problems = _check_mod().check(str(d))
    assert problems and any("meta.json" in p for p in problems)


def test_class_order_mismatch_is_caught(tmp_path):
    """类别顺序错了要拦住——概率会安到别的类别上，不报错。"""
    d = tmp_path / "generated_order"
    d.mkdir()
    (d / "tm_model.h").write_text(
        "#define TM_N_T 16\n#define TM_N_CH 8\n"
        'static const char *const TM_CLASS_NAMES[] = {"睡觉", "活动", "抓挠"};\n',
        encoding="utf-8")
    (d / "tm_model.c").write_text("x" * 80_000, encoding="utf-8")
    (d / "tm_golden.h").write_text("x", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(
        {"classes": ["活动", "睡觉", "抓挠"], "window_size": 16, "n_channels": 8,
         "ch_mean": [0] * 8, "ch_std": [1] * 8}), encoding="utf-8")
    problems = _check_mod().check(str(d))
    assert any("类别对不上" in p for p in problems), problems


# ── 干净 clone 也能起服务 ─────────────────────────────────────────────────


def test_edge_models_prefers_the_in_repo_meta():
    """meta 的第一候选必须是仓库里那份。

    只写 ~/imu_train/... 的话，换台机器 clone 下来服务起不来——
    而模型就在仓库里。这条防的是"顺手把候选列表改回单个路径"。
    """
    with open(os.path.join(ROOT, "edge_models.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for m in cfg["models"]:
        meta = m["meta"]
        assert isinstance(meta, list), f"{m['tag']} 的 meta 不是候选列表"
        assert not meta[0].startswith("~"), \
            f"{m['tag']} 的第一候选是 {meta[0]}，应该是仓库里那份"
        assert meta[0].startswith(m["gen"]), \
            f"{m['tag']} 的第一候选该在它自己的导出目录里"


def test_meta_candidates_fall_back_to_the_training_box():
    """但也要保留训练机上那份作为后备——训练机上重新导出之后
    不用先拷 meta 就能起服务。"""
    with open(os.path.join(ROOT, "edge_models.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for m in cfg["models"]:
        assert len(m["meta"]) >= 2, f"{m['tag']} 没有后备候选"
        assert any("imu_train" in c for c in m["meta"][1:])


def test_blessed_dirs_are_not_gitignored():
    """上板那两份必须能提交。

    .gitignore 里 core/models/generated*/ 会把它们一起挡掉，
    要有对应的 ! 反选——而"挡掉了"的表现是 git add 静默什么都不做。
    """
    for d in BLESSED:
        r = subprocess.run(["git", "check-ignore", f"{d}/tm_model.h"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode != 0, f"{d} 还被 .gitignore 挡着，提交不进去"


def test_other_generated_dirs_are_still_ignored():
    """**别的 generated* 目录仍然要挡住。**

    跟上一条成对：反选写得太宽（比如直接 !core/models/generated*/）的话，
    自测的演示导出也能被 git add -A 扫进来，而那正是这套检查要防的事。
    """
    r = subprocess.run(["git", "check-ignore", "core/models/generated_demo/tm_model.h"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, "core/models/generated_demo/ 没被挡住，演示导出会被扫进仓库"
