"""把 Booster 导成 C + golden vector。"""

import numpy as np

from .gbdt import Booster, flash_bytes


def _f32(v):
    """C99 十六进制浮点字面量——精确，不用靠"打印够多位再解析回来"。"""
    f = float(np.float32(v))
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"导出的浮点里有 {f}，模型有问题，别往板上带")
    return f"{f.hex()}f"


def _f32_input(v):
    """golden **输入**专用：允许 NaN。

    模型参数里出现 NaN 是坏了（_f32 会拦），但输入里的 NaN 是合法的测试数据——
    它钉住"特征出 NaN 时往哪边走"这条行为，而那正是最容易在板上跟训练时不一致、
    又最不会被发现的地方。用 C99 的 NAN 宏（生成的头文件会 include math.h）。
    """
    f = float(np.float32(v))
    if f != f:
        return "NAN"
    if f in (float("inf"), float("-inf")):
        raise ValueError("golden 输入里有 inf，这不是合法的传感器数据")
    return f"{f.hex()}f"


def _arr(name, vals, ctype, fmt=str):
    return f"static const {ctype} {name}[] = {{{', '.join(fmt(v) for v in vals)}}};\n"


def export(b: Booster, golden_x=None, name="tm_gbdt") -> dict:
    n_nodes = len(b.node_feature)
    n_leaves = len(b.leaf_value)
    if b.n_features > 0xFFFF or n_leaves > 0xFFFF:
        raise ValueError(
            f"叶子 {n_leaves} 个、特征 {b.n_features} 维，超出 node_feature 的 uint16。"
            "要么减轮数/限深，要么把这个字段换成 uint32（flash 多 2B/节点）")

    c = ['#include "tm_gbdt.h"\n\n']
    c.append(_arr(f"{name}_off", b.tree_offset, "int32_t"))
    c.append(_arr(f"{name}_feat", b.node_feature, "uint16_t"))
    c.append(_arr(f"{name}_thr", b.node_threshold, "float", _f32))
    c.append(_arr(f"{name}_left", b.node_left, "int32_t"))
    c.append(_arr(f"{name}_right", b.node_right, "int32_t"))
    c.append(_arr(f"{name}_miss", b.node_missing_left, "uint8_t"))
    c.append(_arr(f"{name}_leaf", b.leaf_value, "float", _f32))
    c.append(
        f"\nconst tm_gbdt_t {name} = {{\n"
        f"    {name}_off, {name}_feat, {name}_thr, {name}_left, {name}_right,\n"
        f"    {name}_miss, {name}_leaf,\n"
        f"    {b.n_trees}, {b.n_features}, {b.n_classes}, {_f32(b.base_score)}\n"
        f"}};\n"
    )

    fb = flash_bytes(b)
    total = sum(fb.values())
    names = b.class_names or tuple(f"class{i}" for i in range(b.n_classes))
    h = [
        "/* 自动生成，别手改。 */\n",
        "#ifndef TM_GBDT_MODEL_H\n#define TM_GBDT_MODEL_H\n\n",
        '#include "tm_gbdt.h"\n\n',
        f"/* 树 {b.n_trees} 棵（{b.n_trees // max(b.n_classes,1)} 轮 × "
        f"{b.n_classes} 类），节点 {n_nodes} 个，叶子 {n_leaves} 个\n",
        f" * flash 约 {total} B（{total / 1024:.1f} KB）："
        + "，".join(f"{k} {v}B" for k, v in fb.items()) + " */\n",
        f"#define TM_G_N_FEATURES {b.n_features}\n",
        f"#define TM_G_N_CLASSES {b.n_classes}\n\n",
        "static const char *const TM_G_CLASS_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in names) + "};\n\n",
        f"extern const tm_gbdt_t {name};\n\n#endif\n",
    ]

    files = {"tm_gbdt_model.c": "".join(c), "tm_gbdt_model.h": "".join(h)}
    if golden_x is not None:
        xs = np.asarray(golden_x, np.float32)
        assert xs.ndim == 2 and xs.shape[1] == b.n_features
        marg = np.stack([b.margins(x) for x in xs])
        cls = marg.argmax(axis=1)
        files["tm_gbdt_golden.h"] = (
            "/* 自动生成。比的是 **margin 的位模式**，不是概率——\n"
            " * 端上不做 softmax（它保序，argmax 结果一样），所以也没有概率可比。 */\n"
            "#ifndef TM_GBDT_GOLDEN_H\n#define TM_GBDT_GOLDEN_H\n\n"
            "#include <stdint.h>\n#include <math.h>   /* 输入里可能有 NAN */\n\n"
            f"#define TM_G_GOLDEN_N {len(xs)}\n\n"
            + _arr("tm_gbdt_golden_in", xs.reshape(-1), "float", _f32_input)
            + _arr("tm_gbdt_golden_margin", marg.reshape(-1), "float", _f32)
            + _arr("tm_gbdt_golden_class", cls, "int8_t")
            + "\n#endif\n")
    return files
