"""把 Forest 导成 C 源文件 + golden vector。

跟 int8 CNN 那条路走一样的纪律：导出的同时把一组「这个输入必须得到这个输出」
钉进头文件，板上第一件事是跑它，对不上先别看准确率。
"""

import numpy as np

from .forest import Forest, flash_bytes


def _arr(name, values, ctype, fmt=str):
    body = ", ".join(fmt(v) for v in values)
    return f"static const {ctype} {name}[] = {{{body}}};\n"


def _f32(v):
    """导出成 C99 的**十六进制浮点字面量**（0x1.8p+1f 这种）。

    不用十进制：十进制要靠"打印够多位 + 编译器正确解析"来还原同一个数，多一步
    就多一个出错的机会，而出错的表现是阈值差最后一位——判决只在边界上翻一小部分
    样本，看起来就是"板上准确率莫名低一点"。十六进制字面量是精确的，没有这一步。
    （顺带解决 %.9g 把 0.0 打成 "0"、加个 f 后缀就编不过的问题。）
    """
    f = float(np.float32(v))
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"导出的浮点里有 {f}，模型有问题，别往板上带")
    return f"{f.hex()}f"


def export(forest: Forest, golden_x=None, name="tm_forest") -> dict:
    n_nodes = len(forest.node_feature)
    n_leaves = len(forest.leaf_proba)
    if n_leaves > 0xFFFF or forest.n_features > 0xFFFF:
        raise ValueError(
            f"叶子 {n_leaves} 个、特征 {forest.n_features} 维，超出 node_feature 的 uint16。"
            "要么限深重训，要么把这个字段换成 uint32（flash 会多 2B/节点）")

    c = ['#include "tm_forest.h"\n\n']
    c.append(_arr(f"{name}_off", forest.tree_offset, "int32_t"))
    c.append(_arr(f"{name}_feat", forest.node_feature, "uint16_t"))
    c.append(_arr(f"{name}_thr", forest.node_threshold, "float", _f32))
    c.append(_arr(f"{name}_left", forest.node_left, "int32_t"))
    c.append(_arr(f"{name}_right", forest.node_right, "int32_t"))
    c.append(_arr(f"{name}_leaf", forest.leaf_proba.reshape(-1), "float", _f32))
    c.append(
        f"\nconst tm_forest_t {name} = {{\n"
        f"    {name}_off, {name}_feat, {name}_thr, {name}_left, {name}_right, {name}_leaf,\n"
        f"    {forest.n_trees}, {forest.n_features}, {forest.n_classes}\n"
        f"}};\n"
    )

    fb = flash_bytes(forest)
    total = sum(fb.values())
    names = forest.class_names or tuple(f"class{i}" for i in range(forest.n_classes))
    h = [
        "/* 自动生成，别手改。 */\n",
        "#ifndef TM_FOREST_MODEL_H\n#define TM_FOREST_MODEL_H\n\n",
        '#include "tm_forest.h"\n\n',
        f"/* 树 {forest.n_trees} 棵，节点 {n_nodes} 个，叶子 {n_leaves} 个\n",
        f" * flash 约 {total} B（{total / 1024:.1f} KB）："
        + "，".join(f"{k} {v}B" for k, v in fb.items()) + " */\n",
        f"#define TM_F_N_FEATURES {forest.n_features}\n",
        f"#define TM_F_N_CLASSES {forest.n_classes}\n\n",
        "static const char *const TM_F_CLASS_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in names) + "};\n\n",
        f"extern const tm_forest_t {name};\n\n#endif\n",
    ]

    files = {"tm_forest_model.c": "".join(c), "tm_forest_model.h": "".join(h)}
    if golden_x is not None:
        files["tm_forest_golden.h"] = _golden(forest, golden_x)
    return files


def _golden(forest: Forest, xs):
    xs = np.asarray(xs, np.float32)
    assert xs.ndim == 2 and xs.shape[1] == forest.n_features
    probs = np.stack([forest.predict_proba(x) for x in xs])
    return (
        "/* 自动生成。每条 = 一个特征向量 + Python 参考实现算出的概率。\n"
        " * 板上跑出来必须**逐位**相同（按 float 的位模式比，不是看小数点后几位）。\n"
        " * 对不上先查编译选项（-ffp-contract、是否被优化成乘倒数），别先怀疑模型。 */\n"
        "#ifndef TM_FOREST_GOLDEN_H\n#define TM_FOREST_GOLDEN_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define TM_F_GOLDEN_N {len(xs)}\n\n"
        + _arr("tm_forest_golden_in", xs.reshape(-1), "float", _f32)
        + _arr("tm_forest_golden_out", probs.reshape(-1), "float", _f32)
        + "\n#endif\n"
    )
