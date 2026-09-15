"""把紧凑森林导成 C。

节点直接导成**字节流**，不导成结构体数组：packed struct 的布局在不同编译器上
不完全一样，而这份代码要在 gcc / arm-none-eabi / 将来可能的 Keil 上给出逐位
相同的结果。字节流谁也没得发挥。

字节序由 forest_compact.pack_nodes 一处决定，导出和 Python 参考都从那里拿——
写两遍就迟早有一遍错，而字节序错了不报错，只会让判决莫名其妙。
"""

import numpy as np

from .export_forest_c import _f32
from .forest_compact import CompactForest, NODE_BYTES, pack_nodes


def _u8_arr(name, arr, per_line=24):
    v = np.asarray(arr, np.uint8).reshape(-1)
    body = []
    for i in range(0, len(v), per_line):
        body.append("    " + ", ".join(str(int(x)) for x in v[i:i + per_line]))
    return (f"static const uint8_t {name}[] = {{\n" + ",\n".join(body) + "\n};\n")


def export(cf: CompactForest, name="tm_forest_c", golden_x=None) -> dict:
    nodes = pack_nodes(cf)
    c = ['#include "tm_forest_c.h"\n\n']
    c.append(f"/* {cf.n_trees} 棵树，{cf.n_nodes} 个节点（{NODE_BYTES} B/节点），"
             f"{cf.n_leaves} 个叶子 × {cf.n_classes} 类 */\n")
    c.append(_u8_arr(f"{name}_nodes", nodes))
    c.append(_u8_arr(f"{name}_leaves", cf.leaf_u8.reshape(-1)))
    c.append(f"static const int32_t {name}_off[] = {{"
             + ", ".join(str(int(v)) for v in cf.tree_offset) + "};\n\n")
    c.append(
        f"const tm_forest_c_t {name} = {{\n"
        f"    {name}_nodes, {name}_leaves, {name}_off,\n"
        f"    {cf.n_trees}, {cf.n_features}, {cf.n_classes}\n"
        f"}};\n")

    sizes = cf.flash_bytes()
    names = cf.class_names or tuple(f"class{i}" for i in range(cf.n_classes))
    h = [
        "/* 自动生成，别手改 —— 改了下次导出就没了。 */\n",
        "#ifndef TM_FOREST_C_MODEL_H\n#define TM_FOREST_C_MODEL_H\n\n",
        '#include "tm_forest_c.h"\n\n',
        f"#define TM_FC_N_TREES {cf.n_trees}\n",
        f"#define TM_FC_N_NODES {cf.n_nodes}\n",
        f"#define TM_FC_N_LEAVES {cf.n_leaves}\n",
        f"#define TM_FC_N_FEATURES {cf.n_features}\n",
        f"#define TM_FC_N_CLASSES {cf.n_classes}\n",
        f"/* flash：节点 {sizes['nodes']:,} B + 叶子 {sizes['leaves']:,} B"
        f" + 树表 {sizes['tree_offset']:,} B = {sum(sizes.values()):,} B"
        f"（{sum(sizes.values()) / 1024:.1f} KB） */\n\n",
        "static const char *const TM_FC_CLASS_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in names) + "};\n\n",
        f"extern const tm_forest_c_t {name};\n\n#endif\n",
    ]

    files = {f"{name}_model.c": "".join(c), f"{name}_model.h": "".join(h)}
    if golden_x is not None:
        files[f"{name}_golden.h"] = _golden(cf, golden_x)
    return files


def _golden(cf: CompactForest, xs):
    """golden vector 存的是**整数票数**，不是概率。

    整数没有浮点那些事，板上跟 PC 逐位一致是必然的——所以这份 golden 一旦对不上，
    就一定是编码/解析错了，不可能是"数值误差"。这比浮点的 golden 更有诊断力。
    """
    xs = np.asarray(xs, np.float32)
    assert xs.ndim == 2 and xs.shape[1] == cf.n_features
    votes = np.stack([cf.votes(x) for x in xs]).astype(np.int32)
    body = ", ".join(str(int(v)) for v in votes.reshape(-1))
    # 用十六进制浮点字面量，不是 %.9g。
    # %.9g 把 0.0 打成 "0"，加个 f 后缀就是 "0f" —— 非法的整数后缀，编不过。
    # （export_forest_c._f32 的注释里写了这个坑，我在这个新文件里又犯了一遍，
    #  所以改成直接复用那个函数，而不是再写一份格式化。）
    feats = ", ".join(_f32(v) for v in xs.reshape(-1))
    return (
        "/* 自动生成。每条 = 一个特征向量 + Python 参考实现算出的**整数票数**。\n"
        " * 整数累加跟指令集无关，所以板上必须一模一样——对不上就是编码或解析错了，\n"
        " * 不可能是数值误差。 */\n"
        "#ifndef TM_FOREST_C_GOLDEN_H\n#define TM_FOREST_C_GOLDEN_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define TM_FC_GOLDEN_N {len(xs)}\n\n"
        f"static const float tm_forest_c_golden_in[] = {{{feats}}};\n"
        f"static const int32_t tm_forest_c_golden_out[] = {{{body}}};\n"
        "\n#endif\n"
    )
