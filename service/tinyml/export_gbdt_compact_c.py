"""把 CompactBooster 导成 C。"""

import numpy as np

from .gbdt_compact import NODE_BYTES, CompactBooster


def _f32_bytes(v):
    return np.frombuffer(np.float32(v).tobytes(), dtype=np.uint8)


def pack_nodes(c: CompactBooster) -> np.ndarray:
    """→ uint8 [n_nodes * 6]。字节序跟 C 结构体一致（小端，packed）。

    在 Python 这边显式打包、而不是让 C 去拼，是为了让**打包这一步本身**
    也能被测试覆盖：打包错了（比如字段顺序反了）在 C 那边表现成算出一堆
    看起来正常的垃圾，而不是崩。
    """
    n = c.n_nodes
    buf = np.zeros(n * NODE_BYTES, np.uint8)
    for i in range(n):
        o = i * NODE_BYTES
        buf[o] = c.node_feature[i]
        buf[o + 1:o + 5] = _f32_bytes(c.node_value[i])
        buf[o + 5] = c.node_right[i]
    return buf


def export(c: CompactBooster, golden_x=None, name="tm_gbdt_c_model") -> dict:
    buf = pack_nodes(c)
    fb = c.flash_bytes()
    total = sum(fb.values())

    def arr(nm, vals, ctype, per_line=16):
        out = [f"static const {ctype} {nm}[] = {{"]
        vals = list(vals)
        for i in range(0, len(vals), per_line):
            out.append("    " + ", ".join(str(int(v)) for v in vals[i:i + per_line]) + ",")
        out.append("};\n")
        return "\n".join(out)

    src = ['#include "tm_gbdt_c.h"\n\n']
    src.append(arr(f"{name}_nodes", buf, "uint8_t"))
    src.append(arr(f"{name}_off", c.tree_offset, "int32_t", 12))
    src.append(
        f"\nconst tm_gbdt_c_t {name} = {{\n"
        f"    {name}_nodes, {name}_off,\n"
        f"    {c.n_trees}, {c.n_features}, {c.n_classes}, "
        f"{float(np.float32(c.base_score)).hex()}f\n"
        f"}};\n")

    names = c.class_names or tuple(f"class{i}" for i in range(c.n_classes))
    hdr = [
        "/* 自动生成，别手改。紧凑 + AoS 布局，6 字节一个节点。 */\n",
        "#ifndef TM_GBDT_C_MODEL_H\n#define TM_GBDT_C_MODEL_H\n\n",
        '#include "tm_gbdt_c.h"\n\n',
        f"/* 树 {c.n_trees} 棵（{c.n_trees // max(c.n_classes, 1)} 轮 × "
        f"{c.n_classes} 类），节点 {c.n_nodes} 个\n",
        f" * flash {total} B（{total / 1024:.1f} KB）= 节点 {fb['nodes']} B "
        f"+ 树偏移 {fb['tree_offset']} B */\n",
        f"#define TM_GC_N_FEATURES {c.n_features}\n",
        f"#define TM_GC_N_CLASSES {c.n_classes}\n\n",
        "static const char *const TM_GC_CLASS_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in names) + "};\n\n",
        f"extern const tm_gbdt_c_t {name};\n\n#endif\n",
    ]

    files = {"tm_gbdt_c_model.c": "".join(src), "tm_gbdt_c_model.h": "".join(hdr)}
    if golden_x is not None:
        xs = np.asarray(golden_x, np.float32)
        marg = np.stack([c.margins(x) for x in xs])
        def f32(v):
            f = float(np.float32(v))
            return "NAN" if f != f else f"{f.hex()}f"
        files["tm_gbdt_c_golden.h"] = (
            "/* 自动生成。比 margin 的位模式。 */\n"
            "#ifndef TM_GBDT_C_GOLDEN_H\n#define TM_GBDT_C_GOLDEN_H\n\n"
            "#include <stdint.h>\n#include <math.h>\n\n"
            f"#define TM_GC_GOLDEN_N {len(xs)}\n\n"
            + "static const float tm_gc_golden_in[] = {"
            + ", ".join(f32(v) for v in xs.reshape(-1)) + "};\n"
            + "static const float tm_gc_golden_margin[] = {"
            + ", ".join(f32(v) for v in marg.reshape(-1)) + "};\n"
            + "\n#endif\n")
    return files
