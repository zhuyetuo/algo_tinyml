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


def pipeline_golden(c: CompactBooster, windows, hz, nperseg) -> str:
    """整条链的 golden：原始窗口 → 193 维特征 → GBDT → **类别**。

    比只验模型（从特征进）多覆盖**特征的排列顺序**——错位不会崩，只会让每个阈值
    对到别的特征上。比判决而不比 margin：特征那段 Python 参考和 C 末位可能差一点，
    只要没恰好落在阈值上就走同一条路径；真错位的话判决成片地翻。
    """
    from .features import extract_one

    ws = np.asarray(windows, np.float32)
    if ws.ndim != 3:
        raise ValueError(f"windows 要 [N, T, C]，给的是 {ws.shape}")
    feats = np.stack([extract_one(w, hz, nperseg) for w in ws])
    if feats.shape[1] != c.n_features:
        raise ValueError(f"算出来 {feats.shape[1]} 维特征，模型要 {c.n_features} 维。"
                         "通道数或窗口长度跟训练时不一致")
    cls = [c.predict(f) for f in feats]
    flat_in = np.stack([w.T.reshape(-1) for w in ws]).reshape(-1)   # 板上是 [n_ch][n_t]
    n, t_len, n_ch = ws.shape

    def f32(v):
        return f"{float(np.float32(v)).hex()}f"
    return (
        "/* 自动生成。整条链：原始窗口 → 特征 → GBDT → 类别。 */\n"
        "#ifndef TM_GBDT_C_PIPELINE_GOLDEN_H\n#define TM_GBDT_C_PIPELINE_GOLDEN_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define TM_GCP_GOLDEN_N {n}\n#define TM_GCP_N_CH {n_ch}\n#define TM_GCP_N_T {t_len}\n\n"
        "static const float tm_gbdt_c_pipeline_in[] = {" + ", ".join(f32(v) for v in flat_in) + "};\n"
        "static const int32_t tm_gbdt_c_pipeline_cls[] = {" + ", ".join(str(int(x)) for x in cls) + "};\n"
        "\n#endif\n")
