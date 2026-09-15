"""把量化好的模型 + 一组 golden vector 导成 C 源文件。

golden vector 是这套工具链的核心产物，不是附赠品：
量化和定点实现最常见的失败是「板上结果跟训练时不一样，但不报错」。没有一组
"这个输入必须得到这个输出"的样例钉在那里，这类 bug 只会在准确率里表现成
几个百分点，而你会去怀疑模型、怀疑数据、怀疑传感器——就是不会怀疑舍入。
"""

import numpy as np

from .net import QDense, QNet, QPool, forward_int


def _c_array(name, arr, ctype):
    body = ", ".join(str(int(v)) for v in np.asarray(arr).reshape(-1))
    return f"static const {ctype} {name}[] = {{{body}}};\n"


def _arena_bytes(qnet: QNet) -> int:
    """两块乒乓缓冲各要多大：取所有中间张量里最大的那个。

    在这里算而不是在 C 里写死：改一层通道数就要跟着改的常量，迟早会忘——
    忘了的表现是 tm_invoke 返回 -1（好），或者更糟，改小了却没人发现越界。
    """
    ch, t = qnet.n_ch, qnet.n_t
    biggest = ch * t
    for lyr in qnet.layers:
        if isinstance(lyr, QPool):
            t = t // lyr.pool
        elif isinstance(lyr, QDense):
            ch, t = lyr.w.shape[0], 1
        else:
            oc, ic, k = lyr.w.shape
            ch, t = oc, t - k + 1
        biggest = max(biggest, ch * t)
    return biggest


def export(qnet: QNet, golden_x_i8=None, model_name="tm_model") -> dict:
    """返回 {文件名: 内容}。分成 .h / .c 两个文件是给固件工程用的：
    权重只在一个编译单元里，别的地方 include 头文件就行。"""
    lines = ['#include "tm_runtime.h"\n', "\n"]
    layer_entries = []

    for i, lyr in enumerate(qnet.layers):
        if isinstance(lyr, QPool):
            layer_entries.append(
                f"    {{ TM_MAXPOOL1D, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {lyr.pool} }}"
            )
            continue
        p = f"L{i}"
        lines.append(_c_array(f"{p}_w", lyr.w, "int8_t"))
        lines.append(_c_array(f"{p}_b", lyr.bias, "int32_t"))
        lines.append(_c_array(f"{p}_m", lyr.mult, "int32_t"))
        lines.append(_c_array(f"{p}_s", lyr.shift, "int32_t"))
        if isinstance(lyr, QDense):
            op, out_ch, in_ch, k = "TM_DENSE", lyr.w.shape[0], lyr.w.shape[1], 0
        else:
            op, (out_ch, in_ch, k) = "TM_CONV1D", lyr.w.shape
        layer_entries.append(
            f"    {{ {op}, {p}_w, {p}_b, {p}_m, {p}_s, {out_ch}, {in_ch}, {k}, "
            f"{lyr.in_zp}, {lyr.out_zp}, {1 if lyr.relu else 0}, 0 }}"
        )

    lines.append(f"\nstatic const tm_layer_t {model_name}_layers[] = {{\n")
    lines.append(",\n".join(layer_entries))
    lines.append("\n};\n\n")
    lines.append(
        f"const tm_model_t {model_name} = {{\n"
        f"    {model_name}_layers, {len(layer_entries)},\n"
        f"    {qnet.n_ch}, {qnet.n_t}, {qnet.n_classes}, {qnet.in_zp},\n"
        f"    {qnet.in_scale!r}f, {qnet.out_scale!r}f, {qnet.out_zp}\n"
        f"}};\n"
    )

    arena = _arena_bytes(qnet)
    names = qnet.class_names or [f"class{i}" for i in range(qnet.n_classes)]
    header = [
        "/* 自动生成，别手改 —— 改了下次导出就没了。 */\n",
        "#ifndef TM_MODEL_H\n#define TM_MODEL_H\n\n",
        '#include "tm_runtime.h"\n\n',
        f"#define TM_ARENA_BYTES {2 * arena}\n",
        f"#define TM_N_CH {qnet.n_ch}\n",
        f"#define TM_N_T {qnet.n_t}\n",
        f"#define TM_N_CLASSES {qnet.n_classes}\n\n",
        "/* 类别顺序就是模型输出的下标顺序，跟训练时的 label 编码一致 */\n",
        "static const char *const TM_CLASS_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in names) + "};\n\n",
        f"extern const tm_model_t {model_name};\n\n",
        "#endif\n",
    ]

    files = {"tm_model.c": "".join(lines), "tm_model.h": "".join(header)}
    if golden_x_i8 is not None:
        files["tm_golden.h"] = _golden(qnet, golden_x_i8)
    return files


def _golden(qnet: QNet, xs):
    xs = np.asarray(xs, np.int8)
    assert xs.ndim == 3 and xs.shape[1:] == (qnet.n_ch, qnet.n_t)
    outs = np.stack([forward_int(qnet, x)[0] for x in xs])
    n = xs.shape[0]
    return (
        "/* 自动生成。每条 golden vector = 一个输入窗口 + Python 参考实现算出的 int8 输出。\n"
        " * 板上跑出来必须**逐位**相同，不是「接近」——差 1 个 LSB 就足以让 argmax 翻。 */\n"
        "#ifndef TM_GOLDEN_H\n#define TM_GOLDEN_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define TM_GOLDEN_N {n}\n\n"
        + _c_array("tm_golden_in", xs, "int8_t")
        + _c_array("tm_golden_out", outs, "int8_t")
        + "\n#endif\n"
    )
