"""整条链的 golden vector：**原始窗口** → 特征 → 森林 → 概率。

为什么不是只导"特征的 golden"和"森林的 golden"两份：
  1. 分开导验不到中间那道接缝——特征的排列顺序跟模型训练时对不对得上。顺序错了
     每一维都对到别的特征上，而森林照样给得出一个类别，只是准确率莫名其妙地差。
  2. 板上链接时 --gc-sections 会把没人调的代码整段丢掉。只有自检真的走完整条链，
     tm_features 才会被链进镜像——否则量出来的固件体积是假的（实测过：不走整条链
     的话 tm_features/tm_invoke/tm_window_push 在最终镜像里根本不存在）。
"""

import numpy as np

from .features import extract_one
from .forest import Forest


def _f32(v):
    f = float(np.float32(v))
    return f"{f.hex()}f"


def export(forest: Forest, windows, hz, nperseg=32, name="tm_pipeline") -> dict:
    """windows: [N, T, C] 的原始窗口（实数，不是特征）。"""
    ws = np.asarray(windows, np.float32)
    assert ws.ndim == 3, "windows 应该是 [N, T, C]"
    n, t_len, n_ch = ws.shape

    feats = np.stack([extract_one(w, hz, nperseg) for w in ws])
    if feats.shape[1] != forest.n_features:
        raise ValueError(
            f"算出来 {feats.shape[1]} 维特征，森林要 {forest.n_features} 维。"
            "通道数或窗口长度跟训练时不一致——这个必须先解决，不然板上每一维都会错位")
    probs = np.stack([forest.predict_proba(f) for f in feats])
    preds = probs.argmax(axis=1)

    # 板上是 [n_ch][n_t]（通道在前），跟 tm_window 的输出一致
    flat_in = np.stack([w.T.reshape(-1) for w in ws]).reshape(-1)

    def arr(nm, vals, ctype, fmt=str):
        return f"static const {ctype} {nm}[] = {{{', '.join(fmt(v) for v in vals)}}};\n"

    body = (
        "/* 自动生成。整条链的 golden vector：原始窗口 → 特征 → 森林 → 概率。\n"
        " * 板上必须逐位相同（按 float 位模式比）。 */\n"
        "#ifndef TM_PIPELINE_GOLDEN_H\n#define TM_PIPELINE_GOLDEN_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define TM_P_GOLDEN_N {n}\n"
        f"#define TM_P_N_T {t_len}\n"
        f"#define TM_P_N_CH {n_ch}\n\n"
        + arr(f"{name}_in", flat_in, "float", _f32)
        + arr(f"{name}_proba", probs.reshape(-1), "float", _f32)
        + arr(f"{name}_class", preds, "int8_t")
        + "\n#endif\n"
    )
    return {"tm_pipeline_golden.h": body}
