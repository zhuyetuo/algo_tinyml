"""导出特征提取需要的常量表（Hann 窗、FFT 旋转因子、位反序下标）。

**表从 Python 导出、不让 C 现算**：`cosf` 属于 libm，各实现不保证正确舍入，
两边各算一遍就可能差最后一位。把它变成一张常量表，这一类不一致就从根上没了。
代价是几百字节 flash（nperseg=32 时：窗 128B + 旋转因子 128B + 位反序 64B ≈ 320B）。
"""

import numpy as np

from .features import _bit_reverse, fft_twiddles, hann_periodic


def _f32(v):
    """C99 十六进制浮点字面量。十进制要靠"打印够多位 + 正确解析"来还原同一个数，
    多一步就多一个出错的机会，而这里要的恰恰是一个 bit 都不差。"""
    f = float(np.float32(v))
    return f"{f.hex()}f"


def export(n_t, n_ch, nperseg, fs, name="tm_feat_cfg") -> dict:
    assert nperseg & (nperseg - 1) == 0, f"nperseg={nperseg} 不是 2 的幂（基-2 FFT 的限制）"
    assert nperseg <= n_t, f"nperseg={nperseg} 比窗口 {n_t} 还长"
    win = hann_periodic(nperseg)
    cos_t, sin_t = fft_twiddles(nperseg)
    br = _bit_reverse(nperseg)

    def arr(nm, vals, ctype, fmt=str):
        return f"static const {ctype} {nm}[] = {{{', '.join(fmt(v) for v in vals)}}};\n"

    c = ['#include "tm_features.h"\n\n']
    c.append(arr(f"{name}_win", win, "float", _f32))
    c.append(arr(f"{name}_cos", cos_t, "float", _f32))
    c.append(arr(f"{name}_sin", sin_t, "float", _f32))
    c.append(arr(f"{name}_br", br, "int16_t"))
    c.append(
        f"\nconst tm_feat_cfg_t {name} = {{\n"
        f"    {n_t}, {n_ch}, {nperseg}, {_f32(fs)},\n"
        f"    {name}_win, {name}_cos, {name}_sin, {name}_br\n"
        f"}};\n"
    )

    from .features import n_features
    h = [
        "/* 自动生成，别手改。 */\n",
        "#ifndef TM_FEAT_CFG_H\n#define TM_FEAT_CFG_H\n\n",
        '#include "tm_features.h"\n\n',
        f"#define TM_FEAT_N_T {n_t}\n",
        f"#define TM_FEAT_N_CH {n_ch}\n",
        f"#define TM_FEAT_NPERSEG {nperseg}\n",
        f"#define TM_FEAT_DIM {n_features(n_ch)}\n\n",
        f"extern const tm_feat_cfg_t {name};\n\n#endif\n",
    ]
    return {"tm_feat_cfg.c": "".join(c), "tm_feat_cfg.h": "".join(h)}
