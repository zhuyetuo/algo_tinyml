/* 加速开关：要不要把热点换成 Arm 官方的 CMSIS 内核。**默认全关**，一行汇编都不用，
 * 板上结果跟 PC 上逐位一样（tests/ 里钉着）。
 *
 *   -DTM_USE_CMSIS        两个一起开（下面两个各自也能单独开）
 *   -DTM_CMSIS_DSP=1      特征提取：FFT 换 arm_cfft_f32，均值/功率/RMS/极值/点积换
 *                         arm_mean_f32 / arm_power_f32 / arm_rms_f32 / arm_min|max_no_idx_f32 /
 *                         arm_dot_prod_f32。M4F 上 FFT 大致快 3～5 倍。
 *                         **不再逐位一致**：CMSIS 的 FFT 是基-8、累加顺序也不同，特征差在
 *                         1e-5 相对误差这一级，森林的判决偶尔会翻（留出集上 <1%，见
 *                         tests/test_cmsis_c.py 量出来的数）。
 *   -DTM_CMSIS_NN=1       int8 CNN：卷积 / 池化 / 全连接换 arm_convolve_s8 / arm_max_pool_s8。
 *                         整数累加完全一样；只有重量化的**舍入平局**不同（gemmlowp 负数平局向远离零，
 *                         CMSIS 统一向上），所以个别输出会差 1 LSB（实测 <5% 的元素），argmax 不变。
 *                         golden 自检在这条路上要按 ±1 容差比（host/ 和 board/ 的自检都认 TM_CMSIS_NN）。
 *                         张量布局从 [C][T] 变成 [T][C]（CMSIS 只认 NHWC），权重由导出脚本同时给
 *                         两种排法，编译期按这个开关取一种，flash 不翻倍。
 *
 * 需要的东西（third_party/cmsis/ 里有裁好的一份，README 写了从哪个 commit 拿的）：
 *   CMSIS-DSP：arm_math.h + cfft / 统计 / 点积那几个 .c，FFT 表只带 16～256 点的（抽出来的，官方表文件 7 MB）
 *   CMSIS-NN ：arm_nnfunctions.h + 卷积 / 池化那几个 .c
 *   CMSIS-Core：cmsis_compiler.h / cmsis_gcc.h（GR551x SDK 自带；PC 上编加 -D__GNUC_PYTHON__ 就不用）
 *
 * 编译选项：ARM 上 -mcpu=cortex-m4 会自动定义 __ARM_FEATURE_DSP，CMSIS 据此走 SIMD 路径；
 * 要 -ffunction-sections -fdata-sections -Wl,--gc-sections，tm_features 只引用 TM_FEAT_MAX_NPERSEG
 * 以内的 FFT 长度，链接器把其余的表丢掉（-DTM_FEAT_MAX_NPERSEG=16 → 只剩 16 点那张 0.2 KB）。
 */

#ifndef TM_ACCEL_H
#define TM_ACCEL_H

#ifdef TM_USE_CMSIS
#ifndef TM_CMSIS_DSP
#define TM_CMSIS_DSP 1
#endif
#ifndef TM_CMSIS_NN
#define TM_CMSIS_NN 1
#endif
#endif

#ifndef TM_CMSIS_DSP
#define TM_CMSIS_DSP 0
#endif
#ifndef TM_CMSIS_NN
#define TM_CMSIS_NN 0
#endif

#endif /* TM_ACCEL_H */
