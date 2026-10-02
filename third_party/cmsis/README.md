# CMSIS 子集（可选加速路线用）

从 Arm 官方仓库裁出来的最小子集，只放 `core/` 的可选加速路线（`-DTM_USE_CMSIS`，见
`core/tm_accel.h`）用得到的文件。**默认不编译、不链接**，不开开关时跟这个目录没关系。

| 来源 | 仓库 | commit | 许可 |
|---|---|---|---|
| CMSIS-DSP | https://github.com/ARM-software/CMSIS-DSP | `83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b`（2026-09-10） | Apache-2.0 |
| CMSIS-NN | https://github.com/ARM-software/CMSIS-NN | `71cbe3d5a686c875c88a4d112633b13fae94b5a2`（2026-09-22） | Apache-2.0 |
| CMSIS-Core | https://github.com/ARM-software/CMSIS_6 | `26206e47dcf0abfbdc64eb753a0b6334b24439f6`（2026-09-11） | Apache-2.0 |

## 带了什么

- `dsp/Include/`：整个头文件目录（互相 include，拆不开）；`dsp/PrivateInclude/arm_compiler_specific.h`
- `dsp/Source/`：`arm_cfft_f32.c` `arm_cfft_init_f32.c` `arm_cfft_radix8_f32.c` `arm_bitreversal2.c`
  `arm_mean_f32.c` `arm_power_f32.c` `arm_rms_f32.c` `arm_max_no_idx_f32.c` `arm_min_no_idx_f32.c`
  `arm_dot_prod_f32.c`
- `dsp/Source/tm_cmsis_tables.c`：**我们生成的**。官方 `arm_common_tables.c` 有 7 MB（16～4096 点
  的表全在），这里只抽 `twiddleCoef_{16,32,64,128,256}`、`armBitRevIndexTable{16..256}` 和
  `arm_cfft_sR_f32_len{16..256}` 五档，数值逐字照抄。tm_features 按 `TM_FEAT_MAX_NPERSEG`
  只引用需要的那一档，链接器把其余的丢掉。
- `nn/Include/`：整个头文件目录
- `nn/Source/`：`arm_convolve_s8.c` `arm_convolve_get_buffer_sizes_s8.c`
  `arm_nn_mat_mult_kernel_s8_s16.c` `arm_nn_mat_mult_kernel_row_offset_s8_s16.c`
  `arm_nn_mat_mult_nt_t_s8.c` `arm_q7_to_q15_with_offset.c` `arm_s8_to_s16_unordered_with_offset.c`
  `arm_max_pool_s8.c`
- `core/Include/`：`cmsis_compiler.h` `cmsis_gcc.h` `cmsis_version.h` `m-profile/cmsis_gcc_m.h`——
  交叉编译时 CMSIS-DSP 要它；GR551x SDK 自带同样的东西，工程里已经有就不用这份。
  PC 上编（跑测试）用 `-D__GNUC_PYTHON__` 绕开，不需要 CMSIS-Core。

## 升级

换 commit 时把上面三个 .c/.h 清单重新拷一遍，`tm_cmsis_tables.c` 用
`service/tinyml/cmsis.py` 旁边注释里那段脚本从新的 `arm_common_tables.c` / `arm_const_structs.c`
重新抽；然后跑 `tests/test_cmsis_c.py`，CNN 那条必须逐位不变，特征那条相对误差要在 1e-5 以内。
