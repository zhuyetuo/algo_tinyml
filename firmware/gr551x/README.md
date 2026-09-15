# GR551x 工程接入

把 `firmware/tinyml/` 那几个 .c 挂进 Goodix 的 GR551x SDK。

- SDK 下载：<https://www.goodix.com/zh/software_tool/gr551x_sdk>（GitHub 镜像：
  [goodix-ble/GR551x.SDK](https://github.com/goodix-ble/GR551x.SDK)）
- 开发板：[GR5515 Starter Kit](https://www.goodix.com/zh/kit/gr5515_starter_kit)
- 量产测试：[PLT Lite](https://www.goodix.com/zh/kit/plt_lite)

**这里不复制 SDK 的任何文件**，全部用 `SDK_ROOT` 指过去。复制一份进来迟早跟上游分叉，
而且出了问题没人分得清是我们改的还是 SDK 本来如此。

---

## 先跑一下（不需要板子，不需要 SDK 里的 BLE 部分）

```bash
cd firmware/gr551x/tinyml_app/GCC
make SDK_ROOT=/path/to/GR551x_SDK GEN_DIR=/path/to/generated objs
```

它把我们这部分交叉编译成 .o 并报体积。**这一步能先确认两件事**：我们的代码在
`arm-none-eabi-gcc` 下编得过（跟 PC 上的 gcc 不是同一个编译器），以及占多少。

实测（`arm-none-eabi-gcc 13.2`，`-Os`，`-mcpu=cortex-m4 -mfpu=fpv4-sp-d16`）：

| 模块 | flash (text) | RAM (bss) |
|---|---|---|
| `tm_features.c` 193 维特征 | 3152 B | 2320 B |
| `tm_forest.c` 森林推理 | 202 B | 0 |
| `tm_runtime.c` int8 CNN | 780 B | 0 |
| `tm_window.c` 窗口 | 326 B | 0 |
| `tm_feat_cfg.c` 窗/旋转因子表 | 348 B | 0 |
| `tinyml_selftest.c` | 591 B | 0 |
| `tinyml_task.c` 事件聚合 | 174 B | 0 |
| **小计（不含模型）** | **约 5.6 KB** | **约 2.3 KB** |

模型另算：int8 CNN 约 1.3 KB；随机森林要量（一棵 20 树 × 深 6 的玩具森林是
11.6 KB，真实的 200 棵不限深会大得多，用 `python/rf_footprint.py` 量）。

`TM_FEAT_MAX_T` / `TM_FEAT_MAX_NPERSEG` 默认 32（16Hz × 2 秒）。**按实际窗口调小
能省一半 RAM**：上限 64 时 bss 是 4624 B，32 时是 2320 B。

---
## 完整固件：链得出来，实测过

**不是"应该能链"——这里真的用 `arm-none-eabi-gcc 13.2` 链出了 .bin。**

```bash
cd firmware/gr551x/tinyml_app/GCC
make SDK_ROOT=/path/to/GR551x_SDK GEN_DIR=/path/to/generated firmware
# → build/tinyml_app.{elf,bin,hex,map}
```

实测（RF 路线，含 20 棵树 × 深 6 的玩具森林 + 193 维特征 + golden vector）：

| | flash (text) | RAM (bss) |
|---|---|---|
| 基线：BLE 协议栈 + 最小应用，不含 tinyml | 104,124 B | 22,128 B |
| 加上整条 tinyml | **136,008 B** | **25,312 B** |
| 差值 | +31,884 B | +3,184 B |

512KB flash 用掉 26%，112KB 可用 RAM 用掉 23%。**这还是带着玩具森林的数**，
真实的 200 棵不限深会大得多——所以 `rf_footprint.py` 那个数还是得量。

### 这 31.9KB 里一多半不是代码

```
玩具森林的表                11,604 B
golden vector               14,560 B   ← 几乎一半
特征的窗/旋转因子表             348 B
代码（特征+森林+自检+聚合）    约 5,400 B
```

**golden vector 占 14.5KB 值得单独说**：整条链的那组是 8 个原始窗口 ×8 通道 ×32 点
×4 字节 = 8KB，森林那组是 8 条 ×193 维 ×4 字节 = 6.2KB。量产固件可以只留 2~4 条
（`export_rf.py --golden 4`），或者把自检做成产测固件单独烧。**但别删干净**——
没有自检的固件，算错了没有任何人会知道。

### 一个链接时才会暴露的坑

`--gc-sections` 会把**没人调的代码整段丢掉**。第一次链出来的镜像里
`tm_features` / `tm_invoke` / `tm_window_push` **根本不存在**——因为还没有 IMU 驱动，
没有东西调它们，自检当时也只验森林不验特征。那时候量出来的 +15.9KB 是假的。

修法不是"想办法留住它"，而是**让自检真的走完整条链**（原始窗口 → 特征 → 森林）。
这样既让体积数变真，又顺便验到了中间那道接缝。上面 136,008 那个数是修完之后的。

---

## 挂进 BLE 工程

第一版建议从 `projects/ble/ble_peripheral/ble_app_gus`（Goodix 透传服务）复制一份，
它自带一条可以往手机推数据的通道，不用先自己定义 GATT。

1. 复制工程目录，把 `Src/user/tinyml_task.c`、`tinyml_selftest.c` 和
   `firmware/tinyml/*.c`、导出的 `generated/*.c` 加进它的 `GCC/Makefile`
   的 `PRJ_C_SRC_FILES`，头文件路径加进 `PRJ_C_INCLUDE_PATH`。
2. **在它的 `COMMON_COMPILE_FLAGS` 里加上 `-ffp-contract=off`。**
3. 链接脚本用 `GCC/gcc_linker_gr5513.lds`（Keil 用 `Keil_5/` 里对应的 sct）。

### 第 2 步为什么单独强调

SDK 示例工程的 `COMMON_COMPILE_FLAGS` 里**没有** `-ffp-contract=off`（去
`projects/ble/ble_peripheral/ble_app_hrs/GCC/Makefile` 看，只有 `-std=gnu99 --inline
-ggdb3 -ffunction-sections -fdata-sections -mfloat-abi=softfp -mfpu=fpv4-sp-d16 ...`）。
不加的话编译器可以把 `a*b+c` 合成一条 FMA，中间少一次舍入，浮点结果跟 PC 上的
参考实现末位就不同 —— golden vector 会红，而人第一反应多半是去怀疑模型。

`ARCH_FLAGS` 那几个（尤其 `-mfloat-abi=softfp`）**要跟 SDK 保持一致**，改了跟预编译的
`libble_sdk.a` 的 ABI 就可能对不上。

---

## 三个来自 SDK 本身、不看就会踩的点

**1. 链接脚本的 FLASH 区是 8MB，不是 512KB。**

`platform/soc/linker/gcc/gcc_linker_gr5513.lds` 里写的是：

```
FLASH (rx) : ORIGIN = 0x01002000, LENGTH = (0x00800000 - 0x00002000)
RAM  (rwx) : ORIGIN = 0x30000000 + 0x4000, LENGTH = (0x00020000 - 0x4000)
```

那个 8MB 是**内存映射的 flash 窗口**，不是 GR5513 片上真实的 512KB。**所以固件超出
512KB 时链接不会报错**，要到烧录或运行时才出问题。这对"RF 能不能塞进去"这个问题
很关键：不能靠"链接过了"当证据，必须看 `arm-none-eabi-size` 的 text 值。

**2. GR5513 和 GR5515 的链接脚本只差 RAM 一行。**

`0x20000-0x4000`（GR5513，128KB）vs `0x40000-0x4000`（GR5515，256KB）。用错的话
GR5513 烧 GR5515 的固件**能链过**，跑起来才踩到不存在的内存。Starter Kit 上的是
GR5515，项圈上是 GR5513BENDU —— 在 SK 上验过的固件不能直接认为在项圈上没问题。

**3. `CHIP_TYPE` 默认是 4（GR5515RGBD），不是我们的芯片。**

`custom_config.h` 里 `CHIP_TYPE` 的取值表：`6 = GR5513BEND`、**`7 = GR5513BENDU`**
（项圈上这颗），而 SDK 示例给的默认值是 `4`。直接拿示例改的话极容易漏掉这一行，
它影响一串派生配置——配错了在 GR5515 开发板上跑得好好的，换到项圈上才出问题。
本仓库的 `Src/config/custom_config.h` 已经设成 7。

**4. 留给应用的 RAM 是 112KB 不是 128KB。**

链接脚本从 `0x30000000 + 0x4000` 开始，前 16KB 是系统栈
（`custom_config.h` 里 `SYSTEM_STACK_SIZE 0x4000`）。这 112KB 还要跟 BLE 协议栈的
堆共用，留给我们的实际余量得编出来看 map 文件。

---

## 上板第一件事：跑自检，不是跑真实数据

```c
#include "tinyml_selftest.h"

tm_selftest_report_t rep;
if (tm_selftest_run(&rep) != 0) {
    APP_LOG_ERROR("自检失败：%s（第 %d 条）", tm_selftest_explain(&rep), rep.fail_index);
    /* 到这里就别往下看准确率了 */
} else {
    APP_LOG_INFO("自检通过，%d 条 golden vector", rep.n_checked);
}
```

为什么先跑它：准确率低有十几种可能（模型不行、特征错、编译选项错、传感器量程
不对……），而 golden vector 对不上**只有一种解释**——工具链或编译选项的问题，
跟模型无关。先排除掉这一层，后面的数才有意义。

`tm_selftest_run` 返回非 0 时 `tm_selftest_explain()` 会给出排查顺序：
①编译选项漏了 `-ffp-contract=off` 或别处塞了 `-ffast-math`；②导出的 `tm_*.c` 跟
PC 上验过的不是同一份；③int 宽度/对齐的假设在这个编译器上不成立。

注意 `n_checked == 0` **算失败**，不算通过——没导 golden vector、或者路线宏没配对
的时候会走到这里，而那正是最需要有人知道的情况。

---

## 采样与上报

```c
#include "tinyml_task.h"

static tm_task_t task;
static const tm_task_cfg_t cfg = {
    .event_class = 2,        /* 抓挠的类别号，看导出的 TM_CLASS_NAMES */
    .min_windows = 3,        /* 累计命中 3 个窗口才算一次 */
    .max_gap_windows = 2,    /* 中间最多空 2 个窗口还算同一次 */
};

tm_task_init(&task, &cfg);

/* 每出一个窗口的判决调一次 */
tm_event_t ev;
if (tm_task_on_window(&task, cls, now_ms, &ev)) {
    /* 这里才发 BLE 通知 */
}

/* 要睡 / 要上报汇总之前必须调一次，否则最后一次事件永远发不出去 */
if (tm_task_flush(&task, &ev)) { /* ... */ }
```

**聚合这一层才是端侧推理省电的地方。** 每个窗口发一条 BLE 通知的话，射频开销比
直接把原始数据传上去还大——端侧推理的意义就整个抵消了。人关心的是"今天抓了几次、
什么时候"，不是每秒一条。

`max_gap_windows` 不能是 0：抓挠中间会停顿（换姿势、挠另一边），一停就切断会把
一次连续抓挠拆成好几个事件，上报次数变多，统计出来的"抓了几次"也是错的。

IMU 那一侧建议用 QMI8658B 的 **FIFO 水位中断**批量取数，让 M4F 大部分时间睡着，
比定时器逐点取省得多。具体 FIFO 深度查手册再定 batch 大小。

---

## 还没做

- **没有 QMI8658B 的驱动。** SDK 的 `app_i2c` / `app_spi` 现成，但寄存器配置
  （量程、ODR、FIFO 水位）要对着手册写，写错了表现成"特征量纲不对、模型全错"，
  所以没凭印象写。
- **没有 Keil / IAR 的工程文件。** SDK 的 `build/gcc/keil2makefile.py` 是反过来的
  （Keil → Makefile）。要 Keil 工程的话，从 SDK 示例复制 `.uvprojx` 再加文件最稳妥，
  手写一个 XML 工程文件出错了很难查。
- **没量实际功耗和耗时。** 要板子。
