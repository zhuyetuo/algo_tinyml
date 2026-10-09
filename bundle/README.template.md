# 端侧模型包 @TAG@

@KIND_NAME@，@N_CH@ 通道 × @N_T@ 点 @ @HZ@ Hz（窗口 @WIN_S@ s，步长 @HOP@ 点 = @HOP_S@ s），
类别（下标顺序）：@CLASSES@

这个包是**自包含**的：运行时 C（`core/`）+ 这一版模型导出（`model/`）+ PC 参考实现（`pc/`）。
不用再找算法要任何文件。

```
core/          跟模型无关的运行时，板上和 PC 共用同一份 C
  tm_imu.c/h       IMU @N_SENSOR@ 轴 → 模型 @N_CH@ 通道（重力对齐 + pitch/roll），见下面「输入映射」
@CORE_LIST@
model/         这一版模型导出（自动生成，别手改）
@MODEL_LIST@
pc/
  edge_cli.c       PC 参考实现：CSV → 逐窗口状态序列；--selftest 跑 golden 自检
  tm_edge_cfg.h    采样率 / 步长 / 轴数（edge_cli 用）
  sample.csv       合成的示例数据（@HZ@ Hz，g / °/s），只用来验证能跑通
Makefile       Linux 上编 edge_cli
@EXTRA_LIST@```

---

## 1. Linux 上编译、命令行测试（不要板子、不要 Python）

只要 `gcc` 和 `make`（Ubuntu：`sudo apt install build-essential`）。

```bash
unzip <下载的包>.zip -d edge && cd edge
make            # 编出 ./edge_cli
make test       # golden 自检 + 跑一遍 pc/sample.csv
```

`make test` 应该看到：

```
./edge_cli --selftest
selftest: N/N 通过
./edge_cli pc/sample.csv --out build/sample_result.csv
...
```

**selftest 不过 = 编译选项不对**（最常见是没带 `-ffp-contract=off`），不是模型问题。
golden 是导出时 Python 参考实现算的，C 必须逐位一样。

### 拿自己的采集数据跑

```bash
./edge_cli data.csv                          # 结果打到屏幕（CSV）
./edge_cli data.csv --out result.csv         # 写文件；汇总（各类窗口数、占比、耗时）打在 stderr
./edge_cli --help
```

CSV 要求：

| 项 | 要求 |
|---|---|
| 列 | 表头里有 `acc_x,acc_y,acc_z`@GYR_COLS@（`AccX` / `ax` 这类写法也认，大小写不敏感）；其它列（时间戳等）忽略。没表头就按 `ax,ay,az@GYR_NOHDR@` 的列序读 |
| 单位 | 加速度 **g**，角速度 **°/s**。原始 int16 计数用 `--acc-scale` / `--gyr-scale` 换算（见下） |
| 采样率 | 默认按 @HZ@ Hz。采集是它的整数倍时加 `--in-hz 50` 之类，会按块平均降到 @HZ@ Hz |
| 缺失 | 空格 / `nan` 当缺失：前向填充，一个窗口缺 > 30% 就跳过（跟平台一致） |

int16 原始计数的换算（量程 / 32768）：

```bash
# 例：加速度 ±16g、陀螺 ±2000°/s
./edge_cli raw.csv --acc-scale 0.00048828125 --gyr-scale 0.06103515625
# 加速度是 m/s²
./edge_cli data.csv --acc-scale 0.10197
```

加速度模长不像 1g 时 edge_cli 会在 stderr 打警告——单位错了模型照样给结论，只是全错，所以别忽略它。

输出每行一个窗口：

```
window,start_s,end_s,class_id,class,conf,p_<类别0>,p_<类别1>,...
```

`conf` / `p_*`：@SCORE_NOTE@

> 跟平台「端侧模型 · 板上 C」的结果比：模型、特征、输入映射是同一份 C，
> 采集本来就是 @HZ@ Hz 时逐窗口判决应当一致。降采样不同：平台用 scipy `resample_poly`（FIR），
> 这里是块平均，所以 `--in-hz` 不等于 @HZ@ 时边缘窗口可能有个别差异。
> 板上建议直接把 IMU 输出配成 @HZ@ Hz，或者自己做低通 + 抽取。

---

## 2. 输入映射：IMU @N_SENSOR@ 轴 → 模型 @N_CH@ 通道

**这一步不做或做错，模型照样出类别，只是准确率莫名其妙地掉。** 已经实现在 `core/tm_imu.c`，
照着调就行；下面是它做的事（跟训练时 `imu_train/src/gravity_align.py` 一致）。

每个 IMU 样本：`acc_x acc_y acc_z@GYR_SAMPLE@`，单位 **g**@GYR_UNIT@。凑满一个窗口（@N_T@ 个样本）之后：

1. **pitch / roll**（弧度），用**未旋转**的原始加速度逐样本算：
   `pitch = atan2(-ax, sqrt(ay² + az²))`，`roll = atan2(ay, az)`
2. **重力对齐**：窗口内加速度均值当重力方向 **g**，求把 **g** 转到 +Z 的旋转 R（Rodrigues 公式），
   R 同时作用到这个窗口的 acc@GYR_ROT@。|g|≈0 不转；已经朝 +Z 不转；朝 -Z 时绕 X 轴转 180°。
3. 按通道在前 `[通道][时间]` 排：

| 通道 | 内容 |
|---|---|
@CH_TABLE@

```c
#include "tm_imu.h"
/* 一次性：一个窗口的原始采样 [n_t][@N_SENSOR@]（时间在前）→ [@N_CH@][n_t]（通道在前） */
tm_imu_channels(raw, @N_T@, @N_SENSOR@, x);

/* 或者流式：每来一个样本喂一次，每 @HOP@ 个样本吐一个窗口 */
static float ring[@N_T@ * @N_SENSOR@], lin[@N_T@ * @N_SENSOR@], x[@N_CH@ * @N_T@];
static tm_imu_stream_t s;
tm_imu_stream_init(&s, ring, lin, @N_SENSOR@, @N_T@, @HOP@);
...
float sample[@N_SENSOR@] = {ax_g, ay_g, az_g@GYR_SAMPLE_C@};
if (tm_imu_push(&s, sample, x)) {
    /* x 就是模型输入，往下走推理 */
}
```

---

## 3. 板上调用顺序

@API@

接法二选一：

- **A. 源码**（推荐）：把 `core/*.c` 和 `model/*.c` 加进工程一起编，编译选项跟自己的 SDK 一定一致。
- **B. 静态库**：链 `lib/libtinyml.a`，头文件在 `include/`。预编选项见 `lib/BUILD_FLAGS.txt`，
  Cortex-M4F + softfp（跟 GR551x SDK 一致）；工程是 hard ABI 的话链不上，用 A。包里没有 `lib/`
  说明打包的机器上没装 arm-none-eabi-gcc，用 A。

编译务必：

- `-ffp-contract=off`（否则浮点末位跟 PC 对不上，golden 自检会红）
- `-DTM_FEAT_MAX_T=@N_T@ -DTM_FEAT_MAX_NPERSEG=@N_T@`：按真实窗口开缓冲，不给的话按 64 编、RAM 多占
- 核心代码不依赖任何 OS 接口，裸机或 RTOS 任务里都能调；只用 libm；无 malloc。

板上自检：把 golden 头文件（`model/*golden.h`）编进自检程序，跟 `pc/edge_cli.c` 里 `selftest()` 一样逐位比，
怎么挂进 GR551x 工程见 `board/README.md`。

@ACCEL@
---

## 4. 资源占用（@TOOLCHAIN@）

@FOOTPRINT@
