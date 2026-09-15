# algo_tinyml

宠物项圈‑微型机器学习，嵌入式端侧推理：把 IMU 行为识别模型量化成 int8，跑在项圈的
蓝牙芯片 **GR5513BENDU**（Goodix GR551x，Cortex-M4F）上。

芯片怎么选的、为什么不是 nRF9151 / nRF7002，见 [docs/chip_choice.md](docs/chip_choice.md)。
一句话：推理放**本来就醒着、离 IMU 最近**的那颗；nRF7002 是 Wi-Fi companion IC，
里面根本没有通用 MCU。

---

## 这个仓库解决的是哪个问题

不是"怎么训一个模型"——那在 [imu_train](https://github.com/zhuyetuo/imu_train) 里。

是这一个：**量化 + 定点实现最典型的失败是「板上结果跟训练时不一样，但不报错」。**
它不崩、不打日志，只让准确率掉几个点。等你发现的时候，会去怀疑模型、怀疑数据、
怀疑传感器——就是不会怀疑某处舍入方向反了。

所以这里每一层都有**两份实现和一组 golden vector**：Python 一份（参考），C 一份
（板上真跑的那份），测试现场用 gcc 把 C 编出来，逐位对答案。

这套东西不是摆设。写的过程中它逮到了三个真问题：

| 谁错了 | 错在哪 | 不查会怎样 |
|---|---|---|
| Python 参考实现 | `np.atleast_2d` 把 dense 的 `acc` 从 `[3]` 变成 `[1,3]`，三个类别全套用了通道 0 的乘子 | 输出差几个 LSB，argmax 偶尔翻 |
| C | `rounding_divide_by_pot` 里 `1 << 31` 是未定义行为（UBSan 逮的） | x86 上"能跑"，换到 Cortex-M 上换一种错法 |
| 测试本身 | 整网 golden vector 对负数舍入**不敏感**——故意把 nudge 改错，全部照过 | 以为测过了，其实没有 |

第三条最值得记：端到端对答案只能证明"这组输入下两边一样"，证不了算子本身对。
所以定点原语另外单独扫（`tests/test_fixedpoint_c.py`）。

---

## 目录

```
python/
  tinyml/fixedpoint.py   定点重量化（gemmlowp / TFLite 那套规则）
  tinyml/net.py          小 1D-CNN：float 前向 + 训练后量化 + int8 前向（参考实现）
  tinyml/window.py       采样 → 窗口（参考实现）
  tinyml/export_c.py     导出 tm_model.c/h + tm_golden.h
  train_torch.py         训练（**唯一依赖 torch 的文件**），输出纯 numpy 的 model.npz
  quantize_and_export.py 量化 + 导出，只要 numpy
firmware/tinyml/
  tm_runtime.c/h         int8 推理（conv1d / maxpool / dense），无 malloc、无 float
  tm_window.c/h          环形缓冲 → 窗口 → 量化
tests/                   C ↔ Python 逐位对照（现场用 gcc 编）
docs/chip_choice.md      芯片选型
```

**训练框架跟板上那一侧是隔离的**：`train_torch.py` 最后只交出一个
`{名字: numpy 数组}` 的 npz，量化、导出、对照那一整条链只依赖 numpy。所以换框架、
换训练机器都不波及固件，而且工具链能用随机权重自测——不用先有一个训好的模型
才能验证它。

---

## 跑测试

```bash
bash run_tests.sh
```

需要 python3 + numpy + pytest + gcc。**不需要 torch，也不需要板子。**

---

## 从数据到板上

```bash
# 1. 训练（要 torch）。复用 imu_train 预处理好的窗口
python python/train_torch.py \
    --data ~/imu_train/data/processed_custom \
    --channels 6 --window 64 --epochs 40 --out model.npz

# 2. 量化 + 导出 C（只要 numpy）
python python/quantize_and_export.py \
    --model model.npz --out firmware/generated \
    --classes sleep,active,scratch

# 3. 把这几个文件加进 GR551x 的工程
#    firmware/tinyml/tm_runtime.c  tm_window.c
#    firmware/generated/tm_model.c tm_model.h tm_golden.h
```

第 2 步会打印两个数，**都要看**：

- `int8 与 float 判别一致率` —— 量化掉了多少
- `输入饱和比例` —— 高于 2% 说明校准集没覆盖到剧烈动作

第二个尤其要紧：只拿安静片段校准的话，抓挠那一段会整段饱和到 ±127，模型在最该
判对的时候是瞎的，**而且不报任何错**。这件事在 `tests/test_quantize.py` 里被钉成了
一个测试（先复现出饱和，再证明覆盖之后饱和消失），不只是一句注释。

---

## 板上怎么用

```c
#include "tm_runtime.h"
#include "tm_window.h"
#include "tm_model.h"

static int8_t ring[TM_N_CH * TM_N_T];
static int8_t win[TM_N_CH * TM_N_T];
static int8_t arena[TM_ARENA_BYTES];
static int8_t out[TM_N_CLASSES];
static tm_window_t w;

void app_init(void) {
    /* hop = TM_N_T/2 → 窗口重叠一半。重叠是为了不让一次动作正好被切在两窗之间 */
    tm_window_init(&w, ring, TM_N_CH, TM_N_T, TM_N_T / 2,
                   tm_model.in_scale, tm_model.in_zp);
}

/* 从 QMI8658B 的 FIFO 里取出来的一个样本：acc xyz + gyr xyz */
void on_imu_sample(const float s[6]) {
    if (tm_window_push(&w, s, win)) {
        if (tm_invoke(&tm_model, win, out, arena, sizeof arena) == 0) {
            int cls = tm_argmax(out, TM_N_CLASSES);
            (void)cls;
            /* 这里做事件聚合：连续 N 个窗口都判成抓挠才算一次。
               别一个窗口一个 BLE 通知——那就把端侧推理省下来的射频又花回去了 */
        }
    }
}
```

**上板第一件事是跑 golden vector**，别直接上真实数据：

```c
for (int i = 0; i < TM_GOLDEN_N; i++) {
    tm_invoke(&tm_model, tm_golden_in + i * TM_N_CH * TM_N_T, out, arena, sizeof arena);
    /* out 必须跟 tm_golden_out + i * TM_N_CLASSES 逐字节相同 */
}
```

对不上就先别看模型——那是工具链或编译选项的问题（比如开了 `-ffast-math`，
或者某个 int 宽度假设不成立）。对得上，才轮到讨论准确率。

---

## 现在还没有的

说清楚边界，免得看目录以为已经齐了：

- **没有真实模型。** 测试跑的是随机权重——工具链是验过的，模型还没训。要
  `imu_train` 那边先定下端侧用哪几类、窗口多长。
- **没有 GR551x 的工程文件。** `firmware/tinyml/` 是不依赖芯片头文件的纯 C，
  怎么挂进 GR551x SDK 的工程还没做。
- **没做性能优化。** 算子是最朴素的三重循环，没用 CMSIS-NN、没用 M4F 的 DSP 指令。
  这是有意的：第一版要的是"板上跟 PC 一模一样"。换 CMSIS-NN 之后，这套 golden
  vector 正好用来证明结果没变——**先有对照，再谈优化**。
- **没量实际功耗和耗时。** 要板子。

---

## RF 能不能上端侧

平台（label_service）现在跑的是随机森林，16Hz / 3 分类。**「稳定版 v2」不是另一个模型**，
是同一次推理结果的后处理解码模式（`raw` 逐窗口原始输出 / `stable` 滞回+合并 /
`viterbi` 动态规划），见 imu_train 的 `label_service/postprocess.py`。

RF 要搬到 GR5513 上，三个障碍，难度递增：

1. **体积**——sklearn 默认 `max_depth=None`，节点数完全由数据决定，只能量：
   ```bash
   python python/rf_footprint.py --model ~/imu_train/results/.../rf/xxx.pkl
   ```
   它会打印节点数和三种编码下的 flash 占用，以及预算内能放几棵。
2. **特征**——193 维手工特征里有 Welch PSD，端上要做 FFT。算得动（M4F 有 DSP 指令
   + CMSIS-DSP），但那是 float 的，两边**做不到逐位一致**，只能定容差。这比 int8 CNN
   的一致性问题难一个量级。
3. **后处理搬不过来**——`viterbi` 要看完整条时间轴才解码，端侧是流式的、看不到未来。
   必须改成有限延迟的在线版，而改完**结果跟平台上不一样**。这一点要先想清楚：
   否则同一段数据，项圈说是抓挠、平台说不是，而两边都"没错"。

所以这个仓库的默认路线是 int8 CNN，不是把 RF 搬过去——端侧要的是流式 + 定点可验证。
平台那边继续用 RF 不受影响，两者本来就是不同约束下的不同选择。
