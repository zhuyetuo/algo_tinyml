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
  tinyml/forest.py       随机森林：从 sklearn 抽成扁平数组 + 参考前向
  tinyml/features.py     193 维手工特征的参考实现（纯 numpy float32）
  tinyml/export_forest_c.py / export_features_c.py   导出森林和特征常量表
  export_rf.py           sklearn 随机森林 → 板上的 C（要 sklearn）
  export_gbdt.py         **XGBoost → 板上的 C**（要 xgboost）
  rf_footprint.py        量 RF 搬过去占多少 flash（要 sklearn）
  verify_against_scipy.py  量端侧特征 vs scipy 版差多少、**判别翻了多少**（要 scipy）
firmware/tinyml/         跟芯片无关的纯 C，PC 上也能编（tests/ 就是这么测的）
  tm_runtime.c/h         int8 推理（conv1d / maxpool / dense），无 malloc、无 float
  tm_window.c/h          环形缓冲 → 窗口 → 量化
  tm_forest.c/h          随机森林推理（照抄 sklearn 的概率平均，不是多数投票）
  tm_gbdt.c/h            XGBoost 推理（判决是 < 不是 <=；叶子相加；端上不做 softmax）
  tm_features.c/h        193 维手工特征（含基-2 FFT、Welch、时域统计）
firmware/gr551x/         挂进 Goodix SDK 的工程（见它自己的 README）
  tinyml_app/GCC/Makefile      交叉编译 + 报体积，SDK 用 SDK_ROOT 指过去不复制
  tinyml_app/Src/user/         上板自检（golden vector）、逐窗口判决 → 事件聚合
tests/                   C ↔ Python 逐位对照（现场用 gcc 编）
docs/chip_choice.md      芯片选型
docs/rf_size.md          RF 体积怎么算、要不要量化剪枝
docs/frameworks.md       Cortex-M4F 上有哪些推理框架、为什么我们都没用
docs/market.md           市面上的项圈（FitBark/Fi/Tractive/Maven/Whistle）是不是端侧推理
docs/model_choice.md     端侧选哪个模型（1D-CNN / GBDT / LR / RF 的体积对照）
docs/features_cost.md    193 维特征贵在哪、能不能不用 float32、怎么砍
docs/xgb_result.md       **xgb 200 轮实测 + 怎么塞进 128KB**（轮数曲线/编码/二分类）
docs/ram_and_cache.md    运行内存实测（模型不占 RAM）+ 8KB cache 那笔账
  python/tune_operating_point.py  扫 min_windows × 类别偏置（两个零 flash 成本的旋钮）
tools/host_sim.c         在 PC 上跑板上那份 C，喂真实数据
```

**训练框架跟板上那一侧是隔离的**：`train_torch.py` 最后只交出一个
`{名字: numpy 数组}` 的 npz，量化、导出、对照那一整条链只依赖 numpy。所以换框架、
换训练机器都不波及固件，而且工具链能用随机权重自测——不用先有一个训好的模型
才能验证它。

---

## 在 Ubuntu 服务器上先看效果（不需要板子、不需要交叉编译器）

跑的是 `firmware/tinyml/` 下**板上那份一模一样的 C**，编译选项也跟固件一致
（`-ffp-contract=off`），所以**判决结果跟板上逐位相同**：

```bash
python python/run_host_sim.py --gen firmware/generated \
    --data ~/imu_train/data/processed_custom/test.npz
```

会打印每个窗口的类别和概率、各类占比、以及耗时。**耗时那一栏是 x86 的数，
跟 Cortex-M4F 没有可比性**，只能用来横向比 RF 和 CNN 哪个贵。

资源报告（模型多大、固件多大、运行内存、能存几天）：

```bash
python python/resource_report.py --gen firmware/generated \
    --elf firmware/gr551x/tinyml_app/GCC/build/tinyml_app.elf
```

每一行都标了是实测还是估算——体积和内存可以照着用，耗时只能当数量级。

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

走 RF 那一路的话，窗口这一层不做量化，直接攒 float：

```c
#include "tm_features.h"
#include "tm_feat_cfg.h"
#include "tm_forest.h"
#include "tm_forest_model.h"

static float win[TM_FEAT_N_CH * TM_FEAT_N_T];   /* 通道在前 */
static float feat[TM_FEAT_DIM];
static float proba[TM_F_N_CLASSES];

/* 攒满一个窗口之后 */
tm_features(&tm_feat_cfg, win, feat);
int cls = tm_forest_predict(&tm_forest, feat, proba);
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
- ~~没有真正链出 .bin~~ **链得出来，实测过**：用 `arm-none-eabi-gcc 13.2` 对着真
  SDK 链出了 `tinyml_app.bin`。基线（BLE 协议栈+最小应用）104KB flash / 22KB RAM，
  加上整条 tinyml 是 136KB / 25KB。细节和坑见
  [firmware/gr551x/README.md](firmware/gr551x/README.md)。
- **没有 QMI8658B 的驱动。** 寄存器配置（量程、ODR、FIFO 水位）要对着手册写，
  写错了表现成"特征量纲不对、模型全错"，没凭印象写。
- **没做性能优化。** 算子是最朴素的三重循环，没用 CMSIS-NN、没用 M4F 的 DSP 指令。
  这是有意的：第一版要的是"板上跟 PC 一模一样"。换 CMSIS-NN 之后，这套 golden
  vector 正好用来证明结果没变——**先有对照，再谈优化**。
- **没量实际功耗和耗时。** 要板子。

---

## RF 能不能上端侧

平台（label_service）现在跑的是随机森林，16Hz / 3 分类。**「稳定版 v2」不是另一个模型**，
是同一次推理结果的后处理解码模式（`raw` 逐窗口原始输出 / `stable` 滞回+合并 /
`viterbi` 动态规划），见 imu_train 的 `label_service/postprocess.py`。

**RF 端侧能跑。** 算力根本不是问题，这一点容易想当然地搞反：

- 窗口是 16Hz × 2 秒 = **32 个点**，`features.py` 里 `welch(nperseg=min(len(x), 32))`
  就是个 32 点 FFT。整条特征链（十来次 32 元素排序 + 8 路 32 点 FFT + 一堆统计量）
  撑死几万次浮点运算，一两秒才跑一次，占空比不到 0.5%。
- RF 推理本身**比 CNN 还便宜**——只有比较，没有乘法。200 棵树 × 十几二十层
  ≈ 几千次比较。

真正的约束只有一条，剩下的是工作量：

1. **flash 体积（唯一可能真卡住的）**——`configs/ml.yaml` 是
   `n_estimators: 200, max_depth: null`，不限深意味着节点数完全由训练数据量决定，
   可能几十 KB，也可能上 MB。**可测**：
   ```bash
   python python/rf_footprint.py --model ~/imu_train/results/.../rf/xxx.pkl
   ```
   超了多半也可解：**限深往往比砍树掉点少**——不限深的树尾部都是只覆盖几个样本的
   过拟合分支，那部分是噪声不是信息。
2. ~~193 维特征要在 C 里重写一遍~~ **已经做完了**（`tm_features.c`）。写的时候
   踩到的几处"写错了不会报错"：`np.percentile` 默认是线性插值不是取最近点、
   `find_peaks` 的平台算一个峰、`np.sign` 对正好等于 0 给 0、Welch 的窗是**周期**
   Hann、单边谱除首尾外要乘 2、峰度是 Fisher 的（减 3）。每一条都写成了
   变异测试能逮住的用例。

关于**浮点能不能两边一致**：算术核心能——全程锁 float32、关掉 FMA 合并
（`-ffp-contract=off`），IEEE-754 的加减乘除和 sqrt 都是精确定义的。对不齐的是 libm
的超越函数（频谱熵的 `logf`、偏度峰度的 `powf`、窗函数的正弦表）在不同实现上
末位可能不同。这是要处理的细节，不是墙；而且对 RF 的影响比对 CNN 还小——只有
正好卡在阈值边上的样本会翻分支，200 棵树投票会摊掉。

**流式解码那件事跟 RF 无关**，别记到它头上：`viterbi` 要看完整条时间轴才解码，
端侧看不到未来，必须改成有限延迟的在线版，改完结果会跟平台不一样。这一条
**跑什么模型都一样**，CNN 也躲不掉。

## 两条路线都做了

| | int8 CNN | 随机森林 |
|---|---|---|
| 模型体积 | ~1KB | 要量（`rf_footprint.py` / `export_rf.py` 都会打印） |
| 前处理 | 只要窗口 + 量化 | 193 维手工特征（含 FFT/Welch），已实现 |
| 算术 | 定点，板上跟 PC **逐位相同** | float32，逐位相同要靠 `-ffp-contract=off`，已验证 |
| 跟平台一致 | 两个模型、两套表现 | **同一个模型、同一套结论** |
| 代码 | `tm_runtime.c` + `tm_window.c` | `tm_forest.c` |
| 导出 | `quantize_and_export.py` | `export_rf.py` |

RF 路线的 golden vector 比的是概率的**位模式**（`%08x`），不是 argmax——只比 argmax
的话，一个已经算错、只是恰好还没把类别翻过去的实现能一路混到量产。

### RF 的一致性分两层，别混

```
imu_train/src/ml/features.py   scipy + float64    ← 平台在跑的，是基准
tinyml/features.py             numpy + float32    ← 参考实现，逐行对着上面写
firmware/tinyml/tm_features.c  C + float          ← 板上跑的
```

**参考实现 ↔ C 是逐位一致的**（除频谱熵——它用 `logf`，属于 libm，各实现不保证
正确舍入，测试单独给它 1 ULP 容差，其余全部逐位）。

**参考实现 ↔ scipy 不是，而且做不到**：scipy 全程 float64，M4F 只有单精度 FPU；
FFT 算法也不同，浮点加法不满足结合律。所以这一层只能量：

```bash
python python/verify_against_scipy.py --windows windows.npy --model xxx.pkl --hz 16
```

它的**主输出不是特征差多少，是判别翻了多少**——特征差第几位小数不重要，森林
判别翻没翻才重要。脚本还会看翻掉的样本原本置信度多高：翻的都是低置信度样本
就是边界抖动，有高置信度样本被翻那是 bug。

**RF 的体积怎么估、要不要量化剪枝**：见 [docs/rf_size.md](docs/rf_size.md)。
一句话——**约 20 B/节点，每加一层深度翻倍；剪枝是前提，量化基本不用做**
（超了 100 倍的时候，量化省的那 20% 没有意义）。`python/prune_rf.py` 能
**不重训**就给出「深度 → 体积 → macro-F1」整张表。

怎么选，等体积数出来：

```bash
python python/rf_footprint.py --model ~/imu_train/results/.../rf/xxx.pkl
```

限深之后能压进预算的话，**RF 是更稳的选择**——理由是表里最后一行：端上跑的就是
平台在跑的那个模型，项圈和平台给同一套结论。换 CNN 就是两个模型，以后每次对不上
都要先查是模型差异还是实现差异。

（`viterbi` 那种整条时间轴的解码两条路线都搬不过来，端侧看不到未来，必须改成
有限延迟的在线版——这跟选哪个模型无关。）
