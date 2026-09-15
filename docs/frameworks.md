# Cortex-M4F 上的推理框架：调研

> 查于 2026-09。结论放前面：**我们这两条路线都不建议换框架**，但 CNN 那条应该用
> CMSIS-NN 的算子，RF 那条应该借鉴 emlearn 的节点编码。理由在下面。

先说一件容易被忽略的事：**"TinyML 框架"这个词几乎只覆盖神经网络**。我们有两条路线，
随机森林那条完全不在那些框架的射程内，得看另一批工具（emlearn / m2cgen 这类）。
拿 TFLM 的资料去规划 RF 的部署，方向从一开始就是错的。

---

## 一、神经网络那条（int8 CNN）

### TFLite Micro / LiteRT Micro（+ CMSIS-NN）

事实上的标准。2024 年 Google 把 TensorFlow Lite 改名叫 LiteRT，微控制器这条线
现在也以 [CMSIS Pack](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
的形式发布，接 Cortex-M 比以前顺。

**但它不适合我们**，理由是体积比例：

| | 大小 |
|---|---|
| 解释器内核 | 约 2 KB |
| **算子实现（按用到的 op 算）** | **50–200 KB** |
| tensor arena（小 CNN） | 20–60 KB |
| **我们的模型本身** | **约 1.3 KB** |

运行时比模型大两个数量级。TFLM 的设计点是"一套运行时跑各种模型"——模型会换、
op 会换，那时候解释器是划算的。我们的模型是固定的一条链（conv→relu→pool→conv→
relu→pool→dense），为它背一个 50KB+ 的通用解释器，换不回任何东西。

**什么时候该换过去**：模型结构开始频繁变、或者要同时跑好几个模型的时候。
那时 golden vector 那套对照正好用来证明"换完之后结果没变"。

### CMSIS-NN（Arm）—— 这个该用

它**不是框架，是算子库**：`arm_convolve_*`、`arm_fully_connected_*` 这些，用
Cortex-M4 的 SIMD/DSP 指令写的。Arm 给的数据是相对参考实现
**4.6 倍吞吐、4.9 倍能效**（[TF 博客](https://blog.tensorflow.org/2021/02/accelerated-inference-on-arm-microcontrollers-with-tensorflow-lite.html)）。

它是我们 CNN 那条路线**唯一值得现在就考虑的东西**：保留自己的运行时和 golden
vector，只把三重循环换成 CMSIS-NN 的核。换完之后 golden vector 必须原样通过——
这正是当初坚持"先有对照，再谈优化"的用处。

注意 CMSIS-NN 的量化约定跟 TFLite 一致（per-channel 乘子 + shift），而我们的
`fixedpoint.py` 照抄的就是 gemmlowp/TFLite 那套，所以对接是顺的。这不是巧合，
当初选这套规则就是为了留这条路。

### microTVM / Glow

AOT 编译器：把模型编译成特定的 C/汇编，没有解释器。体积和速度都比解释器好，
代价是工具链复杂度——要引入一整套编译流程，而且调试时"生成的代码"和"模型"
之间隔了一层。资料上普遍认为它们适合 **Cortex-M4 及以上**，所以不是不能用。

对我们：**收益和自己写的三重循环差不多**（模型太小，AOT 的优势主要体现在大模型的
算子融合上），成本高得多。不建议。

### TinyMaix

Sipeed 的极简推理库，静态分配、代码量极小、**刻意不用 CMSIS-NN**，能在 Arduino UNO
级别的芯片上跑基本 CNN。设计哲学跟我们自己写的那份很像——如果当初不想自己写，
它是最接近的选择。现在已经写完并且逐位验过了，没有理由换。

### NNoM

比 TinyMaix 高一层，更像框架。同上。

### ExecuTorch（PyTorch）

它在 MCU 上最完整的路径是**配合 Ethos-U 这类 NPU**；没有 NPU 的纯 MCU 只能退回
CPU 算子。GR5513 没有 NPU。不是我们这个场景的东西。

### X-CUBE-AI / STM32Cube.AI

ST 自家的，绑 STM32。我们是 Goodix GR551x。用不了。

### Edge Impulse（EON Compiler）

能用，而且工程化程度高。但它把数据、训练、部署整条链都拉到他们的平台上，
而我们的训练侧在 `imu_train`、标注在自己的平台上——引入它等于在链条中间插一个
外部依赖。第一版不值得。

---

## 二、随机森林那条 —— 另一批工具

### emlearn

最相关的一个，专门做 sklearn → C，支持决策树/随机森林/朴素贝叶斯/MLP，
不用动态内存、不依赖 stdlib。**它的节点编码值得直接对比**：

```c
/* emlearn */                          /* 我们的 */
int8_t  feature;   /* 1 B */           uint16_t feature;  /* 2 B */
int16_t value;     /* 2 B，量化过 */    float    threshold;/* 4 B */
int16_t left;      /* 2 B */           int32_t  left;     /* 4 B */
int16_t right;     /* 2 B */           int32_t  right;    /* 4 B */
/* = 7 B（对齐后 8 B） */               /* = 14 B */
叶子：1 B（类别号）或 n_classes B（uint8 概率）   叶子：n_classes × 4 B = 12 B
```

摊下来 **emlearn 约 9.5 B/节点，我们约 20 B/节点——它密一倍多**。

**但对我们的模型有一个硬限制**：emlearn 的 `loadable` 方式（就是上面那个紧凑的
表结构）**最多 127 个特征**，而我们的模型是 **193 维**。另一种 `inline` 方式支持到
一万个特征，但它把树生成成 if/else 代码而不是数据表——对几千个节点的森林，
代码体积通常比表还大。

所以：**想用 emlearn 的紧凑编码，得先把特征砍到 127 维以内**。这不一定是坏事
（193 维里肯定有冗余），但它是一个额外的、会影响模型精度的改动，要单独验证。

### m2cgen / sklearn-porter / micromlgen / EmbML

都是 sklearn → 各种语言的代码生成。m2cgen 和 sklearn-porter **没有针对 MCU 的适配**，
生成的是 if/else 代码，对大森林会把 flash 撑爆。emlearn 的随机森林实测比
sklearn-porter 快。这几个不如 emlearn 对口。

---

## 三、所以怎么做

**短期（等你那两个数出来之前）：什么都不换。**

我们现在这套的位置很清楚：模型体积上比 emlearn 差一倍，但**整条链在真芯片上逐位
验过**，而且不引入任何外部依赖。一倍的体积差在"可能超 100 倍"面前不是重点——
这跟 [rf_size.md](rf_size.md) 里"先剪枝再谈量化"是同一个道理。

**RF 的体积真的卡住的时候，按这个顺序**：

1. **剪枝**（`prune_rf.py`）。100 倍的杠杆只有这一个。
2. **借 emlearn 的编码，但不照抄**。把孩子下标 `int32 → int16`（省 4 B/节点，
   **不改任何判决**，前提是森林 < 32767 个节点），叶子概率 `f32 → uint8`
   （3 分类省 9 B/叶子，会轻微改判决但只影响最后相加、不影响路径选择）。
   两条加起来 **20 → 约 11.5 B/节点**，拿到 emlearn 大部分的密度，
   **而且不用把特征砍到 127 维**。128KB 预算下能放多少节点：

   | 编码 | B/节点 | 128KB 能放 | 改判决吗 |
   |---|---|---|---|
   | 现在（u16 feat + f32 thr + i32 孩子 + f32 叶子） | 20.0 | 6,553 | — |
   | 孩子换 i16 | 16.0 | 8,192 | **不改** |
   | 再把叶子概率换 u8 | 11.5 | 11,397 | 轻微（只影响相加，不影响路径） |
   | emlearn（阈值也量化成 i16） | 9.5 | 13,797 | 改路径，且限 127 特征 |

   从 6,553 到 11,397 是 1.7 倍。**但注意这一列的量级**：如果剪枝之后还差 10 倍，
   这 1.7 倍救不了；如果剪完只差 1.5 倍，它就正好够。所以顺序不能反。
3. **阈值量化到 int16** 放最后。它直接改路径选择（特征值落在新旧阈值之间的样本
   会走反），收益却只有 2 B/节点。

**CNN 那条要提速的话**：上 CMSIS-NN 的算子，别上 TFLM。运行时比模型大两个数量级
这件事不会因为"它是标准"而变得合理。

---

## 参考

- [LiteRT CMSIS Pack](https://www.keil.arm.com/packs/tensorflow-lite-micro-tensorflow/overview/)
- [tflite-micro 的 Arm 支持说明](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/arm.md)
- [CMSIS-NN 加速数据（TF 博客）](https://blog.tensorflow.org/2021/02/accelerated-inference-on-arm-microcontrollers-with-tensorflow-lite.html)
- [emlearn](https://github.com/emlearn/emlearn)（节点结构见 `emlearn/eml_trees.h`，
  特征数上限见 `emlearn/trees.py`）
- [TinyMaix](https://github.com/sipeed/TinyMaix)
- [Machine Learning for Microcontroller-Class Hardware: A Review](https://arxiv.org/pdf/2205.14550)
- [Deep Learning on Microcontrollers: The State of Embedded ML in 2025](https://shawnhymel.com/2994/deep-learning-on-microcontrollers-the-state-of-embedded-ml-in-2025/)
