# DL 那条：1D-CNN 上端侧要改什么

## 一、先算体积：**默认配置正好卡在预算上**

`imu_train` 的 `cnn` 是 `Conv1d(padding=k//2) + BatchNorm + ReLU + MaxPool(2)` ×3，
然后一个全连接。`configs/dl.yaml` 默认 `filters: [64,128,256], kernel_size: 3`。

窗口 16 点 × 8 通道 × 5 分类：

| 配置 | 参数量 | int8 体积 | MAC/窗口 |
|---|---|---|---|
| **默认 `[64,128,256]`** | 127,877 | **124.9 KB** ⚠ 正好卡满 | 616,960 |
| filters 减半 `[32,64,128]` | 33,221 | **32.4 KB** ✓ | 161,024 |
| 再减半 `[16,32,64]` | 8,933 | **8.7 KB** ✓ | 43,648 |
| `collar_cnn` 默认（pool=4） | 209,541 | 204.6 KB ✗ | 369,920 |

参数量几乎全在第三层：`conv 128→256 k3` 一层就 98,304 个权重，占 77%。
**减 filters 是这条路线上最直接的杠杆**，而且减半只掉四分之三的参数。

## 二、CNN 相对 GBDT 的两个优势

| | 体积 | 访存 |
|---|---|---|
| GBDT 200 轮（紧凑） | 393 KB | **随机**——每个节点往哪走取决于上一个节点，预取无效 |
| GBDT 50 轮（紧凑） | 114 KB | 同上 |
| **CNN 默认** | 125 KB | **顺序**——权重是流式读的，cache 和预取都能用上 |
| CNN filters 减半 | 32 KB | 顺序 |

在只有 8KB cache 的 GR5513 上，**访存模式这一项对 CNN 明显有利**
（见 [ram_and_cache.md](ram_and_cache.md)）。这一点在纯看体积的比较里是看不见的。

另外 CNN 那条是 **int8**，GBDT 是 float32 —— int8 卷积还能上 CMSIS-NN
（Arm 给的数是 4.6 倍吞吐 / 4.9 倍能效），树遍历没有对应的加速库。

## 三、但要部署，我这边缺两样

现在的 `tm_runtime.c` 跑不了 `imu_train` 的 CNN，差两个东西：

**1. padding。** 我的卷积是 valid（不补零），`imu_train` 用的是 `padding=k//2`。
   16 点窗口在 valid 下走三层会被压没，所以这个必须补。
   量化域里补的不是 0 而是 **zero_point**——补 0 的话相当于补了一个真实的
   负值进去，边界那几个输出会系统性偏，而且不报错。

**2. BatchNorm。** 推理时 BN 可以**折进卷积的权重和偏置**，折完完全等价、
   运行时零开销。这一步在 Python 侧做，C 不用知道 BN 存在。
   折错了的表现是"训练时好好的，导出之后准确率莫名其妙掉一截"。

两样都做完之后，`imu_train` 训的 CNN 就能直接导出上板，跟 GBDT 走同一套
golden vector 对照。

## 四、建议先训哪几个

```bash
cd ~/imu_train
for M in cnn collar_cnn filternet; do
  python src/dl/train.py --hz 16 --model $M \
    --date "2026_8_11-2026_8_27_raw" --source_hz 50 \
    --extra_date "2026_7_17-2026_7_29:16" \
    --window_s 1 --stride_s 0.5 --label_mode majority \
    --missing_strategy drop_window \
    --remap configs/remap_custom_3class.yaml
done
```

**别用 `cnn_lstm`**：LSTM 要跨窗口维护隐状态，端上丢包/漏窗口/重启之后状态就脏了，
**而且不报错，只会让结果慢慢变差**。卷积是无状态的，一个窗口进一个结果出。

拿到结果之后，重点看**抓挠那一列**（跟 xgb 的 0.74/0.83/0.78 比），别只看 macro-F1。
之前那个 `cnn_lstm` 跑出来抓挠是 0.75/0.68/0.72——recall 比 xgb 低不少。

如果默认 filters 的效果够好但 125 KB 太满，先试 `[32,64,128]`（32 KB）看掉多少；
**参数量掉四分之三，效果未必掉那么多**——这类任务上 16 点窗口的信息量有限，
256 个通道大概率是过参数化的。
