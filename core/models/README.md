# 上板的模型导出

这里放**会烧进板子、也会被服务加载**的那两份模型导出：

    core/models/edge_cnn_i8/    CNN int8
    core/models/edge_rf_d10/    RF（compact，20 棵 × 深度 10）

每个目录里是 `export_cnn.py` / `export_rf.py --compact` 的产物，
外加一份 **`meta.json`**（训练产出那个 .json 的副本）。

## meta.json 不是可选的

没有它就**没法核对这是不是真模型**。自测时会造随机权重的演示导出，
文件名、结构、类别名跟真的一模一样，编得过、跑得通、golden 自检也过
（拿同一份假权重生成的，当然自己跟自己对得上）。提交进去之后谁也分不出来，
烧到板上只会得到一堆看着正常的错结论。

`scripts/check_export.py` 拿导出结果跟 meta.json 对棵数、类别、窗口几何。
`./board.sh build` 会先跑它，没过不让编。

## 别的 generated* 目录不进仓库

`.gitignore` 只放行上面那两个具体目录，别的 `core/models/generated*/`
一律挡住——自测的演示导出不该被 `git add -A` 扫进来。

## 怎么更新

见仓库根 README 的「换模型」一节，或者：

    python3 service/export_rf.py --compact --out core/models/edge_rf_d10 ...
    cp <训练产出>/ml_rf.json core/models/edge_rf_d10/meta.json
    python3 scripts/check_export.py core/models/edge_rf_d10
