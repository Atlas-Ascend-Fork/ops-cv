# KeepRatioResizeBilinearFusion

## 融合模式

该融合将图中的 `KeepRatioResizeBilinear` 算子（按 `min_dimension`/`max_dimension` 保持宽高比的双线性缩放，原型注册于 CANN 内置 IR）改写为 `ResizeBilinearV2` + int32 常量 `size`。融合规则注册于 `CustomPassStage::kAfterBuiltinFusionPass` 阶段，图编译时自动调度；CANN 内置同名 legacy pass 先行完成融合时，本规则自然旁路，legacy 移除后自动接管。

Pattern 为单节点，按 `KeepRatioResizeBilinear` 节点类型匹配。

**目标尺寸计算（keep-ratio，与原版逐行一致，`floor` 为向下取整）**

```text
scale_min = min(H, W) / min_dimension
候选1 = { floor(H / scale_min + 0.5), floor(W / scale_min + 0.5) }

scale_max = max(H, W) / max_dimension
候选2 = { floor(H / scale_max + 0.5), floor(W / scale_max + 0.5) }

newH, newW = (候选1 的长边 > max_dimension) ? 候选2 : 候选1
```

即先按"短边缩放到 `min_dimension`"计算候选，若其长边超出 `max_dimension` 则改按"长边缩放到 `max_dimension`"计算。H/W 取自输入 `images`：NHWC 时取 `dims[1,2]`，其余任何格式（含 NCHW）取 `dims[2,3]`。

**改写产物**

```text
        images                          images
          |                               |
  KeepRatioResizeBilinear   ==>   ResizeBilinearV2
          |                        /    \        （size 为 int32 常量
          y                       y      Const size {newH, newW}）
```

- `align_corners`/`half_pixel_centers` 属性透传（缺省 `false`）；
- 改写前先探测 `ResizeBilinearV2D`（经 `GeUtils::CheckNodeSupportOnAicore`）：支持时产出 V2D 形态（`size` 为属性、无常量输入）；当前各目标平台 V2D 均无 kernel，恒回退 V2 + 常量，与原版 const2attr 尝试后回退的行为一致；
- 目标节点沿用原节点的 images/y desc（`y` dtype 固定为 FLOAT，源算子 IR 定义）。

## 使用约束

- 平台约束：经 `rtGetSocSpec` 读取 `NpuArch`，除 `DAV_5102`（MC62CM12A）外全芯片生效——复刻原版注册机制（标准架构默认注册，注册白名单 3510 含本 pass、5102 不含，其余芯片不受白名单约束）。
- 结构约束：`images` 输入 shape 为空或非 4 维时，与原版一致放行至改写阶段失败，整图构建在本 pass 内失败（正常图不可达：内置 infershape 先行拦截）；H/W 维未知（-1 或 -2）跳过融合。
- 属性约束：`min_dimension`/`max_dimension` 必须存在且非 0，否则跳过（缺失时变量保持初值 0 走 ==0 检查，与原版不检查 GetInt 返回值的路径一致）。
- 版本约束：GE 编译器版本不低于 8.5.0 时融合生效（`CustomPassStage::kAfterBuiltinFusionPass` 自 8.5.0 起支持）。

## 支持的型号

本融合规则在除 `DAV_5102`（MC62CM12A）外的全芯片生效，可用型号由改写目标算子 `ResizeBilinearV2` 的算子库支持情况决定：

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->

<!-- npu="910b" id3 -->
Atlas A2系列产品
<!-- end id3 -->

<!-- npu="310p" id4 -->
Atlas推理系列产品
<!-- end id4 -->

<!-- npu="910" id5 -->
Atlas训练系列产品
<!-- end id5 -->

Atlas 200I/500 A2推理产品上 `ResizeBilinearV2` 无 kernel，融合产物不可执行，不在支持范围内。
