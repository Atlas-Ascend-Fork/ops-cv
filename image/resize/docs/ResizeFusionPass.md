# ResizeFusionPass

## 融合模式

该融合将图中的 `Resize` 算子（ONNX Resize 的 GE 适配算子，原型注册于 CANN 内置 IR）按 `mode` 属性改写为具体的目标算子，并裁剪冗余输入。融合规则注册于 `CustomPassStage::kAfterBuiltinFusionPass` 阶段，图编译时自动调度；CANN 内置同名 legacy pass 先行完成融合时，本规则匹配不到 `Resize` 节点、自然旁路，legacy 移除后自动接管。

Pattern 按 `Resize` 节点类型匹配：IR 为 1 个必选输入 `x` 加 3 个可选输入 `roi`/`scales`/`sizes`，按实际接入的可选输入个数共 4 种输入形态，每种形态各建一个 pattern。

**场景一：静态输出 shape（常量分支）**

`size` 取自输出 shape 按输入格式提取的空间维，生成 int32 常量输入（NHWC→`dims[1,2]`、NCHW→`dims[2,3]`、NCDHW→`dims[2,3,4]`；空间格式不在其中时产出 0 元素 `size` 常量，保持与原版一致）：

```text
        x                                x
        |                                |
      Resize      ==>   ResizeNearestNeighborV2 / ResizeBilinearV2
        |                                |   （size 来自 int32 Const 节点）
        y                                y
```

- `nearest` → `ResizeNearestNeighborV2`：改写前先探测 `ResizeNearestNeighborV2D`（经 `GeUtils::CheckNodeSupportOnAicore`），支持时产出 V2D 形态（`size` 为属性、无常量输入）；当前各目标平台 V2D 均无 kernel，恒回退 V2 + 常量，与原版 `CheckOpSupported` 行为一致；
- `linear` → `ResizeBilinearV2`。

**场景二：动态输出 shape（`resize_original_type == "onnx_resize"` 来源）或 `linear` 且 5 维输入（Slice 分支）**

在最后一个接入的输入（`sizes`，缺失时依次回退 `scales`/`roi`）前插入 `Slice(offsets={2}, size={2 或 3})` 提取空间维：

```text
        x     roi  scales  sizes        x                sizes
         \     |     |     /            |                  |
          \    |     |    /     ==>    |    Slice(offsets=2) /
            Resize                     ResizeNearestNeighborV2 / ResizeBilinearV2 / ResizeTrilinear
              |                          |
              y                          y
```

- `nearest` → `ResizeNearestNeighborV2`；
- `linear` 4 维 → `ResizeBilinearV2`；
- `linear` 5 维 → `ResizeTrilinear`（由 AICPU 算子库提供，Ascend 950 上实测可执行）。

**属性与输入语义**

- `coordinate_transformation_mode == "pytorch_half_pixel"` → `half_pixel_centers=true`；`== "align_corners"` → `align_corners=true`；其余均为 `false`；
- 目标算子的 `size` 输入 desc 显式为 int32（Slice 分支沿用源输入 dtype）；
- `roi`/`scales` 等多余输入作为替换图的悬空边界输入保留（等价原版 `DeleteInput` 语义，由 GE 死代码消除）；
- 进入原节点的控制边由框架子图替换机制保留（经插入的 NoOp 节点中转）。

## 使用约束

- 结构约束：按 `Resize` 节点类型匹配，1~4 输入形态均参与；无平台门控，全芯片生效。
- 跳过条件：
  - 指定 DVPP 引擎（`_specified_engine_name == "DNN_VM_DVPP"` 或 `_specified_kernel_lib_name == "dvpp_ops_kernel"`）的节点不改写；
  - 输出 `y` 维度数 < 4、`mode == "cubic"`、动态 shape 且非 `onnx_resize` 来源的节点跳过；
  - `nearest` 模式 `x` dtype ∉ {FLOAT16, FLOAT} 跳过。
- `mode` 属性缺失或非法时，与原版一致放行至改写阶段失败，整图构建在本 pass 内失败。
- 产物可执行性：改写产物是否可执行取决于目标算子（`ResizeNearestNeighborV2`/`ResizeBilinearV2`/`ResizeTrilinear`）在对应平台算子库的支持情况；`linear` 5 维的 `ResizeTrilinear` 当前仅 Ascend 950 可执行，其余平台该形态构建失败（与原版一致）。
- 版本约束：GE 编译器版本不低于 8.5.0 时融合生效（`CustomPassStage::kAfterBuiltinFusionPass` 自 8.5.0 起支持）。

## 支持的型号

本融合规则无平台门控，可用型号由改写目标算子的算子库支持情况决定：

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

Atlas 200I/500 A2推理产品上上述目标算子无 kernel，融合产物不可执行，不在支持范围内。
