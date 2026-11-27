# ResizeDFusionPass

## 融合模式

该融合将图中的 `ResizeD` 算子（动态空间尺寸 Resize，原型注册于 CANN 内置 IR）按 `mode` 与 `data_format` 改写为具体的目标算子。融合规则注册于 `CustomPassStage::kAfterBuiltinFusionPass` 阶段，图编译时自动调度；CANN 内置同名 legacy pass 先行完成融合时，本规则自然旁路，legacy 移除后自动接管。

Pattern 为单节点，按 `ResizeD` 节点类型匹配（`sizes` 为必选属性）。

**场景一：`linear`（`data_format == "NCHW"`，输入 `[N, C, 1, W]`）**

`sizes` 属性转 int32 常量输入；在 `[N, C, 1, W]` 与 `[N, C, W]` 之间以 Squeeze/Unsqueeze 变换：

```text
        x                             x
        |                             |
      ResizeD        ==>          Squeeze(axis={-2})
        |                             |
        y                        ResizeLinear(scale=scales[0])
                                      |
                                  Unsqueeze(axes={-2})
                                      |
                                      y
```

- `scales` 取首个元素转为 `ResizeLinear` 的 `scale` 标量；
- Squeeze/Unsqueeze 的属性值保持原版 `{-2}`（负索引指向被压缩的 H 维）。

**场景二：`cubic` + NCHW**

直接改写为 `ResizeBicubicV2`（`scales` 以 ListFloat 透传）。

**场景三：`cubic` + HWNC**

先插入 `Transpose(perm=[2, 3, 0, 1])` 将输入转为 NCHW，再改写为 `ResizeBicubicV2`：

```text
        x                             x
        |                             |
      ResizeD        ==>          Transpose(perm=[2,3,0,1])
        |                             |
        y                        ResizeBicubicV2(scales)
                                      |
                                      y
```

**属性语义**

- `coordinate_transformation_mode == "align_corners"` 时目标算子 `align_corners=true`，其余均为 `false`；
- `sizes` 属性（`linear` 为 1 个目标 W 元素、`cubic` 为 2 个目标 H/W 元素）转 int32 常量 `size` 输入。

## 使用约束

- 平台约束：经 `rtGetSocSpec` 读取 `NpuArch`，仅 `DAV_3510`（Ascend 950 系列）生效；其余平台本规则匹配后跳过，`ResizeD` 不经融合、按平台算子库提供情况以原生 kernel 执行（Atlas A2 系列实测如此）。该门控复刻原版"代码内 `IsRegBase()` 门禁 ∩ 注册白名单仅含 3510"的交集语义。
- 结构约束：`mode ∈ {linear, cubic}`（默认 `nearest` 不融合）；已知秩时 `x`/`y` 为 4 维，未知秩（`[-2]`）跳过维度检查仍参与融合。
- 属性约束：
  - `data_format`：`linear` 仅支持 `NCHW`；`cubic` 支持 `NCHW` 或 `HWNC`；
  - `sizes`：`linear` 恰 1 个正数、`cubic` 恰 2 个正数；`scales`：`linear` 恰 1 个、`cubic` 恰 2 个（缺失时按默认值参与检查）；
  - `linear` 模式输入 H 维为 1（`[N, C, 1, W]`，由内置 infershape 保证），已知秩时 C/H/W 维非 0。
- dtype 约束：`x` ∈ {FLOAT16, FLOAT, BFLOAT16}（与原版一致）；`y` 的 dtype/format 必须与 `x` 一致。
- 版本约束：GE 编译器版本不低于 8.5.0 时融合生效（`CustomPassStage::kAfterBuiltinFusionPass` 自 8.5.0 起支持）。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
