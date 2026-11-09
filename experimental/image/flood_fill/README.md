# FloodFill

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

产品表对应 `ascend910b`、`ascend910_93` 构建配置；当前上板验收环境为 Ascend 910B4 / CANN 9.1.0-beta.3。

## 功能说明

- 算子功能：从二维单通道图像的种子位置出发，按像素值容差和连通性确定填充区域，将区域内像素替换为 `new_value`，输出图像 `y`。
- 支持 4 连通和 8 连通；种子未被输入 mask 遮挡时属于填充区域，未填充位置保持输入值。
- 容差判定使用输入图像的原始像素值，填充后的值不参与后续区域判断。
- `NEIGHBOR`：候选像素与已属于填充区域的相邻像素比较；`FIXED_RANGE`：候选像素与种子像素比较。判定条件为：

  ```text
  delta = candidate - reference
  -lo_diff <= delta <= up_diff

  NEIGHBOR:    reference 为已接受的相邻像素原值
  FIXED_RANGE: reference 为种子像素原值
  ```

- 可选输入 `mask_in` 提供障碍与历史标记：内部任意非零值都会阻止填充。种子被遮挡时不填充，`y` 与 `x` 一致。
- 可选输出 `mask_out`：保留输入 mask 的内部原有标记，本次新填充位置写 1，未填充零值保持 0；外侧一圈边框统一写 1。未提供 `mask_in` 时等价于内部全零 mask。
- 可以将一次调用的 `mask_out` 作为下一次调用的 `mask_in`，避免重复填充；输入 mask 本身不会被修改。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| x | 输入 | 二维单通道图像，形状为 `[H, W]`。 | UINT8 / INT32 / FLOAT32 | ND |
| mask_in | 可选输入 | UINT8 障碍与历史标记，形状 `[H+2, W+2]`。非零内部位置不可填充；外圈值不参与判定。ACLNN 参数为 `maskInOptional`，无障碍输入可传 `nullptr`。 | UINT8 | ND |
| seed | 属性 | 必填，长度为 2，顺序为 `[row, col]`，即 `[行, 列]`；ACLNN 使用 `aclIntArray`。 | LIST_INT | - |
| new_value | 属性 | 必填，长度为 1 的填充值列表；ACLNN 使用 `aclFloatArray`，参数名为 `newValue`。 | LIST_FLOAT | - |
| lo_diff | 属性 | 必填，长度为 1 的下侧容差列表；ACLNN 使用 `aclFloatArray`，参数名为 `loDiff`。 | LIST_FLOAT | - |
| up_diff | 属性 | 必填，长度为 1 的上侧容差列表；ACLNN 使用 `aclFloatArray`，参数名为 `upDiff`。 | LIST_FLOAT | - |
| connectivity | 属性 | 取值 4 或 8，默认值为 4；ACLNN 调用时显式传入对应整数。 | INT | - |
| diff_mode | 属性 | 取值 `NEIGHBOR` 或 `FIXED_RANGE`，默认值为 `NEIGHBOR`；ACLNN 参数名为 `diffModeOptional`。 | STRING | - |
| y | 输出 | 填充后的图像，形状和数据类型均与 `x` 一致。 | UINT8 / INT32 / FLOAT32 | ND |
| mask_out | 可选输出 | 合并已有标记和本次填充标记，形状为 `[H+2, W+2]`；`x[row, col]` 对应 `mask_out[row+1, col+1]`。ACLNN 不需要该输出时传 `nullptr`。 | UINT8 | ND |

## 约束说明

- 当前支持 `[H, W]` 的单张、单通道图像，数据格式为 `ND`；不支持批量维度或三通道图像。
- 每维满足 `0 <= H, W <= 65535`，总像素数满足 `H * W <= 2147483647`。
- 非空图像的种子必须满足 `0 <= seed[0] < H`、`0 <= seed[1] < W`。
- `new_value`、`lo_diff`、`up_diff` 均通过 float32 列表传入，即使图像为整数类型也使用 `aclFloatArray`。三个列表长度均为 1，值必须有限，两个容差必须非负；零容差需显式传入 `[0.0]`。
- `y` 必须与 `x` 同形状、同数据类型；调用时使用互不重叠的输入和输出缓冲区。`mask_in`、`mask_out` 均使用对应尺寸的 UINT8 Tensor，输入和输出缓冲区互不重叠。
- 非空输入的 `y` 和绑定的 `mask_out` 均完整写出，输出缓冲区原有内容不作为输入条件。省略 `mask_out` 不改变 `y` 的填充结果。
- `H=0` 或 `W=0` 时为兼容性 no-op，不写 `y` 或 `mask_out`；上述种子范围和填充结果说明适用于非空输入，空图不生成 mask 边框。
- 数值属性的处理规则如下；具体实现见 [Host 属性解析](./op_host/flood_fill_tiling.cpp)。

| 输入类型 | new_value | lo_diff / up_diff |
| -------- | --------- | ---------------- |
| UINT8 | 使用 `std::llround` 取整后限制到 `[0, 255]`。 | 使用 `std::llround` 取整后限制到 `[0, 255]`。 |
| INT32 | 使用 `std::llround` 取整后限制到 INT32 值域。 | 使用 `std::llround` 取整后限制到 `[0, INT32_MAX]`。 |
| FLOAT32 | 使用传入的 float32 值。 | 使用 float32 减法计算 `candidate - reference`，再比较上下界。 |

`std::llround` 的半整数向远离零的方向取整，例如 `0.5 -> 1`、`-0.5 -> -1`。整数图像的差值使用 int64 计算；INT32 图像的属性仍受 float32 表示精度限制，不能借助列表传入任意精确的 int32 属性值。

## 与 OpenCV floodFill 的参数对应关系

OpenCV 的接口形式和 mask/flags 含义见 [官方 floodFill 文档](https://docs.opencv.org/4.13.0/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140839f18717)。迁移调用时需同时转换参数形式与输出处理方式。

| 项目 | OpenCV | 本算子 |
| ---- | ------ | ------ |
| 图像 | `image` 是输入输出参数，通常原地改写。 | `x` 为输入，`y` 为独立输出。 |
| 种子坐标 | `seedPoint=(x, y)`，即 `(列, 行)`。 | `seed=[row, col]`，即 `[行, 列]`。例如 OpenCV `(3, 1)` 对应本算子 `[1, 3]`。 |
| 填充值和容差 | C++ 使用 `Scalar`；Python 单通道调用可传标量，容差可省略为 0。 | 三个独立的、必填的长度 1 float32 列表。例如 `200` 对应 `[200.0]`，`0` 对应 `[0.0]`。 |
| 连通性 | 编码在 `flags` 的低 8 位。 | 独立参数 `connectivity=4` 或 `8`，不能传整个 OpenCV `flags`。 |
| 范围模式 | `FLOODFILL_FIXED_RANGE` 标志选择固定范围；未设置时为浮动范围。 | 分别对应 `diff_mode="FIXED_RANGE"` 和 `"NEIGHBOR"`，不按位组合。 |
| mask | 输入输出参数；输入非零位置可作为障碍，已有 mask 可供多次调用使用。 | 拆为只读的 `mask_in` 和独立的 `mask_out`；支持障碍、标记保留及跨调用复用。本次 mask 填充值固定为 1。 |
| 只生成 mask | 支持 `FLOODFILL_MASK_ONLY`。 | `y` 是必需输出，无 mask-only 模式；`mask_out=nullptr` 表示省略 mask 输出。 |
| 区域统计 | 返回填充数量，并可输出最小包围矩形 `rect`。 | ACLNN 返回状态码，无数量或 `rect` 输出；如需本次数量，统计内部 `(mask_in == 0) && (mask_out != 0)`；未提供输入 mask 时统计输出内部非零元素。均须排除边框。 |

整数图像的小数参数还存在数值语义差异：本算子对容差使用 `std::llround`；[OpenCV 5.0.0 实现](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/floodfill.cpp)对 UINT8/INT32 容差使用 `cvFloor`。例如图像 `[[10, 11]]`、种子为第一个像素、`lo_diff=0`、`up_diff=0.5` 时，本算子的有效上容差为 1，可填充两个像素；OpenCV 的有效上容差为 0，只填充一个。`new_value` 的标量转换也应按双方实现确认；做逐像素对拍时优先使用双方可精确表示的整数填充值和整数容差。

## 性能边界

搜索使用容量为 `2 * max(H, W) + 256` 的 span 表。梳齿、密集交替细碎区域等连通图案可能在合法输入尺寸内耗尽该容量；此时会先将延迟记录的行区间写入 visited 位图，再由核心 0 执行 `FallbackSweep`，逐轮扫描全图直到不再新增可达像素。输出图像和 mask 按最终位图生成，保证完整填充。

每轮回退扫描的工作量为 `O(H * W)`；轮数取决于传播方向与连通图案，反向长通路可能需要很多轮，因此回退不具有常规 span 搜索的性能保证。设轮数为 `R`，扫描成本为 `O(R * H * W)`；每个发生变化的轮次至少增加一个可达像素，保守最坏界为 `O((H * W)^2)`。多核调用在这段串行搜索期间，其余核心等待同步，增加核心数量不能加速该回退。尺寸上限仅表示支持的参数范围，不代表极端碎片图案具有可接受的延迟；有延迟要求的应用应使用实际图案评估。

## 调用说明

`aclnnFloodFillGetWorkspaceSize` 的第二个参数为 `maskInOptional`；不使用输入 mask 时传 `nullptr`。调用程序须使用匹配的生成头文件、动态库和算子包。

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| ACLNN 调用 | [test_aclnn_flood_fill.cpp](./examples/test_aclnn_flood_fill.cpp) | Tensor 和属性列表创建、两阶段调用、固定小图结果校验及资源释放。 |
| 编译与运行 | [examples/README.md](./examples/README.md) | 编译安装、示例运行命令和验证范围。 |

Host UT 位于 `tests/ut/op_host`，覆盖形状、类型、属性校验及 tiling。调用示例覆盖三种 dtype、4/8 连通、两种容差模式和可选 mask。

随代码维护的 [回归测试](./tests/README.md) 包含 `op_kernel` CPU 仿真 UT 和 ACLNN 实机 ST，覆盖自然 span 溢出回退、多核比较平面与暂停/恢复、行种子采纳、输入/输出 mask 组合。两者均对照独立逐像素 BFS，检查完整图像、mask 边框及原标签，并检查输入未修改。
