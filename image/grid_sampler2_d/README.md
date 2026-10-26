# GridSampler2D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |


## 功能说明

- 算子功能：根据`grid`提供的归一化坐标，对四维输入`x`进行二维网格采样，支持双线性、最近邻和双三次插值。
- 输入输出尺寸：

  $$
  x: (N, C, H_{in}, W_{in})
  $$

  $$
  grid: (N, H_{out}, W_{out}, 2)
  $$

  $$
  y: (N, C, H_{out}, W_{out})
  $$

  `grid`最后一维依次存放x和y坐标，坐标通常归一化到[-1, 1]。实际输入坐标由`align_corners`决定：

  - `align_corners=true`：

    $$
    x' = \frac{grid_x + 1}{2}(W_{in}-1), \quad y' = \frac{grid_y + 1}{2}(H_{in}-1)
    $$

  - `align_corners=false`：

    $$
    x' = \frac{(grid_x + 1)W_{in}-1}{2}, \quad y' = \frac{(grid_y + 1)H_{in}-1}{2}
    $$

  越界坐标按照`padding_mode`指定的zeros、border或reflection方式处理，再按照`interpolation_mode`指定的bilinear、nearest或bicubic方式计算输出。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入特征图，shape为(N, C, H<sub>in</sub>, W<sub>in</sub>)。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NCHW</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>采样网格，shape为(N, H<sub>out</sub>, W<sub>out</sub>, 2)，数据类型必须与x一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>采样结果，shape为(N, C, H<sub>out</sub>, W<sub>out</sub>)，数据类型与x一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NCHW</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td>插值模式，支持"bilinear"、"nearest"和"bicubic"，默认值为"bilinear"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td>填充模式，支持"zeros"、"border"和"reflection"，默认值为"zeros"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td>是否将输入和输出的角像素中心对齐，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `x`与`grid`均为4维张量，数据类型一致且仅支持`float16`或`float32`。
- `grid`的最后一维必须为2，且`x`与`grid`的batch维必须一致。
- `x`的`H_in`和`W_in`必须大于0；`grid`产生空输出时支持空tensor。
- `x`与`grid`的各维大小均不能超过`INT32_MAX`。
- `x`的高宽乘积（`H_in * W_in`）不能超过`INT32_MAX`。
- 输出像素总数（`N * H_out * W_out`）不能超过`INT32_MAX`。
- `x`的元素总数（`N * C * H_in * W_in`）必须在64位有符号整型可表示范围内（否则拒绝）。
- `interpolation_mode`仅支持`bilinear`、`nearest`、`bicubic`。
- `padding_mode`仅支持`zeros`、`border`、`reflection`。
- 形状推导（InferShape）支持动态形状与动态秩：`x`的`H_in`/`W_in`可用`-1`表示未知维（输出高宽由`grid`推导），整张量形状未知时以`-2`表示未知秩；Tiling消费的是运行期已具体化的shape，未具体化的`-1`/`-2`维/秩不进入Tiling校验。batch维在`x`与`grid`间保持一致。
- 算子仅注册`x`=NCHW、`grid`=ND、`y`=NCHW逻辑格式；其他逻辑格式的接受/拒绝行为未声明。
- `grid`坐标为归一化坐标（通常落在[-1, 1]）；对非有限坐标（NaN/±Inf），行为与PyTorch `grid_sample`一致（`bilinear`/`bicubic`在`zeros`下输出NaN，`nearest`映射到合法索引）。对量级极大（约`|grid| ≳ 1e8`）的退化反射坐标，本实现不保证与PyTorch逐位一致。
- 超过容量上界（任一维大于`INT32_MAX`、`H_in * W_in`大于`INT32_MAX`、`N * H_out * W_out`大于`INT32_MAX`，或`N * C * H_in * W_in`超出64位有符号整型范围）的输入被拒绝；Kernel通路该拒绝不暴露独立错误码。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| GE图模式 | [test_geir_grid_sampler2_d](examples/arch35/test_geir_grid_sampler2_d.cpp) | 通过[算子IR](op_graph/grid_sampler2_d_proto.h)构图方式调用GridSampler2D算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。样例位于`examples/arch35`，编译运行须显式指定`--soc=ascend950`（如`bash build.sh --run_example grid_sampler2_d graph --soc=ascend950`）。 |

- 本算子以GE图模式注册；自定义包按`ACLNNTYPE aclnn_exclude`构建且不含TensorFlow插件，因此包内不生成`aclnnGridSampler2D`与TensorFlow接口。ACLNN/TF数值通路使用框架内置同名接口，自定义包示例的`cust`路由不适用。
