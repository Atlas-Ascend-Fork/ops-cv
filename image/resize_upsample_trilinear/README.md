# ResizeUpsampleTrilinear

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1. 将目标图像的每一个点映射回原图，得到一个带小数点的坐标。
    2. 根据这个浮点数坐标，计算前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐（alignCorners为true）表示按照原始图片左上角像素中心点对齐，边对齐（alignCorners为false）表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时存在差异。对于一个三维插值点$(N, C, D, H, W)$，则有以下公式：

    $$
    scale\_d =\begin{cases}
    1 & outputSize[0] = self.dim[2] \\
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true\&outputSize[0]>1 \\
    0 & alignCorners=true\&outputSize[0]=1 \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    1 & outputSize[1] = self.dim[3] \\
    (self.dim[3]-1) / (outputSize[1]-1) & alignCorners=true\&outputSize[1]>1 \\
    0 & alignCorners=true\&outputSize[1]=1 \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    self.dim[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    1 & outputSize[2] = self.dim[4] \\
    (self.dim[4]-1) / (outputSize[2]-1) & alignCorners=true\&outputSize[2]>1 \\
    0 & alignCorners=true\&outputSize[2]=1 \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    self.dim[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

    因此，对于output的某个方向上的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系：

    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - 记输入在D、H、W轴上的长度分别为$self.dim[2]$、$self.dim[3]$、$self.dim[4]$。将每个轴的两个邻点限制在有效下标范围内，并按以下公式计算权重：

      $$
      x_{0} = min(int(x'), self.dim[2]-1), x_{1} = min(x_{0}+1, self.dim[2]-1), lambda_{1} = min(max(x'-x_{0},0),1), lambda_{0} = 1-lambda_{1}
      $$

      $$
      y_{0} = min(int(y'), self.dim[3]-1), y_{1} = min(y_{0}+1, self.dim[3]-1), lambdb_{1} = min(max(y'-y_{0},0),1), lambdb_{0} = 1-lambdb_{1}
      $$

      $$
      z_{0} = min(int(z'), self.dim[4]-1), z_{1} = min(z_{0}+1, self.dim[4]-1), lambdc_{1} = min(max(z'-z_{0},0),1), lambdc_{0} = 1-lambdc_{1}
      $$

    - 则有以下公式：

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}}
      $$

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
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>可选属性</td>
      <td><ul><li>表示指定`output`在D、H和W维度上的空间大小，对应公式中的`outputSize`。size为3，且各元素均大于零。</li><li>默认值为空。</li></ul></td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>决定是否对齐角像素点，对应公式中的`alignCorners`。align_corners为true，则输入和输出张量的角像素点会被对齐，否则输入和输出张量的左上角顶点及两条边对齐。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_d</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的depth维度乘数，对应公式中的`scales_d`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_h</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的height维度乘数，对应公式中的`scales_h`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scales_w</td>
      <td>可选属性</td>
      <td><ul><li>指定空间大小的width维度乘数，对应公式中的`scales_w`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式描述中的`output`。数据类型和数据格式与入参`input`的数据类型和数据格式保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

<term>Atlas推理系列产品</term>：

- 输入和输出的数据类型不支持BFLOAT16。
- 入参`input`不支持输入inf、-inf。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_upsample_trilinear3d](examples/test_aclnn_upsample_trilinear3d.cpp) | 通过[aclnnUpsampleTrilinear3d](docs/aclnnUpsampleTrilinear3d.md)接口方式调用ResizeUpsampleTrilinear算子。 |
