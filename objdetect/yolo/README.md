# Yolo

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：对YOLOv2/v3目标检测网络的检测特征图进行数据重组和激活处理，将原始卷积输出转换为检测框坐标、目标置信度和类别概率三个输出，供后续Yolov3DetectionOutput算子使用。

- 计算公式：

输入x的shape为(N, boxes*(coords+1+classes), H, W)，通道维度C按(elem, box)排列，即先排列所有锚框的x坐标，再排列所有锚框的y坐标，依此类推。

坐标处理：

- x, y坐标做sigmoid激活
- w, h坐标直接搬移（move），输出时w和h位置交换，输出排列为(x, y, h, w)

$$
sigmoid(x) = \frac{1}{1 + exp(-x)}
$$

目标置信度和类别概率根据yolo_mode处理，yolo_mode由yolo_version、softmax和background三个属性决定：

- YOLO_MODE_1（V3，或V2且softmax=false且background=false）：obj=sigmoid, classes=sigmoid
- YOLO_MODE_2（V2且softmax=true且background=false）：obj=sigmoid, classes=softmax
- YOLO_MODE_3（V2且softmax=false且background=true）：obj=move, classes=sigmoid
- YOLO_MODE_4（V2且softmax=true且background=true）：obj和classes一起做softmax

## 参数说明

<table><thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>卷积层输出特征图，shape为(N, boxes*(coords+1+classes), H, W)，通道维度C=boxes*(coords+1+classes)，按(elem, box)排列。支持NCHW数据排布。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>coord_data</td>
    <td>输出</td>
    <td>检测框坐标，shape为(N, boxes*coords, CeilX(H*W*2+32,32)/2)，通道排列为(x, y, h, w)。输出最后一维使用ceilx内存对齐，有效数据为前H*W个元素。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>obj_prob</td>
    <td>输出</td>
    <td>目标置信度，shape为(N, CeilX(boxes*H*W*2+32,32)/2)。输出最后一维使用ceilx内存对齐，有效数据为前boxes*H*W个元素。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>classes_prob</td>
    <td>输出</td>
    <td>类别概率，shape为(N, classes, CeilX(boxes*H*W*2+32,32)/2)。输出最后一维使用ceilx内存对齐，有效数据为前boxes*H*W个元素。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>boxes</td>
    <td>属性</td>
    <td>锚框数量，取值大于0，默认值为3（YOLOv2默认5）。</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>coords</td>
    <td>属性</td>
    <td>坐标参数数，固定为4，对应x, y, w, h，默认值为4。</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>classes</td>
    <td>属性</td>
    <td>预测类别数，取值范围[1, 1024]，默认值为80。</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yolo_version</td>
    <td>属性</td>
    <td>YOLO版本，取值为"V2"或"V3"，默认值为"V3"。</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmax</td>
    <td>属性</td>
    <td>是否对类别概率做softmax，仅yolo_version为"V2"时有效，默认值为false。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>background</td>
    <td>属性</td>
    <td>obj和classes操作类型控制，仅yolo_version为"V2"时有效，默认值为false。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmaxtree</td>
    <td>属性</td>
    <td>固定为false，未使用，默认值为false。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入x必须为4维NCHW格式，shape为(N, boxes*(coords+1+classes), H, W)。
- 通道维度C必须满足C = boxes*(coords+1+classes)，否则结果未定义。
- coords固定为4，对应x, y, w, h四个坐标值。
- boxes必须大于0。
- classes取值范围为[1, 1024]。
- H和W必须大于0，即H*W > 0，不支持空空间维度。
- yolo_version仅支持"V2"或"V3"。
- 图模式支持动态Shape（-1未知维）和动态Rank（-2未知秩）。
- 输入x必须为连续tensor。
- float16输入时中间计算提升到float32精度。
- 输出coord_data的通道排列为(x, y, h, w)，注意w和h位置与输入相比发生交换。
- softmax和background属性仅当yolo_version为"V2"时有效。
- 输出shape的最后一维使用ceilx内存对齐，与canndev的1.0 infershape一致：
  - coord_data最后一维：CeilX(H*W*2+32, 32)/2，有效数据为前H*W个元素，padding区域补零。
  - obj_prob最后一维：CeilX(boxes*H*W*2+32, 32)/2，有效数据为前boxes*H*W个元素，padding区域补零。
  - classes_prob最后一维：CeilX(boxes*H*W*2+32, 32)/2，有效数据为前boxes*H*W个元素，padding区域补零。
  - CeilX(size, align) = (size + align - 1) / align * align，其中size为元素数乘以2（float16占2字节），align为32（32字节对齐）。
- 算子为确定性实现：sigmoid使用标准公式1/(1+exp(-x))，softmax使用max-subtraction数值稳定的3-pass实现（每线程独立顺序累加，无跨线程归约），相同输入多次运行结果完全一致。
- float16输入时中间计算提升到float32精度，最终结果转回float16。float16 subnormal值（[2^-24, 2^-14)）通过手动构造位模式保留，绕过硬件FTZ（Flush-To-Zero）模式。
- 算子采用多核多线程并行：基于GetBlockIdx/GetBlockNum的全局核索引，每核512线程，Grid-Stride循环分配工作量。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/arch35/test_geir_yolo.cpp">test_geir_yolo</a></td>
    <td>通过<a href="./op_graph/yolo_proto.h">算子IR</a>构图方式调用Yolo算子，参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
