# NonMaxSuppressionV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：按照分数递减顺序，采用贪心策略选择候选框（bounding boxes）子集。

## 参数说明

| 参数名               | 输入/输出/属性 | 描述                                                                                | 数据类型          | 数据格式 |
|-------------------|----------|-----------------------------------------------------------------------------------|---------------|------|
| boxes             | 输入       | 二维Tensor，其shape为(num_boxes, 4)。输入格式为(x1, y1, x2, y2)，并且要求x1 < x2且y1 < y2。 | FLOAT16、FLOAT | ND   |
| scores            | 输入       | 一维Tensor，其shape为(num_boxes)。表示与每个候选框对应的分数（与boxes的每一行一一对应）。                  | FLOAT16、FLOAT | ND   |
| max_output_size   | 输入       | 标量整数Tensor，表示非极大值抑制（Non-Maximum Suppression）最多选择的候选框数量。                          | INT32         | ND   |
| iou_threshold     | 输入       | 标量浮点Tensor，表示用于判断候选框是否在IoU（交并比）上重叠过多的阈值。                                        | FLOAT16、FLOAT | ND   |
| score_threshold   | 输入       | 标量浮点Tensor，表示用于根据分数移除候选框的阈值。                                                     | FLOAT16、FLOAT         | ND   |
| offset            | 可选属性     | • 可选整数。<br>• 默认值为0。                                                              | INT           | -    |
| selected_indices  | 输出       | 一维整数Tensor，shape为(M)，表示从输入boxes Tensor中选中的索引，其中M <= max_output_size。        | INT32 | ND   |

## 约束说明

- boxes和scores必须使用相同的数据类型，均为FLOAT16或FLOAT。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_non_max_suppression_v3](./examples/test_geir_non_max_suppression_v3.cpp)   | 通过[算子IR](./op_graph/non_max_suppression_v3_proto.h)构图方式调用NonMaxSuppressionV3算子。 |
