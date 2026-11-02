# CheckValid

[English](./README_en.md) | 简体中文

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

检查候选框是否在图像有效范围内。给定候选框`bbox_tensor=(x0, y0, x1, y1)`与图像元信息`img_metas=(H, W, r)`，判断每个框是否满足：

$$
\text{valid} = (x_0 \geq 0) \wedge (y_0 \geq 0) \wedge (x_1 \leq W \cdot r - 1) \wedge (y_1 \leq H \cdot r - 1)
$$

计算时，两路输入均转换为float32；图像上界的乘法、减1及四个比较均在float32中完成。临界值按float32舍入后的结果判定，可能与实数或double中间量的结果不同。

输出`valid_tensor`为int8类型，1表示合法、0表示非法。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<td>bbox_tensor</td>
<td>输入</td>
<td>候选框坐标(x0,y0,x1,y1)，shape=(N,4)。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>img_metas</td>
<td>输入</td>
<td>一维图像元信息，shape=(G,)，G≥3；前三个元素为(H,W,r)，其余元素忽略。</td>
<td>FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>valid_tensor</td>
<td>输出</td>
<td>合法性掩码，1=合法/0=非法，shape=(N,1)。</td>
<td>INT8</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- bbox_tensor的shape必须为(N, 4)，末维固定为4。
- bbox_tensor与img_metas的dtype必须一致，支持组合为(FLOAT16, FLOAT16)和(FLOAT, FLOAT)。
- img_metas必须是一维张量(G,)，G≥3；前三个元素为H、W、r，其余元素忽略。bbox_tensor为空时，该约束同样适用。
- 输入仅支持ND数据格式。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 180px">
<col style="width: 200px">
<col style="width: 620px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE图模式</td>
<td><a href="examples/arch35/test_geir_check_valid.cpp">test_geir_check_valid.cpp</a></td>
<td>通过<a href="op_graph/check_valid_proto.h">算子IR</a>构图方式调用CheckValid算子</td>
</tr>
</tbody>
</table>

### GE 示例执行与结果

示例使用 FLOAT 输入：`bbox_tensor=[[0,0,2,2],[-1,0,2,2]]`，`img_metas=[2,3,1.5]`。
预期输出为 INT8 张量 `[[1],[0]]`，shape 为 `(2,1)`，分别覆盖合法框和非法框。

加载 CANN 环境和包含 CheckValid 的算子包后，在仓库根目录执行：

```bash
ASCEND_DEVICE_ID=0 bash build.sh --soc=ascend950 --run_example check_valid graph
```

设备编号可通过 `ASCEND_DEVICE_ID` 设置，默认是 0；也可直接向 `build/` 中编译后的程序传入设备编号：

```bash
./build/test_geir_check_valid 0
```

示例会逐元素比较输出，只有执行、结果比较和资源释放均成功时才打印
`CheckValid GE example passed` 并返回 0。输入和输出二进制文件保存在运行目录下，文件名分别为
`tc_ge_irrun_test_npu_input_0.bin`、`tc_ge_irrun_test_npu_input_1.bin` 和
`tc_ge_irrun_test_npu_output_0.bin`；文件写入或关闭失败时返回非零值。
