# Crop

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: This operator crops a sub-region from the input tensor `x`. The size of the sub-region is determined by the shape of the reference tensor `size`. The `axis` and `offsets` attributes control the start position of cropping. `axis` specifies the start dimension of cropping. Dimensions before this dimension remain the same as those of `x`. Starting from the `axis` dimension, each dimension is cropped with the offset specified by `offsets` to the size of the corresponding dimension of `size`. This operator is compatible with the Crop layer of mainstream deep learning frameworks. It is mainly used for cropping and aligning multi-scale feature maps in object detection networks such as SSD.

- Computation formula:

$$
y[i_0, i_1, \ldots, i_{n-1}] = x[j_0, j_1, \ldots, j_{n-1}]
$$

where:

$$
j_k = \begin{cases} i_k, & k < \text{axis} \\ i_k + \text{offsets}[k - \text{axis}], & k \geq \text{axis} \end{cases}
$$

Output shape:

$$
\text{y.shape}[k] = \begin{cases} \text{x.shape}[k], & k < \text{axis} \\ \text{size.shape}[k], & k \geq \text{axis} \end{cases}
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to be cropped.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Reference tensor for cropping. Its shape determines the dimensions of the output shape from axis onward. Each dimension Si cannot exceed the corresponding dimension Di of x. The rank must be the same as that of x.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after cropping. Its dtype is the same as that of x/size.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attr</td>
      <td>Start dimension of cropping. The default value is 2. Dimensions before this dimension remain the same as those of x. Value range: [-rank(x), rank(x)-1]. Negative values are converted to positive values.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Attr</td>
      <td>Cropping offset of each dimension. When the length is 1, all cropped dimensions use the same offset. When the length is greater than 1, the length must equal rank(x)-axis. Each dimension must satisfy 0 <= offsets[i] <= x.shape[i] - size.shape[i].</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The number of dimensions (rank) of the inputs `x` and `size` must be the same. 1D to 8D are supported.
- The data types of `x`, `size`, and `y` must be the same.
- The value range of `axis` is [-rank(x), rank(x)-1]. A negative value is automatically converted to a positive value (axis += rank(x)).
- When the length of `offsets` is 1, all dimensions starting from `axis` use the same offset value. When the length of `offsets` is greater than 1, the length must equal rank(x) - axis, and each value corresponds to one dimension.
- The offset of each dimension must satisfy `offsets[i] + size.shape[i] <= x.shape[i]` and `offsets[i] >= 0`. Otherwise, an error is reported.
- For dimensions before `axis`, the output shape inherits the corresponding dimensions of `x`. The dimensions of `size` before `axis` do not participate in cropping.
- For the output shape, dimensions before `axis` inherit the corresponding dimensions of `x`, and the dimensions from `axis` onward use the corresponding dimensions of `size`.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Sample Code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_crop.cpp">test_geir_crop</a></td>
    <td>Calls the Crop operator by constructing a graph in the <a href="./op_graph/crop_proto.h">operator IR</a> mode. Refer to the <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation Guide</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
