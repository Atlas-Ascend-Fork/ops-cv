# Col2im

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

- Operator function: This operator implements the backward mapping of Im2col (Col2Im). It rearranges gradOutput, which is expanded by sliding windows, into the gradient tensor specified by inputSize. When multiple sliding windows map to the same output position, the operator sums the corresponding elements.
- Computation formula:

  Consider a gradOutput tensor with the shape $(C×\prod(kernel_size),L)$ or $(N,C×\prod(kernel_size),L)$, where $N$ is the batch dimension and $C$ is the channel dimension.

  This operation rearranges gradOutput into the out tensor with the shape $(C,spatial\_size[0],spatial\_size[1])$ or $(N,C,spatial\_size[0],spatial\_size[1])$.

  where:
  - $C×\prod(kernel_size)$ is the number of values in each sliding window block, and $L$ is the total number of these blocks:

    $$
    L=\prod_d⌊{\frac{spatial\_size[d]+2×padding[d]−dilation[d]×(kernel\_size[d]−1)−1}{stride[d]}+1}⌋
    $$

  - spatial_size is composed of inputSize, and $d$ iterates over all spatial dimensions.
  Therefore, the last dimension of gradOutput is the column dimension. Each index in this dimension corresponds to a sliding window block. When these sliding window blocks are mapped to out, the operator accumulates the elements at overlapping positions.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Backward input. It corresponds to output in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>5HD</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>Attr</td>
      <td>Shape of the input tensor. It corresponds to spatial_size in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Attr</td>
      <td>Size of the convolution kernel. It corresponds to kernel_size in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Attr</td>
      <td>Dilation parameter. It corresponds to dilation in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attr</td>
      <td>Padding size of the convolution. It corresponds to padding in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Attr</td>
      <td>Stride of the convolution. It corresponds to stride in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient computation result. It corresponds to input in the formula. Its data type and data format must be the same as those of `gradOutput`.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>5HD</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_im2col_backward](examples/test_aclnn_im2col_backward.cpp) | Calls the Col2im operator through the [aclnnIm2colBackward](docs/aclnnIm2colBackward_en.md) API. |
