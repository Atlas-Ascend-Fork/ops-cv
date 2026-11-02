# CheckValid

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

The operator checks whether candidate boxes are within the valid image range. Given the candidate boxes `bbox_tensor=(x0, y0, x1, y1)` and the image metadata `img_metas=(H, W, r)`, the operator determines whether each box satisfies:

$$
\text{valid} = (x_0 \geq 0) \wedge (y_0 \geq 0) \wedge (x_1 \leq W \cdot r - 1) \wedge (y_1 \leq H \cdot r - 1)
$$

Both inputs are converted to float32 before calculation. The image bound multiplication, subtraction of 1, and all four comparisons are performed in float32. Boundary cases are decided using float32 rounding and may differ from results computed with real numbers or double-precision intermediates.

The output `valid_tensor` is of the int8 type. 1 indicates valid, and 0 indicates invalid.

## Parameter Description

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
<th>Parameter name</th>
<th>Input/Output/Attr</th>
<th>Description</th>
<th>Data type</th>
<th>Data format</th>
</tr>
</thead>
<tbody>
<tr>
<td>bbox_tensor</td>
<td>Input</td>
<td>Candidate box coordinates (x0,y0,x1,y1). The shape is (N,4).</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>img_metas</td>
<td>Input</td>
<td>One-dimensional image metadata with shape (G,), where G ≥ 3. The first three elements are (H, W, r); any remaining elements are ignored.</td>
<td>FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>valid_tensor</td>
<td>Output</td>
<td>Validity mask. 1 indicates valid and 0 indicates invalid. The shape is (N,1).</td>
<td>INT8</td>
<td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- The shape of bbox_tensor must be (N, 4), and the last dimension is fixed to 4.
- bbox_tensor and img_metas must have the same dtype. The supported pairs are (FLOAT16, FLOAT16) and (FLOAT, FLOAT).
- img_metas must be a one-dimensional tensor with shape (G,), where G ≥ 3. Its first three elements are H, W, and r; any remaining elements are ignored. This constraint also applies when bbox_tensor is empty.
- Only the ND data format is supported for the input.

## Invocation Description

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 180px">
<col style="width: 200px">
<col style="width: 620px">
</colgroup>
<thead>
<tr>
<th>Invocation Method</th>
<th>Sample Code</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE graph mode</td>
<td><a href="examples/arch35/test_geir_check_valid.cpp">test_geir_check_valid.cpp</a></td>
<td>Call the CheckValid operator by constructing a graph in the <a href="op_graph/check_valid_proto.h">operator IR</a> mode</td>
</tr>
</tbody>
</table>

### Running the GE Example and Checking the Result

The example uses FLOAT inputs: `bbox_tensor=[[0,0,2,2],[-1,0,2,2]]` and `img_metas=[2,3,1.5]`.
The expected output is an INT8 tensor `[[1],[0]]` with shape `(2,1)`, covering one valid box and one invalid box.

After loading the CANN environment and an operator package containing CheckValid, run the following command from the repository root:

```bash
ASCEND_DEVICE_ID=0 bash build.sh --soc=ascend950 --run_example check_valid graph
```

Set `ASCEND_DEVICE_ID` to select the device; the default is 0. You can also pass the device ID directly to the compiled program in `build/`:

```bash
./build/test_geir_check_valid 0
```

The example compares every output element. It prints `CheckValid GE example passed` and returns 0 only after execution, result comparison, and resource cleanup succeed. Input and output binaries are written to the working directory as `tc_ge_irrun_test_npu_input_0.bin`, `tc_ge_irrun_test_npu_input_1.bin`, and `tc_ge_irrun_test_npu_output_0.bin`. A file write or close failure causes a nonzero return value.
