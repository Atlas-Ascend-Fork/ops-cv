# PasteSubImg

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

PasteSubImg 是一个面向图像或特征图 patch 拼接场景的区域累加算子，计算范式归为 layout_transform。算子从源图像 `patch_img` 中按 `core_area_coord` 指定的矩形子区域提取像素，经 `scale` 坐标缩放与 `patch_coord` 给出的平移量映射到目标画布 `combine_img` 的对应矩形位置，对目标区域内每个像素执行逐元素累加（`combine_img[dst] += patch_img[src]`），目标区域之外的像素保持原值。`combine_img` 以原地方式同时作为输入与输出，适用于图像拼接、超分辨率回填、滑窗推理特征图聚合等需要重叠区域累加的场景。

计算公式如下，其中 `patch_coord = [px1, py1, px2, py2]`（仅 px1、py1 参与计算），`core_area_coord = [cx1, cy1, cx2, cy2]`（右/下边界 exclusive），scale 为坐标缩放因子：

$$
s_{cy1}=\lfloor cy1 \cdot scale \rfloor,\ s_{cy2}=\lfloor cy2 \cdot scale \rfloor,\ s_{cx1}=\lfloor cx1 \cdot scale \rfloor,\ s_{cx2}=\lfloor cx2 \cdot scale \rfloor
$$

$$
d_{cy1}=\lfloor (cy1+py1) \cdot scale \rfloor,\ d_{cx1}=\lfloor (cx1+px1) \cdot scale \rfloor
$$

$$
combine\_img\_out[h, w, c] =
\begin{cases}
combine\_img[h, w, c] + patch\_img[s_{cy1}+(h-d_{cy1}),\ s_{cx1}+(w-d_{cx1}),\ c] & (h,w) \in [d_{cy1},\ d_{cy1}+\Delta h) \times [d_{cx1},\ d_{cx1}+\Delta w) \\
combine\_img[h, w, c] & \text{otherwise}
\end{cases}
$$

其中 $\Delta h = s_{cy2} - s_{cy1}$、$\Delta w = s_{cx2} - s_{cx1}$，C 维完全对齐逐元素相加。输出与输入同名（原地写回），公式中将累加结果记作 combine_img_out。

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
<td>patch_img</td>
<td>输入</td>
<td>源图像或特征图 patch，3D (H, W, C)，HWC 行优先；对应公式中 patch_img。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>patch_coord</td>
<td>输入</td>
<td>目标画布中的粘贴位置坐标 [px1, py1, px2, py2]，1D (4,)；仅 px1（列偏移）与 py1（行偏移）参与计算，对应公式中 patch_coord。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>core_area_coord</td>
<td>输入</td>
<td>源提取区域 [cx1, cy1, cx2, cy2]，1D (4,)，右/下边界 exclusive；对应公式中 core_area_coord。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img</td>
<td>输入</td>
<td>目标画布，3D (H_out, W_out, C)，HWC 行优先；以原地方式同时作为输入与输出，累加结果直接写回该张量。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>combine_img</td>
<td>输出</td>
<td>累加后的画布，与输入 combine_img 同名、同形、同数据类型，原地写回；公式中记作 combine_img_out。</td>
<td>UINT8、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>scale</td>
<td>属性</td>
<td>坐标缩放因子，乘到所有坐标值上把坐标空间映射到像素空间；必填属性，无默认值。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- patch_img 与 combine_img 的数据类型必须一致，C 维度必须相同。
- patch_coord 与 core_area_coord 的 shape 均为 [4]；core_area_coord 须满足 cx2 > cx1 且 cy2 > cy1。
- 源区域须完全位于 `patch_img` 内：`0 ≤ s_cy1 < s_cy2 ≤ H` 且 `0 ≤ s_cx1 < s_cx2 ≤ W`，其中 H、W 为 `patch_img` 的高、宽。
- 目标区域须完全位于 `combine_img` 内：`d_cy1 ≥ 0`、`d_cy1 + Δh ≤ H_out`、`d_cx1 ≥ 0`、`d_cx1 + Δw ≤ W_out`，其中 H_out、W_out 为 `combine_img` 的高、宽。
- scale 取值范围为 [0.0, 256.0]，非整数结果截断为 int 索引。
- 数据布局为 HWC 行优先 ND，不支持 NCHW 等其他布局。
- combine_img 以原地方式写回，调用前需保证为连续内存。
- 不支持空 Tensor（各维度长度须 ≥ 1）。
- uint8 路径经 float16 中转完成加法，溢出按饱和处理（>255 截断为 255），与部分框架的回绕语义不同。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_paste_sub_img.cpp](examples/arch35/test_geir_paste_sub_img.cpp) | 通过 [算子IR](op_graph/paste_sub_img_proto.h) 构图调用，并将输入/输出 dump 为 .bin 文件。 |

样例依赖 CANN GE 图模式的公开头库（graph.h、ge_api.h、array_ops.h、experiment_ops.h、nn_other.h 等），需在已安装 CANN 开发包的环境中，使用仓库根目录 `build.sh` 完成算子构建后，编译链接该样例可执行文件并运行；样例只做数据 dump，不包含精度比对。
