# DistanceTransform

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算二值图像中每个前景像素到最近背景像素的距离，并输出与输入shape相同的距离图。输入值为0的像素表示背景，非0像素表示前景。
- 距离类型：支持L1距离、L2距离和棋盘距离。设前景像素坐标为$p=(p_h,p_w)$，图像内背景像素集合为$B$，则：

  - L1距离（`distance_type="L1"`）：

    `mask_size`仅校验是否为0、3或5，不参与算法选择，实际固定按3×3邻域计算。

    $$
    y(p)=\min_{q\in B}\left(|p_h-q_h|+|p_w-q_w|\right)
    $$

  - 精确L2距离（`distance_type="L2"`且`mask_size=0`）：

    $$
    y(p)=\min_{q\in B}\sqrt{(p_h-q_h)^2+(p_w-q_w)^2}
    $$

  - 棋盘距离（`distance_type="CHESSBOARD"`）：

    `mask_size`仅校验是否为0、3或5，不参与算法选择，实际固定按3×3邻域计算。

    $$
    y(p)=\min_{q\in B}\max\left(|p_h-q_h|,|p_w-q_w|\right)
    $$

- 当`distance_type="L2"`且`mask_size`为3或5时，算子使用与`cv2.distanceTransform`对应的3×3或5×5 chamfer掩码计算近似L2距离。
- 背景像素的输出为0。Rank 3和Rank 4输入中的各幅图像相互独立计算。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1000px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 170px">
    <col style="width: 380px">
    <col style="width: 180px">
    <col style="width: 120px">
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
      <td>待进行距离变换的二值图像Tensor。支持(H, W)、(N, H, W)和(N, 1, H, W)三种shape。输入值0表示背景，非0值表示前景。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>distance_type</td>
      <td>可选属性</td>
      <td><ul><li>指定距离类型，支持“L1”、“L2”和“CHESSBOARD”，区分大小写。</li><li>默认值为“L2”。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_size</td>
      <td>可选属性</td>
      <td><ul><li>支持0、3和5。L2模式下，0表示精确距离，3和5分别表示使用3×3和5×5 chamfer掩码。</li><li>L1和CHESSBOARD模式下，mask_size仅作合法性校验，不参与算法选择，实际固定按3×3邻域计算。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>距离图Tensor，shape与输入x相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入`x`的rank必须为2、3或4，分别表示`(H, W)`、`(N, H, W)`或`(N, 1, H, W)`；Rank 4输入的通道维必须为1。
- `N`、`H`和`W`必须大于0，`H`不能超过INT32的最大值，`W`不能超过1073741824（2^30），Tensor总元素数不能超过UINT32的最大值。实际可处理规模还受设备和主机可用内存限制。
- 输入应为二值图像，推荐元素取值为0或1；实现中所有非0值均按前景处理。
- 图像外部不作为背景，分批输入中的各幅图像不会相互传播距离。
- 全前景图像不存在真实背景。此时L1返回`H + W`，CHESSBOARD返回`max(H, W)`，精确L2返回`sqrt(H² + W²)`，L2的3×3或5×5掩码路径返回`H + W`。
- 精确L2路径的中间状态保存在AI Core的UB中，过大的`H`可能因UB容量不足而不受支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_distance_transform](examples/test_aclnn_distance_transform.cpp) | 通过[aclnnDistanceTransform](docs/aclnnDistanceTransform.md)接口调用DistanceTransform算子。 |
| 图模式 | - | 通过[算子IR](op_graph/distance_transform_proto.h)构图方式调用DistanceTransform算子。 |

`aclnnDistanceTransformGetWorkspaceSize`、`aclnnDistanceTransform`接口及头文件`aclnn_distance_transform.h`由[op_host/CMakeLists.txt](op_host/CMakeLists.txt)中的`add_modules_sources(OPTYPE distance_transform ACLNNTYPE aclnn)`配置，在构建时依据[算子定义](op_host/distance_transform_def.cpp)自动生成，因此源码目录中没有单独维护的`op_api`目录。

运行aclnn样例前，需先构建并安装包含DistanceTransform的自定义算子包。编译样例时使用该安装包`op_api/include`下的头文件，并链接同一安装包的`op_api/lib/libcust_opapi.so`。样例先调用`aclnnDistanceTransformGetWorkspaceSize`获取workspace大小和执行器，申请workspace后调用`aclnnDistanceTransform`执行计算，最后同步stream并检查输出结果。
