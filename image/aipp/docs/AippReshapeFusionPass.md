# AippReshapeFusionPass

## 功能说明

该Pass在图编译Shape推导（InferShape）完成后执行，用于修正Aipp算子的输入/输出描述信息。

网络中的Aipp节点，其images输入通常按AIPP处理后特征图的尺寸与格式来描述，与输入源图的真实排布不一致。该Pass解析Aipp节点aipp\_config\_path属性中的AIPP配置（JSON格式），将images输入的描述改写为源图的真实描述，并将格式与尺寸信息同步到features输出，保证后续按源图的真实排布进行内存分配和数据搬运。

该Pass的处理特点如下：

- 不删除、不新增、不替换任何节点，仅改写Aipp节点自身的输入/输出描述与属性，不改变图结构。
- 已带有has\_infered\_verified属性的Aipp节点会被跳过，保证该Pass可重复执行。
- 图中不存在待处理的Aipp节点时，图保持不变。

## 改写模式

根据AIPP配置中aipp\_mode的取值，分为静态AIPP和动态AIPP两种改写模式：

- 静态AIPP：源图的尺寸和格式在编译期完全确定。
- 动态AIPP：源图尺寸运行时才能确定，编译期仅已知输入图像的最大字节数max\_src\_image\_size。

两种模式下各改写项的规则如下：

| 改写项 | 静态AIPP | 动态AIPP |
| --- | --- | --- |
| images输入的shape | NCHW/NHWC布局：{N, srcH, srcW, C}；NC1HWC0\_C04布局：{N, C1, srcH, srcW, C0} | {1, max\_src\_image\_size} |
| images输入的format | NCHW/NHWC布局改写为NHWC；NC1HWC0\_C04布局保持原格式 | NHWC |
| images输入的dtype | 由input\_format决定，对应关系见下表 | UINT8 |
| images输入的size | 由input\_format与源图尺寸换算，对应关系见下表 | max\_src\_image\_size |
| features输出的format、size | format继承改写前images输入的format，size改写为源图的真实size | format继承改写前images输入的format，size保持不变 |
| features输出的shape、dtype | 保持不变 | 保持不变 |
| 节点属性 | 写入input\_dims和has\_infered\_verified | 仅写入has\_infered\_verified |

静态AIPP下，images输入的通道数C、dtype、size以及写入的input\_dims均由input\_format决定，完整对应关系如下：

| input\_format | 通道数C | 改写后dtype | 改写后size（字节） | input\_dims |
| --- | --- | --- | --- | --- |
| YUV420SP\_U8 | 3 | UINT8 | N×3×H×W/2 | {N, 3H/2, W, 1} |
| XRGB8888\_U8、ARGB8888\_U8、AYUV444\_U8 | 4 | UINT8 | N×4×H×W | {N, H, W, 4} |
| RGB888\_U8 | 3 | UINT8 | N×3×H×W | {N, H, W, 3} |
| YUV400\_U8 | 1 | UINT8 | N×H×W | {N, H, W, 1} |
| YUYV\_U8 | 3 | UINT8 | N×2×H×W | {N, H, W, 2} |
| YUV422SP\_U8 | 3 | UINT8 | N×2×H×W | {N, 2H, W, 1} |
| NC1HWC0DI\_FP16 | 3 | FLOAT16 | N×C1×H×W×4×2 | {N, H, W, 3} |
| NC1HWC0DI\_S8 | 3 | INT8 | N×C1×H×W×4 | {N, H, W, 3} |
| RAW10、RAW12、RAW16 | 1 | UINT16 | N×H×W×2 | {N, H, W, 1} |
| RAW24 | 1 | UINT32 | N×H×W×4 | {N, H, W, 1} |
| RGB16 | 3 | UINT16 | N×H×W×3 | {N, H, W, 3} |
| RGB20、RGB24 | 3 | UINT32 | N×H×W×3 | {N, H, W, 3} |
| RGB8\_IR | 4 | UINT8 | N×H×W×4 | {N, H, W, 4} |
| RGB16\_IR | 4 | UINT16 | N×H×W×4 | {N, H, W, 4} |
| RGB24\_IR | 4 | UINT32 | N×H×W×4 | {N, H, W, 4} |

表中N为batch维，H/W为源图高/宽，C1为images输入NC1HWC0\_C04布局中的C1维度；通道数C仅在images输入布局为NCHW/NHWC时用于shape改写。

补充说明：

- shape与format的改写同时作用于对应的origin shape/origin format。
- 静态AIPP下，源图高/宽（srcH/srcW）优先取配置项src\_image\_size\_h/src\_image\_size\_w，未配置时沿用images输入描述中的H/W。
- input\_dims为源图的NHWC逻辑尺寸，供acl接口侧使用。
- 动态AIPP下，仅当images输入带有4维shape range时，才将其改写为[{1, 1}, {1, max\_src\_image\_size}]，否则保持不变。

## 使用约束

- 仅处理Aipp类型节点，其他节点不受影响。
- aipp\_config\_path属性必须存在，且内容为合法的JSON对象；其中aipp\_mode为必选字段，取值仅支持static或dynamic。
- 静态AIPP约束：
    - images输入的格式仅支持NCHW、NHWC或NC1HWC0\_C04；NCHW/NHWC布局至少4维，NC1HWC0\_C04布局至少5维。
    - input\_format为必选配置项，仅支持下表所列取值。
- 动态AIPP约束：
    - max\_src\_image\_size必须大于0。
    - images输入不支持unknown rank（shape为[-2]）的场景。
- 配置非法或上述约束不满足时，图编译报错。

静态AIPP支持的input\_format取值：

| 类别 | input\_format取值 |
| --- | --- |
| YUV | YUV420SP\_U8、YUV422SP\_U8、YUV400\_U8、YUYV\_U8、AYUV444\_U8 |
| RGB | RGB888\_U8、XRGB8888\_U8、ARGB8888\_U8、RGB16、RGB20、RGB24 |
| RAW | RAW10、RAW12、RAW16、RAW24 |
| IR | RGB8\_IR、RGB16\_IR、RGB24\_IR |
| DI | NC1HWC0DI\_FP16、NC1HWC0DI\_S8 |

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
