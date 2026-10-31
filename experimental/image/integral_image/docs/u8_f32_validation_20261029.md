# IntegralImage ACLNN 与单元测试验证报告

## 环境与构建

- 日期：2026-10-29。
- 真机平台：Atlas A2 / Ascend910B4。
- CANN：8.5.0 beta.1。
- 构建范围：IntegralImage 单算子，包含 OpDef、ACLNN、L0、Tiling、Kernel；宿主源码按该范围收集。
- ACLNN 示例加载本次算子包安装的 `libcust_opapi.so`。

```bash
CMAKE_BUILD_PARALLEL_LEVEL=16 bash build.sh --pkg --experimental --ops=integral_image --soc=ascend910b -j16
```

单算子包构建、安装通过，构建命令退出码为 0。验证产物 `cann-ops-cv-custom-linux.aarch64.run` SHA-256：

```text
a33aaac22b11068736570f19d3fdce95f69942743a7e4ad00a7a7fc418843430
```

## 类型支持

OpDef 的 DataType、Format、UnknownShapeFormat 按四组 ND 类型签名配置，生成的 ACLNN 支持表包含以下组合：

| image | sdepth | sat | sqsum（提供时） | tilted（提供时） |
| --- | --- | --- | --- | --- |
| UINT8 | -1 / 0 | INT32 | FLOAT32 | INT32 |
| FLOAT16 | -1 / 1 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT32 | -1 / 1 | FLOAT32 | FLOAT32 | FLOAT32 |
| UINT8 | 1 | FLOAT32 | FLOAT32 | FLOAT32 |

## UINT8→FLOAT32 专项验证

每项依次执行 `GetWorkspaceSize`、Workspace 申请、算子执行、Stream 同步及输出回读，并逐元素比较 CPU 参考结果。参考值按接口数学定义计算，包含 `tilted` 的三角累加及末行、末列零边界；浮点比较检查结果为有限值。

```bash
./test_aclnn_integral_image u8_f32
```

| 输入 Shape | 输出组合 | 覆盖内容 |
| --- | --- | --- |
| (8,64) | 形态 1/2/3 | 单通道窄图 |
| (8,131) | 形态 1/2/3 | two-phase、多核、非对齐尾块 |
| (17,131) | 形态 1/2/3 | 多行前缀与对角累加 |
| (2,1031) | 形态 1/2/3 | 跨批次输出与尾部 |
| (1,1) | 形态 1/2/3 | 单像素 |
| (9,1) | 形态 1/2/3 | 单列 |
| (4,6,3) | 形态 3 | HWC 逐通道计算 |
| (9,33,3) | 形态 3 | HWC 非对齐与通道复用 |

形态 1 的 sqsum、tilted，以及形态 2 的 tilted 均传 `nullptr`。完整功能入口包含本组 20 项专项用例，结果为：

```text
IntegralImage UINT8->FLOAT32 ACLNN: 20 / 20 PASS
```

## 参数校验验证

```bash
./test_aclnn_integral_image validation
```

共 13 项反例，仅调用 `GetWorkspaceSize` 并检查返回非零错误码，覆盖：

- sat、sqsum、tilted 的 Rank、空间尺寸和 HWC 通道数。
- 元素总数相同但维度不同的输出。
- `sdepth` 与 sat、tilted 类型的一致性。
- sqsum 的 FLOAT32 类型要求。

```text
IntegralImage parameter validation: 13 / 13 PASS
```

有效参数对照使用 UINT8 单像素输入 `7`，覆盖 `sdepth=-1/0/1` 与形态 1/3，共 6 组；sat 结果均为 `[[0,0],[0,7]]`，三路输出时平方和与倾斜累加结果也逐元素一致。

## 完整 ACLNN 与大尺寸验证

```bash
./test_aclnn_integral_image
./test_aclnn_integral_image verify
```

完整入口共 58 项通过，包含 20 项 UINT8→FLOAT32 专项、13 项输出参数校验，以及三种输入类型、不同输出组合、HWC、多核、非对齐和退化尺寸等用例。

大尺寸普通积分图对 UINT8、FLOAT32 分别执行 1024×1024、2048×2048、4096×4096，共 6 组；每组逐元素比较的失败数量均为 0。

## Host 与 Kernel UT

```bash
bash build.sh -u --ophost --opkernel --experimental --ops=integral_image -j16
```

| 测试 | 通过数 / 总数 | 结果 |
| --- | --- | --- |
| Host Tiling | 27 / 27 | PASS |
| Host InferShape | 8 / 8 | PASS |
| Kernel UT（CPU 仿真与比较器验证） | 17 / 17 | PASS |

Kernel UT 包含 14 项算子数值用例与 3 项比较器验证；其中 Python 比较器验证执行 7 项子测试，覆盖 NaN、正负无穷、数值误差、长度和进程退出码。C++ 比较器同时检查实际值与参考值为有限值。

构建及测试命令退出码为 0。结果摘录：

```text
[==========] 35 tests from 2 test suites ran. (49 ms total)
[  PASSED  ] 35 tests.
[==========] 17 tests from 1 test suite ran. (22651 ms total)
[  PASSED  ] 17 tests.
```
