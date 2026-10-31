# IntegralImage Host UT 测试报告

## 测试环境

- CANN：8.5.0 beta.1。
- 测试平台配置：Ascend910B。
- 更新日期：2026-10-29。

## 测试指令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -u --ophost --experimental --ops=integral_image -j16
```

## 覆盖内容

- UINT8、FLOAT16、FLOAT32 输入及 `sdepth` 对应的模板选择与输出类型校验。
- 二维输入和三维 HWC 输入，包含 C=9、C=33、C=100。
- two-phase 与 legacy 分块策略、非对齐尾块和共享缓冲容量约束。
- InferShape：H、W 各加 1，HWC 通道维保持不变，动态维度保持未知。
- 输出 Rank、空间尺寸、通道数及 StorageShape 校验。
- 可选输出缺省、空 Tensor 占位及不同输出组合。
- UB 容量边界、非法维度、属性取值及列块数量约束。

## 测试结果

| 测试套件 | 通过数 / 总数 | 结果 |
| --- | --- | --- |
| IntegralImageTiling | 27 / 27 | PASS |
| IntegralImageInfershape | 8 / 8 | PASS |
| 合计 | 35 / 35 | PASS |

构建及测试命令退出码为 0。结果摘录：

```text
[==========] Running 35 tests from 2 test suites.
[==========] 35 tests from 2 test suites ran. (49 ms total)
[  PASSED  ] 35 tests.
```
