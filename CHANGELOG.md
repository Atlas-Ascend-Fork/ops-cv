# CHANGELOG

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## v8.5.0-beta.1

发布日期：2025-12-30

ops-cv算子首个Beta版本v8.5.0-beta.1现已发布。
本版本引入了多项新增特性、问题修复及性能改进，目前仍处于测试阶段。
我们诚挚欢迎社区反馈，以进一步提升ops-cv的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/ops-cv/blob/9.3.0-beta.1/README.md)。

### 🔗 版本地址

[CANN 8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
版本目录说明如下：
├── aarch64                 # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```

### 📌 版本配套

**ops-cv子包及相关组件与CANN版本配套关系**

| CANN子包版本 | 版本源码标签   | 配套CANN版本|
|--|--|--|
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 关键特性

- 【工程能力】cv类onnx算子插件支持。([#97](https://gitcode.com/cann/ops-cv/pull/97))
- 【工程能力】增加编译选项oom、asan、mssanitizer、build-type等工程级稳定性与可调试性能力。([#62](https://gitcode.com/cann/ops-cv/pull/62)、[#70](https://gitcode.com/cann/ops-cv/pull/70)、[#80](https://gitcode.com/cann/ops-cv/pull/80))
- 【算子实现】图像采样与处理算子新增对Kirin X90处理器系列产品支持。([#105](https://gitcode.com/cann/ops-cv/pull/105)、[#125](https://gitcode.com/cann/ops-cv/pull/125))
- 【算子实现】three_interpolate_backward算子新增确定性计算特性。([#39](https://gitcode.com/cann/ops-cv/pull/39))
- 【资料优化】增加QUICK_START，离线编译模式，aicore/aicpu/graph模式下开发指南完善。([#40](https://gitcode.com/cann/ops-cv/pull/40)、[#156](https://gitcode.com/cann/ops-cv/pull/156))
- 【资料优化】优化贡献指南中新算子贡献流程。([#57](https://gitcode.com/cann/ops-cv/pull/57))
- 【性能优化】增加asc_opc算子并行编译能力，优化编译效率；增加ccache，优化编译时长。([#80](https://gitcode.com/cann/ops-cv/pull/80))

### 🐛 问题修复

- add_example样例算子执行调用问题修复。([#16](https://gitcode.com/cann/ops-cv/issues/16))
- 修复编译不存在算子时提示编译成功问题。([#25](https://gitcode.com/cann/ops-cv/issues/25))
- 使用constexpr修饰if开启编译优化。([#32](https://gitcode.com/cann/ops-cv/issues/32))
