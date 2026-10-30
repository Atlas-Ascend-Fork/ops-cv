# CHANGELOG

> This document records the important changes of each version. Versions are listed in reverse chronological order.

## v8.5.0-beta.1

Release date: 2025-12-30

The first Beta version v8.5.0-beta.1 of the ops-cv operators has been released.
This version introduces multiple new features, bug fixes, and performance improvements, and is still in the testing stage.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-cv.
For the usage, refer to the [official documentation](https://gitcode.com/cann/ops-cv/blob/master/README_en.md).

### 🔗 Version Address

[CANN 8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directories are described as follows:
├── aarch64                 # ARM CPU
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
├── x86_64                   # X86 CPU
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**Version compatibility between ops-cv subpackages, related components, and CANN versions**

| CANN subpackage version | Source code tag   | Compatible CANN version|
|--|--|--|
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 Key Features

- [Engineering capability] Plugin support for cv ONNX operators. ([#97](https://gitcode.com/cann/ops-cv/pull/97))
- [Engineering capability] Added compilation options such as oom, asan, mssanitizer, and build-type to provide engineering-level stability and debuggability. ([#62](https://gitcode.com/cann/ops-cv/pull/62), [#70](https://gitcode.com/cann/ops-cv/pull/70), [#80](https://gitcode.com/cann/ops-cv/pull/80))
- [Operator implementation] Image sampling and processing operators now support the Kirin X90 processor series products. ([#105](https://gitcode.com/cann/ops-cv/pull/105), [#125](https://gitcode.com/cann/ops-cv/pull/125))
- [Operator implementation] The three_interpolate_backward operator adds the deterministic computation feature. ([#39](https://gitcode.com/cann/ops-cv/pull/39))
- [Documentation optimization] Added QUICK_START and the offline compilation mode, and improved the development guides for the aicore, aicpu, and graph modes. ([#40](https://gitcode.com/cann/ops-cv/pull/40), [#156](https://gitcode.com/cann/ops-cv/pull/156))
- [Documentation optimization] Optimized the new operator contribution process in the contribution guide. ([#57](https://gitcode.com/cann/ops-cv/pull/57))
- [Performance optimization] Added the parallel operator compilation capability of asc_opc to improve compilation efficiency, and added ccache to reduce compilation time. ([#80](https://gitcode.com/cann/ops-cv/pull/80))

### 🐛 Bug Fixes

- Fixed the execution and invocation issue of the add_example sample operator. ([#16](https://gitcode.com/cann/ops-cv/issues/16))
- Fixed the issue where successful compilation was reported when a non-existent operator was compiled. ([#25](https://gitcode.com/cann/ops-cv/issues/25))
- Used constexpr to decorate if statements to enable compilation optimization. ([#32](https://gitcode.com/cann/ops-cv/issues/32))
