# AddExamplePypto

`examples/add_example` 的 PyPTO 版本：功能同为逐元素加法，区别在于 kernel 由 PyPTO DSL
(`op_kernel/add_example_pypto.py`) 编写，而不是 AscendC C++。本算子用于演示 ops-cv 中 PyPTO 算子的接入方式。

## 编译

```bash
source <ascend-toolkit>/set_env.sh
bash build.sh --pkg --ops=add_example_pypto --soc=ascend950
```

## 接入方式

在算子的 `op_host/CMakeLists.txt` 中，于 `add_modules_sources`（或 `add_all_modules_sources`）**之前**加两行：

```cmake
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

add_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
```

- `require_pypto_pro(<op_name>)`：环境中没有 `pypto_pro` 时告警并跳过本算子，不会让整个 configure 失败。
- `enable_pypto_kernel(<op_file>)`：`<op_file>` 需与 `op_kernel/<op_file>.py`、以及算子定义里
  `ExtendCfgInfo("opFile.value", ...)` 三者保持一致。

## 构建流程

`enable_pypto_kernel` 在 **configure 阶段**调用 `scripts/util/pypto_codegen.py`，对 kernel 的 `.py` 做 codegen：

| 产物 | 用途 |
| --- | --- |
| `<Tiling>_tiling.h` | tiling 结构体，host 侧 tiling 直接使用 |
| `<TilingKey>_tilingkey.h` | `ASCENDC_TPL` tiling key 声明 |
| `<op_file>_pypto_infer.cpp` | 供 `.i` infer 使用的自包含源码（build 阶段产出，configure 阶段不生成） |

tiling 的 codegen 目录会以 `target_include_directories` 注入 tiling 目标的 include 路径（CMake 3.16 兼容），
`*_tiling*.cpp` 需**显式 include** 生成的头文件（顺序：tilingkey 在前，tiling 在后）：

```cpp
// 显式 include configure 阶段生成的头文件
#include "AddExamplePyptoTilingKey_tilingkey.h"
#include "AddExamplePyptoTiling_tiling.h"
// AddExamplePyptoTiling 来自 op_kernel/add_example_pypto.py
AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
```

host 与 kernel 因此共享同一份 tiling 布局与 tiling key 位分配，无需手写副本。

**build 阶段**，本算子生成的 wrapper (`<build>/tbe/dynamic/<op_file>.py`) 会调用 `pypto_compile_op`
而不是 `compile_op`，由它完成按 tiling key 的 codegen 并复用 AscendC 后端编出 kernel 二进制。

## 芯片与 arch

PyPTO 的 codegen arch 由平台自动推导，pypto-pro 仅支持 `dav-3510` arch（canonical token 为
`__NPU_ARCH__` 版本号 `3510`）。本算子只注册了 `ascend950`，与当前 toolkit 自动推导的 arch 一致。
编译 kernel 二进制时，cmake 会按所编 soc 导出 `PYPTOPRO_JIT_ARCH`（`ascend950` → `3510`），
将 codegen arch 显式固定为目标芯片。

## 调用说明

AddExamplePypto 的 aclnn 接口由算子定义自动生成，**无需在 `op_host/op_api` 目录下手写源码**；生成的头文件为
`aclnn_add_example_pypto.h`，两段式接口为 `aclnnAddExamplePyptoGetWorkspaceSize` 与 `aclnnAddExamplePypto`。

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_add_example_pypto.cpp">test_aclnn_add_example_pypto</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>

## 运行验证

测试命令调用方式：[build.sh](../../docs/zh/invocation/quick_op_invocation.md)

<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_add_example_pypto.cpp"> test_aclnn_add_example_pypto.cpp</a></td><td>通过aclnn调用的方式调用AddExamplePypto算子。</td>
    </tr>
</table>

编译、安装并运行该样例：

```bash
# 编译（仅支持ascend950）
bash build.sh --pkg --soc=ascend950 --ops=add_example_pypto -j16

# 安装
./build_out/cann-ops-cv-*linux*.run

# 运行样例（运行方式与标准算子一致）
bash build.sh --run_example add_example_pypto eager cust --vendor_name=custom --soc=ascend950
```

预期输出：`first input` 为 1、`second input` 为 2，则 `result` 为 3，表明算子已成功部署并正确执行。
