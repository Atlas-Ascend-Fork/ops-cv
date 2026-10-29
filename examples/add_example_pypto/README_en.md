# AddExamplePypto

English | [简体中文](./README.md)

PyPTO version of `examples/add_example`: the function is also element-wise addition, but the kernel is written in the PyPTO DSL
(`op_kernel/add_example_pypto.py`) instead of AscendC C++. This operator demonstrates how to integrate a PyPTO operator into ops-cv.

## Compilation

```bash
source <ascend-toolkit>/set_env.sh
bash build.sh --pkg --ops=add_example_pypto --soc=ascend950
```

## Integration Method

In `op_host/CMakeLists.txt` of the operator, add the following two lines **before** `add_modules_sources` (or `add_all_modules_sources`):

```cmake
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

add_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
```

- `require_pypto_pro(<op_name>)`: prints a warning and skips this operator when `pypto_pro` does not exist in the environment, without failing the entire configure.
- `enable_pypto_kernel(<op_file>)`: `<op_file>` must be consistent with `op_kernel/<op_file>.py` and
  `ExtendCfgInfo("opFile.value", ...)` in the operator definition.

## Build Process

`enable_pypto_kernel` calls `scripts/util/pypto_codegen.py` in the **configure phase** to perform codegen on the `.py` file of the kernel:

| Output | Usage |
| --- | --- |
| `<Tiling>_tiling.h` | Tiling struct, directly used by host-side tiling |
| `<TilingKey>_tilingkey.h` | `ASCENDC_TPL` tiling key declaration |
| `<op_file>_pypto_infer.cpp` | Self-contained source code used by `.i` infer |

The preceding two header files are **force-included** by `-include` into the `*_tiling*.cpp` files of this operator (in the order: tilingkey first, then tiling).
Therefore, host-side tiling does not need explicit includes and can directly use the tiling struct defined in the `.py` file:

```cpp
// AddExamplePyptoTiling comes from op_kernel/add_example_pypto.py and is visible through force-include
AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
```

The host and the kernel therefore share the same tiling layout and tiling key bit allocation, so no handwritten copy is required.

In the **build phase**, the wrapper (`<build>/tbe/dynamic/<op_file>.py`) generated for this operator calls `pypto_compile_op`
instead of `compile_op`. It performs codegen based on the tiling key and reuses the AscendC backend to compile the kernel binary.

## Chip and arch

The codegen arch of PyPTO is automatically derived by the platform (`a2`/`a3`/`a5`). This operator registers only `ascend950`(a5), which is consistent with the arch
automatically derived by the current toolkit. When compiling the kernel binary, cmake exports `PYPTO_JIT_ARCH` based on the soc being compiled
(`ascend910b`/`ascend910_93` → `a3`, `ascend950` → `a5`). This prevents `pypto_compile_op` from falling back to the default `a5`
and generating instructions not supported by the target chip.
