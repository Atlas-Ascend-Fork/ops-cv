# FloodFill ACLNN 调用示例

[test_aclnn_flood_fill.cpp](./test_aclnn_flood_fill.cpp) 展示 Tensor 和属性列表创建、工作区查询、算子执行、结果校验及资源释放。支持范围和参数迁移说明见 [FloodFill README](../README.md)。

## 示例与验证范围

输入图像为 `[3,5]`，种子为 `[row,col]=[1,3]`，填充值为 200，上下容差均为 1：

```text
x = [[10,  0,  0,  0,  0],
     [ 0, 10, 11, 12,  0],
     [ 0,  0,  0,  0, 12]]
```

4 连通、无输入 mask 时，`FIXED_RANGE` 填充中间行的 11 和 12；`NEIGHBOR` 沿 12→11→10 填充中间行三个像素。8 连通还可到达对角位置：右下角的 12 在两种模式下均被填充，左上角的 10 仅在 `NEIGHBOR` 下被填充。

示例枚举 UINT8 / INT32 / FLOAT32、4/8 连通、两种模式、三种输入 mask 情况和是否绑定输出 mask，共 72 个小图调用。输入 mask 情况为不提供、在像素 `[1,2]` 设置标签 7 障碍、在种子位置设置标签 9 障碍。程序比较显式预期图像、完整输出 mask（含标签与边框），并检查输入图像及输入 mask 未修改。

示例使用独立输入/输出缓冲区；无输入或输出 mask 时，对应 ACLNN 参数传 `nullptr`。属性 `seed` 使用 `aclIntArray`，填充值和两个容差使用长度为 1 的 `aclFloatArray`。

这些用例用于演示接口与基础行为。大图、多核与溢出回退验证见 [回归测试](../tests/README.md)；本示例不包含空图、非连续输入或性能测量。

## 编译与安装

先加载当前 CANN 安装目录的 `set_env.sh`，确保 CANN 构建所需的 Python、CMake 和 g++ 已配置。以下命令从 ops-cv 仓库根目录开始；`FF_CANN` 应包含 `include/` 和 `lib64/`，驱动路径按实际安装位置调整。

```bash
FF_REPO="$PWD"
FF_CANN="${ASCEND_HOME_PATH:?请先加载 CANN set_env.sh}"
FF_INSTALL="$HOME/ascend_opp/flood_fill"
export LD_LIBRARY_PATH="$FF_CANN/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

bash build.sh --pkg --experimental --ops=flood_fill --soc=ascend910b
mkdir -p "$FF_INSTALL"
bash build_out/cann-ops-cv-custom-linux.aarch64.run --install-path="$FF_INSTALL"

FF_VENDOR="$FF_INSTALL/vendors/custom_cv"
export ASCEND_CUSTOM_OPP_PATH="$FF_VENDOR"
export LD_LIBRARY_PATH="$FF_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
```

`FF_VENDOR` 下应直接存在 `op_api/include`、`op_api/lib` 和 `op_impl`。修改 Host/kernel 后须重新打包安装；仅修改示例时重新编译示例即可。后续命令沿用同一终端的环境变量。

## 编译与运行示例

```bash
cd "$FF_REPO/experimental/image/flood_fill/examples"
g++ -O2 -std=c++17 -Wall -Wextra -Werror test_aclnn_flood_fill.cpp \
    -I "$FF_CANN/include" -I "$FF_CANN/include/aclnnop" \
    -I "$FF_VENDOR/op_api/include" \
    -L "$FF_VENDOR/op_api/lib" -L "$FF_CANN/lib64" \
    -lcust_opapi -lascendcl -lnnopbase \
    -Wl,-rpath="$FF_VENDOR/op_api/lib:$FF_CANN/lib64" \
    -o test_aclnn_flood_fill

./test_aclnn_flood_fill
```

使用设备 0，预期末尾输出 `SUMMARY: 72/72 PASS`，退出码为 0。

## Host UT

```bash
cd "$FF_REPO"
bash build.sh -u --ophost --experimental --ops=flood_fill --soc=ascend910b
```

预期 `46 tests` 通过，覆盖形状、数据类型、属性校验、输入 mask、工作区与 tiling 数据，包含极大有限整数属性及半整数舍入回归。
