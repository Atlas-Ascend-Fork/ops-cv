# FloodFill 回归测试

所有命令从仓库根目录执行，先加载 CANN 的 `set_env.sh`，并配置该 CANN 版本所需的编译器、Python 和驱动库路径。测试固定构造输入，不依赖外部数据或 OpenCV。

## Host 与 kernel UT

```bash
bash build.sh -u --ophost --experimental --ops=flood_fill --soc=ascend910b
bash build.sh -u --opkernel --experimental --ops=flood_fill --soc=ascend910b
```

预期 Host UT 为 46 项，kernel UT 为 10 项。Host UT 覆盖参数、tiling 与 workspace。`tests/ut/op_kernel` 使用 tikicpulib 执行真实 `KernelFloodFill`，从现有 `FLOOD_FILL_PROFILE` 诊断记录断言关键路径；该宏只在测试翻译单元启用，不改变正式算子包。

| kernel 用例 | 必须满足的行为 |
| --- | --- |
| 65×65 梳齿，UINT8/INT32/FLOAT32 | 保持生产 span 容量 386，自然溢出并完成 BFS；含 workspace visited、输出 mask 和输入 mask |
| 512×513 宽带，4 核 | 比较平面生成、暂停/恢复、行种子实际采纳 |
| 512×513 非对称容差宽带，4 核 | 多次比较窗口生成与恢复，不使用无向行种子 |
| 512×513 梳齿，4 核，输入及输出 mask | 自然溢出，回退结果发布及完整 mask 合并 |
| 512×513 标签障碍/种子遮挡，4 核 | 不越过障碍，保留标签，正确处理空填充 |
| 恢复的 97×129 梳齿/孤岛 | 自然耗尽容量 514，验证回退补全且不误填不连通的孤岛 |

65×65 梳齿由整行和隔列像素行交替构成，全部连通，共有 1089 个最大水平区间，超过生产容量 386。kernel UT 同时断言 `overflowed=1` 和完整 BFS 输出，避免仅凭图案规模推断覆盖。大图宽度为 513，总像素为 262656，跨过 262144 像素的多核门槛，并覆盖非对齐行尾。

## ACLNN 实机 ST

先从当前源码构建并安装到独立目录。构建前清除外部 `ASCEND_CUSTOM_OPP_PATH`，安装后用同一包的头文件、库和 binary 执行测试：

```bash
FF_INSTALL="$HOME/ascend_opp/flood_fill_review"
unset ASCEND_CUSTOM_OPP_PATH
bash build.sh --pkg --experimental --ops=flood_fill --soc=ascend910b
mkdir -p "$FF_INSTALL"
bash build_out/cann-ops-cv-custom-linux.aarch64.run --install-path="$FF_INSTALL"
bash experimental/image/flood_fill/tests/st/run.sh "$FF_INSTALL/vendors/custom_cv" 0
```

预期 `SUMMARY: 197/197 PASS`、退出码 0。`run.sh` 使用 `-std=c++17 -Wall -Wextra -Werror` 编译，临时可执行文件运行后自动清理；第二个参数为设备编号，默认为 0。

ST 枚举三种 dtype、4/8 连通、`NEIGHBOR`/`FIXED_RANGE`、输出 mask 有/无；梳齿还枚举输入 mask 有/无。范围包含小图溢出、大图溢出、多核宽带、非对称容差、标签障碍及种子遮挡。输出预填哨兵，逐元素比较完整 `y`、完整 `mask_out`（含边框和标签），并回读确认 `x`/`mask_in` 未修改。参考实现在 [flood_fill_cases.h](./flood_fill_cases.h)，采用原始图像上的像素 BFS，独立于 kernel 的 span 算法。

其中 192 项为组合矩阵，另 5 项恢复自精简提交 `1a3ed752` 之前的示例：`span_overflow_u8`、`span_overflow_f32`、`span_overflow_nomask`、`overflow_fixed_island`、`overflow_fixed_island_nomask`。保留原 97×129 输入、齿距 4、种子 `[48,64]`、dtype、填充值、容差、连通性与 mask 输出选择；前三项可达像素为 7905，后两项为 7581，末尾孤岛保持原值。旧测试当时位于 ACLNN 示例中，随规模精简移出；本次恢复关键场景并补充原先没有的 kernel 分支断言。

实机 ST 检查正式包的数值与多核同步；关键分支的显式触发断言由 kernel UT 提供。测试为正确性回归，不提供最坏延迟保证；性能边界见 [算子 README](../README.md#性能边界)。
