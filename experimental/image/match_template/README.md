# MatchTemplate

## 状态

本目录包含面向 Ascend 910B4 和 CANN 9.1.0 优化的 MatchTemplate 实现。实现内容包括：自适应输出 halo 分块、模板在 UB 中复用、按方法专门化的得分计算、面向 FP32、FP16 和 UINT8（包括多通道输入）的 Vector 路径、多核分块调度、按批次确定性的跨核极值归约，以及当模板和 halo 无法装入可用 UB 时使用的 GM 回退路径。

测试套件包含 22 个 Host UT、28 个 kernel UT、47 个 ACLNN 用例（45 个有效功能用例和 2 个非法 `method` 用例），以及 168 个 Golden 用例。覆盖 1x1、图像相等、负值 FP 数据、并列最优、近似并列、批处理、非对齐形状、GM 回退，以及 C3 专用路径的并列最优和尾行回退行为。

kernel UT 使用 AscendC CPU 调试库执行真实 `KernelMatchTemplate::Init/Process`，通过独立的双精度参考公式检查得分、最佳得分和位置。测试采用单核，覆盖三种数据类型、六种方法、halo/GM 路径、零方差、右下边界、跨 512 元素 Gather 和 UINT8 预打包；不将 CPU 调试结果作为多核硬件同步的验证证据。

kernel 入口通过 `AscendC::GetUserWorkspace` 获取用户区地址。用户 workspace 依次存放局部最佳得分、对齐后的局部索引和可选的预打包图像，不再申请未使用的同步缓冲。生产版 Vector kernel 仍使用两处硬件 `SyncAll<true>()`、`KERNEL_TYPE_MIX_AIV_1_0` 和调度模式 1，系统预留区保持不变。

历史验收中，45 次有效 kernel 启动的 msSanitizer memcheck 和 initcheck 均未报告错误；FP16 CCORR 和 UINT8 CCOEFF_NORMED C3 压力用例的完整分块 synccheck 报告 `No error detected`。这些结果属于此前版本，本次资源清理后的验证结果需与对应源码、构建和日志一起记录，不能直接复用历史 Sanitizer 或性能结果。

2026-10-20 在 CANN 9.1.0 环境重新构建后的定向验证：

| 验证层 | 结果 | 说明 |
|---|---:|---|
| Host UT | 16/16 通过 | 包含不同批次/核数的紧凑 workspace 布局检查 |
| kernel UT | 26/26 通过 | Ascend910B1 CPU 调试库，按 FP32/FP16/UINT8/边界四组执行 |
| ACLNN | 47/47 通过 | Ascend 910B4 NPU 实机，独立安装本次构建的 vendor 包 |
| C3 并列 Golden | 16/16 通过 | NPU 实机，常数与非恒定重复模板各 8 条 |

首次整套 CPU 调试运行达到 300 秒执行时限，未将超时计为通过；随后按上述四组独立运行，全部正常结束。C3 实机结果支持当前硬件 `ReduceMax` 的最小索引行为，本次未修改归约算法。

2026-10-23 复审修复：FP16 CCORR 的 C3 调度在最后不足 16 行时仍回退到通用 Vector 路径，因此为该路径保留完整的 `tileH × 对齐行宽 × sizeof(float)` 得分缓冲（16×512 时为 32 KiB）。Host 的 C3、普通 halo 和 GM 回退预算均显式包含实际得分缓冲及两个局部结果缓冲，同时保留 16 KiB 算子余量；容量不足时改选较小分块或拒绝无法容纳的配置。其他无需通用得分缓冲的 C3 路径保留 32 字节占位。

InferShape 改从输入描述读取 dtype，只依赖 shape 和描述信息。Host UT 覆盖无输入数据时的正常推导、未知维度保留、类型不匹配及非法类型拒绝；新增 UB 预算检查覆盖三种 dtype、六种 method 和三档容量。

新增两个 kernel UT 分别覆盖 497 列的单尾行，以及完整 16 行分块之后再处理一行。Golden 可通过 `--scenario c3_tail_best` 单独运行真实 512×512 FP16 CCORR 尾行用例：最佳位置唯一位于 `[496,496]`、最佳得分为 768，逐元素参考由独立 NumPy 空间公式产生。数值通过不能替代缓冲边界检查，尾行用例应同时进行内存检测。

本轮重新构建后的验证：Host UT 22/22、定向 kernel UT 4/4（两个新增尾行用例及 FP16/UINT8 C3 CCORR 保护用例）、ACLNN 47/47、C3 Golden 17/17 通过。其余 kernel UT、完整 168 条 Golden 和性能矩阵本轮未全量重跑。旧代码运行新增单尾行 UT 时，调试库在索引 8 处报告超出长度 8；修复后该用例正常完成。

插桩构建的全核 memcheck 达到 300 秒时限，结果记为 TIMEOUT。针对尾行所在 block 31 的 memcheck 正常结束，未报告越界，输出逐元素精确匹配参考结果；但工具报告 `VECTOR_MASK_1` 和 `FFTS_BASE_ADDR` 未恢复默认值的警告，因此该诊断保留为有警告结果，不记作 Sanitizer 完全无告警通过。

2026-10-27 测试夹具修正：InferShape UT 改用仓库共享的 `InfershapeContextPara` / `ExecuteTestCase`，删除上下文强制转换和原始输入槽替换。共享夹具使用仅携带元数据、无输入数据的 Tensor，因此测试不再宣称模拟“无 Tensor 对象”。本算子的 Host UT 配置检查拒绝 InferShape 源码直接调用 `GetInputTensor()`，防止重新引入已修复的依赖；该检查只在启用 Host UT 时生效，源码变更会触发重新配置。已验证临时重新引入该调用时构建按预期失败，恢复后 Host UT 22/22 通过。产品源码未修改，本轮未重跑 Kernel、NPU 和性能测试。

## 接口

```text
MatchTemplate(image, template_image, method) -> score, best_score, best_location
```

- `image`：`[N, H, W, C]`，在连续 ND 存储中逻辑布局为 NHWC，支持 `uint8`、`float16` 或 `float32`。
- `template_image`：`[1, h, w, C]`，在连续 ND 存储中逻辑布局为 NHWC，数据类型必须与 `image` 相同。
- `method`：必需的整数属性，取值范围为 `[0, 5]`。
- `score`：`[N, H-h+1, W-w+1]`，数据类型为 `float32`。
- `best_score`：`[N]`，数据类型为 `float32`。
- `best_location`：`[N, 2]`，数据类型为 `int64`，存储顺序为 `[x, y]`。

六种方法分别为 SQDIFF、SQDIFF_NORMED、CCORR、CCORR_NORMED、CCOEFF 和 CCOEFF_NORMED。

## 构建与验证

```bash
# 在 ops-cv 仓库根目录执行；先将 ASCEND_INSTALL_PATH 设置为本机 CANN 安装目录。
source "${ASCEND_INSTALL_PATH}/set_env.sh"
export OPS_CV_ROOT="$(pwd)"
export MATCH_TEMPLATE_INSTALL_ROOT="$HOME/ascend_custom/match_template"
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=match_template --vendor_name=custom_match_template -O3 -j8
./build_out/cann-ops-cv-custom_match_template-linux.aarch64.run \
  --quiet --install-path="$MATCH_TEMPLATE_INSTALL_ROOT"
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export MATCH_TEMPLATE_VENDOR_ROOT="$MATCH_TEMPLATE_INSTALL_ROOT/vendors/custom_match_template_cv"
source "$MATCH_TEMPLATE_VENDOR_ROOT/bin/set_env.bash"

bash build.sh --ophost_test --opkernel_test --experimental --ops=match_template \
  --soc=ascend910b --noexec -O0 -j8
export BUILD_PATH="$OPS_CV_ROOT/build"
./build/tests/ut/op_host/cv_op_host_ut --gtest_filter=MatchTemplateTiling.*:MatchTemplateInferShape.*
./build/tests/ut/op_kernel/cv_op_kernel_ut_ascend910B1 --gtest_filter='MatchTemplateKernel*'
```

kernel CPU 调试比 NPU 执行慢，可以按 `MatchTemplateKernelTest/0.*`、`MatchTemplateKernelTest/1.*`、`MatchTemplateKernelTest/2.*` 和 `MatchTemplateKernelBoundaryTest.*` 分组运行。Host UT 还会验证普通路径、预打包路径、不同批次和核数下 workspace 的对齐与精确增长，防止删除缓冲后仍保留无用空间。

### CMake 示例与 CTest

示例默认关闭，因此不会改变软件包构建结果。需要显式启用示例，并让 CMake 指向已安装的 vendor 软件包。由于仓库根目录未启用 CTest，CTest 入口位于该模块目录下方。

```bash
cmake -S "$OPS_CV_ROOT" -B "$OPS_CV_ROOT/build" \
  -DENABLE_TEST=TRUE -DMATCH_TEMPLATE_BUILD_EXAMPLES=ON -DBUILD_TESTING=ON \
  -DMATCH_TEMPLATE_CANN_ROOT="$ASCEND_HOME_PATH" \
  -DMATCH_TEMPLATE_VENDOR_ROOT="$MATCH_TEMPLATE_VENDOR_ROOT"
cmake --build "$OPS_CV_ROOT/build" --target match_template_examples -j4
ctest --test-dir "$OPS_CV_ROOT/build/experimental/image/match_template/examples" \
  --output-on-failure
```

### OpenCV Golden 与性能报告

请在验证环境中安装 `opencv-python-headless==4.12.0.88`。Golden 运行器会写入原始二进制输入和输出、`results.csv` 以及 `summary.json`；使用 `--reuse-actual` 可以基于已有的原始运行结果重新生成报告，无需重新运行 NPU。

```bash
python experimental/image/match_template/tests/golden/run_match_template_golden.py \
  --runner build/experimental/image/match_template/examples/match_template_aclnn_case_runner \
  --output artifacts/match_template_golden

python experimental/image/match_template/tests/performance/summarize_match_template_performance.py \
  --input artifacts/match_template_performance
python experimental/image/match_template/tests/performance/summarize_match_template_profiles.py \
  --input artifacts/match_template_profiles
```

新增的 16 个 C3 Golden 用例使用 `512×512×3` 图像、`16×16` 模板，覆盖 FP16/UINT8 与方法 2–5，确保在目标 UB 容量允许时进入 C3 专用路径：

- `--scenario c3_constant_ties`：8 个全常数用例，所有输出位置并列，期望 `[x,y]=[0,0]`。
- `--scenario c3_pattern_ties`：8 个非恒定重复模板用例，并列位置跨向量段、256 列和 tile 行，期望最小索引位置 `[17,5]`。

新增用例采用独立 NumPy FP64 空间公式生成参考结果，避免 FFT 舍入误差改变精确并列关系。校验同时要求 `best_location` 与参考结果一致，并且等于实际 `score` 中按行优先顺序出现的第一个极值位置；仅最大值相同而位置错误仍判失败。硬件归约的并列行为需要通过 NPU 执行验证，不能只靠 CPU 参考生成器通过来证明。

`run_profile_case.sh` 是 `msopprof` 的应用包装脚本；在通过 `msopprof --application` 传入该脚本前，请设置 `MATCH_TEMPLATE_BENCHMARK_BIN` 和 `MATCH_TEMPLATE_PROFILE_CASE`。

该 kernel 不依赖单独的 IntegralImage 算子。

历史 v19 性能矩阵包含 67 个用例，覆盖小型和大型图像、720p/1080p/4K、C1/C3、非对齐形状、大模板以及批处理场景。FP32 通过 18/18，FP16 通过 22/22，UINT8 通过 27/27；在 Ascend 910B4 上，所有用例均同时满足 Host 和 Device 的 `torch/custom >= 1.05x`。Device 最低加速比为 1.145x，Host 最低加速比为 1.071x。本次资源清理没有重新测量该性能矩阵，不将空间节约直接解释为性能提升。

生产实现仅保留经过验证的 Vector halo 分块路径。未使用的 Cube 和 Matmul 原型，以及与其对应的非活动 dispatch/tiling 数据，均已移除。
