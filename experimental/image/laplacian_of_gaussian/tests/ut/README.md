# LaplacianOfGaussian 原生 UT

从仓库根目录，在已激活的CANN环境内运行：

```bash
bash experimental/image/laplacian_of_gaussian/tests/run_ut.sh /absolute/private-output /absolute/private-third-party
```

LANG/LC_ALL使用C.UTF-8。现有入口用仓库原生build.sh的Host/infer/opapi目标，输出发现列表、GTest XML和日志。保留23个Host/infer、13个API测试；无禁用测试。测试使用生产注册回调及wrapper，只stub inner设备调用。LoG的UT CMake为平台预算测试链接CANN tiling_api，不需NPU。

覆盖项包括完整header/权重公式和padding、三dtype多档UB精确边界、平台256B预留及workspace、legacy八宽子段、raw写入哨兵、动态shape、属性与metadata非法输入、地址范围及双向紧邻、输出指针约定、inner状态透传。发现数不代表分支全覆盖：有限float自动半径的非finite条件、已通过FP32输出字节约束后的API周期拒绝、有效shape后的null attrs和冗余halo回退赋值各有前置约束。

隔离覆盖采集应在单独CMake构建目录启用ENABLE_COVERAGE、CMAKE_EXPORT_COMPILE_COMMANDS并构建cv_op_host_ut/cv_op_api_ut，设置BUILD_PATH指向该目录再执行。只清除该私有目录gcda。分别对ophost_cv_tiling_obj、ophost_cv_infer_obj及API cases对象执行gcov -b -c，避免将未执行的生产副本和已执行wrapper聚合到同一gcov文件。保存源码hash、生产so链接映射、编译命令、发现列表、XML和退出码。
