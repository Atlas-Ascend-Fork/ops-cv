/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_to_absolute_b_box_tiling.cpp
 * \brief ToAbsoluteBBox arch35 host Tiling UT。
 *
 * 覆盖点：
 *   - ValidateInputShapes：rank=3 / shape_hw=(4,) / 坐标维==4（false: shape[2]；true: shape[1]）；
 *   - reversed_box 属性分支：num 提取（false: shape[1]；true: shape[2]）与 TilingKey 选择（0/1）；
 *   - blockFormer 512 元素对齐 + 每核 >= 4KB 多核切分逻辑；
 *   - ubFormer 256B 对齐 UB 分块（fp16=128 元素 / fp32=64 元素对齐）；
 *   - 首/尾 block 循环次数与尾块大小（含单 block、多 block、非对齐尾块、小 UB 多次循环）；
 *   - workspace slot（0）声明；空 tensor / 非法 dtype 拒绝。
 *
 * 说明：核数由 tiling 运行时 GetCoreNumAiv 决定，故不硬编码 coreNum/blockFormer 绝对值，
 * 而是以 tiling 代码承诺的数学关系做字段级断言（对齐 / CeilDiv / 首尾块重建 dim0）。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/to_absolute_b_box_tiling_data.h"

namespace ToAbsoluteBBoxUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "ToAbsoluteBBox";

// 切分常量（须与 op_host/arch35/to_absolute_b_box_tiling_arch35.cpp 保持一致）
static constexpr int64_t ELEM_ALIGN_FACTOR = 512;   // blockFormer 元素对齐因子
static constexpr int64_t FP32_ALIGN_ELEM = 256 / 4; // ubFormer 对齐（256B → 64 fp32 元素）
static constexpr int64_t FP16_ALIGN_ELEM = 256 / 2; // ubFormer 对齐（256B → 128 fp16 元素）
static constexpr uint64_t KEY_REVERSED_OFF = 0UL;   // reversed_box=false → TilingKey 0
static constexpr uint64_t KEY_REVERSED_ON = 1UL;    // reversed_box=true → TilingKey 1

static inline int64_t CeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

// 与 tiling 侧 TilingParse 的 CompileInfo 类型对应（空结构体即可，Build 需非空指针）
struct ToAbsoluteBBoxCompileInfo {
} g_compileInfo;

// 构造 normalized_boxes + shape_hw 输入 + reversed_box 属性的 tiling 上下文
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

static gert::TilingContextPara MakePara(const std::vector<int64_t>& boxShape, bool reversedBox, uint64_t coreNum = 64,
                                        uint64_t ubSize = 262144, ge::DataType dtype = ge::DT_FLOAT,
                                        const std::vector<int64_t>& hwShape = {
                                            4,
                                        })
{
    const gert::StorageShape boxS = MakeStorageShape(boxShape);
    const gert::StorageShape hwS = MakeStorageShape(hwShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs(
        {{boxS, dtype, ge::FORMAT_ND}, {hwS, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::TensorDescription> outputs({{boxS, dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(reversedBox)}});
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, &g_compileInfo, "Ascend950", coreNum, ubSize, 4096);
}

// 运行 tiling 并把 raw tiling data 还原为结构体
static const ToAbsoluteBBoxTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(ToAbsoluteBBoxTilingData));
    return reinterpret_cast<const ToAbsoluteBBoxTilingData*>(info.tilingData.get());
}

// 对成功路径统一校验首/尾块计算的数学关系
static void CheckTilingInvariants(const ToAbsoluteBBoxTilingData* t, int64_t expectDim0)
{
    ASSERT_NE(t, nullptr);
    // dim0
    EXPECT_EQ(t->dim0, expectDim0);
    // 核数 >= 1
    EXPECT_GE(t->coreNum, 1);
    // blockFormer 512 元素对齐且 > 0
    EXPECT_GT(t->blockFormer, 0);
    EXPECT_EQ(t->blockFormer % ELEM_ALIGN_FACTOR, 0);
    // blockNum = CeilDiv(dim0, blockFormer)
    EXPECT_EQ(t->blockNum, CeilDiv(expectDim0, t->blockFormer));
    // 首 block：循环次数与尾块大小
    EXPECT_EQ(t->ubLoopOfFormerBlock, CeilDiv(t->blockFormer, t->ubFormer));
    int64_t expectUbTailFormer = t->blockFormer - (t->ubLoopOfFormerBlock - 1) * t->ubFormer;
    EXPECT_EQ(t->ubTailOfFormerBlock, expectUbTailFormer);
    EXPECT_GT(t->ubTailOfFormerBlock, 0);
    EXPECT_LE(t->ubTailOfFormerBlock, t->ubFormer);
    // 尾 block：blockTail = dim0 - (blockNum-1)*blockFormer
    int64_t blockTail = expectDim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_GT(blockTail, 0);
    EXPECT_LE(blockTail, t->blockFormer);
    EXPECT_EQ(t->ubLoopOfTailBlock, CeilDiv(blockTail, t->ubFormer));
    int64_t expectUbTailTail = blockTail - (t->ubLoopOfTailBlock - 1) * t->ubFormer;
    EXPECT_EQ(t->ubTailOfTailBlock, expectUbTailTail);
    EXPECT_GT(t->ubTailOfTailBlock, 0);
    EXPECT_LE(t->ubTailOfTailBlock, t->ubFormer);
    // 用首/尾块重建 dim0，验证切分自洽
    int64_t rebuilt = (t->blockNum - 1) * t->blockFormer + blockTail;
    EXPECT_EQ(rebuilt, expectDim0);
}

class ToAbsoluteBBoxTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ToAbsoluteBBoxTiling SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "ToAbsoluteBBoxTiling TearDown." << std::endl; }
};

// case 1：fp32 单 block 小 shape（reversed_box=false），验证 num 提取 + 基础切分 + TilingKey=0
TEST_F(ToAbsoluteBBoxTiling, tiling_fp32_single_block)
{
    auto para = MakePara({2, 8, 4}, false); // dim0 = 64，num = shape[1] = 8
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_OFF));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 64);
    EXPECT_EQ(t->num, 8);      // false → num = shape[1]
    EXPECT_EQ(t->blockNum, 1); // 数据量很小 → 单 block（首块与尾块为同一 block）
    // blockDim（框架 blockNum）应等于 tiling 结构体 blockNum
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), t->blockNum);
    // 尾块（唯一实际 block）的元素数应等于 dim0
    EXPECT_EQ(t->ubTailOfTailBlock, 64);
}

// case 2：fp16 单 block（reversed_box=false），验证 fp16 UB 对齐（128 元素）与 bufferDivisor=10
TEST_F(ToAbsoluteBBoxTiling, tiling_fp16_single_block)
{
    auto para = MakePara({1, 16, 4}, false, 64, 262144, ge::DT_FLOAT16); // dim0 = 64，num = 16
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_OFF));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 64);
    EXPECT_EQ(t->num, 16);
    // fp16：bufferDivisor = 5*2 = 10 → maxElem = (262144-64)/10 = 26208 → FloorAlign(26208,128) = 26112
    EXPECT_EQ(t->ubFormer, 26112);
    EXPECT_EQ(t->ubFormer % FP16_ALIGN_ELEM, 0);
}

// case 3：reversed_box=true 常规 shape (2,4,8)，验证 num = shape[2] 与 TilingKey=1
TEST_F(ToAbsoluteBBoxTiling, tiling_reversed_box_layout)
{
    auto para = MakePara({2, 4, 8}, true); // dim0 = 64，坐标维 = shape[1] = 4，num = shape[2] = 8
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_ON));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 64);
    EXPECT_EQ(t->num, 8); // true → num = shape[2]
    EXPECT_EQ(t->blockNum, 1);
}

// case 4：reversed_box=true 边界 shape (1,4,1)（num=1 下边界），验证最小坐标布局 tiling 支持
TEST_F(ToAbsoluteBBoxTiling, tiling_reversed_single_element)
{
    auto para = MakePara({1, 4, 1}, true); // dim0 = 4，坐标维 = shape[1] = 4，num = shape[2] = 1
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_ON));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 4);
    EXPECT_EQ(t->num, 1);
    EXPECT_EQ(t->blockNum, 1);
    EXPECT_EQ(t->ubTailOfTailBlock, 4);
}

// case 5：大 shape 多核多 block，验证多核切分与首尾块一致（对齐满块）
TEST_F(ToAbsoluteBBoxTiling, tiling_multi_block_large_shape)
{
    // dim0 = 1024*4096*4 = 16777216，跨多核；每核 >=4KB 切分生效
    auto para = MakePara({1024, 4096, 4}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_OFF));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 16777216);
    EXPECT_EQ(t->num, 4096);
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), t->blockNum);
    // fp32：bufferDivisor = 5*4 = 20 → maxElem = (262144-64)/20 = 13104 → FloorAlign(13104,64) = 13056
    EXPECT_EQ(t->ubFormer, 13056);
    EXPECT_EQ(t->ubFormer % FP32_ALIGN_ELEM, 0);
    // 满对齐场景尾块为完整 block
    int64_t blockTail = t->dim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_EQ(blockTail, t->blockFormer);
}

// case 6：小 UB 强制多次 UB 循环，验证 UB 切分与首/尾 block 精确值
TEST_F(ToAbsoluteBBoxTiling, tiling_ub_multi_loop_small_ub)
{
    // dim0 = 2048；ubSize=8192 → maxElem = (8192-64)/20 = 406 → ubFormer = FloorAlign(406,64) = 384
    // 理论核数 = CeilDiv(2048*32, 32768) = 2 → blockFormer = CeilAlign(1024,512) = 1024 → blockNum = 2
    auto para = MakePara({2, 256, 4}, false, 8, 8192);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_OFF));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 2048);
    EXPECT_EQ(t->ubFormer, 384);
    EXPECT_GT(t->ubLoopOfFormerBlock, 1);   // 需要多次 UB 循环
    EXPECT_EQ(t->ubLoopOfFormerBlock, 3);   // CeilDiv(1024, 384)
    EXPECT_EQ(t->ubTailOfFormerBlock, 256); // 1024 - 2*384
    EXPECT_EQ(t->ubLoopOfTailBlock, 3);
    EXPECT_EQ(t->ubTailOfTailBlock, 256);
}

// case 7：非 512 对齐 dim0 → 尾块非满，验证尾块尺寸计算（reversed_box=true 下的非对齐切分）
TEST_F(ToAbsoluteBBoxTiling, tiling_unaligned_tail_reversed)
{
    // (33333,4,1)：坐标维 = shape[1] = 4，num = shape[2] = 1，dim0 = 133332（非 512 对齐）
    auto para = MakePara({33333, 4, 1}, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_REVERSED_ON));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 133332);
    EXPECT_EQ(t->num, 1);
    // 尾块严格小于满 block（非对齐场景）
    int64_t blockTail = t->dim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_LT(blockTail, t->blockFormer);
}

// case 8：workspace slot 声明为 0（即使无 workspace 也须声明 1 个 slot，值为 0）
TEST_F(ToAbsoluteBBoxTiling, tiling_workspace_slot)
{
    auto para = MakePara({2, 8, 4}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0u);
}

// ========== 非法输入拒绝路径 ==========

// case 9：rank != 3 拒绝
TEST_F(ToAbsoluteBBoxTiling, tiling_reject_invalid_rank)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 8}, false), info));
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 4, 8, 1}, false), info));
}

// case 10：shape_hw != (4,) 拒绝
TEST_F(ToAbsoluteBBoxTiling, tiling_reject_invalid_shape_hw)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 8, 4}, false, 64, 262144, ge::DT_FLOAT, {5}), info));
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 8, 4}, false, 64, 262144, ge::DT_FLOAT, {4, 1}), info));
}

// case 11：坐标维 != 4 拒绝（false: shape[2]；true: shape[1]）
TEST_F(ToAbsoluteBBoxTiling, tiling_reject_invalid_coord_axis)
{
    TilingInfo info;
    // reversed_box=false → 坐标维取 shape[2]=1
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 8, 1}, false), info));
    // reversed_box=true → 坐标维取 shape[1]=1
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 1, 8}, true), info));
}

// case 12：空 tensor（dim0=0）拒绝
TEST_F(ToAbsoluteBBoxTiling, tiling_reject_empty_tensor)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara({0, 8, 4}, false), info));
}

// case 13：非法 dtype（normalized_boxes 非 fp16/fp32）拒绝
TEST_F(ToAbsoluteBBoxTiling, tiling_reject_invalid_dtype)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara({2, 8, 4}, false, 64, 262144, ge::DT_INT32), info));
}

} // namespace ToAbsoluteBBoxUT
