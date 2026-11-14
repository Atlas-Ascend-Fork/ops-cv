/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_data.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_key.h"

using namespace std;
using namespace ge;

constexpr int32_t kInt32Max = 2147483647;

struct CropAndResizeUtCompileInfo {};

class CropAndResizeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CropAndResizeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CropAndResizeTiling TearDown" << std::endl; }
};

static gert::TilingContextPara::TensorDescription MkTd(std::initializer_list<int64_t> shape, ge::DataType dtype,
                                                       bool isConst = false, const void* constValue = nullptr,
                                                       ge::Format format = ge::FORMAT_ND)
{
    return gert::TilingContextPara::TensorDescription(gert::StorageShape(shape, shape), dtype, format, isConst,
                                                      const_cast<void*>(constValue));
}

// 构建标准 TilingContextPara。输入顺序：x(0), boxes(1), box_index(2), crop_size(3，值依赖)。
static gert::TilingContextPara BuildPara(std::initializer_list<int64_t> xShape,
                                         std::initializer_list<int64_t> boxesShape,
                                         std::initializer_list<int64_t> boxIndexShape,
                                         std::initializer_list<int64_t> cropSizeShape, const int32_t* cropSizeData,
                                         ge::DataType xDtype, ge::DataType boxesDtype, ge::DataType yDtype,
                                         float extrapolationValue = 0.0f, uint64_t coreNum = 64,
                                         uint64_t ubSize = 262144, ge::Format xFormat = ge::FORMAT_ND,
                                         const std::string& method = "bilinear")
{
    static CropAndResizeUtCompileInfo compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MkTd(xShape, xDtype, false, nullptr, xFormat),                                   // x
        MkTd(boxesShape, boxesDtype),                                                    // boxes
        MkTd(boxIndexShape, ge::DT_INT32),                                               // box_index
        MkTd(cropSizeShape, ge::DT_INT32, true, static_cast<const void*>(cropSizeData)), // crop_size (value dependency)
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MkTd({1, 1, 1, 1}, yDtype), // y (shape not used in tiling)
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("extrapolation_value",
                                        Ops::Cv::AnyValue::CreateFrom<float>(extrapolationValue)),
        gert::TilingContextPara::OpAttr("method", Ops::Cv::AnyValue::CreateFrom<std::string>(method)),
    };
    return gert::TilingContextPara("CropAndResize", inputs, outputs, attrs, &compileInfo, "Ascend950", coreNum, ubSize,
                                   4096);
}

// 正例公共校验：x=[2,4,4,256], boxes=[64,4], crop_size={8,8}
//   P0 warp 级并行语义：totalPositions = 64*8*8 = 4096（每 warp 1 个输出位置）
//   推导：coreNum=64 -> perCoreWarps = CeilDiv(4096,64) = 64（>= WARP_PER_CORE_MIN=32，不触发下限钳）
//         -> needCore = CeilDiv(4096,64) = 64
//   tilingKey = CROP_AND_RESIZE_MODE_BILINEAR_NHWC(0)
static void CheckPositiveTiling(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    // tp=4096 -> perCoreWarps=CeilDiv(4096,64)=64 -> needCore=CeilDiv(4096,64)=64
    EXPECT_EQ(info.blockNum, 64u);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_GE(info.workspaceSizes[0], 0);
    ASSERT_GE(info.tilingDataSize, sizeof(CropAndResizeTilingData));
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->totalPositions, 4096);
    EXPECT_EQ(td->batch, 2);
    EXPECT_EQ(td->imageHeight, 4);
    EXPECT_EQ(td->imageWidth, 4);
    EXPECT_EQ(td->depth, 256);
    EXPECT_EQ(td->cropHeight, 8);
    EXPECT_EQ(td->cropWidth, 8);
    EXPECT_EQ(td->numBoxes, 64);
    EXPECT_FLOAT_EQ(td->extrapolationValue, 0.0f);
}

static void CheckTilingFields(const TilingInfo& info, int64_t totalPositions, int64_t batch, int64_t imageHeight,
                              int64_t imageWidth, int64_t depth, int64_t cropHeight, int64_t cropWidth,
                              int64_t numBoxes, size_t blockNum)
{
    ASSERT_GE(info.tilingDataSize, sizeof(CropAndResizeTilingData));
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->totalPositions, totalPositions);
    EXPECT_EQ(td->batch, batch);
    EXPECT_EQ(td->imageHeight, imageHeight);
    EXPECT_EQ(td->imageWidth, imageWidth);
    EXPECT_EQ(td->depth, depth);
    EXPECT_EQ(td->cropHeight, cropHeight);
    EXPECT_EQ(td->cropWidth, cropWidth);
    EXPECT_EQ(td->numBoxes, numBoxes);
    EXPECT_EQ(info.blockNum, blockNum);
}

// ===== 正例：4 种 dtype 组合 =====
// #0: x=FP32, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp32_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #1: x=FP16, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp16_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #2: x=FP16, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp16_fp16)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #3: x=FP32, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp32_fp16)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// ===== 负例：约束 1 x 必须 4D =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_x_not_4d)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：num_boxes=50 不再被下限 50 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_50)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {50, 4}, {50}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 3200, 2, 4, 4, 256, 8, 8, 50, 64u);
}

// ===== 正例：C=128 不再被下限 256 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_128)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 128}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 4, 4, 128, 8, 8, 64, 64u);
}

// ===== 正例：crop_h=17 不再被 <=16 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_17)
{
    int32_t cropSize[2] = {17, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 8704, 2, 4, 4, 256, 17, 8, 64, 64u);
}

// ===== 正例：FP16 H*W=65536 不再被 65530 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_hw_65536)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 256, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 256, 256, 256, 8, 8, 64, 64u);
}

// ===== 正例：FP32 H*W=40000 不再被 32765 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_float32_hw_40000)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 200, 200, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 200, 200, 256, 8, 8, 64, 64u);
}

// ===== 负例：double 仍由 AiCpu 承接，AiCore tiling 拒绝 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_unsupported_dtype)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：op_proto 除 double 外全部 x dtype（boxes f32/f16 各一组） =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_all_dtypes)
{
    int32_t cropSize[2] = {8, 8};
    const std::vector<std::pair<ge::DataType, ge::DataType>> combos = {
        {ge::DT_UINT8, ge::DT_FLOAT},    {ge::DT_UINT8, ge::DT_FLOAT16}, {ge::DT_UINT16, ge::DT_FLOAT},
        {ge::DT_UINT16, ge::DT_FLOAT16}, {ge::DT_INT8, ge::DT_FLOAT},    {ge::DT_INT8, ge::DT_FLOAT16},
        {ge::DT_INT16, ge::DT_FLOAT},    {ge::DT_INT16, ge::DT_FLOAT16}, {ge::DT_INT32, ge::DT_FLOAT},
        {ge::DT_INT32, ge::DT_FLOAT16},  {ge::DT_INT64, ge::DT_FLOAT},   {ge::DT_INT64, ge::DT_FLOAT16},
    };
    for (const auto& [xDtype, boxesDtype] : combos) {
        auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, xDtype, boxesDtype, boxesDtype);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info))
            << "xDtype=" << static_cast<int>(xDtype) << " boxesDtype=" << static_cast<int>(boxesDtype);
        CheckPositiveTiling(info);
    }
}

// ===== 正例：nearest NHWC/NCHW TilingKey =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nearest_nhwc)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f,
                          64, 262144, ge::FORMAT_ND, "nearest");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_NEAREST_NHWC);
    CheckTilingFields(info, 4096, 2, 4, 4, 256, 8, 8, 64, 64u);
}

TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nearest_nchw)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f,
                          64, 262144, ge::FORMAT_NCHW, "nearest");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_NEAREST_NCHW);
    CheckTilingFields(info, 4096, 2, 4, 4, 256, 8, 8, 64, 64u);
}

// ===== 负例：bilinear_v2 继续走 AiCpu，AiCore tiling 拒绝 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_bilinear_v2_rejected)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f,
                          64, 262144, ge::FORMAT_ND, "bilinear_v2");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 10 boxes.shape[0] == box_index.shape[0] =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_boxes_ne_box_index)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {63}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 11 boxes.shape[1] == 4 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_boxes_dim1_ne_4)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 3}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 12 crop_size.shape == (2,) =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_size_shape_ne_2)
{
    int32_t cropSize[3] = {8, 8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {3}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：crop_h=1，验证中心点坐标路径（SE §1.4 crop_height=1 → in_y=0.5*(y1+y2)*(H-1)）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_eq_1)
{
    int32_t cropSize[2] = {1, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->cropHeight, 1);
    EXPECT_EQ(td->cropWidth, 8);
    EXPECT_EQ(td->totalPositions, 512); // 64*1*8
}

// ===== 正例：crop_w=1，验证中心点坐标路径 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_eq_1)
{
    int32_t cropSize[2] = {8, 1};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->cropHeight, 8);
    EXPECT_EQ(td->cropWidth, 1);
    EXPECT_EQ(td->totalPositions, 512); // 64*8*1
}

// ===== 正例：extrapolation_value 非默认值（SE §1.4 超出边界 → 输出 extrapolation_value）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_extrapolation_nonzero)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.5f);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->extrapolationValue, 1.5f);
}

// ===== 正例：num_boxes=4001 不再被上限 4000 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_4001)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {4001, 4}, {4001}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 256064, 2, 4, 4, 256, 8, 8, 4001, 64u);
}

// ===== 正例：C=2049 不再被上限 2048 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_2049)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 2049}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 4, 4, 2049, 8, 8, 64, 64u);
}

// ===== 正例：crop_w=17 不再被 <=16 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_17)
{
    int32_t cropSize[2] = {8, 17};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 8704, 2, 4, 4, 256, 8, 17, 64, 64u);
}

// ===== 负例：约束 4 前置 crop_h <= 0（SE §1.4/§5.4 边界条件）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_le_0)
{
    int32_t cropSize[2] = {0, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 4 前置 crop_w <= 0（SE §1.4/§5.4 边界条件）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_le_0)
{
    int32_t cropSize[2] = {8, 0};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：NCHW x=(N,C,H,W)=(2,256,4,4)，dims 按正确 dim 解析 + NCHW TilingKey；若误按 NHWC 解析则 C=4 违反
// [256,2048] 直接失败 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nchw_fp16_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_NCHW);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NCHW);
    // tp=64*8*8=4096 -> perCoreWarps=CeilDiv(4096,64)=64 -> needCore=CeilDiv(4096,64)=64（核数与 layout 无关）
    EXPECT_EQ(info.blockNum, 64u);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->depth, 256);     // C 来自 x.shape[1]（NCHW dims[1]）
    EXPECT_EQ(td->imageHeight, 4); // H 来自 x.shape[2]
    EXPECT_EQ(td->imageWidth, 4);  // W 来自 x.shape[3]
    EXPECT_EQ(td->batch, 2);
    EXPECT_EQ(td->totalPositions, 4096); // 64*8*8，与 layout 无关
}

// ===== 正例：NCHW C=255 不再被下限 256 拦截 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nchw_depth_255)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 255, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_NCHW);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NCHW);
    CheckTilingFields(info, 4096, 2, 4, 4, 255, 8, 8, 64, 64u);
}

// ===== 正例：空 boxes，启动 1 个 block，kernel 入口直接返回 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_empty_boxes)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {0, 4}, {0}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 0, 2, 4, 4, 256, 8, 8, 0, 1u);
}

// ===== 正例：num_boxes=1 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_1)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {1, 4}, {1}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 64, 2, 4, 4, 256, 8, 8, 1, 2u);
}

// ===== 正例：depth=1 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_1)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 1}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 4, 4, 1, 8, 8, 64, 64u);
}

// ===== 正例：crop=32x32 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_32)
{
    int32_t cropSize[2] = {32, 32};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 65536, 2, 4, 4, 256, 32, 32, 64, 64u);
}

// ===== 正例：depth > INT32_MAX，走 64 位索引路径 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_gt_int32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, static_cast<int64_t>(kInt32Max) + 1}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT,
                          ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4096, 2, 4, 4, static_cast<int64_t>(kInt32Max) + 1, 8, 8, 64, 64u);
}

// ===== 正例：imageHeight > INT32_MAX =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_image_height_gt_int32)
{
    int32_t cropSize[2] = {1, 1};
    auto para = BuildPara({1, static_cast<int64_t>(kInt32Max) + 1, 1, 1}, {1, 4}, {1}, {2}, cropSize, ge::DT_FLOAT,
                          ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 1, 1, static_cast<int64_t>(kInt32Max) + 1, 1, 1, 1, 1, 1, 1u);
}

// ===== 正例：batch > INT32_MAX（xTotal 超限）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_batch_gt_int32)
{
    int32_t cropSize[2] = {1, 1};
    auto para = BuildPara({static_cast<int64_t>(kInt32Max) + 1, 1, 1, 1}, {1, 4}, {1}, {2}, cropSize, ge::DT_FLOAT,
                          ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 1, static_cast<int64_t>(kInt32Max) + 1, 1, 1, 1, 1, 1, 1, 1u);
}

// ===== 正例：boxesTotal > INT32_MAX，但 yTotal 仍在 int32 内 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_boxes_total_gt_int32)
{
    constexpr int64_t kBoxes = static_cast<int64_t>(kInt32Max) / 4 + 1;
    int32_t cropSize[2] = {1, 1};
    auto para = BuildPara({1, 1, 1, 1}, {kBoxes, 4}, {kBoxes}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, kBoxes, 1, 1, 1, 1, 1, 1, kBoxes, 64u);
}

// ===== 正例：totalPositions > INT32_MAX =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_total_positions_gt_int32)
{
    int32_t cropSize[2] = {kInt32Max, 2};
    auto para = BuildPara({1, 1, 1, 1}, {1, 4}, {1}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, 4294967294LL, 1, 1, 1, 1, kInt32Max, 2, 1, 64u);
}

// ===== 正例：yTotal > INT32_MAX，但 totalPositions 仍在 int32 内 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_y_total_gt_int32)
{
    int32_t cropSize[2] = {kInt32Max, 1};
    auto para = BuildPara({1, 1, 1, 2}, {1, 4}, {1}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckTilingFields(info, kInt32Max, 1, 1, 1, 2, kInt32Max, 1, 1, 64u);
}

// ===== 负例：totalPositions 乘法溢出 int64 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_total_positions_overflow)
{
    int32_t cropSize[2] = {kInt32Max, kInt32Max};
    auto para = BuildPara({2, 4, 4, 256}, {kInt32Max, 4}, {kInt32Max}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT,
                          ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：H*W*C 寻址乘积溢出 int64 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_addressing_overflow)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, kInt32Max, kInt32Max, kInt32Max}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT,
                          ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：非法 format（HWCN 非 ND/NHWC/NCHW，ExtractInputInfo 新增拦截分支）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_invalid_format)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_HWCN);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：x 含 -1（动态维）。AiCore 认领后由 tiling 按实际 shape 兜底；
// 此处直接以 -1 调用 tiling，all-dims-positive 拒绝（取 H=-1 代表）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_x_dim_unknown)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, -1, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：num_boxes=-1（boxes.shape[0] 动态），tiling 兜底拒绝 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_unknown)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {-1, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
