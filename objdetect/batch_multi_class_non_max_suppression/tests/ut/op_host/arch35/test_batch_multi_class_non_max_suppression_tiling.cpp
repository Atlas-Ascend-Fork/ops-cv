/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/batch_multi_class_non_max_suppression_tiling_arch35.h"
#include "../../../../op_kernel/arch35/batch_multi_class_non_max_suppression_tiling_data.h"

namespace {

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

struct BatchMultiClassNonMaxSuppressionCompileInfoForTest {
    uint32_t coreNum = 64;
    uint64_t ubSize = 262144;
};

gert::TilingContextPara MakeContext(const std::vector<int64_t>& boxesShape, const std::vector<int64_t>& scoresShape,
                                    int64_t maxSizePerClass, int64_t maxTotalSize, bool changeCoordinateFrame = false,
                                    bool transposeBox = false, ge::DataType boxesDtype = ge::DT_FLOAT,
                                    ge::DataType scoresDtype = ge::DT_FLOAT,
                                    const std::vector<int64_t>& clipWindowShape = {},
                                    const std::vector<int64_t>& numValidBoxesShape = {},
                                    const std::vector<int64_t>& outputBoxesShape = {},
                                    const std::vector<int64_t>& outputNumShape = {})
{
    const int64_t batch = boxesShape[0];
    const int64_t innerBoxes = transposeBox ? 4 : maxTotalSize;
    const int64_t innerCoords = transposeBox ? maxTotalSize : 4;
    const int64_t numSecondDim = transposeBox ? 8 : -1;
    const gert::StorageShape boxes = MakeStorageShape(boxesShape);
    const gert::StorageShape scores = MakeStorageShape(scoresShape);
    const gert::StorageShape outputBoxes = MakeStorageShape(
        outputBoxesShape.empty() ? std::vector<int64_t>{batch, innerBoxes, innerCoords} : outputBoxesShape);
    const gert::StorageShape outputScores = MakeStorageShape({batch, maxTotalSize});
    const gert::StorageShape outputClasses = MakeStorageShape({batch, maxTotalSize});
    const gert::StorageShape outputNum = MakeStorageShape(
        outputNumShape.empty() ?
            (transposeBox ? std::vector<int64_t>{batch, numSecondDim} : std::vector<int64_t>{batch}) :
            outputNumShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {boxes, boxesDtype, ge::FORMAT_ND},
        {scores, scoresDtype, ge::FORMAT_ND},
    };
    if (!numValidBoxesShape.empty()) {
        inputs.push_back({MakeStorageShape(numValidBoxesShape), ge::DT_INT32, ge::FORMAT_ND});
    } else if (!clipWindowShape.empty()) {
        inputs.push_back({MakeStorageShape(clipWindowShape), boxesDtype, ge::FORMAT_ND});
    }
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputBoxes, boxesDtype, ge::FORMAT_ND},
        {outputScores, boxesDtype, ge::FORMAT_ND},
        {outputClasses, boxesDtype, ge::FORMAT_ND},
        {outputNum, ge::DT_INT32, ge::FORMAT_ND},
    };
    static BatchMultiClassNonMaxSuppressionCompileInfoForTest defaultCompileInfo;
    return gert::TilingContextPara(
        "BatchMultiClassNonMaxSuppression", inputs, outputs,
        {gert::TilingContextPara::OpAttr("score_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.25F)),
         gert::TilingContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5F)),
         gert::TilingContextPara::OpAttr("max_size_per_class", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxSizePerClass)),
         gert::TilingContextPara::OpAttr("max_total_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxTotalSize)),
         gert::TilingContextPara::OpAttr("change_coordinate_frame",
                                         Ops::Cv::AnyValue::CreateFrom<bool>(changeCoordinateFrame)),
         gert::TilingContextPara::OpAttr("transpose_box", Ops::Cv::AnyValue::CreateFrom<bool>(transposeBox))},
        &defaultCompileInfo, "Ascend950", defaultCompileInfo.coreNum, defaultCompileInfo.ubSize);
}

TEST(BatchMultiClassNonMaxSuppressionTiling, derives_tiling_for_shared_boxes)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(BatchMultiClassNonMaxSuppressionTilingData));
    const auto* tiling = reinterpret_cast<const BatchMultiClassNonMaxSuppressionTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->batch, 2);
    EXPECT_EQ(tiling->boxesNum, 5);
    EXPECT_EQ(tiling->classesNum, 3);
    EXPECT_EQ(tiling->boxClassesNum, 1);
    EXPECT_EQ(tiling->maxSizePerClass, 2);
    EXPECT_EQ(tiling->maxTotalSize, 4);
    EXPECT_EQ(tiling->usedCoreNum, 6);
    EXPECT_EQ(tiling->hasClipWindow, 0);
    EXPECT_EQ(tiling->hasNumValidBoxes, 0);
    EXPECT_EQ(tiling->changeCoordinateFrame, 0);
    EXPECT_EQ(tiling->transposeBox, 0);
    EXPECT_FLOAT_EQ(tiling->scoreThreshold, 0.25F);
    EXPECT_FLOAT_EQ(tiling->iouThreshold, 0.5F);
    EXPECT_EQ(info.blockNum, 6U);
    EXPECT_FALSE(info.workspaceSizes.empty());
}

TEST(BatchMultiClassNonMaxSuppressionTiling, accepts_clip_window_with_change_coordinate_frame)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, true, false, ge::DT_FLOAT, ge::DT_FLOAT, {2, 4});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = reinterpret_cast<const BatchMultiClassNonMaxSuppressionTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->hasClipWindow, 1);
    EXPECT_EQ(tiling->changeCoordinateFrame, 1);
}

TEST(BatchMultiClassNonMaxSuppressionTiling, accepts_compacted_num_valid_boxes)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, false, false, ge::DT_FLOAT, ge::DT_FLOAT, {}, {2});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = reinterpret_cast<const BatchMultiClassNonMaxSuppressionTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->hasClipWindow, 0);
    EXPECT_EQ(tiling->hasNumValidBoxes, 1);
}

TEST(BatchMultiClassNonMaxSuppressionTiling, accepts_transposed_boxes)
{
    auto context = MakeContext({2, 3, 4, 5}, {2, 3, 5}, 2, 4, false, true);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = reinterpret_cast<const BatchMultiClassNonMaxSuppressionTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->batch, 2);
    EXPECT_EQ(tiling->boxesNum, 5);
    EXPECT_EQ(tiling->classesNum, 3);
    EXPECT_EQ(tiling->boxClassesNum, 3);
    EXPECT_EQ(tiling->transposeBox, 1);
    EXPECT_EQ(info.blockNum, 6U);
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_wrong_boxes_rank)
{
    auto context = MakeContext({2, 5, 4}, {2, 5, 3}, 2, 4);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_non_positive_dims)
{
    auto context = MakeContext({2, 0, 1, 4}, {2, 0, 3}, 2, 4);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_out_of_range_max_sizes)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 0);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_non_four_coordinates)
{
    auto context = MakeContext({2, 5, 1, 5}, {2, 5, 3}, 2, 4);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_mismatched_batch)
{
    auto context = MakeContext({2, 5, 1, 4}, {3, 5, 3}, 2, 4);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_dtype_mismatch)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, false, false, ge::DT_FLOAT, ge::DT_FLOAT16);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_bad_clip_window_shape)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, false, false, ge::DT_FLOAT, ge::DT_FLOAT, {2, 3});
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_change_coordinate_frame_without_clip_window)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, true);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_output_contract_violation)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4, false, false, ge::DT_FLOAT, ge::DT_FLOAT, {}, {},
                               {2, 8, 4});
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BatchMultiClassNonMaxSuppressionTiling, rejects_zero_ub_in_compile_info)
{
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, 2, 4);
    // Swap in a compile info whose platform UB query returned 0 so the
    // resource guard fires through the platform-report path.
    static BatchMultiClassNonMaxSuppressionCompileInfoForTest zeroUb{64, 0};
    context.compileInfo_ = &zeroUb;
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}
} // namespace
