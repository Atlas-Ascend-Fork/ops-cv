/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "op_infer_datatype_context_builder.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
std::vector<gert::InfershapeContextPara::OpAttr> GetAttrs(int64_t maxTotalSize, bool transposeBox = false)
{
    return {
        gert::InfershapeContextPara::OpAttr("score_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.3F)),
        gert::InfershapeContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5F)),
        gert::InfershapeContextPara::OpAttr("max_size_per_class", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
        gert::InfershapeContextPara::OpAttr("max_total_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(maxTotalSize)),
        gert::InfershapeContextPara::OpAttr("change_coordinate_frame", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        gert::InfershapeContextPara::OpAttr("transpose_box", Ops::Cv::AnyValue::CreateFrom<bool>(transposeBox)),
    };
}
} // namespace

TEST(BatchMultiClassNonMaxSuppressionInfershape, derives_public_output_layout)
{
    gert::InfershapeContextPara context("BatchMultiClassNonMaxSuppression",
                                        {{{{2, 8, 1, 4}, {2, 8, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 8, 3}, {2, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                        GetAttrs(5));
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 5, 4}, {2, 5}, {2, 5}, {2}});
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, rejects_out_of_range_max_total_size)
{
    for (const int64_t maxTotalSize : {-1, 0, 1001}) {
        SCOPED_TRACE(::testing::Message() << "max_total_size=" << maxTotalSize);
        gert::InfershapeContextPara context("BatchMultiClassNonMaxSuppression",
                                            {{{{2, 8, 1, 4}, {2, 8, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{2, 8, 3}, {2, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                            {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                            GetAttrs(maxTotalSize));
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, accepts_max_total_size_limits)
{
    for (const int64_t maxTotalSize : {1, 1000}) {
        SCOPED_TRACE(::testing::Message() << "max_total_size=" << maxTotalSize);
        gert::InfershapeContextPara context("BatchMultiClassNonMaxSuppression",
                                            {{{{2, 8, 1, 4}, {2, 8, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{2, 8, 3}, {2, 8, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                            {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                             {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                            GetAttrs(maxTotalSize));
        ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, maxTotalSize, 4}, {2, maxTotalSize}, {2, maxTotalSize}, {2}});
    }
}

namespace {
void ExpectLegacyShape(const gert::StorageShape& boxes, const gert::StorageShape& scores, bool transposeBox,
                       int64_t expectedBatch)
{
    constexpr int64_t kMaxTotalSize = 5;
    constexpr int64_t kBoxCoordinateCount = 4;
    constexpr int64_t kPaddedCountStride = 8;
    gert::InfershapeContextPara context("BatchMultiClassNonMaxSuppression",
                                        {{boxes, ge::DT_FLOAT, ge::FORMAT_ND}, {scores, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                        GetAttrs(kMaxTotalSize, transposeBox));
    const std::vector<int64_t> expectedBoxes = transposeBox ? std::vector<int64_t>{expectedBatch, kBoxCoordinateCount,
                                                                                   kMaxTotalSize} :
                                                              std::vector<int64_t>{expectedBatch, kMaxTotalSize,
                                                                                   kBoxCoordinateCount};
    const std::vector<int64_t> expectedNum = transposeBox ? std::vector<int64_t>{expectedBatch, kPaddedCountStride} :
                                                            std::vector<int64_t>{expectedBatch};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                    {expectedBoxes, {expectedBatch, kMaxTotalSize}, {expectedBatch, kMaxTotalSize}, expectedNum});
}
} // namespace

TEST(BatchMultiClassNonMaxSuppressionInfershape, derives_legacy_transposed_outputs_for_fused_internal_node)
{
    // The public verifier rejects transpose_box=true, but the GE fusion pass
    // sets it on the transformed internal node after transposing the inputs.
    ExpectLegacyShape({{2, 1, 4, 8}, {2, 1, 4, 8}}, {{2, 3, 8}, {2, 3, 8}}, true, 2);
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, derives_batch_from_scores_with_unknown_boxes_rank)
{
    ExpectLegacyShape({{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}}, {{2, 8, 3}, {2, 8, 3}}, false, 2);
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, preserves_unknown_batch_for_dynamic_scores)
{
    const std::vector<gert::StorageShape> dynamicScores = {{{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}},
                                                           {{ge::UNKNOWN_DIM, 8, 3}, {ge::UNKNOWN_DIM, 8, 3}},
                                                           {{2, ge::UNKNOWN_DIM, 3}, {2, ge::UNKNOWN_DIM, 3}},
                                                           {{2, 8, ge::UNKNOWN_DIM}, {2, 8, ge::UNKNOWN_DIM}}};
    for (const bool transposeBox : {false, true}) {
        for (size_t index = 0; index < dynamicScores.size(); ++index) {
            SCOPED_TRACE(::testing::Message() << "transpose_box=" << transposeBox << ", case=" << index);
            ExpectLegacyShape({{2, 8, 1, 4}, {2, 8, 1, 4}}, dynamicScores[index], transposeBox, ge::UNKNOWN_DIM);
        }
    }
}

TEST(BatchMultiClassNonMaxSuppressionInfershape, floating_outputs_follow_scores_dtype)
{
    constexpr size_t kInputCount = 2;
    constexpr size_t kOutputCount = 4;
    constexpr size_t kFloatingOutputCount = 3;
    constexpr size_t kBoxesInputIndex = 0;
    constexpr size_t kScoresInputIndex = 1;
    constexpr size_t kCountOutputIndex = kFloatingOutputCount;
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("BatchMultiClassNonMaxSuppression");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    for (const ge::DataType scoresType : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        SCOPED_TRACE(scoresType);
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("BatchMultiClassNonMaxSuppression").OpName("BatchMultiClassNonMaxSuppression");
        builder.IONum(kInputCount, kOutputCount);
        // Deliberately differ so this detects accidentally taking the boxes dtype.
        const ge::DataType boxesType = scoresType == ge::DT_FLOAT ? ge::DT_FLOAT16 : ge::DT_FLOAT;
        builder.InputTensorDesc(kBoxesInputIndex, boxesType, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(kScoresInputIndex, scoresType, ge::FORMAT_ND, ge::FORMAT_ND);
        for (size_t index = 0; index < kOutputCount; ++index) {
            builder.OutputTensorDesc(index, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        for (size_t index = 0; index < kFloatingOutputCount; ++index) {
            EXPECT_EQ(context->GetOutputDataType(index), scoresType);
        }
        EXPECT_EQ(context->GetOutputDataType(kCountOutputIndex), ge::DT_INT32);
    }
}
