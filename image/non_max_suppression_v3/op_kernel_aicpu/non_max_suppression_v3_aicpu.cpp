/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "non_max_suppression_v3_aicpu.h"

#include <queue>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_attr_value.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "log.h"
#include "status.h"
#include "allocator_utils.h"
#include "utils/kernel_util.h"

namespace {
const char* const kNonMaxSuppressionV3 = "NonMaxSuppressionV3";
constexpr uint32_t Two = 2;

bool IsScalarOrSingleElement(const aicpu::TensorShape& shape)
{
    const int32_t rank = shape.GetDims();
    return rank == 0 || (rank == 1 && shape.GetDimSize(0) == 1);
}
} // namespace

namespace aicpu {
uint32_t NonMaxSuppressionV3CpuKernel::GetInputAndCheck(const CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("GetInputAndCheck start.");
    // get input tensors
    // get boxes with size [num_boxes, 4]
    boxes_ = ctx.Input(kFirstInputIndex);
    KERNEL_CHECK_FALSE((boxes_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "input:0 boxes failed.");
    std::shared_ptr<TensorShape> boxes_shape = boxes_->GetTensorShape();
    KERNEL_CHECK_FALSE((boxes_shape != nullptr), KERNEL_STATUS_PARAM_INVALID, "The boxes_shape couldn't be null.");
    int32_t boxes_rank = boxes_shape->GetDims();
    KERNEL_LOG_DEBUG("Input dim size of boxes is %d(boxes_rank) ", boxes_rank);

    if (boxes_rank != 2 || boxes_shape->GetDimSize(1) != 4) {
        KERNEL_LOG_ERROR("The input dim size of boxes must be 2-D and must have 4 columns, "
                         "while %d, %ld",
                         boxes_rank, boxes_shape->GetDimSize(1));
        return KERNEL_STATUS_PARAM_INVALID;
    }
    num_boxes_ = boxes_shape->GetDimSize(0);
    AttrValue* offset_ptr = ctx.GetAttr("offset");
    offset_ = (offset_ptr == nullptr) ? 0 : (offset_ptr->GetInt());
    // get scores with size [num_boxes]
    scores_ = ctx.Input(kSecondInputIndex);
    KERNEL_CHECK_FALSE((scores_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "input:1 scores failed.");
    std::shared_ptr<TensorShape> scores_shape = scores_->GetTensorShape();
    KERNEL_CHECK_FALSE((scores_shape != nullptr), KERNEL_STATUS_PARAM_INVALID, "The scores_shape couldn't be null.");
    int32_t scores_rank = scores_shape->GetDims();
    KERNEL_LOG_DEBUG("Input dim size of scores is %d(scores_rank) ", scores_rank);
    KERNEL_CHECK_FALSE((scores_rank == 1), KERNEL_STATUS_PARAM_INVALID,
                       "The input dim size of scores must be 1-D, while %d.", scores_rank);
    KERNEL_CHECK_FALSE((scores_shape->GetDimSize(0) == num_boxes_), KERNEL_STATUS_PARAM_INVALID,
                       "The len of scores must be equal to the number of boxes, while dims[%ld], num_boxes_[%ld].",
                       scores_shape->GetDimSize(0), num_boxes_);

    // get max_output_size : scalar
    Tensor* max_output_size_tensor = ctx.Input(kThirdInputIndex);
    KERNEL_CHECK_FALSE((max_output_size_tensor != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "input:2 max_output_size failed.");
    auto max_output_size_shape = max_output_size_tensor->GetTensorShape();
    KERNEL_CHECK_FALSE((max_output_size_shape != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "The max_output_size_shape couldn't be null.");
    KERNEL_CHECK_FALSE(IsScalarOrSingleElement(*max_output_size_shape), KERNEL_STATUS_PARAM_INVALID,
                       "The input dim size of max_output_size must be scalar or have one element.");
    max_output_size_ = *static_cast<int32_t*>(max_output_size_tensor->GetData());
    KERNEL_CHECK_FALSE((max_output_size_ >= 0), KERNEL_STATUS_PARAM_INVALID,
                       "max_output_size must be non-negative, but is [%d]", max_output_size_);

    // get iou_threshold : scalar
    iou_threshold_tensor_ = ctx.Input(kFourthInputIndex);
    KERNEL_CHECK_FALSE((iou_threshold_tensor_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "input:3 iou_threshold failed.");
    auto iou_threshold_shape = iou_threshold_tensor_->GetTensorShape();
    KERNEL_CHECK_FALSE((iou_threshold_shape != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "The iou_threshold_shape couldn't be null.");
    KERNEL_CHECK_FALSE(IsScalarOrSingleElement(*iou_threshold_shape), KERNEL_STATUS_PARAM_INVALID,
                       "The input dim size of iou_threshold must be scalar or have one element.");

    // get score_threshold: scalar
    score_threshold_tensor_ = ctx.Input(kFifthInputIndex);
    KERNEL_CHECK_FALSE((score_threshold_tensor_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "input:4 score_threshold failed.");
    auto score_threshold_shape = score_threshold_tensor_->GetTensorShape();
    KERNEL_CHECK_FALSE((score_threshold_shape != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "The score_threshold_shape couldn't be null.");
    KERNEL_CHECK_FALSE(IsScalarOrSingleElement(*score_threshold_shape), KERNEL_STATUS_PARAM_INVALID,
                       "The input dim size of score_threshold must be scalar or have one element.");

    // get output tensors
    output_indices_ = ctx.Output(kFirstOutputIndex);
    KERNEL_CHECK_FALSE((output_indices_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                       "GetInputAndCheck: get "
                       "output:0 output_indices failed.");

    boxes_scores_dtype_ = static_cast<DataType>(boxes_->GetDataType());
    if (static_cast<DataType>(scores_->GetDataType()) != boxes_scores_dtype_) {
        KERNEL_LOG_ERROR("The dtype of input[0]boxes and input[1]scores must be the same.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (boxes_scores_dtype_ != DT_FLOAT16 && boxes_scores_dtype_ != DT_FLOAT) {
        KERNEL_LOG_ERROR("The dtype of input[0]boxes and scores must be float16 or float32.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    threshold_dtype_ = static_cast<DataType>(iou_threshold_tensor_->GetDataType());
    if (threshold_dtype_ != DT_FLOAT16 && threshold_dtype_ != DT_FLOAT) {
        KERNEL_LOG_ERROR("The dtype of input[3]iou_threshold must be float16 or float32.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_CHECK_FALSE((score_threshold_tensor_->GetDataType() == threshold_dtype_), KERNEL_STATUS_PARAM_INVALID,
                       "The dtype of iou_threshold and score_threshold must be the same.");

    const auto score_threshold_dtype = static_cast<DataType>(score_threshold_tensor_->GetDataType());
    if (threshold_dtype_ != score_threshold_dtype) {
        KERNEL_LOG_ERROR("The dtypes of input[3]iou_threshold and input[4]score_threshold must be the same.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    KERNEL_LOG_INFO("GetInputAndCheck end.");
    return KERNEL_STATUS_OK;
}

template <typename T>
inline float NonMaxSuppressionV3CpuKernel::IOUSimilarity(const T* box_1, const T* box_2, const float offset,
                                                         const CorrectBox& correct_box) const
{
    const float ymin_i = static_cast<float>(box_1[correct_box.ymin_index]);
    const float xmin_i = static_cast<float>(box_1[correct_box.xmin_index]);
    const float ymax_i = static_cast<float>(box_1[correct_box.ymax_index]);
    const float xmax_i = static_cast<float>(box_1[correct_box.xmax_index]);
    const float ymin_j = Eigen::numext::mini<float>(static_cast<float>(box_2[0]), static_cast<float>(box_2[2]));
    const float xmin_j = Eigen::numext::mini<float>(static_cast<float>(box_2[1]), static_cast<float>(box_2[3]));
    const float ymax_j = Eigen::numext::maxi<float>(static_cast<float>(box_2[0]), static_cast<float>(box_2[2]));
    const float xmax_j = Eigen::numext::maxi<float>(static_cast<float>(box_2[1]), static_cast<float>(box_2[3]));

    const float area_i = (ymax_i - ymin_i + offset) * (xmax_i - xmin_i + offset);
    const float area_j = (ymax_j - ymin_j + offset) * (xmax_j - xmin_j + offset);
    if (area_i <= 0.0f || area_j <= 0.0f) {
        return 0.0f;
    }
    const float intersection_ymin = Eigen::numext::maxi<float>(ymin_i, ymin_j);
    const float intersection_xmin = Eigen::numext::maxi<float>(xmin_i, xmin_j);
    const float intersection_ymax = Eigen::numext::mini<float>(ymax_i, ymax_j);
    const float intersection_xmax = Eigen::numext::mini<float>(xmax_i, xmax_j);
    const float intersection_area = Eigen::numext::maxi<float>(intersection_ymax - intersection_ymin + offset, 0.0f) *
                                    Eigen::numext::maxi<float>(intersection_xmax - intersection_xmin + offset, 0.0f);
    if (IsValueEqual<float>((area_i + area_j - intersection_area), 0.0f)) {
        return 0.0f;
    }
    return intersection_area / (area_i + area_j - intersection_area);
}

template <typename T, typename T_threshold>
uint32_t NonMaxSuppressionV3CpuKernel::DoCompute()
{
    KERNEL_LOG_INFO("DoCompute start!!");

    Eigen::TensorMap<Eigen::Tensor<T, Two, Eigen::RowMajor>> boxes_map(reinterpret_cast<T*>(boxes_->GetData()),
                                                                       num_boxes_, 4);
    std::vector<T> scores_data(num_boxes_);
    std::copy_n(reinterpret_cast<T*>(scores_->GetData()), num_boxes_, scores_data.begin());

    auto iou_threshold = static_cast<T>(*(static_cast<T_threshold*>(iou_threshold_tensor_->GetData())));
    auto score_threshold = static_cast<T>(*(static_cast<T_threshold*>(score_threshold_tensor_->GetData())));

    std::unique_ptr<int32_t[]> indices_data(new (std::nothrow) int32_t[max_output_size_]);
    if (indices_data == nullptr) {
        KERNEL_LOG_ERROR("DoCompute: new indices_data failed");
        return KERNEL_STATUS_INNER_ERROR;
    }
    if (iou_threshold < static_cast<T>(0.0) || iou_threshold > static_cast<T>(1.0)) {
        KERNEL_LOG_ERROR("DoCompute: input[3]iou_threshold must be in the "
                         "range [0, 1].");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    struct Candidate {
        int box_index;
        T score;
        int suppress_begin_index;
    };

    auto cmp = [](const Candidate& bs_i, const Candidate& bs_j) {
        // NMS requires a strict score-descending order. Use the index only for exact ties;
        // a tolerance here can make the comparator non-transitive for close scores.
        if (bs_i.score < bs_j.score) {
            return true;
        }
        if (bs_j.score < bs_i.score) {
            return false;
        }
        return bs_i.box_index > bs_j.box_index;
    };

    std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidate_priority_queue(cmp);
    for (uint32_t i = 0; i < scores_data.size(); ++i) {
        if (scores_data[i] > score_threshold) {
            candidate_priority_queue.emplace(Candidate({static_cast<int>(i), scores_data[i], 0}));
        }
    }

    float similarity = 0.0f;
    Candidate next_candidate = {.box_index = 0, .score = static_cast<T>(0.0), .suppress_begin_index = 0};
    KERNEL_CHECK_FALSE(CheckFloatAddOverflow(static_cast<float>(offset_), 0.0f), KERNEL_STATUS_INNER_ERROR,
                       "float overflow");
    float offset_cast = static_cast<float>(offset_);
    T original_score;
    int32_t indices_data_size = 0;
    while (indices_data_size < max_output_size_ && !candidate_priority_queue.empty()) {
        next_candidate = candidate_priority_queue.top();
        original_score = next_candidate.score;
        candidate_priority_queue.pop();
        // iterate through the previously selected boxes backwards to see if
        // `next_candidate` should be suppressed.
        bool should_suppress = false;
        const T* box = &boxes_map(next_candidate.box_index, 0);
        CorrectBox correct_box = {0, 1, 2, 3};
        if (box[kFirstInputIndex] > box[kThirdInputIndex]) {
            correct_box.ymin_index = kThirdInputIndex;
            correct_box.ymax_index = kFirstInputIndex;
        }
        if (box[kSecondInputIndex] > box[kFourthInputIndex]) {
            correct_box.xmin_index = kFourthInputIndex;
            correct_box.xmax_index = kSecondInputIndex;
        }
        for (int j = indices_data_size - 1; j >= next_candidate.suppress_begin_index; --j) {
            similarity = IOUSimilarity(&boxes_map(next_candidate.box_index, 0), &boxes_map(indices_data[j], 0),
                                       offset_cast, correct_box);
            next_candidate.score *= (static_cast<T>(similarity) <= iou_threshold) ? static_cast<T>(1.0) :
                                                                                    static_cast<T>(0.0);
            // First decide whether to perform hard suppression
            if (static_cast<T>(similarity) > iou_threshold) {
                should_suppress = true;
                break;
            }
            // If next_candidate survives hard suppression, apply soft suppressin
            if (next_candidate.score <= score_threshold)
                break;
        }
        // next_candidate.suppress_begin_index equals the size of indices_data
        next_candidate.suppress_begin_index = indices_data_size;

        if (!should_suppress) {
            if (IsValueEqual<T>(next_candidate.score, original_score)) {
                indices_data[indices_data_size] = next_candidate.box_index;
                indices_data_size += 1;
                continue;
            }
            if (next_candidate.score > score_threshold) {
                candidate_priority_queue.push(next_candidate);
            }
        }
    }

    int64_t num_valid_outputs = static_cast<int64_t>(indices_data_size);
    std::vector<int64_t> output_shape = {num_valid_outputs};
    KERNEL_LOG_INFO("The num of selected indices is %ld", num_valid_outputs);
    auto ret = CpuKernelAllocatorUtils::UpdateOutputDataTensor(output_shape, DT_INT32, indices_data.get(),
                                                               num_valid_outputs * sizeof(int32_t), output_indices_);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "UpdateOutputDataTensor failed.")

    KERNEL_LOG_INFO("DoCompute end!!");
    return KERNEL_STATUS_OK;
}

uint32_t NonMaxSuppressionV3CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("NonMaxSuppressionV3 kernel in.");
    uint32_t res = GetInputAndCheck(ctx);
    if (res != KERNEL_STATUS_OK) {
        return res;
    }

    if (boxes_scores_dtype_ == DT_FLOAT16 && threshold_dtype_ == DT_FLOAT16) {
        res = DoCompute<Eigen::half, Eigen::half>();
    } else if (boxes_scores_dtype_ == DT_FLOAT && threshold_dtype_ == DT_FLOAT16) {
        res = DoCompute<float, Eigen::half>();
    } else if (boxes_scores_dtype_ == DT_FLOAT16 && threshold_dtype_ == DT_FLOAT) {
        res = DoCompute<Eigen::half, float>();
    } else if (boxes_scores_dtype_ == DT_FLOAT && threshold_dtype_ == DT_FLOAT) {
        res = DoCompute<float, float>();
    }

    KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "Computation failed.");

    KERNEL_LOG_INFO("Compute end!!");
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNonMaxSuppressionV3, NonMaxSuppressionV3CpuKernel);
} // namespace aicpu
