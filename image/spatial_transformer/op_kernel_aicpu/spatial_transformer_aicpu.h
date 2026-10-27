/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_CV_IMAGE_SPATIAL_TRANSFORMER_AICPU_H
#define OPS_CV_IMAGE_SPATIAL_TRANSFORMER_AICPU_H

#include <memory>
#include <vector>
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class SpatialTransformerCpuKernel : public CpuKernel {
public:
    ~SpatialTransformerCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    /**
     * @brief Init params and check valid
     * @param ctx cpu kernel context
     * @return status if success
     */
    KernelStatus GetInputAndCheckValid(const CpuKernelContext& ctx);

    /**
     * @brief compute for all types
     * @param ctx cpu kernel context
     * @return status if success
     */
    template <typename T>
    uint32_t DoCompute(CpuKernelContext& ctx);

    /**
     * @brief compute for NCHW format
     * @param ctx cpu kernel context
     * @return status if success
     */
    template <typename T, typename T1>
    KernelStatus DoCompute4D();

    /**
     * @brief compute for NC1HWC0 format
     * @param ctx cpu kernel context
     * @return status if success
     */
    template <typename T, typename T1>
    KernelStatus DoCompute5D();

    /**
     * @brief compute for NC1HWC0 format
     * @param ctx cpu kernel context
     * @return status if success
     */
    template <typename T, typename T1>
    KernelStatus DoCompute5D_C1();

    /**
     * @brief init theta from input_theta and default_theta
     * @note The index is consumed by value so one batch cannot advance the slice used by the next batch.
     */
    template <typename T1>
    void InitTheta(const T1* input_theta, uint32_t input_theta_idx, std::vector<float>& theta);

    /**
     * @brief compute input grid from output grid using theta
     */
    void ComputeGrid(const std::vector<float>& theta, float* input_grid);

    /**
     * @brief scalar bilinear interpolation for a single pixel
     */
    template <typename T>
    float BilinearInterpolateScalar(const T* data, int32_t base_idx, float x, float y, int32_t row_stride,
                                    int32_t col_stride);

    /**
     * @brief vector bilinear interpolation for NC1HWC0 format
     */
    template <typename T>
    void BilinearInterpolateVector(const T* data, int32_t base_idx, float x, float y, float* res);

    template <typename T>
    void BilinearInterpolFillData(const T* data, int32_t base_idx, float x, float y, float* res);

    /**
     * @brief get attributes from context
     */
    KernelStatus GetAttrs(const CpuKernelContext& ctx);

    Tensor* input_tensor_ = nullptr;
    Tensor* input_theta_ = nullptr;
    Tensor* output_tensor_ = nullptr;
    int32_t input_n_ = 0;
    int32_t input_c_ = 0;
    int32_t input_c1_ = 0;
    int32_t input_c0_ = 0;
    int32_t input_h_ = 0;
    int32_t input_w_ = 0;
    int32_t output_h_ = 0;
    int32_t output_w_ = 0;
    int32_t stn_ori_channel_ = 0;
    std::vector<float> theta_;
    std::vector<int64_t> theta_valid_;
    // Zero for a [2, 3] transform shared by every batch, six for a per-batch [batch, 2, 3] tensor.
    uint32_t theta_batch_stride_ = 0;
    Format date_format_ = FORMAT_ND;
    DataType input_data_type_ = DT_FLOAT;
    DataType input_theta_type_ = DT_FLOAT;
    DataType output_data_type_ = DT_FLOAT;
};
} // namespace aicpu
#endif
