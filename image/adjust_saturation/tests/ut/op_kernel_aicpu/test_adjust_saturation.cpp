/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {
constexpr int64_t kRgbChannelCount{3};
constexpr int64_t kEmptyDimension{0};
constexpr int64_t kImageHeight{2};
constexpr int64_t kInvalidChannelCount{2};
constexpr int64_t kScalarDimension{1};
constexpr int kTestBufferElements{1};
constexpr int kInvalidImageElements{8};
constexpr float kTestSaturationFactor{0.5F};
constexpr float kTestInputValue{1.0F};
constexpr float kTestZeroValue{0.0F};

} // namespace

class TEST_ADJUST_SATURATION_UT : public testing::Test {
protected:
    std::float_t* float_null_{nullptr};
    std::float_t float_0_[0];
    std::float_t float_12_[12]{1.0f};
    std::float_t float_16_[16]{0.0f};
    std::int32_t int32_22_[22]{1};
    std::int64_t int64_22_[22]{0L};
    bool bool_22_[22]{true};
};

inline void RunKernelAdjustSaturation(std::shared_ptr<aicpu::NodeDef> node_def, aicpu::DeviceType device_type,
                                      uint32_t expect)
{
    std::string node_def_str;
    node_def->SerializeToString(node_def_str);
    aicpu::CpuKernelContext ctx(device_type);
    EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KERNEL_STATUS_OK);
    std::uint32_t ret{aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx)};
    EXPECT_EQ(ret, expect);
}

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
    auto node_def = CpuKernelUtils::CreateNodeDef();                       \
    NodeDefBuilder(node_def.get(), "AdjustSaturation", "AdjustSaturation") \
        .Input({"images", (data_types)[0], (shapes)[0], (datas)[0]})       \
        .Input({"delta", (data_types)[1], (shapes)[1], (datas)[1]})        \
        .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]});

// ===== float32 functional test =====
TEST_F(TEST_ADJUST_SATURATION_UT, DATA_TYPE_DT_FLOAT)
{
    // Input: shape [2, 2, 3], 12 elements. delta = 0.5
    float input_f32[] = {0.813376963f, 0.429118544f, 0.673147976f, 0.230282947f, 0.358550310f, 0.223925725f,
                         0.102745622f, 0.619133949f, 0.832032800f, 0.495856494f, 0.687193096f, 0.674335837f};
    float output_f32_exp[] = {0.813376963f, 0.621247768f, 0.743262470f, 0.294416636f, 0.358550310f, 0.291238010f,
                              0.467389226f, 0.725583375f, 0.832032800f, 0.591524780f, 0.687193096f, 0.680764437f};
    float delta[1] = {0.5f};
    float output_f32[12] = {0};

    vector<vector<int64_t>> shapes = {{2, 2, 3}, {1}, {2, 2, 3}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input_f32, delta, output_f32};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output_f32, output_f32_exp, 12);
    EXPECT_EQ(compare, true);
}

// ===== float16 functional test =====
TEST_F(TEST_ADJUST_SATURATION_UT, DATA_TYPE_DT_FLOAT16)
{
    // Input: shape [1, 2, 3], 6 elements. delta = 0.5
    Eigen::half input_f16[6];
    input_f16[0] = Eigen::half(0.929688f);
    input_f16[1] = Eigen::half(0.316406f);
    input_f16[2] = Eigen::half(0.183960f);
    input_f16[3] = Eigen::half(0.204590f);
    input_f16[4] = Eigen::half(0.567871f);
    input_f16[5] = Eigen::half(0.595703f);

    Eigen::half output_f16_exp[6];
    output_f16_exp[0] = Eigen::half(0.929688f);
    output_f16_exp[1] = Eigen::half(0.623047f);
    output_f16_exp[2] = Eigen::half(0.556641f);
    output_f16_exp[3] = Eigen::half(0.400146f);
    output_f16_exp[4] = Eigen::half(0.582031f);
    output_f16_exp[5] = Eigen::half(0.595703f);

    float delta[1] = {0.5f};
    Eigen::half output_f16[6];

    vector<vector<int64_t>> shapes = {{1, 2, 3}, {1}, {1, 2, 3}};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<void*> datas = {input_f16, delta, output_f16};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output_f16, output_f16_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADJUST_SATURATION_UT, EMPTY_IMAGE_IS_SAFE)
{
    float input[kTestBufferElements] = {kTestZeroValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kTestBufferElements] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{kEmptyDimension, kImageHeight, kRgbChannelCount},
                                      {kScalarDimension},
                                      {kEmptyDimension, kImageHeight, kRgbChannelCount}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// ===== Exception test cases =====
TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_SHAPE_EXCEPTION)
{
    float input[12] = {1.0f};
    float delta[1] = {0.5f};
    float output[16] = {0.0f};

    vector<vector<int64_t>> shapes = {{2, 6}, {1}, {2, 8}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, SCALE_SHAPE_EXCEPTION)
{
    float input[kRgbChannelCount] = {kTestInputValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kRgbChannelCount] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{1, 1, kRgbChannelCount}, {1, 1}, {1, 1, kRgbChannelCount}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, OUTPUT_SHAPE_EXCEPTION)
{
    float input[kRgbChannelCount] = {kTestInputValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kRgbChannelCount] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{1, 1, kRgbChannelCount}, {kScalarDimension}, {kRgbChannelCount, 1, 1}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_LAST_DIM_EXCEPTION)
{
    float input[kInvalidImageElements] = {kTestInputValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kInvalidImageElements] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{kImageHeight, kImageHeight, kInvalidChannelCount},
                                      {kScalarDimension},
                                      {kImageHeight, kImageHeight, kInvalidChannelCount}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_RANK_EXCEPTION)
{
    float input[kTestBufferElements] = {kTestInputValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kTestBufferElements] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{}, {kScalarDimension}, {}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_ONE_DIMENSION_EXCEPTION)
{
    float input[kRgbChannelCount] = {kTestInputValue};
    float delta[kTestBufferElements] = {kTestSaturationFactor};
    float output[kRgbChannelCount] = {kTestZeroValue};

    vector<vector<int64_t>> shapes = {{kRgbChannelCount}, {kScalarDimension}, {kRgbChannelCount}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_DTYPE_EXCEPTION)
{
    int32_t input[22] = {1};
    float delta[1] = {0.5f};
    int64_t output[22] = {0L};

    vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};
    vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT64};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_NULL_EXCEPTION)
{
    float* null_ptr = nullptr;
    float delta[1] = {0.5f};

    vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {null_ptr, delta, null_ptr};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, NO_OUTPUT_EXCEPTION)
{
    float input[12] = {1.0f};
    float delta[1] = {0.5f};

    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "AdjustSaturation", "AdjustSaturation")
        .Input({"images", DT_FLOAT, {2, 6}, input})
        .Input({"delta", DT_FLOAT, {1}, delta});
    RunKernelAdjustSaturation(node_def, aicpu::DeviceType::HOST, aicpu::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADJUST_SATURATION_UT, INPUT_BOOL_UNSUPPORT)
{
    bool input[22] = {true};
    float delta[1] = {0.5f};
    bool output[22] = {true};

    vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};
    vector<DataType> data_types = {DT_BOOL, DT_FLOAT, DT_BOOL};
    vector<void*> datas = {input, delta, output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

template <typename T>
void CheckSmallChroma(DataType dataType, T low, T high)
{
    T input[] = {high, low, low, low, low, low, T(0), T(0), T(0)};
    T output[9] = {};
    float scale[] = {100000.0F};
    vector<vector<int64_t>> shapes = {{1, 3, 3}, {1}, {1, 3, 3}};
    vector<DataType> dataTypes = {dataType, DT_FLOAT, dataType};
    vector<void*> datas = {input, scale, output};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    const T expected[] = {high, T(0), T(0), low, low, low, T(0), T(0), T(0)};
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(static_cast<float>(output[i]), static_cast<float>(expected[i]));
    }
}

TEST_F(TEST_ADJUST_SATURATION_UT, SMALL_NONZERO_CHROMA_IS_SCALED)
{
    CheckSmallChroma<float>(DT_FLOAT, 0.5F, 0.5F + 8e-6F);
    CheckSmallChroma<Eigen::half>(DT_FLOAT16, Eigen::half(0.001F), Eigen::half(0.001001F));
}

TEST_F(TEST_ADJUST_SATURATION_UT, ZERO_MAX_DOES_NOT_PRODUCE_NAN)
{
    float input[] = {-8e-6F, 0.0F, 0.0F};
    float output[3] = {};
    float scale[] = {0.0F};
    vector<vector<int64_t>> shapes = {{1, 1, 3}, {1}, {1, 1, 3}};
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {input, scale, output};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (float value : output) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_EQ(value, 0.0F);
    }
}
