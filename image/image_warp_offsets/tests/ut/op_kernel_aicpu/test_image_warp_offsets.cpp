/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

class TEST_IMAGE_WARP_OFFSETS_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num == input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            expect_out[j] = input0_data[j] < input1_data[j] ? true : false;
        }
    }
}

template <typename T>
void CalcExpectWithDiffShape(const NodeDef& node_def, bool expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();
    if (input0_num > input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            int64_t i = j % input1_num;
            expect_out[j] = input0_data[j] < input1_data[i] ? true : false;
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                      \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
    NodeDefBuilder(node_def.get(), "IMGWarpOffsets", "IMGWarpOffsets") \
        .Input({"x1", data_types[0], shapes[0], datas[0]})             \
        .Input({"x2", data_types[1], shapes[1], datas[1]})             \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, NO_KERNEL_FAILED)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 3}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_RANK_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_SHPAE_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, -1}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_DIM_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, 3}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_RANK_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_SHPAE_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2, 2, -1}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_DIM_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 3}, {1, 2, 2, 2}, {1, 4, 2, 2, 2}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, OUTPUT_RANK_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, OUTPUT_SHPAE_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, SHPAE_MISMATCH_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 1}};
    int32_t data[40] = {0};
    vector<void*> datas = {(void*)data, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INVALID_VALUE_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    Eigen::half data1[12] = {Eigen::half(1.1)};
    int32_t data[40] = {30, 30, 30, 30, 30, 30, 20, 20, 20};
    vector<void*> datas = {(void*)data1, (void*)data, (void*)data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, NEGATIVE_INT32_OFFSET_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    Eigen::half input0_data[24] = {Eigen::half(1.0f)};
    int32_t input1_data[16] = {-1, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21};
    Eigen::half output_data[48] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, POSITIVE_INT32_OFFSET_OOB_FAILED)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    Eigen::half input0_data[24] = {Eigen::half(1.0f)};
    int32_t input1_data[16] = {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 22};
    Eigen::half output_data[48] = {Eigen::half(0.0f)};
    vector<void*> data = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, data);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, DT_FLOAT16_DT_FLOAT_DT_FLOAT16_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    Eigen::half input0_data[24] = {Eigen::half(1.0f),  Eigen::half(2.0f),  Eigen::half(3.0f),  Eigen::half(4.0f),
                                   Eigen::half(5.0f),  Eigen::half(6.0f),  Eigen::half(7.0f),  Eigen::half(8.0f),
                                   Eigen::half(9.0f),  Eigen::half(10.0f), Eigen::half(11.0f), Eigen::half(12.0f),
                                   Eigen::half(13.0f), Eigen::half(14.0f), Eigen::half(15.0f), Eigen::half(16.0f),
                                   Eigen::half(17.0f), Eigen::half(18.0f), Eigen::half(19.0f), Eigen::half(20.0f),
                                   Eigen::half(21.0f), Eigen::half(22.0f), Eigen::half(23.0f), Eigen::half(24.0f)};
    float input1_data[16] = {0.0f,  1.0f,  2.0f,  3.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                             12.0f, 13.0f, 14.0f, 15.0f, 18.0f, 19.0f, 20.0f, 21.0f};
    Eigen::half output_data[48] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(static_cast<float>(output_data[0]), 1.0f);
    EXPECT_EQ(static_cast<float>(output_data[1]), 2.0f);
    EXPECT_EQ(static_cast<float>(output_data[2]), 3.0f);
    EXPECT_EQ(static_cast<float>(output_data[3]), 2.0f);
    EXPECT_EQ(static_cast<float>(output_data[4]), 3.0f);
    EXPECT_EQ(static_cast<float>(output_data[5]), 4.0f);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, DT_FLOAT_DT_FLOAT_DT_FLOAT_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    float input0_data[24] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                             13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    float input1_data[16] = {0.0f,  1.0f,  2.0f,  3.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                             12.0f, 13.0f, 14.0f, 15.0f, 18.0f, 19.0f, 20.0f, 21.0f};
    float output_data[48] = {0.0f};
    vector<void*> datas = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(output_data[0], 1.0f);
    EXPECT_EQ(output_data[1], 2.0f);
    EXPECT_EQ(output_data[2], 3.0f);
    EXPECT_EQ(output_data[3], 2.0f);
    EXPECT_EQ(output_data[4], 3.0f);
    EXPECT_EQ(output_data[5], 4.0f);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, DT_UINT8_DT_FLOAT_DT_UINT8_SUCCESS)
{
    vector<DataType> data_types = {DT_UINT8, DT_FLOAT, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    uint8_t input0_data[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    float input1_data[16] = {0.0f,  1.0f,  2.0f,  3.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                             12.0f, 13.0f, 14.0f, 15.0f, 18.0f, 19.0f, 20.0f, 21.0f};
    uint8_t output_data[48] = {0};
    vector<void*> datas = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(output_data[0], 1);
    EXPECT_EQ(output_data[1], 2);
    EXPECT_EQ(output_data[2], 3);
    EXPECT_EQ(output_data[3], 2);
    EXPECT_EQ(output_data[4], 3);
    EXPECT_EQ(output_data[5], 4);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, DT_FLOAT16_DT_INT32_DT_FLOAT16_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 4, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
    Eigen::half input0_data[24] = {Eigen::half(1.0f),  Eigen::half(2.0f),  Eigen::half(3.0f),  Eigen::half(4.0f),
                                   Eigen::half(5.0f),  Eigen::half(6.0f),  Eigen::half(7.0f),  Eigen::half(8.0f),
                                   Eigen::half(9.0f),  Eigen::half(10.0f), Eigen::half(11.0f), Eigen::half(12.0f),
                                   Eigen::half(13.0f), Eigen::half(14.0f), Eigen::half(15.0f), Eigen::half(16.0f),
                                   Eigen::half(17.0f), Eigen::half(18.0f), Eigen::half(19.0f), Eigen::half(20.0f),
                                   Eigen::half(21.0f), Eigen::half(22.0f), Eigen::half(23.0f), Eigen::half(24.0f)};
    int32_t input1_data[16] = {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21};
    Eigen::half output_data[48] = {Eigen::half(0.0f)};
    vector<void*> datas = {(void*)input0_data, (void*)input1_data, (void*)output_data};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_EQ(static_cast<float>(output_data[0]), 1.0f);
    EXPECT_EQ(static_cast<float>(output_data[1]), 2.0f);
    EXPECT_EQ(static_cast<float>(output_data[2]), 3.0f);
    EXPECT_EQ(static_cast<float>(output_data[3]), 2.0f);
    EXPECT_EQ(static_cast<float>(output_data[4]), 3.0f);
    EXPECT_EQ(static_cast<float>(output_data[5]), 4.0f);
}
