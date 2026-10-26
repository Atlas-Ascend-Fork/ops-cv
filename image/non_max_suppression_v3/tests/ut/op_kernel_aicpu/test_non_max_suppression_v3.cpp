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
#include "allocator_utils.h"
#undef private
#undef protected
#include <algorithm>

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {
uint64_t result_summary[4] = {0};

uint32_t RunControlShapeCase(const vector<int64_t>& max_shape, const vector<int64_t>& iou_shape,
                             const vector<int64_t>& score_shape)
{
    vector<float> boxes = {0.0F, 0.0F, 1.0F, 1.0F, 10.0F, 10.0F, 11.0F, 11.0F};
    vector<float> scores = {0.8F, 0.6F};
    int32_t max_output_size_scalar = 2;
    int32_t max_output_size[2] = {2, 999};
    float iou_threshold_scalar = 0.5F;
    float iou_threshold[2] = {0.5F, -1.0F};
    float score_threshold_scalar = 0.75F;
    float score_threshold[2] = {0.75F, 100.0F};
    vector<void*> datas = {
        (void*)boxes.data(),
        (void*)scores.data(),
        (void*)(max_shape.empty() ? static_cast<void*>(&max_output_size_scalar) : static_cast<void*>(max_output_size)),
        (void*)(iou_shape.empty() ? static_cast<void*>(&iou_threshold_scalar) : static_cast<void*>(iou_threshold)),
        (void*)(score_shape.empty() ? static_cast<void*>(&score_threshold_scalar) :
                                      static_cast<void*>(score_threshold)),
        (void*)result_summary};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3")
        .Input({"boxes", DT_FLOAT, {2, 4}, datas[0]})
        .Input({"scores", DT_FLOAT, {2}, datas[1]})
        .Input({"max_output_size", DT_INT32, max_shape, datas[2]})
        .Input({"iou_threshold", DT_FLOAT, iou_shape, datas[3]})
        .Input({"score_threshold", DT_FLOAT, score_shape, datas[4]})
        .Output({"selected_indices", DT_UINT64, {-1}, datas[5]});
    CpuKernelContext context(HOST);
    const uint32_t init_ret = context.Init(node_def.get());
    if (init_ret != KERNEL_STATUS_OK) {
        return init_ret;
    }
    return CpuKernelRegister::Instance().RunCpuKernel(context);
}

uint32_t RunBoxScoreDtypeCase(DataType boxes_dtype, DataType scores_dtype)
{
    std::vector<float> boxes_float = {0.0F, 0.0F, 1.0F, 1.0F, 10.0F, 10.0F, 11.0F, 11.0F};
    std::vector<Eigen::half> boxes_half = {(Eigen::half)0.0F,  (Eigen::half)0.0F,  (Eigen::half)1.0F,
                                           (Eigen::half)1.0F,  (Eigen::half)10.0F, (Eigen::half)10.0F,
                                           (Eigen::half)11.0F, (Eigen::half)11.0F};
    std::vector<float> scores_float = {0.8F, 0.6F};
    std::vector<Eigen::half> scores_half = {(Eigen::half)0.8F, (Eigen::half)0.6F};
    void* boxes_data = boxes_dtype == DT_FLOAT ? static_cast<void*>(boxes_float.data()) :
                                                 static_cast<void*>(boxes_half.data());
    void* scores_data = scores_dtype == DT_FLOAT ? static_cast<void*>(scores_float.data()) :
                                                   static_cast<void*>(scores_half.data());
    int32_t max_output_size = 2;
    float iou_threshold = 0.5F;
    float score_threshold = 0.5F;
    uint64_t local_result_summary[4] = {0, 0, 0, 0};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3")
        .Input({"boxes", boxes_dtype, {2, 4}, boxes_data})
        .Input({"scores", scores_dtype, {2}, scores_data})
        .Input({"max_output_size", DT_INT32, {}, &max_output_size})
        .Input({"iou_threshold", DT_FLOAT, {}, &iou_threshold})
        .Input({"score_threshold", DT_FLOAT, {}, &score_threshold})
        .Output({"selected_indices", DT_UINT64, {-1}, local_result_summary});
    CpuKernelContext context(HOST);
    const uint32_t init_ret = context.Init(node_def.get());
    if (init_ret != KERNEL_STATUS_OK) {
        return init_ret;
    }
    return CpuKernelRegister::Instance().RunCpuKernel(context);
}
} // namespace

class TEST_NON_MAX_SUPPRESSION_V3_UT : public testing::Test {};

#define CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas)                   \
    auto memcpy_node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(memcpy_node_def.get(), "MemCopy", "MemCopy")             \
        .Input({"release_flag", data_types[0], shapes[0], datas[0]})        \
        .Input({"data_size", data_types[1], shapes[1], datas[1]})           \
        .Input({"src_ptr", data_types[2], shapes[2], datas[2]})             \
        .Input({"dst_ptr", data_types[3], shapes[3], datas[3]})             \
        .Attr("num", 2);

#define CREATE_NODEDEF(shapes, data_types, datas)                                \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3") \
        .Input({"boxes", data_types[0], shapes[0], datas[0]})                    \
        .Input({"scores", data_types[1], shapes[1], datas[1]})                   \
        .Input({"max_output_size", data_types[2], shapes[2], datas[2]})          \
        .Input({"iou_threshold", data_types[3], shapes[3], datas[3]})            \
        .Input({"score_threshold", data_types[4], shapes[4], datas[4]})          \
        .Output({"result_summary", data_types[5], shapes[5], datas[5]})

#define CREATE_NODEDEF_WITHATTR(shapes, data_types, datas)                       \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3") \
        .Input({"boxes", data_types[0], shapes[0], datas[0]})                    \
        .Input({"scores", data_types[1], shapes[1], datas[1]})                   \
        .Input({"max_output_size", data_types[2], shapes[2], datas[2]})          \
        .Input({"iou_threshold", data_types[3], shapes[3], datas[3]})            \
        .Input({"score_threshold", data_types[4], shapes[4], datas[4]})          \
        .Output({"result_summary", data_types[5], shapes[5], datas[5]})          \
        .Attr("offset", 0)

#define ADD_CASE(base_type, aicpu_type)                                                                              \
    TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_##aicpu_type)                                     \
    {                                                                                                                \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32, aicpu_type, aicpu_type, DT_UINT64};         \
        vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};                                         \
        base_type boxes[] = {(base_type)24.664701, (base_type)92.11955,  (base_type)6.000731,  (base_type)50.0787,   \
                             (base_type)83.716705, (base_type)3.6205719, (base_type)9.723194,  (base_type)46.209377, \
                             (base_type)77.11508,  (base_type)74.02599,  (base_type)78.719894, (base_type)31.890612, \
                             (base_type)16.0346,   (base_type)21.073357, (base_type)31.530176, (base_type)54.47457,  \
                             (base_type)45.646084, (base_type)26.065968, (base_type)30.528221, (base_type)97.30633}; \
        vector<base_type> scores = {(base_type)0.2390374, (base_type)0.92039955, (base_type)0.05051243,              \
                                    (base_type)0.49574447, (base_type)0.8355223};                                    \
        int32_t max_output_size = 100;                                                                               \
        base_type iou_threshold(0.001);                                                                              \
        base_type score_threshold(0.2);                                                                              \
        std::unique_ptr<int32_t[]> selected_indices(new int32_t[max_output_size]);                                   \
        int32_t output_expect[2] = {1, 0};                                                                           \
        vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,              \
                               (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};               \
        CREATE_NODEDEF(shapes, data_types, datas);                                                                   \
        uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);                                         \
        EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);                                                           \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                \
        check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);                                  \
        EXPECT_EQ(check_ret, KERNEL_STATUS_OK);                                                                      \
        check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);                                  \
        EXPECT_EQ(check_ret, KERNEL_STATUS_OK);                                                                      \
    }

#define ADD_NULL_CASE(base_type, aicpu_type)                                                                         \
    TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_null)                                             \
    {                                                                                                                \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32, aicpu_type, aicpu_type, DT_UINT64};         \
        vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};                                         \
        base_type boxes[] = {(base_type)24.664701, (base_type)92.11955,  (base_type)6.000731,  (base_type)50.0787,   \
                             (base_type)83.716705, (base_type)3.6205719, (base_type)9.723194,  (base_type)46.209377, \
                             (base_type)77.11508,  (base_type)74.02599,  (base_type)78.719894, (base_type)31.890612, \
                             (base_type)16.0346,   (base_type)21.073357, (base_type)31.530176, (base_type)54.47457,  \
                             (base_type)45.646084, (base_type)26.065968, (base_type)30.528221, (base_type)97.30633}; \
        vector<base_type> scores = {(base_type)0, (base_type)0, (base_type)0, (base_type)0, (base_type)0};           \
        int32_t max_output_size = 100;                                                                               \
        base_type iou_threshold(0.001);                                                                              \
        base_type score_threshold(0.2);                                                                              \
        vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,              \
                               (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};               \
        CREATE_NODEDEF(shapes, data_types, datas);                                                                   \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                \
    }

#define ADD_MEMCOPY_CASE_INVALID_PTR(base_type, aicpu_type)                                                        \
    TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_memcopy_invalid_ptr##aicpu_type)                \
    {                                                                                                              \
        vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64};                                \
        vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};                                                     \
        uint64_t release_flag[2] = {1, 1};                                                                         \
        uint64_t data_size[2] = {0x100, 0x100};                                                                    \
        uint64_t src_pt[2] = {0x80, 0x81};                                                                         \
        uint64_t raw_shape_size = 0x100;                                                                           \
        uint64_t raw_data_size = 0x100;                                                                            \
        void* data_buffer = malloc(raw_data_size);                                                                 \
        void* shape_buffer = malloc(raw_shape_size);                                                               \
        uint64_t dst_ptr[2] = {reinterpret_cast<uint64_t>(shape_buffer), reinterpret_cast<uint64_t>(data_buffer)}; \
        vector<void*> datas = {(void*)release_flag, (void*)data_size, (void*)src_pt, (void*)dst_ptr};              \
        CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas);                                                         \
        RUN_KERNEL(memcpy_node_def, HOST, KERNEL_STATUS_INNER_ERROR);                                              \
        free(data_buffer);                                                                                         \
        free(shape_buffer);                                                                                        \
    }

#define ADD_MEMCOPY_NULL_CASE(base_type, aicpu_type)                                                               \
    TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_memcopy_null)                                   \
    {                                                                                                              \
        vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64};                                \
        vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};                                                     \
        uint64_t release_flag[2] = {1, 1};                                                                         \
        uint64_t data_size[2] = {result_summary[1], result_summary[3]};                                            \
        uint64_t src_pt[2] = {result_summary[0], result_summary[2]};                                               \
        uint64_t raw_shape_size = result_summary[1];                                                               \
        uint64_t raw_data_size = result_summary[3];                                                                \
        void* data_buffer = nullptr;                                                                               \
        void* shape_buffer = malloc(raw_shape_size);                                                               \
        uint64_t dst_ptr[2] = {reinterpret_cast<uint64_t>(shape_buffer), reinterpret_cast<uint64_t>(data_buffer)}; \
        vector<void*> datas = {(void*)release_flag, (void*)data_size, (void*)src_pt, (void*)dst_ptr};              \
        CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas);                                                         \
        RUN_KERNEL(memcpy_node_def, HOST, KERNEL_STATUS_OK);                                                       \
        free(shape_buffer);                                                                                        \
    }

ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(float, DT_FLOAT)
ADD_NULL_CASE(float, DT_FLOAT)
ADD_MEMCOPY_CASE_INVALID_PTR(float, DT_FLOAT)

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_RejectsMixedBoxesScoresDtype)
{
    EXPECT_EQ(RunBoxScoreDtypeCase(DT_FLOAT, DT_FLOAT16), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(RunBoxScoreDtypeCase(DT_FLOAT16, DT_FLOAT), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_RejectsNonScalarControlInputs)
{
    EXPECT_EQ(RunControlShapeCase({2}, {1}, {1}), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(RunControlShapeCase({1}, {2}, {1}), KERNEL_STATUS_PARAM_INVALID);
    EXPECT_EQ(RunControlShapeCase({1}, {1}, {2}), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3noattr)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32, DT_FLOAT16, DT_FLOAT16, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    Eigen::half boxes[] = {
        (Eigen::half)24.664701, (Eigen::half)92.11955,  (Eigen::half)6.000731,  (Eigen::half)50.0787,
        (Eigen::half)83.716705, (Eigen::half)3.6205719, (Eigen::half)9.723194,  (Eigen::half)46.209377,
        (Eigen::half)77.11508,  (Eigen::half)74.02599,  (Eigen::half)78.719894, (Eigen::half)31.890612,
        (Eigen::half)16.0346,   (Eigen::half)21.073357, (Eigen::half)31.530176, (Eigen::half)54.47457,
        (Eigen::half)45.646084, (Eigen::half)26.065968, (Eigen::half)30.528221, (Eigen::half)97.30633};
    vector<Eigen::half> scores = {(Eigen::half)0.2390374, (Eigen::half)0.92039955, (Eigen::half)0.05051243,
                                  (Eigen::half)0.49574447, (Eigen::half)0.8355223};
    int32_t max_output_size = 100;
    Eigen::half iou_threshold(0.001);
    Eigen::half score_threshold(0.2);
    std::unique_ptr<int32_t[]> selected_indices(new int32_t[max_output_size]);
    int32_t output_expect[2] = {1, 0};
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF_WITHATTR(shapes, data_types, datas);
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_Iou1_Score0)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 5;
    float iou_threshold(1);
    float score_threshold(0);
    std::unique_ptr<int32_t[]> selected_indices(new int32_t[max_output_size]);
    int32_t output_expect[2] = {1, 0};
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_IouInvalid)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 5;
    float iou_threshold(1.5);
    float score_threshold(0);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_IouNegative)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 5;
    float iou_threshold(-0.5);
    float score_threshold(0);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_SoftSuppression)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{3, 4}, {3}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)0.0,  (float)0.0,  (float)10.0, (float)10.0, (float)5.0,  (float)5.0,
                     (float)15.0, (float)15.0, (float)8.0,  (float)8.0,  (float)18.0, (float)18.0};
    vector<float> scores = {(float)0.9, (float)0.8, (float)0.7};
    int32_t max_output_size = 3;
    float iou_threshold(0.5);
    float score_threshold(0.5);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_InvalidBoxCoords)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)10.0, (float)10.0, (float)0.0,  (float)0.0,
                     (float)5.0,  (float)5.0,  (float)15.0, (float)15.0};
    vector<float> scores = {(float)0.9, (float)0.8};
    int32_t max_output_size = 2;
    float iou_threshold(0.5);
    float score_threshold(0.0);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_MixedDtypes_Float16_Float)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    Eigen::half boxes[] = {
        (Eigen::half)24.664701, (Eigen::half)92.11955,  (Eigen::half)6.000731,  (Eigen::half)50.0787,
        (Eigen::half)83.716705, (Eigen::half)3.6205719, (Eigen::half)9.723194,  (Eigen::half)46.209377,
        (Eigen::half)77.11508,  (Eigen::half)74.02599,  (Eigen::half)78.719894, (Eigen::half)31.890612,
        (Eigen::half)16.0346,   (Eigen::half)21.073357, (Eigen::half)31.530176, (Eigen::half)54.47457,
        (Eigen::half)45.646084, (Eigen::half)26.065968, (Eigen::half)30.528221, (Eigen::half)97.30633};
    vector<Eigen::half> scores = {(Eigen::half)0.2390374, (Eigen::half)0.92039955, (Eigen::half)0.05051243,
                                  (Eigen::half)0.49574447, (Eigen::half)0.8355223};
    int32_t max_output_size = 100;
    float iou_threshold(0.001);
    float score_threshold(0.2);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_MixedDtypes_Float_Float16)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT16, DT_FLOAT16, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 100;
    Eigen::half iou_threshold(0.001);
    Eigen::half score_threshold(0.2);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_SameScores)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{3, 4}, {3}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)0.0,  (float)0.0,  (float)10.0, (float)10.0, (float)20.0, (float)20.0,
                     (float)30.0, (float)30.0, (float)40.0, (float)40.0, (float)50.0, (float)50.0};
    vector<float> scores = {(float)0.9, (float)0.9, (float)0.9};
    int32_t max_output_size = 3;
    float iou_threshold(0.5);
    float score_threshold(0.0);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

namespace {
vector<int32_t> RunScoreOrderingCase(vector<float> scores)
{
    vector<float> boxes = {0.0F,  0.0F,  1.0F,  1.0F,  10.0F, 10.0F, 11.0F, 11.0F,
                           20.0F, 20.0F, 21.0F, 21.0F, 30.0F, 30.0F, 31.0F, 31.0F};
    int32_t max_output_size = 2;
    float iou_threshold = 0.5F;
    float score_threshold = 0.49F;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{4, 4}, {4}, {}, {}, {}, {-1}};
    vector<void*> datas = {boxes.data(),   scores.data(),    &max_output_size,
                           &iou_threshold, &score_threshold, (void*)result_summary};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3")
        .Input({"boxes", data_types[0], shapes[0], datas[0]})
        .Input({"scores", data_types[1], shapes[1], datas[1]})
        .Input({"max_output_size", data_types[2], shapes[2], datas[2]})
        .Input({"iou_threshold", data_types[3], shapes[3], datas[3]})
        .Input({"score_threshold", data_types[4], shapes[4], datas[4]})
        .Output({"selected_indices", data_types[5], shapes[5], datas[5]});
    std::string node_def_str;
    node_def->SerializeToString(node_def_str);
    CpuKernelContext context(HOST);
    if (context.Init(node_def.get()) != KERNEL_STATUS_OK ||
        CpuKernelRegister::Instance().RunCpuKernel(context) != KERNEL_STATUS_OK || result_summary[2] == 0) {
        return {};
    }
    const auto count = result_summary[3] / sizeof(int32_t);
    const auto* output = reinterpret_cast<const int32_t*>(result_summary[2]);
    vector<int32_t> selected(output, output + count);
    CpuKernelAllocatorUtils::DeleteOutputDataPtr(result_summary[0]);
    CpuKernelAllocatorUtils::DeleteOutputDataPtr(result_summary[2]);
    result_summary[0] = 0;
    result_summary[1] = 0;
    result_summary[2] = 0;
    result_summary[3] = 0;
    return selected;
}
} // namespace

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3StrictScoreOrdering)
{
    const vector<float> close_scores = {0.5F, 0.500004F, 0.500002F, 0.500003F};
    const vector<float> equal_scores = {0.5F, 0.5F, 0.5F, 0.5F};
    EXPECT_EQ(RunScoreOrderingCase(close_scores), (vector<int32_t>{1, 3}));
    EXPECT_EQ(RunScoreOrderingCase(equal_scores), (vector<int32_t>{0, 1}));
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_OffsetAttr)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 100;
    float iou_threshold(0.001);
    float score_threshold(0.2);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3")
        .Input({"boxes", data_types[0], shapes[0], datas[0]})
        .Input({"scores", data_types[1], shapes[1], datas[1]})
        .Input({"max_output_size", data_types[2], shapes[2], datas[2]})
        .Input({"iou_threshold", data_types[3], shapes[3], datas[3]})
        .Input({"score_threshold", data_types[4], shapes[4], datas[4]})
        .Output({"result_summary", data_types[5], shapes[5], datas[5]})
        .Attr("offset", 1);
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
    check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);
}
TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_MixedThresholdDtypes_Float16_Float)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    Eigen::half boxes[] = {
        (Eigen::half)24.664701, (Eigen::half)92.11955,  (Eigen::half)6.000731,  (Eigen::half)50.0787,
        (Eigen::half)83.716705, (Eigen::half)3.6205719, (Eigen::half)9.723194,  (Eigen::half)46.209377,
        (Eigen::half)77.11508,  (Eigen::half)74.02599,  (Eigen::half)78.719894, (Eigen::half)31.890612,
        (Eigen::half)16.0346,   (Eigen::half)21.073357, (Eigen::half)31.530176, (Eigen::half)54.47457,
        (Eigen::half)45.646084, (Eigen::half)26.065968, (Eigen::half)30.528221, (Eigen::half)97.30633};
    vector<Eigen::half> scores = {(Eigen::half)0.2390374, (Eigen::half)0.92039955, (Eigen::half)0.05051243,
                                  (Eigen::half)0.49574447, (Eigen::half)0.8355223};
    int32_t max_output_size = 100;
    Eigen::half iou_threshold(0.001);
    float score_threshold(0.2f);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_MixedThresholdDtypes_Float_Float16)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT16, DT_UINT64};
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};
    float boxes[] = {(float)24.664701, (float)92.11955,  (float)6.000731,  (float)50.0787,   (float)83.716705,
                     (float)3.6205719, (float)9.723194,  (float)46.209377, (float)77.11508,  (float)74.02599,
                     (float)78.719894, (float)31.890612, (float)16.0346,   (float)21.073357, (float)31.530176,
                     (float)54.47457,  (float)45.646084, (float)26.065968, (float)30.528221, (float)97.30633};
    vector<float> scores = {(float)0.2390374, (float)0.92039955, (float)0.05051243, (float)0.49574447,
                            (float)0.8355223};
    int32_t max_output_size = 100;
    float iou_threshold(0.001f);
    Eigen::half score_threshold(0.2);
    vector<void*> datas = {(void*)boxes,          (void*)scores.data(),    (void*)&max_output_size,
                           (void*)&iou_threshold, (void*)&score_threshold, (void*)result_summary};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
