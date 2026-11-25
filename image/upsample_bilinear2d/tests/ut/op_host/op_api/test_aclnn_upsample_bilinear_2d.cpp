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
#include "../../../../op_host/op_api/aclnn_upsample_bilinear_2d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsamplebilinear2d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplebilinear2d_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplebilinear2d_test TearDown" << std::endl; }
};

// 空tensor
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_001)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 0, 0};
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> out = {1, 1, 0, 0};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_002)
{
    const double_t scales_h = 1.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

// 异常路径，float64
// TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_003)
// {
//     const double_t scales_h = 2.0;
//     bool align_corners = false;
//     vector<int64_t> inp_dims = {1, 1, 3, 3};
//     vector<int64_t> output_size = {2, 2};
//     vector<int64_t> out = {1, 1, 3, 3};

//     auto self_desc = TensorDesc(inp_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto output_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

//     auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
//         INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h),  // host api输入
//         OUTPUT(output_desc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
// }

// CheckNotNull_1
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_nullptr_self)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = nullptr;
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// Case 2: outputSize=nullptr 应返回 161001（文档返回码表：空指针场景）
// （修复前该检查在 CheckOutputSize 内随调用点返回 161002 并报 EZ0037，与文档/规范不符）
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_nullptr_output_size)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = nullptr;
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_nullptr_out)
// {
//     const double_t scales_h = 2.0;
//     bool align_corners = false;
//     vector<int64_t> inp_dims = {1, 1, 3, 3};
//     vector<int64_t> output_size = {2, 2};
//     vector<int64_t> out = {1, 1, 3, 3};

//     auto self_desc = TensorDesc(inp_dims, ACL_DOUBLE, ACL_FORMAT_NCHW);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto output_desc = nullptr;

//     auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
//         INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h),  // host api输入
//         OUTPUT(output_desc));

//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
// }

// CheckDtypeEqual
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_dtype)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> out = {1, 1, 3, 3};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckNCDimValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ncdim001)
{
    const double_t scalesH = 1.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 1, 3, 3};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 1, 2, 2};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// CheckNCDimValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ncdim002)
{
    const double_t scalesH = 1.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 2, 3, 3};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 1, 2, 2};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckNCDimValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ncdim003)
{
    const double_t scalesH = 1.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 1, 3, 3};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 1, 3, 3};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckNCDimValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ncdim004)
{
    const double_t scalesH = 1.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 1, 3, 4};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 1, 2, 3};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckScalesValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ScalesValid001)
{
    const double_t scalesH = -2.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 3, 3, 1};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 3, 3, 1};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckScalesValid
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_ScalesValid002)
{
    const double_t scalesH = -2.0;
    const double_t scalesW = 1.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 3, 3, 1};
    vector<int64_t> outputSize = {2, 2};
    vector<int64_t> out = {1, 3, 3, 1};

    auto selfDesc = TensorDesc(inpDims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesW, scalesH), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// double 类型仅支持 NHWC 格式：double + NCHW 应在 CheckFormat 被拦截
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_double_nchw)
{
    const double_t scalesH = 2.0;
    const double_t scalesW = 2.0;
    bool alignCorners = false;
    vector<int64_t> inpDims = {1, 3, 3, 3}; // NCHW: N=1, C=3, H=3, W=3
    vector<int64_t> outputSize = {6, 6};    // 输出 H=6, W=6
    vector<int64_t> out = {1, 3, 6, 6};     // NCHW: N=1, C=3, H=6, W=6

    auto selfDesc = TensorDesc(inpDims, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto outputSizeDesc = IntArrayDesc(outputSize);
    auto outputDesc = TensorDesc(out, ACL_DOUBLE, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesW), // host api输入
                        OUTPUT(outputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// double 类型仅支持 NHWC 格式：double + NHWC 为合法路径，走 AICPU。
// 但 UT 环境无 AICPU kernel 注册（TestGetWorkspaceSize 直连真实 GetWorkspaceSize，
// 无 AICPU mock），l0op::ResizeBilinearV2 的 AICPU 分支 ADD_TO_LAUNCHER_LIST_AICPU
// 必然失败返回 ACLNN_ERR_INNER_NULLPTR(561103)，正向路径只能在真机环境验证。
// 与 upsample_nearest3d/trilinear3d/grid_sampler2d 等 AICPU 路径用例的注释处理保持一致。
// TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_double_nhwc)
// {
//     const double_t scalesH = 2.0;
//     const double_t scalesW = 2.0;
//     bool alignCorners = false;
//     vector<int64_t> inpDims = {1, 3, 3, 1}; // NHWC: N=1, H=3, W=3, C=1
//     vector<int64_t> outputSize = {6, 6};    // 输出 H=6, W=6
//     vector<int64_t> out = {1, 6, 6, 1};     // NHWC: N=1, H=6, W=6, C=1
//
//     auto selfDesc = TensorDesc(inpDims, ACL_DOUBLE, ACL_FORMAT_NHWC);
//     auto outputSizeDesc = IntArrayDesc(outputSize);
//     auto outputDesc = TensorDesc(out, ACL_DOUBLE, ACL_FORMAT_NHWC);
//
//     auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
//                         INPUT(selfDesc, outputSizeDesc, alignCorners, scalesH, scalesW), // host api输入
//                         OUTPUT(outputDesc));
//
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
// }

// 正常路径，float32
TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1001)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1002)
{
    const double_t scales_h = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {0, 0};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1003)
{
    const double_t scales_h = 1.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {6, 6};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1004)
{
    const double_t scales_h = 1.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {0, 0};
    vector<int64_t> out = {1, 1, 6, 6};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_1005)
{
    const double_t scales_h = 100.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {0, 0};
    vector<int64_t> out = {1, 1, 300, 300};

    auto self_desc = TensorDesc(inp_dims, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1006)
{
    const double_t scales_h = 3.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {0, 0};
    vector<int64_t> out = {1, 1, 9, 9};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1007)
{
    const double_t scales_h = 0.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {9, 9};
    vector<int64_t> out = {1, 1, 9, 9};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1008)
{
    const double_t scales_h = 5.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {9, 9};
    vector<int64_t> out = {1, 1, 9, 9};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_h), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1009)
{
    const double_t scales_h = 3.0;
    const double_t scales_w = 5.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 3, 3};
    vector<int64_t> output_size = {9, 9};
    vector<int64_t> out = {1, 1, 9, 9};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1010)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 1, 160, 10};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1011)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 1, 160, 10};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1012)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 10.0;
    bool align_corners = true;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 1};
    vector<int64_t> out = {1, 1, 160, 1};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend910B2_l2_upsamplebilinear2d_test_1013)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = true;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 1, 160, 10};

    auto self_desc = TensorDesc(inp_dims, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend950PR_9599_l2_upsamplebilinear2d_test_001)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = true;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 1, 160, 10};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend950PR_9599_l2_upsamplebilinear2d_test_002)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 80, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 1, 160, 10};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend950PR_9599_l2_upsamplebilinear2d_test_003)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 80, 1, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 160, 10, 1};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_upsamplebilinear2d_test, Ascend950PR_9599_l2_upsamplebilinear2d_test_004)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 10.0;
    bool align_corners = true;
    vector<int64_t> inp_dims = {1, 80, 1, 1};
    vector<int64_t> output_size = {160, 10};
    vector<int64_t> out = {1, 160, 10, 1};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 对标 torch CPU：self 的 H/W 为 0 不支持（torch 报 "Input and output sizes should be greater than 0"）。
// self 为空（H==0 且 W==0）且 out 非空：第一段应返回 ACLNN_ERR_PARAM_INVALID。
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_self_nonempty_out)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {2, 2, 0, 0};
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> out = {2, 2, 4, 4};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 对标 torch CPU：self 的 W 为 0 不支持。self 仅 W==0、out 非空：第一段应返回 ACLNN_ERR_PARAM_INVALID。
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_self_w_zero_nonempty_out)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {1, 1, 4, 0};
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> out = {1, 1, 4, 4};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// N==0 时 out 也为空：没有元素可写，早退成功且不需要 workspace
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_out)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {0, 1, 3, 3};
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> out = {0, 1, 2, 2};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

// 对标 torch CPU：self 的 C 为 0 不支持（torch 报 "Non-empty 4D data tensor expected"）。
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_self_c_zero)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {2, 0, 4, 4};
    vector<int64_t> output_size = {8, 8};
    vector<int64_t> out = {2, 0, 8, 8};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 对标 torch CPU：输出 H 为 0 不支持（torch 报 "Input and output sizes should be greater than 0"）。
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_out_h_zero)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {2, 3, 4, 4};
    vector<int64_t> output_size = {0, 8};
    vector<int64_t> out = {2, 3, 0, 8};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 对标 torch CPU：输出 W 为 0 不支持（torch 报 "Input and output sizes should be greater than 0"）。
TEST_F(l2_upsamplebilinear2d_test, l2_upsamplebilinear2d_test_empty_out_w_zero)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> inp_dims = {2, 3, 4, 4};
    vector<int64_t> output_size = {8, 0};
    vector<int64_t> out = {2, 3, 8, 0};

    auto self_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2d,
                        INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w), // host api输入
                        OUTPUT(output_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
