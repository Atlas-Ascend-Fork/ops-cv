/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

namespace aicpu {
namespace {
class CropAndResizeBilinearV2Test : public testing::Test {
protected:
    void ExpectCrop(int64_t imageHeight, int64_t imageWidth, int64_t channels, std::vector<float> image,
                    std::vector<float> box, int32_t cropHeight, int32_t cropWidth, const std::vector<float>& expected)
    {
        ASSERT_EQ(image.size(), static_cast<size_t>(imageHeight * imageWidth * channels));
        ASSERT_EQ(box.size(), 4U);
        ASSERT_EQ(expected.size(), static_cast<size_t>(cropHeight * cropWidth * channels));
        int32_t boxIndex = 0;
        std::vector<int32_t> cropSize = {cropHeight, cropWidth};
        std::vector<float> output(expected.size(), 1234.0F);
        auto nodeDef = CpuKernelUtils::CreateNodeDef();
        NodeDefBuilder(nodeDef.get(), "CropAndResize", "CropAndResize")
            .Input({"x", DT_FLOAT, {1, imageHeight, imageWidth, channels}, image.data()})
            .Input({"boxes", DT_FLOAT, {1, 4}, box.data()})
            .Input({"box_index", DT_INT32, {1}, &boxIndex})
            .Input({"crop_size", DT_INT32, {2}, cropSize.data()})
            .Output({"y", DT_FLOAT, {1, cropHeight, cropWidth, channels}, output.data()})
            .Attr("method", std::string("bilinear_v2"))
            .Attr("extrapolation_value", -1.0F);
        CpuKernelContext context(DEVICE);
        ASSERT_EQ(context.Init(nodeDef.get()), KERNEL_STATUS_OK);
        ASSERT_EQ(CpuKernelRegister::Instance().RunCpuKernel(context), KERNEL_STATUS_OK);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(output[i], expected[i]) << "output index " << i;
        }
    }
};

TEST_F(CropAndResizeBilinearV2Test, NegativeFractionalStartUsesFloor)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {-0.1F, 0, 0.55F, 0}, 1, 1, {5});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, -0.1F, 0, 0.55F}, 1, 1, {5});
}

TEST_F(CropAndResizeBilinearV2Test, NegativeStartExtrapolatesBeforeImageRead)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {-0.1F, 0, 0.55F, 0}, 4, 1, {-1, 0, 10, 20});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, -0.1F, 0, 0.55F}, 1, 4, {-1, 0, 10, 20});
}

TEST_F(CropAndResizeBilinearV2Test, FractionalSampleOutsideImageUsesExtrapolation)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {-0.1F, 0, 0.55F, 0}, 2, 1, {-1, 15});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, -0.1F, 0, 0.55F}, 1, 2, {-1, 15});
}

TEST_F(CropAndResizeBilinearV2Test, FullFrameEndpointMapsToLastPixel)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {0, 0, 1, 0}, 5, 1, {0, 7, 15, 23, 30});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 0, 0, 1}, 1, 5, {0, 7, 15, 23, 30});
}

TEST_F(CropAndResizeBilinearV2Test, FullyOutsideWindowUsesExtrapolation)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {-0.5F, 0, -0.25F, 0}, 2, 1, {-1, -1});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 1.25F, 0, 1.5F}, 1, 2, {-1, -1});
}

TEST_F(CropAndResizeBilinearV2Test, ExtrapolationFillsEveryChannel)
{
    ExpectCrop(4, 1, 2, {0, 1, 10, 11, 20, 21, 30, 31}, {-0.1F, 0, 0.55F, 0}, 4, 1, {-1, -1, 0, 1, 10, 11, 20, 21});
}

TEST_F(CropAndResizeBilinearV2Test, ForwardInterpolationRemainsUnchanged)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {0, 0, 0.75F, 0}, 2, 1, {5, 25});
    ExpectCrop(1, 1, 1, {7}, {0, 0, 0, 0}, 2, 2, {7, 7, 7, 7});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedBoxFlipsBothAxes)
{
    ExpectCrop(4, 4, 1, {1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34}, {0.75F, 0.75F, 0.25F, 0.25F}, 3,
               3, {34, 33, 32, 24, 23, 22, 14, 13, 12});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedSingleAxisPreservesInterpolation)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {0.75F, 0, 0, 0}, 2, 1, {25, 5});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 0.75F, 0, 0}, 1, 2, {25, 5});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedWindowExtrapolatesPastNegativeEndpoint)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {0.75F, 0, -0.25F, 0}, 5, 1, {30, 20, 10, 0, -1});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 0.75F, 0, -0.25F}, 1, 5, {30, 20, 10, 0, -1});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedFullFrameEndpointMapsToLastPixel)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {1, 0, 0, 0}, 5, 1, {30, 23, 15, 7, 0});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 1, 0, 0}, 1, 5, {30, 23, 15, 7, 0});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedWindowStartingOutsideImageUsesExtrapolation)
{
    ExpectCrop(4, 1, 1, {0, 10, 20, 30}, {1.25F, 0, 0, 0}, 6, 1, {-1, -1, 30, 20, 10, 0});
    ExpectCrop(1, 4, 1, {0, 10, 20, 30}, {0, 1.25F, 0, 0}, 1, 6, {-1, -1, 30, 20, 10, 0});
}

TEST_F(CropAndResizeBilinearV2Test, ReversedFractionalEndpointUsesFloorForEveryChannel)
{
    ExpectCrop(4, 1, 2, {0, 1, 10, 11, 20, 21, 30, 31}, {0.55F, 0, -0.1F, 0}, 4, 1, {20, 21, 10, 11, 0, 1, -1, -1});
}
} // namespace
} // namespace aicpu
