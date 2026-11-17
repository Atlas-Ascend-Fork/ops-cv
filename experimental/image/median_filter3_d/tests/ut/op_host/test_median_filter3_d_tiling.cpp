/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "median_filter3_d_test_helpers.h"
TEST(MedianFilter3DTiling, Types)
{
    TileCase({2, 3, 5}, ge::DT_UINT8, true);
    TileCase({2, 2, 3, 5}, ge::DT_FLOAT, true);
    TileCase({2, 3, 2, 3, 5}, ge::DT_FLOAT16, true);
}
TEST(MedianFilter3DTiling, EvenAndMaxWindow)
{
    TileCase({1, 1, 1}, ge::DT_FLOAT, true, {2, 4, 1});
    TileCase({1, 1, 1}, ge::DT_FLOAT, true, {1, 1, 1024});
}
TEST(MedianFilter3DTiling, InvalidShapeAndDtype)
{
    TileCase({0, 2, 3}, ge::DT_FLOAT, false);
    TileCase({-1, 2, 3}, ge::DT_FLOAT, false);
    TileCase({2, 3, 4}, ge::DT_INT32, false);
}
TEST(MedianFilter3DTiling, InvalidAttrs)
{
    TileCase({2, 3, 4}, ge::DT_FLOAT, false, {3, 3, -1});
    TileCase({2, 3, 4}, ge::DT_FLOAT, false, {3, 3, 3}, "MIRROR");
}
TEST(MedianFilter3DTiling, CvalRange)
{
    TileCase({1, 2, 3}, ge::DT_UINT8, true, {2, 2, 2}, "CONSTANT", 0);
    TileCase({1, 2, 3}, ge::DT_UINT8, true, {2, 2, 2}, "CONSTANT", 7.5f);
    TileCase({1, 2, 3}, ge::DT_UINT8, true, {2, 2, 2}, "CONSTANT", 255);
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", -1);
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", 256);
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", 300);
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", std::numeric_limits<float>::infinity());
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", -std::numeric_limits<float>::infinity());
    TileCase({1, 2, 3}, ge::DT_UINT8, false, {2, 2, 2}, "CONSTANT", std::numeric_limits<float>::quiet_NaN());
    TileCase({1, 2, 3}, ge::DT_UINT8, true, {2, 2, 2}, "REFLECT", 300);
    TileCase({1, 2, 3}, ge::DT_FLOAT, true, {2, 2, 2}, "CONSTANT", 300);
    TileCase({1, 2, 3}, ge::DT_FLOAT16, true, {2, 2, 2}, "CONSTANT", 300);
    TileCase({1, 2, 3}, ge::DT_FLOAT, true, {2, 2, 2}, "CONSTANT", std::numeric_limits<float>::infinity());
}
