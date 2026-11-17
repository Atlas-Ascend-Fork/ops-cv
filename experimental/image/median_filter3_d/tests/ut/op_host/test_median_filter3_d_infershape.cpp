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
TEST(MedianFilter3DShape, RanksAndDynamic)
{
    ShapeCase({2, 3, 5}, true);
    ShapeCase({2, 2, 3, 5}, true);
    ShapeCase({2, 3, -1, 4, 5}, true);
}
TEST(MedianFilter3DShape, UnknownRank)
{
    ShapeCase({ge::UNKNOWN_RANK[0]}, true);
    ShapeCase({ge::UNKNOWN_RANK[0]}, false, {0, 3, 3});
}
TEST(MedianFilter3DShape, InvalidRanks)
{
    ShapeCase({2, 3}, false);
    ShapeCase({1, 1, 1, 1, 1, 1}, false);
}
TEST(MedianFilter3DShape, ZeroAndOverflow)
{
    ShapeCase({0, 3, 4}, false);
    ShapeCase({2147483647, 2, 3}, false);
}
TEST(MedianFilter3DShape, EvenAndAnisotropic) { ShapeCase({1, 2, 3}, true, {2, 4, 1}); }
TEST(MedianFilter3DShape, InvalidWindow)
{
    ShapeCase({2, 3, 4}, false, {0, 3, 3});
    ShapeCase({2, 3, 4}, false, {3, 3});
    ShapeCase({2, 3, 4}, false, {11, 11, 11});
}
TEST(MedianFilter3DShape, InvalidModeAndNan)
{
    ShapeCase({2, 3, 4}, false, {3, 3, 3}, "WRAP");
    ShapeCase({2, 3, 4}, false, {3, 3, 3}, "CONSTANT", std::numeric_limits<float>::quiet_NaN());
}
