/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTIVE_THRESHOLD_GAUSSIAN_TEST_DATA_H_
#define ADAPTIVE_THRESHOLD_GAUSSIAN_TEST_DATA_H_

#include <cstdint>
#include <vector>

namespace AdaptiveThresholdTest {

// Frozen Q16 coefficients, independent of the production weight builder.
// Sizes 3/5/7/9 use the exact small Gaussian kernels. Larger sizes use
// sigma = 0.3 * (radius - 1) + 0.8, normalized and rounded to Q16.
// The center restores the total to 65536: size 31 needs 5240 (not 5239),
// and size 255 needs 666 (not 678). Keep these literal regression expectations.
inline std::vector<uint32_t> GaussianWeights(uint32_t blockSize)
{
    switch (blockSize) {
        case 3:
            return {16384, 32768, 16384};
        case 5:
            return {4096, 16384, 24576, 16384, 4096};
        case 7:
            return {2048, 7168, 14336, 18432, 14336, 7168, 2048};
        case 9:
            return {1024, 3328, 7680, 13056, 15360, 13056, 7680, 3328, 1024};
        case 11:
            return {578, 1779, 4267, 7972, 11600, 13144, 11600, 7972, 4267, 1779, 578};
        case 31:
            return {58,   104,  178,  294,  466,  709,  1037, 1457, 1966, 2550, 3178, 3804, 4376, 4836, 5135, 5240,
                    5135, 4836, 4376, 3804, 3178, 2550, 1966, 1457, 1037, 709,  466,  294,  178,  104,  58};
        case 255: {
            // Left half including the center; only reflection is done at runtime.
            std::vector<uint32_t> weights = {
                3,   3,   4,   4,   4,   5,   5,   5,   6,   6,   7,   7,   8,   9,   9,   10,  11,  12,  13,
                14,  15,  16,  17,  18,  19,  21,  22,  24,  25,  27,  29,  31,  33,  35,  37,  40,  42,  45,
                48,  50,  53,  57,  60,  64,  67,  71,  75,  79,  83,  88,  93,  98,  103, 108, 113, 119, 125,
                131, 137, 144, 150, 157, 164, 172, 179, 187, 194, 203, 211, 219, 228, 237, 246, 255, 264, 274,
                283, 293, 303, 313, 323, 333, 344, 354, 365, 375, 386, 396, 407, 418, 428, 439, 449, 460, 470,
                481, 491, 501, 511, 521, 531, 540, 550, 559, 568, 576, 585, 593, 601, 608, 615, 622, 629, 635,
                641, 646, 651, 656, 660, 664, 667, 670, 672, 674, 676, 677, 678, 666};
            weights.resize(255);
            for (uint32_t index = 0; index < 127; ++index) {
                weights[254 - index] = weights[index];
            }
            return weights;
        }
        default:
            return {};
    }
}

} // namespace AdaptiveThresholdTest

#endif // ADAPTIVE_THRESHOLD_GAUSSIAN_TEST_DATA_H_
