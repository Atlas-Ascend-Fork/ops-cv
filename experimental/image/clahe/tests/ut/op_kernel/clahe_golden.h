/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CLAHE_UT_GOLDEN_H
#define CLAHE_UT_GOLDEN_H
#include <array>
#include <cstdint>

// Frozen with OpenCV 4.13.0 createCLAHE on aarch64; the UT has no OpenCV dependency.
// Each pair below records the complete input and exact integer output.
namespace ClaheGolden {
// 7x9, clip_limit=2, tile_grid_size=[3,2].
constexpr std::array<uint8_t, 63> UINT8_REFLECT_INPUT = {
    0,   127, 2,   255, 31,  1,  128, 7,   0,   128, 7,   0,   127, 2,   255, 31, 1,   128, 31,  1,   128,
    7,   0,   127, 2,   255, 31, 2,   255, 31,  1,   128, 7,   0,   127, 2,   0,  127, 2,   255, 31,  1,
    128, 7,   0,   128, 7,   0,  127, 2,   255, 31,  1,   128, 31,  1,   128, 7,  0,   127, 2,   255, 31,
};
constexpr std::array<uint8_t, 63> UINT8_REFLECT_OUTPUT = {
    32,  159, 64,  255, 96,  48, 191, 80,  32,  191, 80,  32,  159, 64,  255, 96, 48,  191, 96,  48,  191,
    80,  32,  159, 64,  255, 96, 64,  255, 96,  48,  191, 80,  32,  159, 64,  32, 159, 64,  255, 96,  48,
    191, 80,  32,  191, 80,  32, 159, 64,  255, 96,  48,  191, 96,  48,  191, 80, 32,  159, 64,  255, 96,
};
// 3x4, clip_limit=0, tile_grid_size=[2,1].
constexpr std::array<uint8_t, 12> UINT8_HALF_INPUT = {
    0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255,
};
constexpr std::array<uint8_t, 12> UINT8_HALF_OUTPUT = {
    128, 255, 106, 255, 128, 255, 106, 255, 128, 255, 255, 255,
};
// 5x7, clip_limit=40, tile_grid_size=[1,1].
constexpr std::array<uint16_t, 35> UINT16_REFLECT_INPUT = {
    0,   32767, 0, 8191,  65535, 255,  32768, 0,   32767, 0, 8191,  65535, 255,  32768, 0,   32767, 0, 8191, 65535,
    255, 32768, 0, 32767, 0,     8191, 65535, 255, 32768, 0, 32767, 0,     8191, 65535, 255, 32768,
};
constexpr std::array<uint16_t, 35> UINT16_REFLECT_OUTPUT = {
    3745,  35576, 3745, 13107, 65535, 5617,  37449, 3745,  35576, 3745, 13107, 65535,
    5617,  37449, 3745, 35576, 3745,  13107, 65535, 5617,  37449, 3745, 35576, 3745,
    13107, 65535, 5617, 37449, 3745,  35576, 3745,  13107, 65535, 5617, 37449,
};
// 3x4, clip_limit=0, tile_grid_size=[2,1].
constexpr std::array<uint16_t, 12> UINT16_HALF_INPUT = {
    0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 65535, 65535,
};
constexpr std::array<uint16_t, 12> UINT16_HALF_OUTPUT = {
    32768, 65535, 27306, 65535, 32768, 65535, 27306, 65535, 32768, 65535, 65535, 65535,
};
} // namespace ClaheGolden
#endif
