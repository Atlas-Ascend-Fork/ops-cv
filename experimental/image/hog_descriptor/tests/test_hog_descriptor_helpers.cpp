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
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "../examples/hog_descriptor_fp16.h"
#include "../op_host/hog_descriptor_ub.h"

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--dump-half") == 0) {
#ifdef _WIN32
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
            return 1;
        }
#endif
        for (uint32_t bits = 0; bits < 65536U; ++bits) {
            const float value = HogExample::Fp16ToFp32(static_cast<uint16_t>(bits));
            if (std::fwrite(&value, sizeof(value), 1, stdout) != 1) {
                return 1;
            }
        }
        return 0;
    }
    for (uint32_t bits = 0; bits < 65536U; ++bits) {
        const auto half = static_cast<uint16_t>(bits);
        const float value = HogExample::Fp16ToFp32(half);
        const auto roundTrip = HogExample::Fp32ToFp16(value);
        const bool isNan = (half & 0x7c00U) == 0x7c00U && (half & 0x03ffU) != 0U;
        const bool encodedNan = (roundTrip & 0x7c00U) == 0x7c00U && (roundTrip & 0x03ffU) != 0U;
        if ((!isNan && roundTrip != half) || (isNan && (!std::isnan(value) || !encodedNan))) {
            std::fprintf(stderr, "FP16 round-trip failed at 0x%04x\n", bits);
            return 1;
        }
    }
    if (HogExample::Fp16ToFp32(1) != 0x1p-24f || HogExample::Fp16ToFp32(0x8001) != -0x1p-24f ||
        HogExample::Fp16ToFp32(0x03ff) != 1023.0f * 0x1p-24f || HogExample::Fp32ToFp16(1.0f + 0x1p-11f) != 0x3c00U ||
        HogExample::Fp32ToFp16(1.0f + 3.0f * 0x1p-11f) != 0x3c02U) {
        std::fprintf(stderr, "FP16 boundary or round-to-even check failed\n");
        return 1;
    }
    const uint64_t ub = 192U * 1024U;
    const uint64_t fixed = optiling::HogFixedUbBytes(512, 3, 4, 64, 40, 8, 8);
    const uint64_t row = 64U * 40U * sizeof(float);
    if (fixed + 10U * row <= ub || fixed + 9U * row > ub ||
        optiling::HogFixedUbBytes(512, 3, 2, 64, 40, 8, 8) >= fixed) {
        std::fprintf(stderr, "UB boundary regression failed\n");
        return 1;
    }
    std::puts("PASS: 65536 FP16 round trips, rounding boundaries, RGB/block UB budget");
    return 0;
}
