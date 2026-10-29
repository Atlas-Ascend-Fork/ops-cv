/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_resize_upsample_trilinear_arch35.cpp
 * \brief ResizeUpsampleTrilinear A950 SIMT kernel UT - direct Simt class invocation
 */

#ifdef UT_ARCH35

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

// Directly include SIMT class (bypass apt.cpp template macros)
#include "../../../op_kernel/arch35/resize_upsample_trilinear_simt.h"
#include "../../../op_kernel/arch35/resize_upsample_trilinear_tiling_data.h"

using namespace std;
using namespace AscendC;
using namespace ResizeUpsampleTrilinear;

class ResizeUpsampleTrilinearArch35Test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ResizeUpsampleTrilinearArch35Test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "ResizeUpsampleTrilinearArch35Test TearDown\n" << endl; }
};

// Manually construct arch35 tiling data for single-core UT execution
// row-based tiling: totalRows = N*C*outD*outH, 1 core processes all rows
static void BuildArch35TilingData(ResizeUpsampleTrilinearRegBaseTilingData& td, int64_t inD, int64_t inH, int64_t inW,
                                  int64_t outD, int64_t outH, int64_t outW, int64_t nc, int32_t alignCorners)
{
    td.inD = inD;
    td.inH = inH;
    td.inW = inW;
    td.outD = outD;
    td.outH = outH;
    td.outW = outW;
    td.alignCorners = alignCorners;

    // scale = input / output (alignCorners=false, scale attr=0 -> use input/output)
    td.scaleD = (outD == inD) ? 1.0f : static_cast<float>(inD) / static_cast<float>(outD);
    td.scaleH = (outH == inH) ? 1.0f : static_cast<float>(inH) / static_cast<float>(outH);
    td.scaleW = (outW == inW) ? 1.0f : static_cast<float>(inW) / static_cast<float>(outW);

    // row-based tiling: 1 core, all rows
    int64_t totalRows = nc * outD * outH;
    td.blkProcessNum = totalRows; // 1 core handles all rows
    td.tailBlockNum = 0;
    td.coreNum = 1; // single core for UT
}

TEST_F(ResizeUpsampleTrilinearArch35Test, test_case_float32)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    // Shape: [1,2,2,4,4] -> [1,2,8,8,16]
    const int64_t inD = 2, inH = 4, inW = 4;
    const int64_t outD = 8, outH = 8, outW = 16;
    const int64_t nc = 2; // N*C
    const int32_t alignCorners = 0;

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float32'");

    size_t inputByteSize = nc * inD * inH * inW * sizeof(float);
    size_t outputByteSize = nc * outD * outH * outW * sizeof(float);

    uint8_t* x = (uint8_t*)GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(outputByteSize);
    uint8_t* tiling = (uint8_t*)GmAlloc(sizeof(ResizeUpsampleTrilinearRegBaseTilingData));

    std::string fileName = "./resize_upsample_trilinear_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    ResizeUpsampleTrilinearRegBaseTilingData* td = reinterpret_cast<ResizeUpsampleTrilinearRegBaseTilingData*>(tiling);
    BuildArch35TilingData(*td, inD, inH, inW, outD, outH, outW, nc, alignCorners);

    // Directly instantiate SIMT class (bypass apt.cpp template macros)
    // Single core (block 0), uint32_t index since shape fits in uint32
    ResizeUpsampleTrilinearSimt<float, uint32_t> op;
    op.Init(x, y, td);
    op.Process();

    fileName = "./resize_upsample_trilinear_data/float32_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    GmFree(x);
    GmFree(y);
    GmFree(tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float32'");
}

TEST_F(ResizeUpsampleTrilinearArch35Test, test_case_float16)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const int64_t inD = 2, inH = 4, inW = 4;
    const int64_t outD = 8, outH = 8, outW = 16;
    const int64_t nc = 2;
    const int32_t alignCorners = 0;

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float16'");

    size_t inputByteSize = nc * inD * inH * inW * sizeof(half);
    size_t outputByteSize = nc * outD * outH * outW * sizeof(half);

    uint8_t* x = (uint8_t*)GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(outputByteSize);
    uint8_t* tiling = (uint8_t*)GmAlloc(sizeof(ResizeUpsampleTrilinearRegBaseTilingData));

    std::string fileName = "./resize_upsample_trilinear_data/float16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    ResizeUpsampleTrilinearRegBaseTilingData* td = reinterpret_cast<ResizeUpsampleTrilinearRegBaseTilingData*>(tiling);
    BuildArch35TilingData(*td, inD, inH, inW, outD, outH, outW, nc, alignCorners);

    ResizeUpsampleTrilinearSimt<half, uint32_t> op;
    op.Init(x, y, td);
    op.Process();

    fileName = "./resize_upsample_trilinear_data/float16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    GmFree(x);
    GmFree(y);
    GmFree(tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float16'");
}

TEST_F(ResizeUpsampleTrilinearArch35Test, test_case_bfloat16)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const int64_t inD = 2, inH = 4, inW = 4;
    const int64_t outD = 8, outH = 8, outW = 16;
    const int64_t nc = 2;
    const int32_t alignCorners = 0;

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'bfloat16'");

    size_t inputByteSize = nc * inD * inH * inW * sizeof(bfloat16_t);
    size_t outputByteSize = nc * outD * outH * outW * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(outputByteSize);
    uint8_t* tiling = (uint8_t*)GmAlloc(sizeof(ResizeUpsampleTrilinearRegBaseTilingData));

    std::string fileName = "./resize_upsample_trilinear_data/bfloat16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    ResizeUpsampleTrilinearRegBaseTilingData* td = reinterpret_cast<ResizeUpsampleTrilinearRegBaseTilingData*>(tiling);
    BuildArch35TilingData(*td, inD, inH, inW, outD, outH, outW, nc, alignCorners);

    ResizeUpsampleTrilinearSimt<bfloat16_t, uint32_t> op;
    op.Init(x, y, td);
    op.Process();

    fileName = "./resize_upsample_trilinear_data/bfloat16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    GmFree(x);
    GmFree(y);
    GmFree(tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float16'");
}

#endif // UT_ARCH35
