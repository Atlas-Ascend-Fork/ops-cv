/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scharr.cpp
 * \brief Scharr kernel CPU 仿真 UT：经 helper 调用真实 tiling 函数生成 tilingData，
 *        再 ICPU_RUN_KF 执行 kernel，与 numpy golden（OpenCV float32 语义的位级复刻）逐位对拍。
 *        平台常量按 910B4：40 核 / UB 192KB。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tikicpulib.h"
#include "data_utils.h"

using uint8 = uint8_t;
using float32 = float;

// Kernel UT builds one entry per dtype inside this single target: the default
// entry uses DTYPE_INPUT from the build options (-DDTYPE_INPUT=uint8, matching
// CI); the float16/float32 entries are produced by re-including the production
// kernel with the type macro rebound and the entry symbol renamed, so every
// dtype case runs in the same executable (the UT infra allows only one target
// per op and SoC).
#ifndef DTYPE_INPUT
#define DTYPE_INPUT uint8
#endif
#include "../../../op_kernel/scharr.cpp"

#undef DTYPE_INPUT
#define DTYPE_INPUT half
#define scharr scharr_fp16
#include "../../../op_kernel/scharr.cpp"
#undef scharr
#undef DTYPE_INPUT

#define DTYPE_INPUT float
#define scharr scharr_fp32
#include "../../../op_kernel/scharr.cpp"
#undef scharr
#undef DTYPE_INPUT

// tiling 由独立 TU 调用真实 TilingFunc 生成（避免 host DT_* 宏与 kernel 头冲突）
#include "scharr_ut_helper.h"

static constexpr size_t UT_WORKSPACE_BYTES = 1U << 20;

class ScharrKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScharrKernel SetUp" << std::endl;
        system("cp -rf ../../../../experimental/image/scharr/tests/ut/op_kernel/scharr_data ./");
        system("chmod -R 755 ./scharr_data/");
    }

    static void TearDownTestCase() { std::cout << "ScharrKernel TearDown" << std::endl; }

    static std::vector<int64_t> ParseShape(const std::string& shapeStr)
    {
        std::vector<int64_t> dims;
        std::string cur;
        for (char ch : shapeStr) {
            if (ch == ',') {
                dims.push_back(std::stoll(cur));
                cur.clear();
            } else {
                cur += ch;
            }
        }
        dims.push_back(std::stoll(cur));
        return dims;
    }

    // dtypeId: 0=uint8 1=float16 2=float32; dirIdx: 0=X 1=Y; borderIdx: 0=REFLECT_101 1=REPLICATE 2=CONSTANT
    static void RunOneCase(const std::string& shapeStr, const std::string& dtypeCmd, int dtypeId, size_t elemBytes,
                           int dirIdx, int borderIdx)
    {
        static const char* DIRS[2] = {"X", "Y"};
        static const char* BORDERS[3] = {"REFLECT_101", "REPLICATE", "CONSTANT"};
        const std::vector<int64_t> shape = ParseShape(shapeStr);
        size_t elems = 1;
        for (int64_t d : shape) {
            elems *= static_cast<size_t>(d);
        }
        const size_t xByteSize = elems * elemBytes;
        const size_t zByteSize = elems * sizeof(float);

        // 1) 真实 tiling（独立 TU）
        ScharrUt::TilingResult tr = ScharrUt::RunTiling(shape, dtypeId, dirIdx, borderIdx);
        ASSERT_TRUE(tr.ok) << "ExecuteTiling failed";
        std::cout << "[Case " << shapeStr << " " << dtypeCmd << " " << DIRS[dirIdx] << " " << BORDERS[borderIdx]
                  << "] key=" << tr.tilingKey << " blocks=" << tr.blockNum << " tilingBytes=" << tr.tilingData.size()
                  << std::endl;

        // 2) 生成输入 + golden
        std::string genCmd = "cd ./scharr_data/ && python3 gen_data.py --shape " + shapeStr + " --dtype " + dtypeCmd +
                             " --direction " + DIRS[dirIdx] + " --border-type " + BORDERS[borderIdx] + " --seed 42";
        int ret = system(genCmd.c_str());
        ASSERT_EQ(ret, 0) << "gen_data.py 执行失败：" << genCmd;
        if (ret != 0) {
            return;
        }

        // 3) 申请内存并读入输入
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
        uint8_t* out = (uint8_t*)AscendC::GmAlloc(zByteSize);
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(UT_WORKSPACE_BYTES);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tr.tilingData.size());
        size_t fileLen = xByteSize;
        ASSERT_TRUE(ReadFile("./scharr_data/input/input_scharr.bin", fileLen, x, xByteSize));
        std::memcpy(tiling, tr.tilingData.data(), tr.tilingData.size());

        // 4) 执行 kernel
        ICPU_SET_TILING_KEY(tr.tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if (dtypeId == 0) {
            ICPU_RUN_KF(scharr, tr.blockNum, x, out, workspace, tiling);
        } else if (dtypeId == 1) {
            ICPU_RUN_KF(scharr_fp16, tr.blockNum, x, out, workspace, tiling);
        } else {
            ICPU_RUN_KF(scharr_fp32, tr.blockNum, x, out, workspace, tiling);
        }

        // 5) 写输出并逐位对拍
        WriteFile("./scharr_data/output.bin", out, zByteSize);
        ret = system("cd ./scharr_data/ && python3 compare_data.py");
        EXPECT_EQ(WEXITSTATUS(ret), 0) << "golden 对拍失败！shape=" << shapeStr;

        AscendC::GmFree(x);
        AscendC::GmFree(out);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

// ============================ uint8 cases ============================
TEST_F(ScharrKernel, test_case_2d_65x64_x_reflect101) { RunOneCase("65,64", "uint8", 0, 1, 0, 0); }
TEST_F(ScharrKernel, test_case_2d_63x65_y_replicate_oddw) { RunOneCase("63,65", "uint8", 0, 1, 1, 1); }
TEST_F(ScharrKernel, test_case_2d_10x130_x_constant) { RunOneCase("10,130", "uint8", 0, 1, 0, 2); }
TEST_F(ScharrKernel, test_case_3d_batch_y_reflect101) { RunOneCase("2,33,45", "uint8", 0, 1, 1, 0); }
TEST_F(ScharrKernel, test_case_fallback_w2) { RunOneCase("16,2", "uint8", 0, 1, 0, 2); }
TEST_F(ScharrKernel, u8_2d_8x2048_x_reflect101_mode1) { RunOneCase("8,2048", "uint8", 0, 1, 0, 0); }

// ============================ float32 cases ============================
TEST_F(ScharrKernel, fp32_2d_65x64_x_reflect101) { RunOneCase("65,64", "float32", 2, 4, 0, 0); }
TEST_F(ScharrKernel, fp32_2d_63x65_y_replicate_oddw) { RunOneCase("63,65", "float32", 2, 4, 1, 1); }
TEST_F(ScharrKernel, fp32_3d_batch_x_constant) { RunOneCase("2,33,45", "float32", 2, 4, 0, 2); }
TEST_F(ScharrKernel, fp32_2d_10x4102_band) { RunOneCase("10,4102", "float32", 2, 4, 1, 0); }
// mode1 (pixel-range Gather) regression: workCoreNum > rowCoreNum shapes must
// keep the identity offset table (tileH==0) end to end.
TEST_F(ScharrKernel, fp32_2d_3x512_x_reflect101_mode1) { RunOneCase("3,512", "float32", 2, 4, 0, 0); }

// ============================ float16 cases ============================
TEST_F(ScharrKernel, fp16_2d_64x127_x_reflect101) { RunOneCase("64,127", "float16", 1, 2, 0, 0); }
TEST_F(ScharrKernel, fp16_3d_batch_y_constant) { RunOneCase("2,31,63", "float16", 1, 2, 1, 2); }
