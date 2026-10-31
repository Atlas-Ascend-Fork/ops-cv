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
 * \file test_integral_image.cpp
 * \brief IntegralImage op_kernel UT（tikicpulib CPU 仿真）
 *
 * 覆盖点：
 *   - 2D UINT8 legacy 分支（W < 128）：sat / sqsum / tilted 三路输出全部比对
 *   - 2D FLOAT32 legacy 分支：CastVec2D + 向量前缀扫描路径
 *   - 3D (H,W,C) HWC legacy 分支：逐通道独立累加，C 不要求 32 字节对齐
 *   - 2D UINT8 two-phase 分支（W >= 128）：workspace 行前缀 + 列向累加 + 可选输出
 *   - 形态 1/2（sqsum / tilted 未使能）：可选分支跳过，未使能输出按 nullptr 下发
 *   - FLOAT16 输入（tilingKey=1）与 UINT8 + sdepth=1（tilingKey=3）模板
 *   - 3D HWC C=9：dstSqBatch 交错快照按 blockWidth*C 分配
 *
 * golden 由 integral_image_data/gen_data.py 按接口数学定义计算，compare_data.py 逐元素比对：
 * 整型输出要求完全一致，浮点输出按 dtype 量化误差给出容差。
 * CPU 仿真使用单核配置（activeCoreNum = 1）；examples 真机用例覆盖多核同步。
 */

#include "../../../op_kernel/integral_image.cpp"
#include "../../../examples/integral_image_test_utils.h"
#include "integral_image_tiling.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

namespace {

// 运行期数据目录（相对 build 目录），输入/golden/输出都放在这里
const string kRunDir = "./integral_image_data";

// UT 数据脚本所在目录（源码目录）：从当前工作目录向上回溯定位，避免依赖编译路径
string SourceDataDir()
{
    const string suffix = "experimental/image/integral_image/tests/ut/op_kernel/integral_image_data";
    char cwdBuf[4096] = {0};
    string cwd = (getcwd(cwdBuf, sizeof(cwdBuf)) != nullptr) ? string(cwdBuf) : string(".");
    string dir = cwd;
    for (int i = 0; i < 16; i++) {
        const string candidate = dir + "/" + suffix;
        struct stat st;
        if (stat(candidate.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            return candidate;
        }
        const size_t pos = dir.find_last_of('/');
        if (pos == string::npos || pos == 0) {
            break;
        }
        dir = dir.substr(0, pos);
    }
    return cwd + "/integral_image_data";
}

bool RunCmd(const string& cmd)
{
    const int ret = system(cmd.c_str());
    if (ret != 0) {
        cout << "[ERROR] command failed(" << ret << "): " << cmd << endl;
        return false;
    }
    return true;
}

struct CaseParam {
    const char* tag;
    int64_t h;
    int64_t w;
    int64_t c;
    const char* inDtype;  // uint8 / float16 / float32
    const char* satDtype; // int32 / float32
    int64_t inBytes;      // sizeof(InT)
    int64_t satBytes;     // sizeof(AccT)
    uint32_t schMode;     // tilingKey：0=U8_I32 1=F16_F32 2=F32_F32 3=U8_F32
    int64_t twoPhase;
    int64_t blockWidth;
    int64_t rowTileWidth;
    int64_t colTileWidth;
    int64_t colTileCount;
    // 可选输出使能，对应 tiling 的 sqsumEnabled / tiltedEnabled：
    // 1 = 调用方提供该输出（形态 2/3），0 = 未使能（kernel 必须整体跳过对应分支）
    int64_t sqsumEnabled;
    int64_t tiltedEnabled;
};

// 单核下发，运行一次 kernel 并与 golden 比对
int RunKernelCase(const CaseParam& p)
{
    const int64_t inBytes = p.h * p.w * p.c * p.inBytes;
    const int64_t outElems = (p.h + 1) * (p.w + 1) * p.c;
    // two-phase 布局：[0,2HW) sum/sqsum 行前缀，[2HW,4HW) tilted 双对角缓冲
    const int64_t wsBytes = p.h * p.w * p.satBytes * 4;
    const uint32_t numBlocks = 1;

    if (!RunCmd("mkdir -p " + kRunDir)) {
        return 1;
    }
    const string genCmd = "python3 " + SourceDataDir() + "/gen_data.py " + kRunDir + " " + p.tag + " " +
                          to_string(p.h) + " " + to_string(p.w) + " " + to_string(p.c) + " " + p.inDtype + " " +
                          p.satDtype;
    if (!RunCmd(genCmd)) {
        return 1;
    }

    // 可选输出未使能时下发 nullptr，并验证 Kernel 跳过对应分支。
    const bool sqsumOn = (p.sqsumEnabled != 0);
    const bool tiltedOn = (p.tiltedEnabled != 0);
    uint8_t* image = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inBytes));
    uint8_t* sat = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outElems * p.satBytes));
    uint8_t* sqsum = sqsumOn ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outElems * 4)) : nullptr;
    uint8_t* tilted = tiltedOn ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outElems * p.satBytes)) : nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wsBytes));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(IntegralImageTilingData)));

    const string inPath = kRunDir + "/" + p.tag + "_input.bin";
    size_t inSize = static_cast<size_t>(inBytes);
    if (!ReadFile(inPath, inSize, image, static_cast<size_t>(inBytes))) {
        cout << "[ERROR] read input failed: " << inPath << endl;
        return 1;
    }

    IntegralImageTilingData* tilingData = reinterpret_cast<IntegralImageTilingData*>(tiling);
    tilingData->height = p.h;
    tilingData->width = p.w;
    tilingData->channel = p.c;
    tilingData->blockWidth = p.blockWidth;
    tilingData->coreNum = numBlocks;
    tilingData->twoPhase = p.twoPhase;
    tilingData->activeCoreNum = numBlocks;
    tilingData->rowTileWidth = p.rowTileWidth;
    tilingData->colTileWidth = p.colTileWidth;
    tilingData->colTileCount = p.colTileCount;
    tilingData->sqsumEnabled = p.sqsumEnabled;
    tilingData->tiltedEnabled = p.tiltedEnabled;

    ICPU_SET_TILING_KEY(static_cast<int32_t>(p.schMode));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    switch (p.schMode) {
        case 0:
            ICPU_RUN_KF(integral_image<0>, numBlocks, image, sat, sqsum, tilted, workspace, tiling);
            break;
        case 1:
            ICPU_RUN_KF(integral_image<1>, numBlocks, image, sat, sqsum, tilted, workspace, tiling);
            break;
        case 2:
            ICPU_RUN_KF(integral_image<2>, numBlocks, image, sat, sqsum, tilted, workspace, tiling);
            break;
        default:
            ICPU_RUN_KF(integral_image<3>, numBlocks, image, sat, sqsum, tilted, workspace, tiling);
            break;
    }

    // Check the HWC zero boundary in-process: NaN must also fail this exact-zero assertion.
    // In particular, C > blockWidth must not read past an undersized zeroSq buffer.
    if (sqsumOn) {
        const float* sqsumValues = reinterpret_cast<const float*>(sqsum);
        for (int64_t y = 0; y <= p.h; y++) {
            for (int64_t c = 0; c < p.c; c++) {
                EXPECT_EQ(sqsumValues[y * (p.w + 1) * p.c + c], 0.0f)
                    << "sqsum left boundary: tag=" << p.tag << " y=" << y << " c=" << c;
            }
        }
    }

    // sat 为必选输出；sqsum/tilted 未使能时不落盘，由 compare_data.py 跳过比对
    bool wroteAll = WriteFile(kRunDir + "/" + p.tag + "_output_sat.bin", sat, outElems * p.satBytes);
    if (sqsumOn) {
        wroteAll = WriteFile(kRunDir + "/" + p.tag + "_output_sqsum.bin", sqsum, outElems * 4) && wroteAll;
    }
    if (tiltedOn) {
        wroteAll = WriteFile(kRunDir + "/" + p.tag + "_output_tilted.bin", tilted, outElems * p.satBytes) && wroteAll;
    }

    AscendC::GmFree(image);
    AscendC::GmFree(sat);
    if (sqsum != nullptr) {
        AscendC::GmFree(sqsum);
    }
    if (tilted != nullptr) {
        AscendC::GmFree(tilted);
    }
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    if (!wroteAll) {
        cout << "[ERROR] write output failed, tag=" << p.tag << endl;
        return 1;
    }
    const string cmpCmd = "python3 " + SourceDataDir() + "/compare_data.py " + kRunDir + " " + p.tag + " " +
                          p.satDtype + " " + to_string(p.sqsumEnabled) + " " + to_string(p.tiltedEnabled);
    return RunCmd(cmpCmd) ? 0 : 1;
}

class IntegralImageKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "IntegralImageKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "IntegralImageKernelTest TearDown" << endl; }
};

} // namespace

// 2D UINT8 legacy：W=64 < 128，走 Process2D 的标量 Cast + 行扫描路径
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_legacy)
{
    CaseParam p = {"2d_u8_legacy", 8, 64, 1, "uint8", "int32", 1, 4, 0, 0, 1024, 0, 0, 0, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 2D FLOAT32 legacy：W=32 < 128，InT == AccT，走向量 Add 前缀扫描路径
TEST_F(IntegralImageKernelTest, integral_image_2d_f32_legacy)
{
    CaseParam p = {"2d_f32_legacy", 4, 32, 1, "float32", "float32", 4, 4, 2, 0, 1024, 0, 0, 0, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 3D HWC legacy：C=3 时逐通道独立累加，通道维不要求 32 字节对齐
TEST_F(IntegralImageKernelTest, integral_image_3d_u8_hwc)
{
    CaseParam p = {"3d_u8_hwc", 4, 6, 3, "uint8", "int32", 1, 4, 0, 0, 1024, 0, 0, 0, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 2D UINT8 two-phase：W=128 恰好 4 个 32 宽列块，覆盖 workspace 行前缀 + 列向累加
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_two_phase)
{
    CaseParam p = {"2d_u8_two_phase", 8, 128, 1, "uint8", "int32", 1, 4, 0, 1, 1024, 1024, 32, 4, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 2D UINT8 two-phase 尾块：W=131 不是 32 的整数倍，最后一个列块只有 3 列（非对齐搬运路径）
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_two_phase_tail)
{
    CaseParam p = {"2d_u8_two_phase_tail", 4, 131, 1, "uint8", "int32", 1, 4, 0, 1, 1024, 1024, 32, 5, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 形态 1（只出 sat）：sqsum/tilted 均未使能，two-phase 的可选分支必须整体跳过
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_two_phase_form1_sat_only)
{
    CaseParam p = {"2d_u8_tp_form1", 8, 128, 1, "uint8", "int32", 1, 4, 0, 1, 1024, 1024, 32, 4, 0, 0};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 形态 2（sat + sqsum）：tilted 未使能，two-phase 的 tilted 分支必须跳过
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_two_phase_form2_no_tilted)
{
    CaseParam p = {"2d_u8_tp_form2", 8, 128, 1, "uint8", "int32", 1, 4, 0, 1, 1024, 1024, 32, 4, 1, 0};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 形态 1（只出 sat）：3D HWC legacy 路径同样必须跳过 sqsum/tilted 分支
TEST_F(IntegralImageKernelTest, integral_image_3d_u8_legacy_form1_sat_only)
{
    CaseParam p = {"3d_u8_form1", 4, 6, 3, "uint8", "int32", 1, 4, 0, 0, 1024, 0, 0, 0, 0, 0};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// FLOAT16 输入（tilingKey=1）：legacy 2D 路径的 Cast 与浮点前缀扫描分支
TEST_F(IntegralImageKernelTest, integral_image_2d_f16_legacy)
{
    CaseParam p = {"2d_f16_legacy", 8, 64, 1, "float16", "float32", 2, 4, 1, 0, 1024, 0, 0, 0, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// UINT8 + sdepth=1（tilingKey=3）：u8 输入、float32 输出的 two-phase 分支
TEST_F(IntegralImageKernelTest, integral_image_2d_u8_sdepth_f32_two_phase)
{
    CaseParam p = {"2d_u8_f32_tp", 8, 128, 1, "uint8", "float32", 1, 4, 3, 1, 1024, 1024, 32, 4, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

TEST_F(IntegralImageKernelTest, integral_image_2d_u8_sdepth_f32_two_phase_tail)
{
    CaseParam p = {"2d_u8_f32_tp_tail", 17, 131, 1, "uint8", "float32", 1, 4, 3, 1, 1024, 1024, 32, 5, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// 3D HWC C=9：blockWidth=512，覆盖逐通道交错结果与相应的 UB 分配。
TEST_F(IntegralImageKernelTest, integral_image_3d_u8_c9_legacy)
{
    CaseParam p = {"3d_u8_c9", 8, 512, 9, "uint8", "int32", 1, 4, 0, 0, 512, 0, 0, 0, 1, 1};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// C=100 exercises an HWC zero boundary wider than the block and uses DataCopyPad.
TEST_F(IntegralImageKernelTest, integral_image_3d_u8_c100_sqsum_boundary)
{
    CaseParam p = {"3d_u8_c100", 2, 1, 100, "uint8", "int32", 1, 4, 0, 0, 64, 0, 0, 0, 1, 0};
    EXPECT_EQ(RunKernelCase(p), 0);
}

// Aligned C exercises DataCopy for the same boundary with float32 input and a partial tile.
TEST_F(IntegralImageKernelTest, integral_image_3d_f32_c128_sqsum_boundary)
{
    CaseParam p = {"3d_f32_c128", 2, 33, 128, "float32", "float32", 4, 4, 2, 0, 64, 0, 0, 0, 1, 0};
    EXPECT_EQ(RunKernelCase(p), 0);
}

TEST_F(IntegralImageKernelTest, integral_image_finite_comparison)
{
    EXPECT_TRUE(IntegralImageTest::IsFiniteClose(1.125, 1.0, 0.125));
    EXPECT_TRUE(IntegralImageTest::IsFiniteClose(-2.0, -2.0, 0.0));
    EXPECT_FALSE(IntegralImageTest::IsFiniteClose(1.25, 1.0, 0.125));
}

TEST_F(IntegralImageKernelTest, integral_image_nonfinite_comparison)
{
    const double values[] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (double value : values) {
        EXPECT_FALSE(IntegralImageTest::IsFiniteClose(value, 1.0, 1e-3));
        EXPECT_FALSE(IntegralImageTest::IsFiniteClose(1.0, value, 1e-3));
        EXPECT_FALSE(IntegralImageTest::IsFiniteClose(value, value, 1e-3));
    }
}

TEST_F(IntegralImageKernelTest, integral_image_python_result_comparison)
{
    EXPECT_TRUE(RunCmd("python3 " + SourceDataDir() + "/test_compare_data.py"));
}
