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
 * \file test_lbp_feature.cpp
 * \brief LbpFeature kernel CPU 仿真 UT：经 helper 调用真实 TilingFunc 生成 tilingData，
 *        再 ICPU_RUN_KF 执行 kernel，与 numpy golden（cv::elbp 语义的位级复刻）逐位对拍。
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

// cce 仿真构建下同一 UT 目标只能带一组 dtype 编译宏；因此在一个可执行内
// 按 dtypeId 参数化实例化 kernel 入口：uint8 变体沿用构建宏
// （-DDTYPE_X=uint8 -DDTYPE_Y=int32），float32 变体通过重包含生产 kernel
// 并重绑类型宏与入口符号生成，避免 AddOpTestCase 单目标限制。
using uint8 = uint8_t;
using int32 = int32_t;
using float32 = float;

#ifndef DTYPE_X
#define DTYPE_X uint8
#endif
#ifndef DTYPE_Y
#define DTYPE_Y int32
#endif
#include "../../../op_kernel/lbp_feature.cpp"
template __global__ __aicore__ void lbp_feature<1>(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
template __global__ __aicore__ void lbp_feature<2>(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X float32
#define DTYPE_Y int32
#define lbp_feature lbp_feature_fp32
#define LbpFeatureTilingKey LbpFeatureTilingKeyFp32
#include "../../../op_kernel/lbp_feature.cpp"
#undef LbpFeatureTilingKey
#undef lbp_feature
#undef DTYPE_Y
#undef DTYPE_X
template __global__ __aicore__ void lbp_feature_fp32<1>(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
template __global__ __aicore__ void lbp_feature_fp32<2>(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

// tiling 由独立 TU 调用真实 TilingFunc 生成（避免 host DT_* 宏与 kernel 头冲突）
#include "lbp_feature_ut_helper.h"

static constexpr size_t UT_WORKSPACE_BYTES = 1U << 20;

// TPL（UI_LIST）编码后的 tilingKey：schMode 1/2 编码为列表下标 0/1
// （Host UT 冻结基线实测：RowSplit -> key=0，Tile2D -> key=1）
static constexpr uint64_t UT_KEY_ROW_SPLIT = 0ULL;
static constexpr uint64_t UT_KEY_TILE_2D = 1ULL;

class LbpFeatureKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LbpFeatureKernel SetUp" << std::endl;
        system("cp -rf ../../../../experimental/image/lbp_feature/tests/ut/op_kernel/lbp_feature_data ./");
        system("chmod -R 755 ./lbp_feature_data/");
    }

    static void TearDownTestCase() { std::cout << "LbpFeatureKernel TearDown" << std::endl; }

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

    // float → 科学计数法字符串。std::to_string 固定 6 位小数，
    // 1e-30 会被静默写成 "0.000000"，极值用例就废了，必须用 %.9g
    static std::string FmtFloat(float v)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", v);
        return std::string(buf);
    }

    // dtypeId: 0 = uint8, 1 = float32
    static void RunOneCase(const std::string& shapeStr, const std::string& dtypeCmd, int dtypeId, size_t inBytesPerElem,
                           int64_t radius, int64_t neighbors, float minVal, float maxVal)
    {
        const std::vector<int64_t> shape = ParseShape(shapeStr);
        size_t elems = 1;
        size_t batch = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            elems *= static_cast<size_t>(shape[i]);
            if (i + 2 < shape.size()) {
                batch *= static_cast<size_t>(shape[i]);
            }
        }
        const uint32_t outH = static_cast<uint32_t>(shape[shape.size() - 2] - 2 * radius);
        const uint32_t outW = static_cast<uint32_t>(shape[shape.size() - 1] - 2 * radius);
        const size_t xByteSize = elems * inBytesPerElem;
        const size_t zByteSize = batch * outH * outW * sizeof(int32_t); // 输出恒为 int32

        // 1) 真实 tiling（独立 TU）
        LbpUt::TilingResult tr = LbpUt::RunTiling(shape, dtypeId, radius, neighbors);
        ASSERT_TRUE(tr.ok) << "ExecuteTiling failed";
        std::cout << "[Case " << shapeStr << " " << dtypeCmd << "] key=" << tr.tilingKey << " blocks=" << tr.blockNum
                  << " tilingBytes=" << tr.tilingData.size() << std::endl;

        // 2) 生成输入 + golden（固定种子，可复现）
        //    注意：min/max 用等号连接 —— argparse 会把独立的 "-1e+10"
        //    误判成未知选项 flag（"expected one argument" 报错的元凶）
        std::string genCmd = "cd ./lbp_feature_data/ && python3 gen_data.py --shape " + shapeStr + " --dtype " +
                             dtypeCmd + " --radius " + std::to_string(radius) + " --neighbors " +
                             std::to_string(neighbors) + " --seed 42";
        if (dtypeCmd == "float32") {
            genCmd += " --min_val=" + FmtFloat(minVal) + " --max_val=" + FmtFloat(maxVal);
        }
        int ret = system(genCmd.c_str());
        ASSERT_EQ(ret, 0) << "gen_data.py 执行失败： " << genCmd;
        if (ret != 0) {
            return;
        }

        // 3) 申请内存并读入输入
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
        uint8_t* out = (uint8_t*)AscendC::GmAlloc(zByteSize);
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(UT_WORKSPACE_BYTES);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tr.tilingData.size());
        size_t fileLen = xByteSize;
        ASSERT_TRUE(ReadFile("./lbp_feature_data/input_lbp_feature.bin", fileLen, x, xByteSize));
        std::memcpy(tiling, tr.tilingData.data(), tr.tilingData.size());

        // 4) 执行 kernel（按 tilingKey 选择编译期分支）
        ICPU_SET_TILING_KEY(tr.tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if (dtypeId == 0) {
            if (tr.tilingKey == UT_KEY_ROW_SPLIT) {
                ICPU_RUN_KF(lbp_feature<1>, tr.blockNum, x, out, workspace, tiling);
            } else {
                ICPU_RUN_KF(lbp_feature<2>, tr.blockNum, x, out, workspace, tiling);
            }
        } else {
            if (tr.tilingKey == UT_KEY_ROW_SPLIT) {
                ICPU_RUN_KF(lbp_feature_fp32<1>, tr.blockNum, x, out, workspace, tiling);
            } else {
                ICPU_RUN_KF(lbp_feature_fp32<2>, tr.blockNum, x, out, workspace, tiling);
            }
        }

        // 5) 写输出并逐位对拍（int32 零容差）
        WriteFile("./lbp_feature_data/int32_output_lbp_feature.bin", out, zByteSize);
        ret = system("cd ./lbp_feature_data/ && python3 compare_data.py 'int32'");
        EXPECT_EQ(WEXITSTATUS(ret), 0) << "golden 对拍失败！shape=" << shapeStr;

        // 6) 释放
        AscendC::GmFree(x);
        AscendC::GmFree(out);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

// ════════════════════════════════════════════════════════════
// UINT8 基础用例（默认构建）：RowSplit 与列并行各一
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureKernel, lbp_feature_kernel_rowsplit_2d) { RunOneCase("6,70", "uint8", 0, 1, 1, 8, 0.f, 255.f); }

TEST_F(LbpFeatureKernel, lbp_feature_kernel_colparallel_2d) { RunOneCase("8,130", "uint8", 0, 1, 1, 8, 0.f, 255.f); }

// ════════════════════════════════════════════════════════════
// FLOAT32 矩阵用例（float32 构建时编译）：
//   2D/3D/4D/5D × 奇偶 × RowSplit/Tile2D × 四档数值量级
//   tilingKey / 核数全部由真实 TilingFunc 推导
// ════════════════════════════════════════════════════════════

// 2D：最小形状，奇/奇，个位数
TEST_F(LbpFeatureKernel, fp32_2d_3x3_unit) { RunOneCase("3,3", "float32", 1, 4, 1, 8, 0.f, 9.f); }
// 2D：奇/偶，两位数
TEST_F(LbpFeatureKernel, fp32_2d_7x4_decades) { RunOneCase("7,4", "float32", 1, 4, 1, 8, -99.f, 99.f); }
// 2D：奇数宽度（127，行切分奇边界），三位数
TEST_F(LbpFeatureKernel, fp32_2d_5x129_hundreds) { RunOneCase("5,129", "float32", 1, 4, 1, 8, 0.f, 999.f); }
// 3D：全奇，个位数带负值
TEST_F(LbpFeatureKernel, fp32_3d_2x3x5_unit) { RunOneCase("2,3,5", "float32", 1, 4, 1, 8, -9.f, 9.f); }
// 3D：宽图（outW=258）触发 Tile2D 列并行，三位数
TEST_F(LbpFeatureKernel, fp32_3d_2x5x260_tile2d) { RunOneCase("2,5,260", "float32", 1, 4, 1, 8, 0.f, 999.f); }
// 3D：float 极值（±3.4e38），插值饱和路径
TEST_F(LbpFeatureKernel, fp32_3d_4x9x9_extreme) { RunOneCase("4,9,9", "float32", 1, 4, 1, 8, -3.4e38f, 3.4e38f); }
// 4D：跨 60 个数量级（1e-30 ~ 1e30）
TEST_F(LbpFeatureKernel, fp32_4d_2x2x3x7_scale) { RunOneCase("2,2,3,7", "float32", 1, 4, 1, 8, 1e-30f, 1e30f); }
// 4D：宽图（outW=298）Tile2D 列并行，三位数
TEST_F(LbpFeatureKernel, fp32_4d_3x4x300_tile2d) { RunOneCase("3,4,300", "float32", 1, 4, 1, 8, -999.f, 999.f); }
// 5D：深度 batch（batch=8），十的十次方量级
TEST_F(LbpFeatureKernel, fp32_5d_2x2x2x3x5_deep) { RunOneCase("2,2,2,3,5", "float32", 1, 4, 1, 8, -1e10f, 1e10f); }
// 5D：batch=24（首三维 2*2*6）跑满 40 核且行不均匀分配（bigRowCoreCount=8），高精度同量级
TEST_F(LbpFeatureKernel, fp32_5d_2x2x6x6x150_40core) { RunOneCase("2,2,6,6,150", "float32", 1, 4, 1, 8, 1e5f, 1e6f); }

// 原大图溢出分支用例：整行超 UB → 溢出分支切列，40 核
TEST_F(LbpFeatureKernel, fp32_2d_10x4102_overflow) { RunOneCase("10,4102", "float32", 1, 4, 1, 8, -3.4e38f, 3.4e38f); }
