/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gaussian_filter_3_d.cpp
 * \brief GaussianFilter3D kernel UT：单核ICPU仿真，与CPU可分离高斯参考实现比对
 */

#include <math.h>
#include <utility>
#include <stdint.h>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void gaussian_filter_3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {

constexpr uint32_t MODE_REFLECT = 0;
constexpr uint32_t MODE_CONSTANT = 1;
constexpr uint32_t MODE_NEAREST = 2;
constexpr uint32_t MODE_MIRROR = 3;
constexpr uint32_t MODE_WRAP = 4;

// 一维高斯权重：w[i] = exp(-0.5*i^2/sigma^2)/sum，i in [-r, r]
// sigma=0 时 r=0、taps=1，公式会算出 exp(0/0)=NaN，必须显式给出恒等权重 [1.0]
void GenWeights(double sigma, uint32_t r, std::vector<double>& wts)
{
    int64_t taps = 2 * static_cast<int64_t>(r) + 1;
    wts.assign(taps, 0.0);
    if (!(sigma > 0.0)) {
        wts[static_cast<size_t>(r)] = 1.0;
        return;
    }
    double sum = 0.0;
    for (int64_t k = 0; k < taps; k++) {
        int64_t i = k - static_cast<int64_t>(r);
        wts[k] = exp(-0.5 * static_cast<double>(i) * static_cast<double>(i) / (sigma * sigma));
        sum += wts[k];
    }
    if (sum > 0.0) {
        for (auto& v : wts) {
            v /= sum;
        }
    }
}

int64_t MapIdx(int64_t i, int64_t n, uint32_t mode)
{
    if (i >= 0 && i < n) {
        return i;
    }
    if (mode == MODE_CONSTANT) {
        return -1;
    }
    if (n == 1) {
        return 0;
    }
    if (mode == MODE_NEAREST) {
        return i < 0 ? 0 : n - 1;
    }
    if (mode == MODE_REFLECT) {
        int64_t p = 2 * n;
        int64_t j = i % p;
        if (j < 0) {
            j += p;
        }
        return j >= n ? p - 1 - j : j;
    }
    if (mode == MODE_MIRROR) {
        int64_t p = 2 * n - 2;
        int64_t j = i % p;
        if (j < 0) {
            j += p;
        }
        return j >= n ? p - j : j;
    }
    int64_t j = i % n;
    if (j < 0) {
        j += n;
    }
    return j;
}

// fp64可分离参考实现（W→H→D，与kernel pass顺序一致）
void RefGaussian3D(const std::vector<float>& in, std::vector<float>& out, int64_t d, int64_t h, int64_t w,
                   const std::vector<double>& wts, uint32_t r, uint32_t mode)
{
    int64_t elem = d * h * w;
    std::vector<double> a(in.begin(), in.end());
    std::vector<double> b(elem, 0.0);
    double* src = a.data();
    double* dst = b.data();
    const int64_t axisLen[3] = {w, h, d};
    const int64_t axisStride[3] = {1, w, h * w};
    for (int32_t pass = 0; pass < 3; pass++) {
        int64_t n = axisLen[pass];
        int64_t st = axisStride[pass];
        int64_t outer = elem / n; // 其它维的元素数
        for (int64_t oi = 0; oi < outer; oi++) {
            // oi 为跳过本轴的一维下标，展开回平坦偏移（H/D轴 stride 不为1）
            int64_t base = (oi / st) * (n * st) + (oi % st);
            for (int64_t i = 0; i < n; i++) {
                double acc = 0.0;
                for (int64_t k = 0; k < static_cast<int64_t>(2 * r + 1); k++) {
                    int64_t j = MapIdx(i + k - static_cast<int64_t>(r), n, mode);
                    if (j < 0) {
                        continue; // CONSTANT越界
                    }
                    acc += wts[k] * src[base + j * st];
                }
                dst[base + i * st] = acc;
            }
        }
        // ping-pong
        std::swap(src, dst);
        // pass0: a->b, pass1: b->a, pass2: a->b；最终结果在src指向的缓冲
        if (pass == 2) {
            // 结果在dst原缓冲（已被swap到src）
        }
    }
    // 三次swap后 src 指向最终结果
    for (int64_t i = 0; i < elem; i++) {
        out[i] = static_cast<float>(src[i]);
    }
}

// half/bfloat16 位模式转换
float Bits16ToFloat(uint16_t h, bool isBf16)
{
    uint32_t bits = isBf16 ? (static_cast<uint32_t>(h) << 16) : 0U;
    if (!isBf16) {
        uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t man = h & 0x03FF;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x0400) == 0) {
                    man <<= 1;
                    exp--;
                }
                man &= 0x03FF;
                bits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000 | (man << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
        }
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

uint16_t FloatToBits16(float f, bool isBf16)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    if (isBf16) {
        return static_cast<uint16_t>((bits >> 16) & 0xFFFF);
    }
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = bits & 0x007FFFFF;
    if (((bits >> 23) & 0xFF) == 0xFF) {
        return static_cast<uint16_t>(sign | 0x7C00 | (man ? 0x0200 : 0));
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00); // 溢出饱和
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        man = (man | 0x00800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((man + 0x1000) >> 13));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((man + 0x1000) >> 13));
}

struct CaseCfg {
    int64_t d;
    int64_t h;
    int64_t w;
    double sigma;
    uint32_t mode;
    uint32_t tileLen;
    uint32_t dtypeKey; // 0 fp32 / 1 fp16 / 2 bf16
};

void RunKernelCase(const CaseCfg& cfg)
{
    const bool isBf16 = cfg.dtypeKey == 2;
    const uint32_t dtypeSize = cfg.dtypeKey == 0 ? 4U : 2U;
    const uint32_t r = static_cast<uint32_t>(4.0 * cfg.sigma + 0.5);
    const int64_t elem = cfg.d * cfg.h * cfg.w;
    const uint32_t chunkNum = static_cast<uint32_t>((cfg.w + cfg.tileLen - 1) / cfg.tileLen);
    const uint32_t lineNum = static_cast<uint32_t>(cfg.d * cfg.h);
    const uint64_t pongOffset = ((elem * 4U + 255U) / 256U) * 256U;

    std::vector<double> wts;
    GenWeights(cfg.sigma, r, wts);

    // tiling
    GaussianFilter3DTilingData tilingData;
    memset(&tilingData, 0, sizeof(tilingData));
    tilingData.batch = 1;
    tilingData.depth = static_cast<uint32_t>(cfg.d);
    tilingData.height = static_cast<uint32_t>(cfg.h);
    tilingData.width = static_cast<uint32_t>(cfg.w);
    tilingData.radius = r;
    tilingData.mode = cfg.mode;
    tilingData.tileLen = cfg.tileLen;
    tilingData.chunkNum = chunkNum;
    tilingData.lineNum = lineNum;
    tilingData.coreNum = 1;
    tilingData.tasksPerCore = lineNum * chunkNum;
    tilingData.pongOffset = static_cast<uint32_t>(pongOffset);
    for (uint32_t k = 0; k < 2 * r + 1; k++) {
        float wf = static_cast<float>(wts[k]);
        memcpy(&tilingData.weightBits[k], &wf, sizeof(float));
    }

    // golden
    std::vector<float> xVal(elem);
    for (int64_t i = 0; i < elem; i++) {
        xVal[i] = static_cast<float>(sin(i * 0.37) * 0.7 + (i % 11) * 0.05 - 0.3);
    }
    if (cfg.dtypeKey != 0) {
        for (auto& v : xVal) {
            v = Bits16ToFloat(FloatToBits16(v, isBf16), isBf16);
        }
    }
    std::vector<float> golden(elem, 0.0f);
    RefGaussian3D(xVal, golden, cfg.d, cfg.h, cfg.w, wts, r, cfg.mode);

    // device buffers
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elem * dtypeSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elem * dtypeSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(pongOffset * 2));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GaussianFilter3DTilingData)));

    for (int64_t i = 0; i < elem; i++) {
        if (cfg.dtypeKey == 0) {
            reinterpret_cast<float*>(x)[i] = xVal[i];
        } else {
            reinterpret_cast<uint16_t*>(x)[i] = FloatToBits16(xVal[i], isBf16);
        }
    }
    memset(y, 0, elem * dtypeSize);
    memcpy(tiling, &tilingData, sizeof(tilingData));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(cfg.dtypeKey);
    ICPU_RUN_KF(gaussian_filter_3_d, 1, x, y, workspace, tiling);

    // 比对
    double maxAbsDiff = 0.0;
    int64_t worstIdx = -1;
    bool hasNonFinite = false;
    for (int64_t i = 0; i < elem; i++) {
        float out;
        if (cfg.dtypeKey == 0) {
            out = reinterpret_cast<float*>(y)[i];
        } else {
            out = Bits16ToFloat(reinterpret_cast<uint16_t*>(y)[i], isBf16);
        }
        // NaN/Inf 必须显式判失败：fabs(NaN) 与任何值比较都是 false，会静默“相等”而漏判
        if (!isfinite(static_cast<double>(out)) || !isfinite(golden[i])) {
            hasNonFinite = true;
            worstIdx = i;
            break;
        }
        double diff = fabs(static_cast<double>(out) - golden[i]);
        if (diff > maxAbsDiff) {
            maxAbsDiff = diff;
            worstIdx = i;
        }
    }
    EXPECT_FALSE(hasNonFinite) << "kernel output or golden contains NaN/Inf, idx=" << worstIdx;
    double tolDbg = cfg.dtypeKey == 0 ? 1e-4 : (isBf16 ? 2e-2 : 1e-2);
    if (maxAbsDiff >= tolDbg) {
        // 调试输出：前24个元素与首个错误的上下文（W行内定位）
        for (int64_t i = 0; i < 24 && i < elem; i++) {
            float out = cfg.dtypeKey == 0 ? reinterpret_cast<float*>(y)[i] :
                                            Bits16ToFloat(reinterpret_cast<uint16_t*>(y)[i], isBf16);
            printf("DBG i=%lld y=%.6f golden=%.6f\n", (long long)i, out, golden[i]);
        }
        if (worstIdx >= 0) {
            int64_t w = cfg.w;
            printf("DBG worst i=%lld (row=%lld col=%lld) y=%.6f golden=%.6f\n", (long long)worstIdx,
                   (long long)(worstIdx / w), (long long)(worstIdx % w),
                   cfg.dtypeKey == 0 ? reinterpret_cast<float*>(y)[worstIdx] :
                                       Bits16ToFloat(reinterpret_cast<uint16_t*>(y)[worstIdx], isBf16),
                   golden[worstIdx]);
        }
    }
    double tol = cfg.dtypeKey == 0 ? 1e-4 : (isBf16 ? 2e-2 : 1e-2);
    EXPECT_LT(maxAbsDiff, tol) << "maxAbsDiff=" << maxAbsDiff << " r=" << r << " mode=" << cfg.mode;

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace

class gaussian_filter_3_d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gaussian_filter_3_d_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "gaussian_filter_3_d_test TearDown" << endl; }
};

TEST_F(gaussian_filter_3_d_test, fp32_reflect_sigma1)
{ // 含r>H(3<r=4? r=4,H=3)、W=9 tileLen=8 尾块=1
    RunKernelCase({4, 3, 9, 1.0, MODE_REFLECT, 8, 0});
}

TEST_F(gaussian_filter_3_d_test, diag_single_row_passW_only)
{ // D=1,H=1：pass H/D 退化为同一行，隔离 pass W
    RunKernelCase({1, 1, 9, 1.0, MODE_REFLECT, 8, 0});
}

TEST_F(gaussian_filter_3_d_test, diag_single_plane_passH)
{ // D=1：隔离 pass H
    RunKernelCase({1, 3, 9, 1.0, MODE_REFLECT, 8, 0});
}

TEST_F(gaussian_filter_3_d_test, fp32_constant_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_CONSTANT, 8, 0}); }

TEST_F(gaussian_filter_3_d_test, fp32_nearest_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_NEAREST, 8, 0}); }

TEST_F(gaussian_filter_3_d_test, fp32_mirror_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_MIRROR, 8, 0}); }

TEST_F(gaussian_filter_3_d_test, fp32_wrap_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_WRAP, 8, 0}); }

TEST_F(gaussian_filter_3_d_test, fp32_sigma0_identity) { RunKernelCase({4, 3, 9, 0.0, MODE_REFLECT, 8, 0}); }

// sigma=0 的直通语义与 dtype 无关：fp16/bf16 也必须是逐元素恒等（权重 [1.0]，不含 NaN）
TEST_F(gaussian_filter_3_d_test, fp16_sigma0_identity) { RunKernelCase({4, 3, 9, 0.0, MODE_REFLECT, 8, 1}); }

TEST_F(gaussian_filter_3_d_test, bf16_sigma0_identity) { RunKernelCase({4, 3, 9, 0.0, MODE_REFLECT, 8, 2}); }

TEST_F(gaussian_filter_3_d_test, fp32_sigma2_tile_eq_w)
{
    // r=8 > D=4/H=3，tileLen=W=9 单chunk，重点覆盖折叠映射
    RunKernelCase({4, 3, 9, 2.0, MODE_REFLECT, 9, 0});
}

TEST_F(gaussian_filter_3_d_test, fp16_reflect_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_REFLECT, 8, 1}); }

TEST_F(gaussian_filter_3_d_test, bf16_reflect_sigma1) { RunKernelCase({4, 3, 9, 1.0, MODE_REFLECT, 8, 2}); }
