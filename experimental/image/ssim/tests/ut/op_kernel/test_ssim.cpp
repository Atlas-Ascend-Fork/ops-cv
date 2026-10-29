/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void ssim(GM_ADDR x1, GM_ADDR x2, GM_ADDR yOut, GM_ADDR workspace, GM_ADDR tiling);

class ssim_test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

namespace {
uint64_t gRngState = 0;
void SeedRng(uint64_t seed) { gRngState = static_cast<uint64_t>(seed) * 2654435761ULL + 1; }
uint64_t NextRng()
{
    gRngState = gRngState * 6364136223846793005ULL + 1442695040888963407ULL;
    return gRngState;
}

// Same RNG as examples/test_aclnn_ssim.cpp so expected values are shared.
void GenFloatData(float* x1, float* x2, int64_t numel)
{
    for (int64_t i = 0; i < numel; i++) {
        uint64_t s = NextRng();
        float v = static_cast<float>((s >> 40) & 0xFFFF) / 65535.0f;
        x1[i] = v;
        x2[i] = v + 0.03f * static_cast<float>((s >> 24) & 0xFF) / 255.0f;
    }
}

// x2 = 1 - x1 keeps the two images strongly anti-correlated, so a window taken across an
// image boundary moves the result far more than the fp32 accumulation noise does.
void GenFloatDataInverted(float* x1, float* x2, int64_t numel)
{
    for (int64_t i = 0; i < numel; i++) {
        uint64_t s = NextRng();
        float v = static_cast<float>((s >> 40) & 0xFFFF) / 65535.0f;
        x1[i] = v;
        x2[i] = 1.0f - v;
    }
}

void RunSsim(uint32_t h, uint32_t w, uint32_t c, uint32_t batch, uint32_t blockDim, uint32_t colTile, uint64_t seed,
             float* outVal, bool repeatImage = false, bool inverted = false, std::vector<float>* hostX1 = nullptr,
             std::vector<float>* hostX2 = nullptr)
{
    if (h < 11U || w < 11U) {
        ADD_FAILURE() << "kernel UT shape must satisfy H >= 11 and W >= 11";
        *outVal = 0.0f;
        return;
    }
    const uint64_t kSysWsBytes = 16ULL * 1024 * 1024;
    const uint64_t kUserWsBytes = 64ULL * 1024;
    uint64_t numel = static_cast<uint64_t>(h) * w * c * batch;
    const size_t elemSize = sizeof(float);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(numel * elemSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(numel * elemSize);
    uint8_t* yOut = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(kSysWsBytes + kUserWsBytes);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SsimTilingData));

    const int64_t perImage = static_cast<int64_t>(h) * w * c;
    for (uint32_t b = 0; b < batch; b++) {
        if (b == 0 || repeatImage) {
            SeedRng(seed);
        }
        float* p1 = reinterpret_cast<float*>(x1) + b * perImage;
        float* p2 = reinterpret_cast<float*>(x2) + b * perImage;
        if (inverted) {
            GenFloatDataInverted(p1, p2, perImage);
        } else {
            GenFloatData(p1, p2, perImage);
        }
    }

    if (hostX1 != nullptr && hostX2 != nullptr) {
        const float* base1 = reinterpret_cast<const float*>(x1);
        const float* base2 = reinterpret_cast<const float*>(x2);
        hostX1->assign(base1, base1 + numel);
        hostX2->assign(base2, base2 + numel);
    }

    uint32_t outH = h - 10;
    uint32_t outW = w - 10;
    uint32_t totalOutRows = batch * outH;
    uint32_t blockRows = (totalOutRows + blockDim - 1) / blockDim;
    if (blockDim > 1 && blockRows * (blockDim - 1) > totalOutRows) {
        blockRows = totalOutRows / blockDim;
    }
    uint32_t lastBlockRows = totalOutRows - blockRows * (blockDim - 1);
    uint64_t totalPixels = static_cast<uint64_t>(batch) * outH * outW * c;
    const float l = 1.0f;
    float c1 = (0.01f * l) * (0.01f * l);
    float c2 = (0.03f * l) * (0.03f * l);

    SsimTilingData* tilingData = reinterpret_cast<SsimTilingData*>(tiling);
    tilingData->totalOutRows = totalOutRows;
    tilingData->perImageOutRows = outH;
    tilingData->perImageInRows = h;
    tilingData->outW = outW;
    tilingData->inW = w;
    tilingData->inH = h;
    tilingData->channels = c;
    tilingData->batch = batch;
    tilingData->blockRows = blockRows;
    tilingData->lastBlockRows = lastBlockRows;
    // Mirror the host tiling rule: channels are packed innermost, so the element count per
    // tile (colTile * C) has to stay inside the UB ring buffer budget.
    uint32_t tileCols = colTile;
    if (c > 1U) {
        tileCols = (colTile / c) & ~7U;
        if (tileCols < 8U) {
            tileCols = 8U;
        }
    }
    tilingData->colTile = tileCols;
    tilingData->totalPixels = static_cast<uint32_t>(totalPixels);
    tilingData->blockDim = blockDim;
    tilingData->invTotalPixels = 1.0f / static_cast<float>(totalPixels);
    tilingData->c1 = c1;
    tilingData->c2 = c2;
    tilingData->isUint8 = 0U;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ssim, blockDim, x1, x2, yOut, workspace, tiling);

    *outVal = *reinterpret_cast<float*>(yOut);

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(yOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Independent fp64 reference: separable 11x11 gaussian window (sigma = 1.5) evaluated over the
// valid (h-10)x(w-10) region, then averaged over every pixel and channel. Computing it here keeps
// the expected value reproducible from the repository alone (no external script, no magic number).
constexpr double kGaussWin[11] = {0.001028380, 0.007598758, 0.036000773, 0.109360687, 0.213005543, 0.266011715,
                                  0.213005543, 0.109360687, 0.036000773, 0.007598758, 0.001028380};

double RefMssim(const std::vector<float>& x1, const std::vector<float>& x2, uint32_t h, uint32_t w, uint32_t c)
{
    const uint32_t outH = h - 10;
    const uint32_t outW = w - 10;
    const double c1 = 0.01 * 0.01;
    const double c2 = 0.03 * 0.03;
    double sum = 0.0;
    for (uint32_t oh = 0; oh < outH; oh++) {
        for (uint32_t ow = 0; ow < outW; ow++) {
            for (uint32_t ch = 0; ch < c; ch++) {
                double mu1 = 0.0;
                double mu2 = 0.0;
                double uxx = 0.0;
                double uyy = 0.0;
                double uxy = 0.0;
                for (uint32_t ky = 0; ky < 11; ky++) {
                    for (uint32_t kx = 0; kx < 11; kx++) {
                        const size_t idx = (static_cast<size_t>(oh + ky) * w + static_cast<size_t>(ow + kx)) * c + ch;
                        const double a = static_cast<double>(x1[idx]);
                        const double b = static_cast<double>(x2[idx]);
                        const double g = kGaussWin[ky] * kGaussWin[kx];
                        mu1 += g * a;
                        mu2 += g * b;
                        uxx += g * a * a;
                        uyy += g * b * b;
                        uxy += g * a * b;
                    }
                }
                const double sig1 = uxx - mu1 * mu1;
                const double sig2 = uyy - mu2 * mu2;
                const double sig12 = uxy - mu1 * mu2;
                const double num = (2.0 * mu1 * mu2 + c1) * (2.0 * sig12 + c2);
                const double den = (mu1 * mu1 + mu2 * mu2 + c1) * (sig1 + sig2 + c2);
                sum += num / den;
            }
        }
    }
    return sum / (static_cast<double>(outH) * static_cast<double>(outW) * static_cast<double>(c));
}

constexpr float kTol = 1e-5f;
constexpr float kBatchTol = 1e-5f;
} // namespace

TEST_F(ssim_test, float32_rank2_64x64_single_core)
{
    float out = 0.0f;
    std::vector<float> hostX1;
    std::vector<float> hostX2;
    RunSsim(64, 64, 1, 1, 1, 256, 12345, &out, false, false, &hostX1, &hostX2);
    EXPECT_NEAR(out, static_cast<float>(RefMssim(hostX1, hostX2, 64, 64, 1)), kTol);
}

TEST_F(ssim_test, float32_rank3_c3_multi_core_matches_reference)
{
    float out = 0.0f;
    std::vector<float> hostX1;
    std::vector<float> hostX2;
    RunSsim(64, 64, 3, 1, 4, 256, 12345, &out, false, false, &hostX1, &hostX2);
    EXPECT_NEAR(out, static_cast<float>(RefMssim(hostX1, hostX2, 64, 64, 3)), kTol);
}

TEST_F(ssim_test, float32_rank4_batch2_keeps_image_boundary)
{
    float single = 0.0f;
    RunSsim(64, 64, 1, 1, 4, 256, 12345, &single, false, true);
    float batched = 0.0f;
    RunSsim(64, 64, 1, 2, 4, 256, 12345, &batched, true, true);
    // Both batch images hold identical content, so the sliding window must restart at the
    // image boundary: the mean has to stay independent of the batch size.
    EXPECT_NEAR(batched, single, kBatchTol);
}
