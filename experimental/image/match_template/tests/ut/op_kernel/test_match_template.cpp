/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "match_template_tiling.h"
#include "../../../op_kernel/match_template.h"

namespace {

constexpr int64_t BLOCK_BYTES = 32;

int64_t AlignBlock(int64_t bytes) { return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES; }

// Invoke the production implementation, including all Vector instructions and GM
// writes. This entry accepts an already resolved user workspace. The framework
// workspace conversion remains the responsibility of the production entry.
template <typename T, uint32_t METHOD>
__global__ __aicore__ void MatchTemplateKernelUt(GM_ADDR image, GM_ADDR templateImage, GM_ADDR score, GM_ADDR bestScore,
                                                 GM_ADDR bestLocation, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MatchTemplateTilingData, data, tiling);
    NsMatchTemplate::KernelMatchTemplate<T, METHOD> op;
    op.Init(image, templateImage, score, bestScore, bestLocation, workspace, &data);
    op.Process();
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : data_(static_cast<uint8_t*>(AscendC::GmAlloc(AlignBlock(bytes))))
    {
        if (data_ != nullptr) {
            std::memset(data_, 0, AlignBlock(bytes));
        }
    }
    ~GmBuffer() { AscendC::GmFree(data_); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    uint8_t* Data() const { return data_; }

private:
    uint8_t* data_;
};

struct Shape {
    int64_t batch;
    int64_t h;
    int64_t w;
    int64_t c;
    int64_t th;
    int64_t tw;
};

template <typename T>
MatchTemplateTilingData MakeTiling(const Shape& shape, uint32_t method, bool halo, int64_t tileH = 2, int64_t tileW = 8,
                                   bool c3 = false)
{
    MatchTemplateTilingData data{};
    data.batch = shape.batch;
    data.imageH = shape.h;
    data.imageW = shape.w;
    data.channels = shape.c;
    data.templateH = shape.th;
    data.templateW = shape.tw;
    data.outputH = shape.h - shape.th + 1;
    data.outputW = shape.w - shape.tw + 1;
    data.method = method;
    data.tileH = tileH;
    data.tileW = tileW;
    data.tileRows = (data.outputH + tileH - 1) / tileH;
    data.tileCols = (data.outputW + tileW - 1) / tileW;
    data.tilesPerBatch = data.tileRows * data.tileCols;
    data.totalTiles = data.batch * data.tilesPerBatch;
    // Single-core numerical UTs do not emulate or replace hardware SyncAll.
    // Multicore ordering is covered by the NPU ACLNN/Golden tests.
    data.usedCores = 1;
    data.useHaloTile = halo;
    data.useC3VectorTile = c3;
    data.templateElements = shape.th * shape.tw * shape.c;
    data.templateStatsBufferBytes = method >= 4 ? AlignBlock(shape.c * sizeof(float)) : 0;
    data.workspaceIndexOffset = AlignBlock(shape.batch * sizeof(float));
    data.workspacePackedImageOffset = data.workspaceIndexOffset + AlignBlock(shape.batch * sizeof(int64_t));
    if (halo) {
        data.templateBufferBytes = AlignBlock(data.templateElements * sizeof(T));
        const int64_t rows = shape.th + tileH - 1;
        const int64_t width = shape.tw + tileW - 1;
        data.haloBufferBytes = rows * AlignBlock(width * shape.c * sizeof(T));
        if constexpr (std::is_same<T, uint8_t>::value) {
            data.templateFloatBufferBytes = AlignBlock(data.templateElements * sizeof(float));
            data.haloFloatBufferBytes = data.haloBufferBytes * sizeof(float);
            if (c3 && (method == 2 || method == 5)) {
                data.prepackScratchBytes = 16 * 1024;
                data.haloBufferBytes = data.prepackScratchBytes;
                data.haloFloatBufferBytes = rows * AlignBlock(width * shape.c * sizeof(half));
                data.packedImageBytes = AlignBlock(shape.batch * shape.h * shape.w * shape.c * sizeof(half));
            }
        }
    }
    return data;
}

// Direct double-precision specification: the coefficient is evaluated from
// centered samples, independently of the kernel's accumulated-sum formulas.
template <typename T>
std::vector<double> Reference(const std::vector<T>& image, const std::vector<T>& pattern,
                              const MatchTemplateTilingData& d)
{
    std::vector<double> scores(d.batch * d.outputH * d.outputW);
    const int64_t spatial = d.templateH * d.templateW;
    for (int64_t n = 0; n < d.batch; ++n) {
        for (int64_t y = 0; y < d.outputH; ++y) {
            for (int64_t x = 0; x < d.outputW; ++x) {
                double diff = 0, correlation = 0, imageSquare = 0, templateSquare = 0;
                double coefficient = 0, imageVariance = 0, templateVariance = 0;
                for (int64_t c = 0; c < d.channels; ++c) {
                    double imageMean = 0, templateMean = 0;
                    for (int64_t ty = 0; ty < d.templateH; ++ty) {
                        for (int64_t tx = 0; tx < d.templateW; ++tx) {
                            const double a = static_cast<float>(
                                image[((n * d.imageH + y + ty) * d.imageW + x + tx) * d.channels + c]);
                            const double b = static_cast<float>(pattern[(ty * d.templateW + tx) * d.channels + c]);
                            imageMean += a;
                            templateMean += b;
                            diff += (a - b) * (a - b);
                            correlation += a * b;
                            imageSquare += a * a;
                            templateSquare += b * b;
                        }
                    }
                    imageMean /= spatial;
                    templateMean /= spatial;
                    for (int64_t ty = 0; ty < d.templateH; ++ty) {
                        for (int64_t tx = 0; tx < d.templateW; ++tx) {
                            const double
                                a = static_cast<float>(
                                        image[((n * d.imageH + y + ty) * d.imageW + x + tx) * d.channels + c]) -
                                    imageMean;
                            const double b = static_cast<float>(pattern[(ty * d.templateW + tx) * d.channels + c]) -
                                             templateMean;
                            coefficient += a * b;
                            imageVariance += a * a;
                            templateVariance += b * b;
                        }
                    }
                }
                double value = 0;
                if (d.method == 0 || d.method == 1) {
                    value = diff;
                } else if (d.method == 2 || d.method == 3) {
                    value = correlation;
                } else {
                    value = coefficient;
                }
                if (d.method == 1 || d.method == 3 || d.method == 5) {
                    const double denominator = std::sqrt(d.method == 5 ? imageVariance * templateVariance :
                                                                         imageSquare * templateSquare);
                    if (denominator <= 1e-12) {
                        value = d.method == 1 && diff > 1e-12 ? 1.0 : 0.0;
                    } else {
                        value = std::clamp(value / denominator, d.method == 1 ? 0.0 : -1.0, 1.0);
                    }
                }
                scores[(n * d.outputH + y) * d.outputW + x] = value;
            }
        }
    }
    return scores;
}

template <typename T>
std::vector<T> Samples(int64_t size, uint32_t seed)
{
    std::vector<T> values(size);
    for (T& value : values) {
        seed = seed * 1664525U + 1013904223U;
        const int32_t sample = static_cast<int32_t>((seed >> 16) % 17);
        if constexpr (std::is_same<T, uint8_t>::value) {
            value = static_cast<T>(sample);
        } else {
            value = static_cast<T>((sample - 8) * 0.25f);
        }
    }
    return values;
}

template <typename T, uint32_t METHOD>
void CheckKernel(const Shape& shape, bool halo, bool zeros = false, bool constant = false, bool constantPattern = false,
                 int64_t tileH = 2, int64_t tileW = 8, bool c3 = false, bool plant = false)
{
    SCOPED_TRACE(::testing::Message() << "method=" << METHOD << " halo=" << halo << " c3=" << c3);
    const auto data = MakeTiling<T>(shape, METHOD, halo, tileH, tileW, c3);
    auto image = Samples<T>(shape.batch * shape.h * shape.w * shape.c, 31);
    auto pattern = Samples<T>(data.templateElements, 177);
    if (zeros || constant) {
        std::fill(image.begin(), image.end(), static_cast<T>(zeros ? 0 : 1));
        std::fill(pattern.begin(), pattern.end(), static_cast<T>(zeros ? 0 : 1));
    } else if (constantPattern) {
        std::fill(pattern.begin(), pattern.end(), static_cast<T>(2));
    }
    if (plant) {
        for (int64_t ty = 0; ty < shape.th; ++ty) {
            for (int64_t tx = 0; tx < shape.tw; ++tx) {
                for (int64_t c = 0; c < shape.c; ++c) {
                    image[((shape.h - shape.th + ty) * shape.w + shape.w - shape.tw + tx) * shape.c +
                          c] = pattern[(ty * shape.tw + tx) * shape.c + c];
                }
            }
        }
    }
    const auto expected = Reference(image, pattern, data);
    GmBuffer imageGm(image.size() * sizeof(T)), patternGm(pattern.size() * sizeof(T));
    GmBuffer scoreGm(expected.size() * sizeof(float)), bestGm(shape.batch * sizeof(float));
    GmBuffer locationGm(shape.batch * 2 * sizeof(int64_t)), tilingGm(sizeof(data));
    GmBuffer workspace(data.workspacePackedImageOffset + data.packedImageBytes);
    ASSERT_NE(imageGm.Data(), nullptr);
    ASSERT_NE(patternGm.Data(), nullptr);
    ASSERT_NE(scoreGm.Data(), nullptr);
    ASSERT_NE(bestGm.Data(), nullptr);
    ASSERT_NE(locationGm.Data(), nullptr);
    ASSERT_NE(tilingGm.Data(), nullptr);
    ASSERT_NE(workspace.Data(), nullptr);
    std::memcpy(imageGm.Data(), image.data(), image.size() * sizeof(T));
    std::memcpy(patternGm.Data(), pattern.data(), pattern.size() * sizeof(T));
    std::memcpy(tilingGm.Data(), &data, sizeof(data));
    auto* actual = reinterpret_cast<float*>(scoreGm.Data());
    auto* best = reinterpret_cast<float*>(bestGm.Data());
    auto* location = reinterpret_cast<int64_t*>(locationGm.Data());
    std::fill(actual, actual + expected.size(), std::numeric_limits<float>::quiet_NaN());
    std::fill(best, best + shape.batch, std::numeric_limits<float>::quiet_NaN());
    std::fill(location, location + shape.batch * 2, -1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kernel = MatchTemplateKernelUt<T, METHOD>;
    ICPU_RUN_KF(kernel, 1U, imageGm.Data(), patternGm.Data(), scoreGm.Data(), bestGm.Data(), locationGm.Data(),
                workspace.Data(), tilingGm.Data());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i])) << "score index " << i;
        EXPECT_NEAR(actual[i], expected[i], 2e-4 + 2e-4 * std::abs(expected[i])) << "score index " << i;
    }
    const int64_t perBatch = data.outputH * data.outputW;
    for (int64_t n = 0; n < shape.batch; ++n) {
        int64_t index = 0;
        for (int64_t i = 1; i < perBatch; ++i) {
            const bool better = METHOD < 2 ? expected[n * perBatch + i] < expected[n * perBatch + index] :
                                             expected[n * perBatch + i] > expected[n * perBatch + index];
            if (better) {
                index = i;
            }
        }
        EXPECT_EQ(location[2 * n], index % data.outputW);
        EXPECT_EQ(location[2 * n + 1], index / data.outputW);
        EXPECT_NEAR(best[n], expected[n * perBatch + index], 2e-4 + 2e-4 * std::abs(expected[n * perBatch + index]));
        if (plant) {
            EXPECT_EQ(index, perBatch - 1);
        }
    }
}

template <typename T>
void CheckAllMethods(const Shape& shape, bool halo, bool ties = false)
{
    CheckKernel<T, 0>(shape, halo, false, ties);
    CheckKernel<T, 1>(shape, halo, false, ties);
    CheckKernel<T, 2>(shape, halo, false, ties);
    CheckKernel<T, 3>(shape, halo, false, ties);
    CheckKernel<T, 4>(shape, halo, false, ties);
    CheckKernel<T, 5>(shape, halo, false, ties);
}

template <typename T>
class MatchTemplateKernelTest : public ::testing::Test {};

using InputTypes = ::testing::Types<float, half, uint8_t>;
TYPED_TEST_SUITE(MatchTemplateKernelTest, InputTypes);

TYPED_TEST(MatchTemplateKernelTest, HaloOddShapeBatchAndAllMethods)
{
    CheckAllMethods<TypeParam>({2, 7, 14, 1, 3, 4}, true);
}

TYPED_TEST(MatchTemplateKernelTest, GmFallbackAndAllMethods) { CheckAllMethods<TypeParam>({1, 7, 14, 3, 3, 4}, false); }

TYPED_TEST(MatchTemplateKernelTest, HaloThreeChannelsAndAllMethods)
{
    CheckAllMethods<TypeParam>({1, 7, 14, 3, 3, 4}, true);
}

TYPED_TEST(MatchTemplateKernelTest, ZeroDenominatorBothPaths)
{
    const Shape shape{1, 7, 14, 3, 3, 4};
    for (bool halo : {false, true}) {
        CheckKernel<TypeParam, 1>(shape, halo, true);
        CheckKernel<TypeParam, 3>(shape, halo, true);
        CheckKernel<TypeParam, 5>(shape, halo, true);
        CheckKernel<TypeParam, 5>(shape, halo, false, false, true);
    }
}

TYPED_TEST(MatchTemplateKernelTest, ExactTiesSelectFirstIndexBothPaths)
{
    CheckAllMethods<TypeParam>({1, 7, 14, 3, 3, 4}, true, true);
    CheckAllMethods<TypeParam>({1, 7, 14, 3, 3, 4}, false, true);
}

TYPED_TEST(MatchTemplateKernelTest, InputSizedTemplateAllMethods)
{
    CheckAllMethods<TypeParam>({1, 3, 5, 1, 3, 5}, true);
}

TYPED_TEST(MatchTemplateKernelTest, UniqueBestAtBottomRightBothPaths)
{
    for (bool halo : {false, true}) {
        CheckKernel<TypeParam, 0>({1, 7, 14, 3, 3, 4}, halo, false, false, false, 2, 8, false, true);
        CheckKernel<TypeParam, 5>({1, 7, 14, 3, 3, 4}, halo, false, false, false, 2, 8, false, true);
    }
}

TEST(MatchTemplateKernelBoundaryTest, FloatC3GatherCrosses512Elements)
{
    const Shape shape{1, 2, 530, 3, 1, 16};
    CheckKernel<float, 0>(shape, true, false, false, false, 1, 512);
    CheckKernel<float, 2>(shape, true, false, false, false, 1, 512);
    CheckKernel<float, 5>(shape, true, false, false, false, 1, 512);
}

// Compact dimensions exercise the real C3 implementation in CPU debug. The
// 512x512 Host dispatch predicate and multicore hardware tie behavior are tested
// separately by the full-size ACLNN Golden cases.
TEST(MatchTemplateKernelBoundaryTest, HalfC3CcorrTie)
{
    CheckKernel<half, 2>({1, 31, 32, 3, 16, 16}, true, false, true, false, 16, 512, true);
}

TEST(MatchTemplateKernelBoundaryTest, HalfC3CcorrSingleTailRow)
{
    // Same 497-column tail as the 512x512/16x16 schedule, but only one
    // output row: this must use the generic score buffer, not C3 storage.
    CheckKernel<half, 2>({1, 16, 512, 3, 16, 16}, true, false, false, false, 16, 512, true);
}

TEST(MatchTemplateKernelBoundaryTest, HalfC3CcorrFullTileAndTailRow)
{
    // outputH=17 exercises a full specialized tile followed by one generic
    // tail row in the same kernel invocation, including nonconstant scores.
    CheckKernel<half, 2>({1, 32, 512, 3, 16, 16}, true, false, false, false, 16, 512, true);
}

TEST(MatchTemplateKernelBoundaryTest, Uint8C3PrepackCcorrTie)
{
    CheckKernel<uint8_t, 2>({1, 31, 32, 3, 16, 16}, true, false, true, false, 16, 512, true);
}

TEST(MatchTemplateKernelBoundaryTest, HalfC3NormalizedZeroVariance)
{
    CheckKernel<half, 5>({1, 19, 32, 3, 16, 16}, true, false, true, false, 4, 512, true);
}

TEST(MatchTemplateKernelBoundaryTest, Uint8C3PrepackNormalizedZeroVariance)
{
    CheckKernel<uint8_t, 5>({1, 19, 32, 3, 16, 16}, true, false, true, false, 4, 512, true);
}

} // namespace
