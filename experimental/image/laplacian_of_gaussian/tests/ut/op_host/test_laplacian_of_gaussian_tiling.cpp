/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include <cmath>
#include <cstring>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/laplacian_of_gaussian_tiling_data.h"
using V = Ops::Cv::AnyValue;
static int compileInfo = 0;
static gert::TilingContextPara Params(std::vector<int64_t> shape = {128, 129}, ge::DataType dt = ge::DT_FLOAT,
                                      float sigma = 1, int64_t radius = 4, float truncate = 4,
                                      std::string border = "REFLECT", uint64_t ub = 196608, uint64_t cores = 40,
                                      uint64_t cap = 2048)
{
    gert::Shape s;
    for (auto dim : shape)
        s.AppendDim(dim);
    gert::StorageShape storage;
    storage.MutableOriginShape() = s;
    storage.MutableStorageShape() = s;
    return gert::TilingContextPara("LaplacianOfGaussian", {{storage, dt, ge::FORMAT_ND}},
                                   {{storage, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{"sigma", V::CreateFrom<float>(sigma)},
                                    {"radius", V::CreateFrom<int64_t>(radius)},
                                    {"truncate", V::CreateFrom<float>(truncate)},
                                    {"border_type", V::CreateFrom<std::string>(border)}},
                                   &compileInfo, "Ascend910B", cores, ub, cap);
}
TEST(LoGTiling, NormalMatrix)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);

    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
        for (auto sh : {std::vector<int64_t>{128, 129}, {2, 128, 129}, {2, 1, 128, 129}})
            for (auto border : {"REFLECT", "REFLECT_101", "REPLICATE", "CONSTANT"})
                for (int radius : {-1, 4}) {
                    auto p = Params(sh, dt, 1, radius, 4, border);
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(p, info));
                    ASSERT_GE(info.tilingDataSize, 192);
                    LaplacianOfGaussianTilingData t;
                    memcpy(&t, info.tilingData.get(), 64);
                    EXPECT_EQ(info.tilingKey, 0);
                    ASSERT_EQ(info.workspaceSizes.size(), 1);
                    EXPECT_GT(info.workspaceSizes[0], 0);
                    EXPECT_EQ(t.tileWidth, 160);
                    EXPECT_EQ(t.reserved & 1u, 1u);
                    EXPECT_EQ(t.radius, 4);
                    EXPECT_EQ(t.k, 9);
                    EXPECT_EQ(t.border, std::string(border) == "REFLECT"     ? 0 :
                                        std::string(border) == "REFLECT_101" ? 1 :
                                        std::string(border) == "REPLICATE"   ? 2 :
                                                                               3);
                    EXPECT_EQ(t.w, 129);
                    EXPECT_EQ(t.h, 128);
                    EXPECT_EQ(t.n, sh.size() == 2 ? 1 : 2);
                    {
                        uint64_t rc = t.reserved >> 1;
                        ASSERT_GE(rc, 1u);
                        EXPECT_EQ(t.tiles, t.n * ((128 + rc - 1) / rc));
                    }
                    EXPECT_EQ(t.blocks, std::min<uint64_t>(t.tiles, 40));
                    EXPECT_EQ(t.weightOffset, 64);
                    EXPECT_EQ(t.weightStride, 64);
                    double sum = 0;
                    for (int i = -4; i <= 4; ++i)
                        sum += std::exp(-0.5 * i * i);
                    const float* g = reinterpret_cast<const float*>(info.tilingData.get() + 64);
                    const float* d = reinterpret_cast<const float*>(info.tilingData.get() + 128);
                    double dsum = 0;
                    for (int i = 0; i < 9; ++i)
                        dsum += d[i];
                    EXPECT_NE(dsum, 0.0);
                    ASSERT_EQ(info.tilingDataSize, 192);
                    for (int j = 36; j < 64; ++j) {
                        EXPECT_EQ(info.tilingData.get()[64 + j], 0);
                        EXPECT_EQ(info.tilingData.get()[128 + j], 0);
                    }
                    for (int i = 0; i < 9; ++i) {
                        double e = std::exp(-.5 * (i - 4) * (i - 4)) / sum;
                        EXPECT_FLOAT_EQ(g[i], float(e));
                        EXPECT_FLOAT_EQ(d[i], float(e * ((i - 4) * (i - 4) - 1)));
                    }
                }
}
static void CoverHalo(std::vector<int>& hits, const LaplacianOfGaussianTilingData& t, uint64_t h, uint64_t w)
{
    uint64_t cols = (w + t.tileWidth - 1) / t.tileWidth, rc = t.reserved >> 1, chunks = (h + rc - 1) / rc,
             perN = cols * chunks;
    for (uint64_t id = 0; id < t.tiles; ++id) {
        uint64_t n = id / perN, rem = id % perN, ci = rem % cols, ck = rem / cols;
        uint64_t r0 = ck * rc, r1 = std::min<uint64_t>(h, r0 + rc);
        for (uint64_t r = r0; r < r1; ++r)
            for (uint64_t j = ci * t.tileWidth; j < std::min<uint64_t>(w, (ci + 1) * t.tileWidth); ++j)
                ++hits[n * h * w + r * w + j];
    }
}
static void CoverLegacy(std::vector<int>& hits, const LaplacianOfGaussianTilingData& t, uint64_t h, uint64_t w)
{
    uint64_t cols = (w + 31) / 32;
    for (uint64_t id = 0; id < t.tiles; ++id) {
        uint64_t col = (id % cols) * 32, row = (id / cols) % h, n = id / (cols * h);
        for (uint64_t j = col; j < std::min<uint64_t>(w, col + 32); ++j)
            ++hits[n * h * w + row * w + j];
    }
}
TEST(LoGTiling, TailCoverage)
{
    for (int w : {1, 31, 32, 33, 127, 128, 129, 159, 160, 161, 4097}) {
        TilingInfo i;
        ASSERT_TRUE(ExecuteTiling(Params({7, w}), i));
        LaplacianOfGaussianTilingData t;
        memcpy(&t, i.tilingData.get(), 64);
        EXPECT_EQ(t.tileWidth % 32, 0);
        EXPECT_LE(t.tileWidth, 4096);
        std::vector<int> hits(7 * w);
        if (t.reserved & 1u) {
            uint64_t cols = (w + t.tileWidth - 1) / t.tileWidth, rc = t.reserved >> 1, chunks = (7 + rc - 1) / rc;
            EXPECT_EQ(t.tiles, cols * chunks);
            CoverHalo(hits, t, 7, w);
        } else {
            uint64_t cols = (w + 31) / 32;
            EXPECT_EQ(t.tiles, 7 * cols);
            CoverLegacy(hits, t, 7, w);
        }
        for (auto hit : hits)
            EXPECT_EQ(hit, 1);
    }
}
TEST(LoGTiling, InvalidAttributes)
{
    for (float s : {0.f, -1.f, INFINITY, NAN})
        ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, s));
    for (float tr : {0.f, -1.f, INFINITY, NAN})
        ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, tr));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, -2));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "reflect"));
}
TEST(LoGTiling, InvalidShapes)
{
    for (auto s : {std::vector<int64_t>{3}, {1, 1, 1, 1, 1}, {2, 2, 3, 4}, {0, 3}, {-1, 3}, {INT64_MAX, 2}})
        ExecuteTestCase(Params(s));
}
TEST(LoGTiling, Capacity)
{
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, 1, 123), info));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 124));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 196608, 40, 191));
    EXPECT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 196608, 40, 192), info));
}
TEST(LoGTiling, Resources)
{
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 8192));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 196608, 0));
    TilingInfo i;
    ASSERT_TRUE(ExecuteTiling(Params({128, 129}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 11000), i));
    LaplacianOfGaussianTilingData t;
    memcpy(&t, i.tilingData.get(), 64);
    EXPECT_LT(t.tileWidth, 160);
}

// CANN 9.1 PlatformAscendC reserves 256 bytes from the supplied UB.
TEST(LoGTiling, ExactBudgetAndFallback)
{
    for (uint64_t ub : {21759, 21760, 21761}) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(Params({128, 31}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", ub + 256), info));
        LaplacianOfGaussianTilingData t;
        memcpy(&t, info.tilingData.get(), 64);
        EXPECT_EQ(t.tileWidth, 32);
        EXPECT_EQ(t.reserved & 1u, ub >= 21760 ? 1u : 0u);
    }
}
TEST(LoGTiling, ExtremeWeights)
{
    for (float sigma : {1e-30f, 1e30f})
        ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, sigma));
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, 1, 0), info));
    for (float sigma : {std::nextafter(1.125f, 0.f), 1.125f, std::nextafter(1.125f, 2.f)}) {
        ASSERT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, sigma, -1), info));
        LaplacianOfGaussianTilingData t;
        memcpy(&t, info.tilingData.get(), 64);
        EXPECT_EQ(t.radius, uint32_t(std::floor(double(sigma) * 4 + .5)));
    }
}

TEST(LoGTiling, MockCoresAndSubtiles)
{
    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
        for (uint64_t cores : {1, 2, 40, 4096})
            for (int h : {1, 7})
                for (int w : {1, 31, 32, 33, 127, 128, 129, 159, 160, 161}) {
                    TilingInfo i;
                    ASSERT_TRUE(ExecuteTiling(Params({2, h, w}, dt, 1, 4, 4, "REFLECT", 196608, cores), i));
                    LaplacianOfGaussianTilingData t;
                    memcpy(&t, i.tilingData.get(), 64);
                    EXPECT_EQ(t.blocks, std::min<uint64_t>(t.tiles, cores));
                    std::vector<int> hits(2 * h * w);
                    if (t.reserved & 1u)
                        CoverHalo(hits, t, h, w);
                    else
                        CoverLegacy(hits, t, h, w);
                    for (auto hit : hits)
                        EXPECT_EQ(hit, 1);
                }
}
TEST(LoGTiling, EachDtypeBudget)
{
    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT}) {
        uint64_t elem = dt == ge::DT_UINT8 ? 1 : dt == ge::DT_FLOAT16 ? 2 : 4;
        uint64_t threshold = (16 + 2 * 9) * 4 * 32 + (elem + 10) * 64 + 128 + 16384;
        for (int delta : {-1, 0, 1}) {
            TilingInfo i;
            ASSERT_TRUE(ExecuteTiling(Params({128, 31}, dt, 1, 4, 4, "REFLECT", threshold + delta + 256), i));
            LaplacianOfGaussianTilingData t;
            memcpy(&t, i.tilingData.get(), 64);
            EXPECT_EQ(t.reserved & 1u, delta >= 0 ? 1u : 0u);
        }
    }
}
TEST(LoGTiling, PeriodAndOutputBytes)
{
    ExecuteTestCase(Params({INT64_MAX / 2 + 1, 1}, ge::DT_UINT8, 1, 0));
    ExecuteTestCase(Params({INT64_MAX / 4 + 1, 1}, ge::DT_FLOAT, 1, 0));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1e10f, -1));
    TilingInfo i;
    ASSERT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, 1, 4, 1), i));
    LaplacianOfGaussianTilingData t;
    memcpy(&t, i.tilingData.get(), 64);
    EXPECT_EQ(t.radius, 4);
}

TEST(LoGTiling, MissingAttributeSuffixes)
{
    for (size_t size = 0; size < 4; ++size) {
        auto p = Params();
        p.attrs_.erase(p.attrs_.begin() + size, p.attrs_.end());
        ExecuteTestCase(p);
    }
}
TEST(LoGTiling, NullContext)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
}
#include "platform/platform_infos_def.h"
TEST(LoGTiling, MissingBuilderInputs)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    gert::StorageShape shape;
    shape.MutableStorageShape().AppendDim(3);
    shape.MutableStorageShape().AppendDim(5);
    shape.MutableOriginShape() = shape.GetStorageShape();
    gert::Tensor input(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                       gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc{{"ai_core_cnt", "40"}}, spec{{"ub_size", "196608"}},
        version{{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");

    for (int missing = 0; missing < 5; ++missing) {
        gert::TilingContextFaker f;
        f.SetOpType("LaplacianOfGaussian").NodeIoNum(1, 1);
        f.InputTensors({&input}).OutputTensors({&input});
        f.Attr("sigma", 1.f)
            .Attr("radius", int64_t(4))
            .Attr("truncate", 4.f)
            .Attr("border_type", ge::AscendString("REFLECT"));
        f.PlatformInfo(&platform).CompileInfo(&compileInfo);
        auto raw = gert::TilingData::CreateCap(2048);
        f.TilingData(raw.get());
        if (missing == 4)
            reinterpret_cast<gert::TilingData*>(raw.get())->Init(2048, nullptr);
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        f.Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
        auto holder = f.Build();
        auto c = holder.GetContext();
        ASSERT_NE(c, nullptr);
        if (missing == 0)
            const_cast<gert::Chain*>(reinterpret_cast<gert::KernelContext*>(c)->GetInput(0))->Set(nullptr, nullptr);
        if (missing == 1)
            const_cast<gert::Chain*>(reinterpret_cast<gert::KernelContext*>(c)->GetInput(3))->Set(nullptr, nullptr);
        if (missing == 2)
            reinterpret_cast<gert::KernelContext*>(c)->GetOutput(3)->Set(nullptr, nullptr);
        if (missing == 3)
            reinterpret_cast<gert::KernelContext*>(c)->GetOutput(4)->Set(nullptr, nullptr);
        EXPECT_EQ(impl->tiling(c), ge::GRAPH_FAILED) << missing;
        if (missing == 3)
            EXPECT_EQ(c->GetRawTilingData()->GetDataSize(), 192);
    }
}

TEST(LoGTiling, BudgetWidthsAndLegacyTails)
{
    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT}) {
        uint64_t elem = dt == ge::DT_UINT8 ? 1 : dt == ge::DT_FLOAT16 ? 2 : 4;
        for (uint64_t width : {32, 64, 160}) {
            uint64_t padded = (width + 8 + 31) / 32 * 32,
                     budget = (16 + 2 * 9) * 4 * width + (10 + elem) * padded + 128 + 16384;
            for (int delta : {-1, 0, 1}) {
                TilingInfo i;
                ASSERT_TRUE(ExecuteTiling(Params({128, 160}, dt, 1, 4, 4, "REFLECT", budget + delta + 256, 1), i));
                LaplacianOfGaussianTilingData t;
                memcpy(&t, i.tilingData.get(), 64);
                EXPECT_EQ(t.reserved & 1u, (width == 32 && delta < 0) ? 0u : 1u);
                EXPECT_EQ(t.tileWidth, delta < 0 ? (width == 32 ? 32 : width - 32) : width);
            }
        }
        uint64_t minimum = 192 + 8192 + 32 * 40;
        ExecuteTestCase(Params({7, 31}, dt, 1, 4, 4, "REFLECT", minimum - 1 + 256));
        for (int delta : {0, 1})
            for (int tail = 1; tail <= 7; ++tail) {
                TilingInfo i;
                ASSERT_TRUE(
                    ExecuteTiling(Params({2, 7, 32 + tail}, dt, 1, 4, 4, "REFLECT", minimum + delta + 256, 40), i));
                LaplacianOfGaussianTilingData t;
                memcpy(&t, i.tilingData.get(), 64);
                ASSERT_EQ(t.reserved, 0);
                ASSERT_EQ(t.tileWidth, 32);
                std::vector<int> hits(2 * 7 * (32 + tail));
                uint64_t cols = 2;
                for (uint64_t b = 0; b < t.blocks; ++b)
                    for (uint64_t id = b; id < t.tiles; id += t.blocks) {
                        uint64_t col = id % cols * 32, row = id / cols, count = std::min<uint64_t>(32, 32 + tail - col);
                        for (uint64_t part = 0; part < count; part += 8)
                            for (uint64_t j = part; j < std::min<uint64_t>(part + 8, count); ++j)
                                ++hits[row * (32 + tail) + col + j];
                    }
                for (int hit : hits)
                    EXPECT_EQ(hit, 1);
            }
    }
}
TEST(LoGTiling, RadiusCapacityAndPadding)
{
    for (int radius : {0, 123}) {
        uint64_t k = 2 * radius + 1, stride = (k * 4 + 31) / 32 * 32, bytes = 64 + 2 * stride;
        ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, radius, 4, "REFLECT", 196608, 40, bytes - 1));
        TilingInfo i;
        ASSERT_TRUE(ExecuteTiling(Params({2, 3}, ge::DT_FLOAT, 1, radius, 4, "REFLECT", 196608, 40, bytes), i));
        ASSERT_EQ(i.tilingDataSize, bytes);
        LaplacianOfGaussianTilingData t;
        memcpy(&t, i.tilingData.get(), 64);
        EXPECT_EQ(t.radius, radius);
        EXPECT_EQ(t.k, k);
        EXPECT_EQ(t.weightStride, stride);
        double sum = 0;
        for (int j = 0; j < k; ++j) {
            double z = j - radius;
            sum += std::exp(-.5 * z * z);
        }
        for (int j = 0; j < k; ++j) {
            double z = j - radius, g = std::exp(-.5 * z * z) / sum;
            float gv, dv;
            memcpy(&gv, i.tilingData.get() + 64 + 4 * j, 4);
            memcpy(&dv, i.tilingData.get() + 64 + stride + 4 * j, 4);
            EXPECT_FLOAT_EQ(gv, float(g));
            EXPECT_FLOAT_EQ(dv, float(g * (z * z - 1)));
        }
        for (uint64_t j = k * 4; j < stride; ++j) {
            EXPECT_EQ(i.tilingData.get()[64 + j], 0);
            EXPECT_EQ(i.tilingData.get()[64 + stride + j], 0);
        }
    }
}

TEST(LoGTiling, FailedWeightsPreserveRawSentinels)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    gert::StorageShape shape;
    shape.MutableStorageShape().AppendDim(2);
    shape.MutableStorageShape().AppendDim(3);
    shape.MutableOriginShape() = shape.GetStorageShape();
    gert::Tensor input(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                       gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc{{"ai_core_cnt", "40"}}, spec{{"ub_size", "196608"}},
        version{{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
    for (int mode = 0; mode < 3; ++mode) {
        float sigma = mode == 0 ? 1e-30f : mode == 1 ? 1e30f : 1.f;
        size_t cap = mode == 2 ? 191 : 2048;
        std::vector<unsigned char> guarded(cap + 32, 0xA5);
        auto raw = gert::TilingData::CreateCap(cap);
        auto td = reinterpret_cast<gert::TilingData*>(raw.get());
        td->Init(cap, guarded.data() + 16);
        auto ws = gert::ContinuousVector::Create<size_t>(1);
        gert::TilingContextFaker f;
        f.SetOpType("LaplacianOfGaussian").NodeIoNum(1, 1).InputTensors({&input}).OutputTensors({&input});
        f.Attr("sigma", sigma)
            .Attr("radius", int64_t(4))
            .Attr("truncate", 4.f)
            .Attr("border_type", ge::AscendString("REFLECT"));
        f.PlatformInfo(&platform)
            .CompileInfo(&compileInfo)
            .TilingData(td)
            .Workspace(reinterpret_cast<gert::ContinuousVector*>(ws.get()));
        auto holder = f.Build();
        ASSERT_NE(holder.GetContext(), nullptr);
        EXPECT_EQ(impl->tiling(holder.GetContext()), ge::GRAPH_FAILED);
        EXPECT_EQ(td->GetDataSize(), 0);
        for (auto byte : guarded)
            EXPECT_EQ(byte, 0xA5);
    }
}

TEST(LoGTiling, PeriodWidthAndTinyUb)
{
    ExecuteTestCase(Params({1, INT64_MAX / 2 + 1}, ge::DT_UINT8, 1, 0));
    ExecuteTestCase(Params({2, 3}, ge::DT_FLOAT, 1, 4, 4, "REFLECT", 256 + 191));
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    EXPECT_EQ(impl->tiling_parse(nullptr), ge::GRAPH_SUCCESS);
}

#include "tiling/platform/platform_ascendc.h"
TEST(LoGTiling, PlatformReportedBudget)
{
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc{{"ai_core_cnt", "40"}}, spec{{"ub_size", "196608"}},
        version{{"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
    platform_ascendc::PlatformAscendC actual(&platform);
    uint64_t ub = 0;
    actual.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    EXPECT_EQ(ub, 196608 - 256);
    EXPECT_EQ(actual.GetCoreNumAiv(), 40);
    TilingInfo i;
    ASSERT_TRUE(ExecuteTiling(Params(), i));
    ASSERT_EQ(i.workspaceSizes.size(), 1);
    EXPECT_EQ(i.workspaceSizes[0], actual.GetLibApiWorkSpaceSize());
}
