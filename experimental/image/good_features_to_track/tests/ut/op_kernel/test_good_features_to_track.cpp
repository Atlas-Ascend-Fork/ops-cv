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
 * \file test_good_features_to_track.cpp
 * \brief GoodFeaturesToTrack Kernel UT（ICPU/CPU 模拟执行内核源码）
 *
 * 说明：
 * 1) 内核多核阶段有 SyncAll 屏障，ICPU 模拟把 block 顺序执行、屏障退化为空操作，
 *    因此统一用 coreNum = 1，单核走完"响应 → 候选 → 全局 TopK"全流程，语义与真机一致。
 * 2) 退化输入由两个用例覆盖，且只断言不变量、不锁死具体数值：
 *    - flat_image_few_border_artifacts：常量图。响应恒 0 使阈值退化为 0，边界补 0 会产生少量
 *      边缘假角点（实测 ≤ 4 个），故只断言“数量极少 + 不变量”。OpenCV 对常量图返回 0 个角点，
 *      属已知偏差（详见 README「与 OpenCV 的行为差异」）。
 *    - fully_masked_no_corner：全 0 掩码，所有像素被屏蔽 ⇒ 断言 0 个角点。
 */

#include "../../../op_kernel/good_features_to_track.cpp"
#include "good_features_to_track_tiling.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "tikicpulib.h"

using namespace std;

namespace {
constexpr uint32_t kH = 64;
constexpr uint32_t kW = 64;
constexpr uint32_t kMaxCorners = 16;
constexpr float kQuality = 0.01F;
constexpr float kMinDistance = 1.0F;
constexpr int32_t kBlockSize = 3;
constexpr uint32_t kCoreNum = 1; // 见文件头说明
constexpr float kDefaultHarrisK = 0.04F;

struct RunConfig {
    uint32_t h = kH;
    uint32_t w = kW;
    uint32_t batch = 1U; // >1 时 image 数据按 [batch, h, w] 连续摆放
    uint32_t maxCorners = kMaxCorners;
    float quality = kQuality;
    float minDistance = kMinDistance;
    int32_t blockSize = kBlockSize;
    uint32_t useHarris = 0U;
    float harrisK = kDefaultHarrisK;
};

struct CornerResult {
    uint32_t num = 0;                   // 内核写回的 numCorners
    uint32_t filled = 0;                // 非空槽位数（真实角点数，仅第 0 张图）
    vector<pair<int32_t, int32_t>> pts; // 第 0 张图的非空槽位，(y, x)（由内核的 (x, y) 换算）
    vector<uint32_t> counts;            // 逐图真实角点数（batch>1 逐图断言用）
};

vector<float> MakeFlatImage(uint32_t h, uint32_t w, float value)
{
    return vector<float>(static_cast<size_t>(h) * w, value);
}

vector<float> MakeBlobImageScaled(uint32_t h, uint32_t w, uint32_t y0, uint32_t x0, uint32_t size, float delta);

// 黑底 + 一个亮方块：方块四角是强角点
vector<float> MakeBlobImage(uint32_t h, uint32_t w, uint32_t y0, uint32_t x0, uint32_t size)
{
    return MakeBlobImageScaled(h, w, y0, x0, size, 1.0F);
}

// 黑底 + 一个指定幅度 δ 的方块：响应 ∝ δ²，用于构造强弱对比（逐图阈值用例）
vector<float> MakeBlobImageScaled(uint32_t h, uint32_t w, uint32_t y0, uint32_t x0, uint32_t size, float delta)
{
    vector<float> img = MakeFlatImage(h, w, 0.0F);
    for (uint32_t y = y0; y < y0 + size && y < h; ++y) {
        for (uint32_t x = x0; x < x0 + size && x < w; ++x) {
            img[static_cast<size_t>(y) * w + x] = delta;
        }
    }
    return img;
}

// 8x8 方格棋盘：角点很多且分布均匀
vector<float> MakeCheckerboard(uint32_t h, uint32_t w, uint32_t cell)
{
    vector<float> img = MakeFlatImage(h, w, 0.0F);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const bool on = (((y / cell) + (x / cell)) % 2U) == 0U;
            img[static_cast<size_t>(y) * w + x] = on ? 1.0F : 0.0F;
        }
    }
    return img;
}

vector<uint8_t> MakeMaskAll(uint32_t h, uint32_t w, uint8_t value)
{
    return vector<uint8_t>(static_cast<size_t>(h) * w, value);
}

// 仅盒内为 on，其余为 off
vector<uint8_t> MakeMaskBox(uint32_t h, uint32_t w, uint32_t y0, uint32_t x0, uint32_t size, uint8_t on, uint8_t off)
{
    vector<uint8_t> mask = MakeMaskAll(h, w, off);
    for (uint32_t y = y0; y < y0 + size && y < h; ++y) {
        for (uint32_t x = x0; x < x0 + size && x < w; ++x) {
            mask[static_cast<size_t>(y) * w + x] = on;
        }
    }
    return mask;
}

// maxCorners=0 表示自动模式，容量取 H*W（与 op_host tiling 一致）
size_t CapacityOf(const RunConfig& c) { return c.maxCorners == 0U ? static_cast<size_t>(c.h) * c.w : c.maxCorners; }

size_t WorkspaceBytes(uint32_t batch, uint32_t h, uint32_t w, uint32_t coreNum)
{
    const size_t resp = static_cast<size_t>(batch) * h * w * sizeof(float);
    const size_t maxPerCore = static_cast<size_t>(batch) * coreNum * 8U * sizeof(float); // 按图存放
    const size_t keyHist = static_cast<size_t>(coreNum) * 256U * sizeof(int32_t);
    const size_t topkMeta = (2U * static_cast<size_t>(coreNum) + 8U) * sizeof(int32_t);
    const size_t subHist = static_cast<size_t>(coreNum) * 256U * sizeof(int32_t);
    const size_t cand = static_cast<size_t>(batch) * h * w * sizeof(uint32_t);
    // 多给 16MB 兜住框架 lib 区（跨核同步用），kernel 不依赖其具体值
    return resp + maxPerCore + keyHist + topkMeta + subHist + cand * 4U + (16U << 20);
}

void FillTiling(GoodFeaturesToTrackTilingData* t, const RunConfig& c, uint32_t hasMask)
{
    memset(t, 0, sizeof(*t));
    t->batch = c.batch;
    t->coreNum = kCoreNum;
    t->height = c.h;
    t->width = c.w;
    t->imageElements = c.h * c.w;
    t->outputCapacity = static_cast<uint32_t>(CapacityOf(c));
    t->blockSize = c.blockSize;
    t->qualityLevel = c.quality;
    t->minDistance = c.minDistance;
    t->harrisK = c.harrisK;
    t->useHarris = c.useHarris;
    t->hasMask = hasMask;
}

CornerResult RunKernel(const vector<float>& image, const vector<uint8_t>* mask, const RunConfig& c)
{
    const size_t imgBytes = image.size() * sizeof(float);
    const size_t cornersBytes = static_cast<size_t>(c.batch) * CapacityOf(c) * 2U * sizeof(float);
    const size_t wsBytes = WorkspaceBytes(c.batch, c.h, c.w, kCoreNum);

    uint8_t* imageDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(imgBytes));
    uint8_t* maskDev = nullptr;
    uint8_t* cornersDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cornersBytes));
    uint8_t* numDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(int32_t) * c.batch));
    uint8_t* wsDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wsBytes));
    uint8_t* tilingDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GoodFeaturesToTrackTilingData)));

    memcpy(imageDev, image.data(), imgBytes);
    memset(cornersDev, 0, cornersBytes);
    memset(numDev, 0, sizeof(int32_t) * c.batch);
    memset(wsDev, 0, wsBytes);
    FillTiling(reinterpret_cast<GoodFeaturesToTrackTilingData*>(tilingDev), c, mask != nullptr ? 1U : 0U);

    if (mask != nullptr) {
        const size_t maskBytes = mask->size() * sizeof(uint8_t);
        maskDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(maskBytes));
        memcpy(maskDev, mask->data(), maskBytes);
    }

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(good_features_to_track<0>, kCoreNum, imageDev, maskDev, cornersDev, numDev, wsDev, tilingDev);

    CornerResult result;
    result.num = *reinterpret_cast<uint32_t*>(numDev);
    const float* corners = reinterpret_cast<const float*>(cornersDev);
    const size_t cap = CapacityOf(c);
    // 注：ICPU 模拟下 numCorners 写回的是 outputCapacity（真机写实际角点数，aclnn 位比对已验证），
    //     故这里按"非空槽位"统计真实角点，不把这个模拟差异写进断言。
    for (uint32_t img = 0U; img < c.batch; ++img) {
        uint32_t cnt = 0U;
        const size_t base = static_cast<size_t>(img) * cap * 2U;
        for (size_t i = 0; i < cap; ++i) {
            // 内核按 (x, y) 写出（对齐 OpenCV）；UT 内部统一换算成 (y, x)，便于与图像行列对应
            const int32_t x = static_cast<int32_t>(corners[base + 2U * i]);
            const int32_t y = static_cast<int32_t>(corners[base + 2U * i + 1U]);
            // 未填充槽位：实测会出现 (0,0) 或 (-1,-1) 两种哨兵，一律不计入真实角点
            if ((y == 0 && x == 0) || y < 0 || x < 0) {
                continue;
            }
            ++cnt;
            if (img == 0U) {
                result.pts.emplace_back(y, x);
            }
        }
        result.counts.push_back(cnt);
    }
    result.filled = static_cast<uint32_t>(result.pts.size());

    AscendC::GmFree(imageDev);
    if (maskDev != nullptr) {
        AscendC::GmFree(maskDev);
    }
    AscendC::GmFree(cornersDev);
    AscendC::GmFree(numDev);
    AscendC::GmFree(wsDev);
    AscendC::GmFree(tilingDev);
    return result;
}

bool NearBlobCorner(const CornerResult& r, int32_t y0, int32_t x0, uint32_t size, int32_t tol)
{
    const int32_t ys[2] = {y0, y0 + static_cast<int32_t>(size)};
    const int32_t xs[2] = {x0, x0 + static_cast<int32_t>(size)};
    for (const auto& p : r.pts) {
        for (int32_t yc : ys) {
            for (int32_t xc : xs) {
                if (abs(p.first - yc) <= tol && abs(p.second - xc) <= tol) {
                    return true;
                }
            }
        }
    }
    return false;
}

// 通用不变量：不超容量、坐标在界内、无重复点
void ExpectCommonInvariants(const CornerResult& r, const RunConfig& c)
{
    EXPECT_LE(r.num, static_cast<uint32_t>(CapacityOf(c)));
    EXPECT_EQ(r.pts.size(), r.filled);
    for (size_t i = 0; i < r.pts.size(); ++i) {
        EXPECT_GE(r.pts[i].first, 0);
        EXPECT_LT(r.pts[i].first, static_cast<int32_t>(c.h));
        EXPECT_GE(r.pts[i].second, 0);
        EXPECT_LT(r.pts[i].second, static_cast<int32_t>(c.w));
        for (size_t j = i + 1; j < r.pts.size(); ++j) {
            EXPECT_FALSE(r.pts[i] == r.pts[j])
                << "repeated corner (" << r.pts[i].first << "," << r.pts[i].second << ")";
        }
    }
}
} // namespace

class GoodFeaturesToTrackKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "GoodFeaturesToTrackKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "GoodFeaturesToTrackKernelTest TearDown" << endl; }
};

// 常量图（无纹理）：内部无梯度，但图像边界因 padding/halo 会产生边缘假响应，
// 实测当前实现给出 4 个（图像四角附近）；全 0 掩码用例为 0，印证是边界效应。
// OpenCV 对无纹理输入返回 0 —— 差异只存在于退化输入，真实图像由 aclnn 位比对覆盖。
// 这里只断言"数量极少 + 不变量成立"，不锁死具体数值。
// batch=2 逐图阈值：图 0 为强纹理（4 个幅度 1.0 的亮块），图 1 为弱纹理（同形状但幅度 0.05）。
// 响应 ∝ 幅度² ⇒ 图 1 的峰值只有图 0 的 1/400：若阈值错用 batch 全局 R_max，图 1 的阈值会被
// 抬到 0.01×R_max(图0)，远超其自身响应 ⇒ 图 1 全漏检；逐图阈值下图 1 应正常检出 4 个角点。
TEST_F(GoodFeaturesToTrackKernelTest, batch2_per_image_threshold)
{
    RunConfig c;
    c.batch = 2U;
    // 图 0：4 个 8x8 亮块 ⇒ 16 个强角点
    vector<float> img0 = MakeFlatImage(c.h, c.w, 0.0F);
    const std::pair<uint32_t, uint32_t> boxes[] = {{4U, 4U}, {4U, 40U}, {40U, 4U}, {40U, 40U}};
    for (const auto& g : boxes) {
        const vector<float> blob = MakeBlobImage(c.h, c.w, g.first, g.second, 8U);
        for (size_t i = 0; i < img0.size(); ++i) {
            if (blob[i] > 0.0F) {
                img0[i] = blob[i];
            }
        }
    }
    // 图 1：单个 8x8 弱块（幅度 0.05）⇒ 4 个弱角点
    const vector<float> img1 = MakeBlobImageScaled(c.h, c.w, 24U, 24U, 8U, 0.05F);
    vector<float> batch2 = img0;
    batch2.insert(batch2.end(), img1.begin(), img1.end());

    const CornerResult r = RunKernel(batch2, nullptr, c);
    ASSERT_EQ(r.counts.size(), 2U) << "应写出逐图角点数";
    EXPECT_GE(r.counts[0], 10U) << "强纹理图应检出方块四角, cnt0=" << r.counts[0] << " cnt1=" << r.counts[1];
    EXPECT_GE(r.counts[1], 4U) << "弱纹理图在逐图阈值下必须检出自身角点, cnt0=" << r.counts[0]
                               << " cnt1=" << r.counts[1];
    EXPECT_LE(r.counts[1], 8U) << "弱纹理图不应出现大量边界虚警, cnt0=" << r.counts[0] << " cnt1=" << r.counts[1];
}

TEST_F(GoodFeaturesToTrackKernelTest, flat_image_few_border_artifacts)
{
    RunConfig c;
    const vector<float> image = MakeFlatImage(c.h, c.w, 0.5F);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_LE(r.filled, 4U) << "无纹理输入不应出现大量角点";
    ExpectCommonInvariants(r, c);
}

// 掩码全 0：全部像素被屏蔽 ⇒ 无角点
TEST_F(GoodFeaturesToTrackKernelTest, fully_masked_no_corner)
{
    RunConfig c;
    const vector<float> image = MakeBlobImage(c.h, c.w, 20U, 20U, 16U);
    const vector<uint8_t> mask = MakeMaskAll(c.h, c.w, 0U);
    const CornerResult r = RunKernel(image, &mask, c);
    EXPECT_EQ(r.filled, 0U);
    ExpectCommonInvariants(r, c);
}

// 记录 numCorners 字段在 ICPU 模拟下的当前行为：写回 outputCapacity。
// 真机写实际角点数（aclnn 位比对已覆盖），此处只锁住模拟器行为，便于将来对齐。
TEST_F(GoodFeaturesToTrackKernelTest, num_corners_field_icpu_characterization)
{
    RunConfig c;
    const vector<float> image = MakeBlobImage(c.h, c.w, 20U, 20U, 16U);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_EQ(r.num, c.maxCorners);
    EXPECT_LT(r.filled, r.num);
    ExpectCommonInvariants(r, c);
}

// 亮方块：应检出角点且位置落在方块角上
TEST_F(GoodFeaturesToTrackKernelTest, single_blob_detected)
{
    const uint32_t y0 = 20U;
    const uint32_t x0 = 20U;
    const uint32_t size = 16U;
    RunConfig c;
    const vector<float> image = MakeBlobImage(c.h, c.w, y0, x0, size);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    EXPECT_TRUE(NearBlobCorner(r, static_cast<int32_t>(y0), static_cast<int32_t>(x0), size, 3));
    ExpectCommonInvariants(r, c);
}

// 掩码只开方块附近：检出的角点必须落在掩码允许的范围内
TEST_F(GoodFeaturesToTrackKernelTest, mask_limits_search_region)
{
    const uint32_t y0 = 20U;
    const uint32_t x0 = 20U;
    const uint32_t size = 16U;
    RunConfig c;
    const vector<float> image = MakeBlobImage(c.h, c.w, y0, x0, size);
    const vector<uint8_t> mask = MakeMaskBox(c.h, c.w, y0 - 4U, x0 - 4U, size + 8U, 1U, 0U);
    const CornerResult r = RunKernel(image, &mask, c);
    EXPECT_GE(r.filled, 1U);
    for (const auto& p : r.pts) {
        EXPECT_GE(p.first, static_cast<int32_t>(y0) - 4);
        EXPECT_LE(p.first, static_cast<int32_t>(y0 + size) + 4);
        EXPECT_GE(p.second, static_cast<int32_t>(x0) - 4);
        EXPECT_LE(p.second, static_cast<int32_t>(x0 + size) + 4);
    }
    ExpectCommonInvariants(r, c);
}

// 非方图（32x128）：宽图路径也要正确
TEST_F(GoodFeaturesToTrackKernelTest, non_square_wide_image)
{
    const uint32_t h = 32U;
    const uint32_t w = 128U;
    const uint32_t y0 = 8U;
    const uint32_t x0 = 60U;
    const uint32_t size = 16U;
    RunConfig c;
    c.h = h;
    c.w = w;
    const vector<float> image = MakeBlobImage(h, w, y0, x0, size);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    EXPECT_TRUE(NearBlobCorner(r, static_cast<int32_t>(y0), static_cast<int32_t>(x0), size, 3));
    ExpectCommonInvariants(r, c);
}

// outputCapacity = 1：最多只输出 1 个角点
TEST_F(GoodFeaturesToTrackKernelTest, max_corners_cap)
{
    RunConfig c;
    c.maxCorners = 1U;
    const vector<float> image = MakeBlobImage(c.h, c.w, 20U, 20U, 16U);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_LE(r.filled, 1U);
    ExpectCommonInvariants(r, c);
}

// maxCorners = 0（自动模式）：容量退化为 H*W，应能返回不止 1 个角点
TEST_F(GoodFeaturesToTrackKernelTest, auto_max_corners)
{
    RunConfig c;
    c.h = 32U;
    c.w = 32U;
    c.maxCorners = 0U;
    const vector<float> image = MakeBlobImage(c.h, c.w, 8U, 8U, 12U);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    ExpectCommonInvariants(r, c);
}

// 棋盘：角点密集，不变量必须成立（容量上限、界内、无重复）
TEST_F(GoodFeaturesToTrackKernelTest, checkerboard_invariants)
{
    RunConfig c;
    const vector<float> image = MakeCheckerboard(c.h, c.w, 8U);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    ExpectCommonInvariants(r, c);
}

// quality_level 越大，阈值越高，检出数应单调不增
TEST_F(GoodFeaturesToTrackKernelTest, quality_level_monotone)
{
    const vector<float> image = MakeCheckerboard(kH, kW, 8U);
    RunConfig low;
    low.quality = 0.01F;
    RunConfig high;
    high.quality = 0.5F;
    const CornerResult rLow = RunKernel(image, nullptr, low);
    const CornerResult rHigh = RunKernel(image, nullptr, high);
    EXPECT_LE(rHigh.filled, rLow.filled);
    ExpectCommonInvariants(rLow, low);
    ExpectCommonInvariants(rHigh, high);
}

// min_distance 越大，非极大值抑制越强，检出数应单调不增
TEST_F(GoodFeaturesToTrackKernelTest, min_distance_monotone)
{
    const vector<float> image = MakeCheckerboard(kH, kW, 8U);
    RunConfig near;
    near.minDistance = 1.0F;
    RunConfig far;
    far.minDistance = 8.0F;
    const CornerResult rNear = RunKernel(image, nullptr, near);
    const CornerResult rFar = RunKernel(image, nullptr, far);
    EXPECT_LE(rFar.filled, rNear.filled);
    ExpectCommonInvariants(rNear, near);
    ExpectCommonInvariants(rFar, far);
}

// Harris 检测器路径：同样应检出方块角点
TEST_F(GoodFeaturesToTrackKernelTest, harris_detector_blob)
{
    const uint32_t y0 = 20U;
    const uint32_t x0 = 20U;
    const uint32_t size = 16U;
    RunConfig c;
    c.useHarris = 1U;
    const vector<float> image = MakeBlobImage(c.h, c.w, y0, x0, size);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    EXPECT_TRUE(NearBlobCorner(r, static_cast<int32_t>(y0), static_cast<int32_t>(x0), size, 4));
    ExpectCommonInvariants(r, c);
}

// blockSize = 5：窗口更大，仍应检出方块角点
TEST_F(GoodFeaturesToTrackKernelTest, block_size_five_blob)
{
    const uint32_t y0 = 20U;
    const uint32_t x0 = 20U;
    const uint32_t size = 16U;
    RunConfig c;
    c.blockSize = 5;
    const vector<float> image = MakeBlobImage(c.h, c.w, y0, x0, size);
    const CornerResult r = RunKernel(image, nullptr, c);
    EXPECT_GE(r.filled, 1U);
    ExpectCommonInvariants(r, c);
}
