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
 * \file test_aclnn_hog_descriptor.cpp
 * \brief HOGDescriptor device E2E test: correctness vs a double-precision host
 * golden implementing the exact operator semantics (pad-first replicate,
 * trilinear voting with fold-back edges, skimage eps normalization forms),
 * plus an optional aclrtEvent benchmark mode.
 *
 * Usage:
 *   ./test_aclnn_hog_descriptor                     -- run all correctness cases
 *   ./test_aclnn_hog_descriptor --bench H W C dtype(1=fp16/2=fp32) iters
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_hog_descriptor.h"
#include "hog_descriptor_fp16.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout);                 \
    } while (0)

// ---------------------------------------------------------------------------
// fp16 (IEEE 754 binary16) conversion with round-to-nearest-even (bf3d-proven).
// ---------------------------------------------------------------------------
using HogExample::Fp16ToFp32;
using HogExample::Fp32ToFp16;

static uint32_t LcgNext(uint32_t& state)
{
    state = state * 1664525U + 1013904223U;
    return state;
}

static float LcgUniform(uint32_t& state) { return static_cast<float>(LcgNext(state) >> 8) * (1.0f / 16777216.0f); }

// ---------------------------------------------------------------------------
// Host golden - exact operator semantics in double precision.
// ---------------------------------------------------------------------------
struct HogCfg {
    int64_t pH = 8, pW = 8, bH = 2, bW = 2, sH = 1, sW = 1, nBins = 9;
    int normMode = 3; // 0=L1 1=L1-sqrt 2=L2 3=L2-Hys
    double l2Hys = 0.2;
    bool gamma = true;
    bool signedGrad = false;
    double eps = 1e-6;
};

static int64_t FloorI(double v)
{
    int64_t i = static_cast<int64_t>(v);
    if (static_cast<double>(i) > v) {
        --i;
    }
    return i;
}

static int64_t ClampI(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// img: n*h*w*c fp32 values (already dtype-rounded); out: n * dimD
static void GoldenHog(const std::vector<float>& img, int64_t n, int64_t h, int64_t w, int64_t c, const HogCfg& g,
                      std::vector<float>& out)
{
    const int64_t nCellsH = (h + g.pH - 1) / g.pH;
    const int64_t nCellsW = (w + g.pW - 1) / g.pW;
    const int64_t nBlocksH = (nCellsH - g.bH) / g.sH + 1;
    const int64_t nBlocksW = (nCellsW - g.bW) / g.sW + 1;
    const int64_t padH = nCellsH * g.pH;
    const int64_t padW = nCellsW * g.pW;
    const double range = g.signedGrad ? 2.0 * M_PI : M_PI;
    const double binScale = static_cast<double>(g.nBins) / range;
    const int64_t dimD = g.nBins * g.bH * g.bW * nBlocksH * nBlocksW;
    out.assign(static_cast<size_t>(n * dimD), 0.0f);

    for (int64_t v = 0; v < n; ++v) {
        // pad-first replicate onto the padded grid
        std::vector<double> P(static_cast<size_t>(padH * padW * c), 0.0);
        for (int64_t y = 0; y < padH; ++y) {
            const int64_t sy = ClampI(y, 0, h - 1);
            for (int64_t x = 0; x < padW; ++x) {
                const int64_t sx = ClampI(x, 0, w - 1);
                for (int64_t ch = 0; ch < c; ++ch) {
                    double val = img[static_cast<size_t>((v * h + sy) * w + sx) * c + ch];
                    P[(static_cast<size_t>(y) * padW + x) * c + ch] = g.gamma ? std::sqrt(val) : val;
                }
            }
        }
        // trilinear histogram
        std::vector<double> hist(static_cast<size_t>(nCellsH * nCellsW * g.nBins), 0.0);
        for (int64_t y = 0; y < padH; ++y) {
            const int64_t ym = ClampI(y - 1, 0, padH - 1);
            const int64_t yp = ClampI(y + 1, 0, padH - 1);
            for (int64_t x = 0; x < padW; ++x) {
                const int64_t xm = ClampI(x - 1, 0, padW - 1);
                const int64_t xp = ClampI(x + 1, 0, padW - 1);
                double gx = 0.0, gy = 0.0, m2 = 0.0;
                for (int64_t ch = 0; ch < c; ++ch) {
                    const double gcx = P[(static_cast<size_t>(y) * padW + xp) * c + ch] -
                                       P[(static_cast<size_t>(y) * padW + xm) * c + ch];
                    const double gcy = P[(static_cast<size_t>(yp) * padW + x) * c + ch] -
                                       P[(static_cast<size_t>(ym) * padW + x) * c + ch];
                    const double m2c = gcx * gcx + gcy * gcy;
                    if (ch == 0 || m2c > m2) { // first channel wins ties
                        gx = gcx;
                        gy = gcy;
                        m2 = m2c;
                    }
                }
                const double mag = std::sqrt(m2);
                double theta = 0.0;
                if (m2 > 0.0) {
                    theta = std::atan2(gy, gx);
                    theta = std::fmod(theta, range);
                    if (theta < 0.0) {
                        theta += range;
                    }
                }
                const double u = theta * binScale - 0.5;
                const int64_t b0f = FloorI(u);
                const double a = u - static_cast<double>(b0f);
                const int64_t b0 = b0f < 0 ? b0f + g.nBins : b0f;
                const int64_t b1raw = b0 + 1;
                const int64_t b1 = b1raw >= g.nBins ? 0 : b1raw;

                const double vv = (static_cast<double>(y) + 0.5) / static_cast<double>(g.pH) - 0.5;
                const int64_t i0f = FloorI(vv);
                const double fy = vv - static_cast<double>(i0f);
                int64_t i0 = i0f, i1 = i0f + 1;
                double rA = 1.0 - fy, rB = fy;
                if (i0 < 0) {
                    rB = 1.0;
                    rA = 0.0;
                }
                if (i1 >= nCellsH) {
                    rA = 1.0;
                    rB = 0.0;
                }

                const double uu = (static_cast<double>(x) + 0.5) / static_cast<double>(g.pW) - 0.5;
                const int64_t j0f = FloorI(uu);
                const double fx = uu - static_cast<double>(j0f);
                int64_t j0 = j0f, j1 = j0f + 1;
                double cA = 1.0 - fx, cB = fx;
                if (j0 < 0) {
                    cB = 1.0;
                    cA = 0.0;
                }
                if (j1 >= nCellsW) {
                    cA = 1.0;
                    cB = 0.0;
                }

                for (int64_t k = 0; k < g.nBins; ++k) {
                    const double wk = (b0 == k ? 1.0 - a : 0.0) + (b1 == k ? a : 0.0);
                    if (wk == 0.0) {
                        continue;
                    }
                    const double m = mag * wk;
                    if (rA != 0.0) {
                        if (cA != 0.0)
                            hist[(i0 * nCellsW + j0) * g.nBins + k] += m * rA * cA;
                        if (cB != 0.0)
                            hist[(i0 * nCellsW + j1) * g.nBins + k] += m * rA * cB;
                    }
                    if (rB != 0.0) {
                        if (cA != 0.0)
                            hist[(i1 * nCellsW + j0) * g.nBins + k] += m * rB * cA;
                        if (cB != 0.0)
                            hist[(i1 * nCellsW + j1) * g.nBins + k] += m * rB * cB;
                    }
                }
            }
        }
        // blocks: normalize and concatenate
        std::vector<double> block(static_cast<size_t>(g.bH * g.bW * g.nBins), 0.0);
        for (int64_t bi = 0; bi < nBlocksH; ++bi) {
            for (int64_t bj = 0; bj < nBlocksW; ++bj) {
                for (int64_t ci = 0; ci < g.bH; ++ci) {
                    for (int64_t cj = 0; cj < g.bW; ++cj) {
                        for (int64_t k = 0; k < g.nBins; ++k) {
                            block[(ci * g.bW + cj) * g.nBins +
                                  k] = hist[((bi * g.sH + ci) * nCellsW + (bj * g.sW + cj)) * g.nBins + k];
                        }
                    }
                }
                const size_t len = block.size();
                if (v == 0 && bi == 0 && bj == 0) {
                    for (size_t i = 0; i < len; ++i) {
                        LOG_PRINT("GH[%zu]=%.6f\n", i, block[i]);
                    }
                }
                auto l2Pass = [&]() {
                    double s = 0.0;
                    for (size_t i = 0; i < len; ++i) {
                        s += block[i] * block[i];
                    }
                    const double denom = std::sqrt(s + g.eps * g.eps);
                    for (size_t i = 0; i < len; ++i) {
                        block[i] /= denom;
                    }
                };
                if (g.normMode == 0 || g.normMode == 1) {
                    double s = 0.0;
                    for (size_t i = 0; i < len; ++i) {
                        s += std::fabs(block[i]);
                    }
                    const double denom = s + g.eps;
                    for (size_t i = 0; i < len; ++i) {
                        block[i] /= denom;
                    }
                    if (g.normMode == 1) {
                        for (size_t i = 0; i < len; ++i) {
                            block[i] = std::sqrt(block[i]);
                        }
                    }
                } else {
                    l2Pass();
                    if (g.normMode == 3) {
                        for (size_t i = 0; i < len; ++i) {
                            block[i] = std::min(block[i], g.l2Hys);
                        }
                        l2Pass();
                    }
                }
                for (size_t i = 0; i < len; ++i) {
                    out[static_cast<size_t>(v * dimD) + static_cast<size_t>((bi * nBlocksW + bj)) * len +
                        i] = static_cast<float>(block[i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// acl helpers
// ---------------------------------------------------------------------------
static int MakeTensor(const std::vector<int64_t>& shape, aclDataType dt, void* dev, size_t bytes, aclTensor** t)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                         shape.size(), dev);
    CHECK_RET(*t != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    (void)bytes;
    return 0;
}

struct CaseRt {
    std::vector<float> hostIn;
    std::vector<float> golden;
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    aclDataType dt = ACL_FLOAT;
    HogCfg cfg;
    void* devX = nullptr;
    void* devY = nullptr;
    aclTensor* tx = nullptr;
    aclTensor* ty = nullptr;
};

static int PrepareCase(const std::vector<int64_t>& shape, aclDataType dt, const HogCfg& cfg, uint32_t seed, CaseRt& rt,
                       bool needGolden, int pattern = 0)
{
    size_t total = 1;
    for (int64_t d : shape) {
        total *= static_cast<size_t>(d);
    }
    rt.inShape = shape;
    rt.dt = dt;
    rt.cfg = cfg;
    rt.hostIn.resize(total);
    uint32_t state = seed;
    for (size_t i = 0; i < total; ++i) {
        if (pattern == 1) {
            rt.hostIn[i] = std::ldexp(static_cast<float>(i % 1024U), -24);
        } else if (pattern == 2) {
            rt.hostIn[i] = 0.25f;
        } else {
            rt.hostIn[i] = LcgUniform(state); // [0,1) like normalized images
        }
    }
    if (pattern == 3 || pattern == 4) {
        CHECK_RET(shape.size() == 2U && dt == ACL_FLOAT && !cfg.gamma, return -1);
        const int64_t height = shape[0], width = shape[1];
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const int64_t axis = pattern == 3 ? y : x;
                const float ramp = pattern == 3 ? static_cast<float>(width - 1 - x) : static_cast<float>(y);
                rt.hostIn[static_cast<size_t>(y * width + x)] = axis % 4 == 0 ? 0.0f :
                                                                axis % 4 == 2 ? 1.0f :
                                                                                ramp * 2.5e-39f;
            }
        }
    }
    const bool isFp16 = (dt == ACL_FLOAT16);
    const size_t inBytes = total * (isFp16 ? sizeof(uint16_t) : sizeof(float));
    CHECK_RET(aclrtMalloc(&rt.devX, inBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);

    int64_t n = 1, h, w, c = 1;
    if (shape.size() == 2U) {
        h = shape[0];
        w = shape[1];
    } else if (shape.size() == 3U && (shape[2] == 1 || shape[2] == 3)) {
        h = shape[0];
        w = shape[1];
        c = shape[2];
    } else if (shape.size() == 3U) {
        n = shape[0];
        h = shape[1];
        w = shape[2];
    } else {
        n = shape[0];
        h = shape[1];
        w = shape[2];
        c = shape[3];
    }
    const int64_t nCellsH = (h + cfg.pH - 1) / cfg.pH;
    const int64_t nCellsW = (w + cfg.pW - 1) / cfg.pW;
    const int64_t nBlocksH = (nCellsH - cfg.bH) / cfg.sH + 1;
    const int64_t nBlocksW = (nCellsW - cfg.bW) / cfg.sW + 1;
    const int64_t dimD = cfg.nBins * cfg.bH * cfg.bW * nBlocksH * nBlocksW;
    rt.outShape = n > 1 ? std::vector<int64_t>{n, dimD} : std::vector<int64_t>{dimD};
    const size_t outBytes = static_cast<size_t>(n * dimD) * sizeof(float);
    CHECK_RET(aclrtMalloc(&rt.devY, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemset(rt.devY, outBytes, 0, outBytes) == ACL_SUCCESS, return -1);

    if (isFp16) {
        std::vector<uint16_t> packed(total);
        for (size_t i = 0; i < total; ++i) {
            packed[i] = Fp32ToFp16(rt.hostIn[i]);
            rt.hostIn[i] = Fp16ToFp32(packed[i]);
        }
        CHECK_RET(aclrtMemcpy(rt.devX, inBytes, packed.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                  return -1);
    } else {
        CHECK_RET(aclrtMemcpy(rt.devX, inBytes, rt.hostIn.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                  return -1);
    }
    CHECK_RET(MakeTensor(shape, dt, rt.devX, inBytes, &rt.tx) == 0, return -1);
    CHECK_RET(MakeTensor(rt.outShape, ACL_FLOAT, rt.devY, outBytes, &rt.ty) == 0, return -1);
    if (needGolden) {
        GoldenHog(rt.hostIn, n, h, w, c, cfg, rt.golden);
    }
    return 0;
}

static void ReleaseCase(CaseRt& rt)
{
    if (rt.tx != nullptr) {
        aclDestroyTensor(rt.tx);
        rt.tx = nullptr;
    }
    if (rt.ty != nullptr) {
        aclDestroyTensor(rt.ty);
        rt.ty = nullptr;
    }
    if (rt.devX != nullptr) {
        (void)aclrtFree(rt.devX);
        rt.devX = nullptr;
    }
    if (rt.devY != nullptr) {
        (void)aclrtFree(rt.devY);
        rt.devY = nullptr;
    }
}

static int RunOnce(const CaseRt& rt, aclrtStream stream)
{
    const int64_t ppcArr[2] = {rt.cfg.pH, rt.cfg.pW};
    const int64_t cpbArr[2] = {rt.cfg.bH, rt.cfg.bW};
    const int64_t bstArr[2] = {rt.cfg.sH, rt.cfg.sW};
    aclIntArray* ppc = aclCreateIntArray(ppcArr, 2);
    aclIntArray* cpb = aclCreateIntArray(cpbArr, 2);
    aclIntArray* bst = aclCreateIntArray(bstArr, 2);
    if (ppc == nullptr || cpb == nullptr || bst == nullptr) {
        LOG_PRINT("aclCreateIntArray failed\n");
        if (ppc != nullptr) {
            aclDestroyIntArray(ppc);
        }
        if (cpb != nullptr) {
            aclDestroyIntArray(cpb);
        }
        if (bst != nullptr) {
            aclDestroyIntArray(bst);
        }
        return -1;
    }
    char norm[16];
    std::snprintf(norm, sizeof(norm), "%s",
                  rt.cfg.normMode == 0 ? "L1" :
                  rt.cfg.normMode == 1 ? "L1-sqrt" :
                  rt.cfg.normMode == 2 ? "L2" :
                                         "L2-Hys");
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnHOGDescriptorGetWorkspaceSize(rt.tx, ppc, cpb, bst, rt.cfg.nBins, norm, rt.cfg.l2Hys, rt.cfg.gamma,
                                                  rt.cfg.signedGrad, 1, rt.cfg.eps, rt.ty, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed, ret=%d\n", (int)ret); aclDestroyIntArray(ppc);
              aclDestroyIntArray(cpb); aclDestroyIntArray(bst); return -1);
    void* ws = nullptr;
    if (wsSize > 0) {
        if (aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            LOG_PRINT("workspace allocation failed\n");
            // The second-stage API has not taken ownership of this executor.
            if (executor != nullptr) {
                const auto destroyRet = aclDestroyAclOpExecutor(executor);
                if (destroyRet != ACL_SUCCESS) {
                    LOG_PRINT("aclDestroyAclOpExecutor failed, ret=%d\n", static_cast<int>(destroyRet));
                }
                executor = nullptr;
            }
            aclDestroyIntArray(ppc);
            aclDestroyIntArray(cpb);
            aclDestroyIntArray(bst);
            return -1;
        }
    }
    ret = aclnnHOGDescriptor(ws, wsSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (ws != nullptr) {
        (void)aclrtFree(ws);
    }
    aclDestroyIntArray(ppc);
    aclDestroyIntArray(cpb);
    aclDestroyIntArray(bst);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHOGDescriptor failed, ret=%d\n", (int)ret); return -1);
    return 0;
}

struct CaseDef {
    const char* name;
    std::vector<int64_t> shape;
    aclDataType dt;
    HogCfg cfg;
    double tol;
    int pattern = 0; // 0=random, 1=half subnormals, 2=constant, 3=tiny gx, 4=tiny gy
};

static int RunCorrectness(aclrtStream stream)
{
    HogCfg def;
    HogCfg gOff = def;
    gOff.gamma = false;
    HogCfg odd = def;
    odd.pH = 5;
    odd.pW = 5;
    HogCfg sgn = def;
    sgn.signedGrad = true;
    HogCfg tinySigned = sgn;
    tinySigned.gamma = false;
    HogCfg l1 = def;
    l1.normMode = 0;
    HogCfg l1s = def;
    l1s.normMode = 1;
    HogCfg l2 = def;
    l2.normMode = 2;
    HogCfg st2 = def;
    st2.sH = 2;
    st2.sW = 2;
    HogCfg b44 = def;
    b44.bH = 4;
    b44.bW = 4;
    HogCfg n6 = def;
    n6.nBins = 6;
    HogCfg ubEdge = def;
    ubEdge.bH = 8;
    ubEdge.bW = 8;
    ubEdge.nBins = 36;

    const std::vector<CaseDef> cases = {
        {"fp32_128_default", {128, 128}, ACL_FLOAT, def, 2e-3},
        {"fp32_64_gamma_off", {64, 64}, ACL_FLOAT, gOff, 2e-3},
        {"fp16_128_default", {128, 128}, ACL_FLOAT16, def, 5e-3},
        {"fp16_100x80_pad", {100, 80}, ACL_FLOAT16, def, 5e-3},
        {"fp32_p5_odd_slow", {64, 64}, ACL_FLOAT, odd, 2e-3},
        {"fp32_signed", {64, 64}, ACL_FLOAT, sgn, 2e-3},
        {"fp32_L1", {64, 64}, ACL_FLOAT, l1, 2e-3},
        {"fp32_L1sqrt", {64, 64}, ACL_FLOAT, l1s, 2e-3},
        {"fp32_L2", {64, 64}, ACL_FLOAT, l2, 2e-3},
        {"fp32_batch", {3, 96, 80}, ACL_FLOAT, def, 2e-3},
        {"fp32_rgb", {64, 64, 3}, ACL_FLOAT, def, 3e-3},
        {"fp16_rgb", {48, 40, 3}, ACL_FLOAT16, def, 5e-3},
        {"fp32_tiny_16", {16, 16}, ACL_FLOAT, def, 2e-3},
        {"fp32_stride2", {64, 64}, ACL_FLOAT, st2, 2e-3},
        {"fp32_block44", {64, 64}, ACL_FLOAT, b44, 2e-3},
        {"fp32_bins6", {64, 64}, ACL_FLOAT, n6, 2e-3},
        {"fp16_subnormals", {32, 40}, ACL_FLOAT16, def, 5e-3, 1},
        {"fp32_constant", {24, 32}, ACL_FLOAT, def, 2e-3, 2},
        {"fp32_nhwc_n1", {1, 24, 32, 1}, ACL_FLOAT, def, 2e-3},
        {"fp32_nhwc_batch", {2, 24, 32, 3}, ACL_FLOAT, def, 3e-3},
        {"fp32_multi_band", {1024, 256}, ACL_FLOAT, def, 2e-3},
        {"fp32_ub_edge", {128, 512, 3}, ACL_FLOAT, ubEdge, 3e-3},
        {"fp32_width_tail", {24, 1023}, ACL_FLOAT, def, 2e-3},
        {"fp32_signed_tiny_gx", {32, 40}, ACL_FLOAT, tinySigned, 2e-3, 3},
        {"fp32_signed_tiny_gy", {32, 40}, ACL_FLOAT, tinySigned, 2e-3, 4},
    };
    int pass = 0;
    int fail = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const CaseDef& c = cases[ci];
        CaseRt rt;
        if (PrepareCase(c.shape, c.dt, c.cfg, 1000U + static_cast<uint32_t>(ci), rt, true, c.pattern) != 0) {
            LOG_PRINT("[FAIL] %s prepare error\n", c.name);
            ++fail;
            ReleaseCase(rt);
            continue;
        }
        if (RunOnce(rt, stream) != 0) {
            LOG_PRINT("[FAIL] %s run error\n", c.name);
            ++fail;
            ReleaseCase(rt);
            continue;
        }
        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ReleaseCase(rt); ++fail; continue);
        const size_t total = rt.golden.size();
        const size_t bytes = total * sizeof(float);
        std::vector<uint8_t> raw(bytes);
        CHECK_RET(aclrtMemcpy(raw.data(), bytes, rt.devY, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                  ReleaseCase(rt);
                  ++fail; continue);
        double maxErr = 0.0;
        double maxRel = 0.0;
        int64_t firstBad = -1;
        for (size_t i = 0; i < total; ++i) {
            float out;
            std::memcpy(&out, raw.data() + i * 4, 4);
            const double diff = std::fabs(static_cast<double>(out) - rt.golden[i]);
            const double bound = c.tol * (1.0 + std::fabs(static_cast<double>(rt.golden[i])));
            if (!(diff <= bound) && firstBad < 0) {
                firstBad = static_cast<int64_t>(i);
            }
            maxErr = std::max(maxErr, diff);
            maxRel = std::max(maxRel, diff / (1.0 + std::fabs(static_cast<double>(rt.golden[i]))));
        }
        if (firstBad < 0) {
            LOG_PRINT("[PASS] %-18s max_err=%.3e max_rel=%.3e\n", c.name, maxErr, maxRel);
            ++pass;
        } else {
            {
                size_t nz = 0;
                int64_t firstNz = -1;
                int shown = 0;
                for (size_t i = 0; i < total; ++i) {
                    float ov;
                    std::memcpy(&ov, raw.data() + i * 4, 4);
                    if (ov != 0.0f) {
                        ++nz;
                        if (firstNz < 0) {
                            firstNz = static_cast<int64_t>(i);
                        }
                        if (shown < 12) {
                            LOG_PRINT("  nz[%ld]=%.6f gold=%.6f\n", static_cast<long>(i), ov, rt.golden[i]);
                            ++shown;
                        }
                    }
                }
                LOG_PRINT("  nonzero=%zu/%zu firstNz=%ld\n", nz, total, static_cast<long>(firstNz));
            }
            LOG_PRINT("[FAIL] %-18s max_err=%.3e max_rel=%.3e first_bad=%ld gold[fb]=%.6f\n", c.name, maxErr, maxRel,
                      static_cast<long>(firstBad), rt.golden[static_cast<size_t>(firstBad)]);
            for (int64_t k2 = firstBad; k2 < firstBad + 40 && k2 < static_cast<int64_t>(total); ++k2) {
                float ov;
                std::memcpy(&ov, raw.data() + static_cast<size_t>(k2) * 4, 4);
                LOG_PRINT("  [%ld] out=%.6f gold=%.6f\n", static_cast<long>(k2), ov,
                          rt.golden[static_cast<size_t>(k2)]);
            }
            ++fail;
        }
        ReleaseCase(rt);
    }
    LOG_PRINT("=== SUMMARY: %d/%d PASS ===\n", pass, pass + fail);
    return fail == 0 ? 0 : 1;
}

static int RunBench(aclrtStream stream, int64_t h, int64_t w, int64_t c, int dtype, int iters, bool signedGrad = false)
{
    CHECK_RET(
        (dtype == 1 || dtype == 2) && iters > 0 && h >= 9 && h <= 10000 && w >= 9 && w <= 1024 && (c == 1 || c == 3),
        LOG_PRINT("bench requires H=9..10000, W=9..1024, C=1/3, dtype=1(fp16)/2(fp32), iters>0\n");
        return -1);
    const aclDataType dt = dtype == 1 ? ACL_FLOAT16 : ACL_FLOAT;
    HogCfg cfg;
    cfg.signedGrad = signedGrad;
    std::vector<int64_t> shape = c == 3 ? std::vector<int64_t>{h, w, c} : std::vector<int64_t>{h, w};
    CaseRt rt;
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    auto cleanup = [&](int status) {
        if (status != 0) {
            (void)aclrtSynchronizeStream(stream);
        }
        if (end != nullptr) {
            (void)aclrtDestroyEvent(end);
            end = nullptr;
        }
        if (begin != nullptr) {
            (void)aclrtDestroyEvent(begin);
            begin = nullptr;
        }
        ReleaseCase(rt);
        return status;
    };
    CHECK_RET(PrepareCase(shape, dt, cfg, 20260827U, rt, false) == 0, return cleanup(-1));
    for (int i = 0; i < 3; ++i) {
        CHECK_RET(RunOnce(rt, stream) == 0, return cleanup(-1));
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return cleanup(-1));
    CHECK_RET(aclrtCreateEvent(&begin) == ACL_SUCCESS, LOG_PRINT("create begin event failed\n"); return cleanup(-1));
    CHECK_RET(aclrtCreateEvent(&end) == ACL_SUCCESS, LOG_PRINT("create end event failed\n"); return cleanup(-1));
    std::vector<float> ms(static_cast<size_t>(iters), 0.0f);
    for (int i = 0; i < iters; ++i) {
        CHECK_RET(aclrtRecordEvent(begin, stream) == ACL_SUCCESS, LOG_PRINT("record begin event failed\n");
                  return cleanup(-1));
        CHECK_RET(RunOnce(rt, stream) == 0, return cleanup(-1));
        CHECK_RET(aclrtRecordEvent(end, stream) == ACL_SUCCESS, LOG_PRINT("record end event failed\n");
                  return cleanup(-1));
        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return cleanup(-1));
        float elapsed = 0.0f;
        CHECK_RET(aclrtEventElapsedTime(&elapsed, begin, end) == ACL_SUCCESS, LOG_PRINT("event elapsed time failed\n");
                  return cleanup(-1));
        ms[static_cast<size_t>(i)] = elapsed;
    }
    std::sort(ms.begin(), ms.end());
    const float median = ms[ms.size() / 2];
    const int64_t inBytes = h * w * c * (dt == ACL_FLOAT16 ? 2 : 4);
    const double outBytes = static_cast<double>(rt.outShape.back()) * 4.0;
    LOG_PRINT(
        "BENCH H=%ld W=%ld C=%ld dtype=%d median_stream_interval=%.3fms logical_io=%.1fMB io_per_interval=%.2fGB/s\n",
        h, w, c, dtype, median, (inBytes + outBytes) / 1e6, (inBytes + outBytes) / 1e6 / (median / 1000.0));
    return cleanup(0);
}

int main(int argc, char** argv)
{
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, LOG_PRINT("aclInit failed\n"); return 1);
    int32_t deviceId = 0;
    (void)aclrtGetDevice(&deviceId);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed\n"); return 1);
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;
    (void)aclrtCreateContext(&ctx, deviceId);
    (void)aclrtCreateStream(&stream);
    int rc = 1;
    if (argc >= 2 && (std::string(argv[1]) == "--bench" || std::string(argv[1]) == "--bench-signed")) {
        CHECK_RET(argc == 7, LOG_PRINT("Usage: --bench or --bench-signed H W C dtype(1=fp16/2=fp32) iters\n");
                  aclrtDestroyStream(stream); aclrtDestroyContext(ctx); aclrtResetDevice(deviceId); aclFinalize();
                  return 1);
        rc = RunBench(stream, std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]), std::atoi(argv[5]),
                      std::atoi(argv[6]), std::string(argv[1]) == "--bench-signed");
    } else {
        LOG_PRINT("=== HOGDescriptor E2E verification (Ascend 910B device) ===\n");
        rc = RunCorrectness(stream);
    }
    (void)aclrtSynchronizeStream(stream);
    (void)aclrtDestroyStream(stream);
    (void)aclrtDestroyContext(ctx);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return rc;
}
