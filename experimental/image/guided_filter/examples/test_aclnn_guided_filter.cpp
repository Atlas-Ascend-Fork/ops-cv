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
 * \file test_aclnn_guided_filter.cpp
 * \brief GuidedFilter aclnn 样例 + 自测：内置 CPU 参考实现（积分图 + double 累加）逐例对拍。
 *
 * 语义对齐 cv::ximgproc::guidedFilter（见 GF_契约与设计_20261025.md）：
 *   box 均值 = (2r+1)^2 归一化均值，边界 BORDER_REFLECT（镜像含边缘）
 *   Σ_cc = box(g_c^2) - μ_c^2 + eps ;  Σ_cd = box(g_c g_d) - μ_c μ_d
 *   α = Σ^{-1} ω ;  β = ν - Σ_c α_c μ_c ;  q = Σ_c box(α_c) g_c + box(β)
 *
 * 用法：
 *   ./test_aclnn_guided_filter            # 跑全部用例
 *   GF_BENCH=1 ./test_aclnn_guided_filter # 额外跑性能
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn_guided_filter.h"

#define CHECK_RET(cond, ret)                                                                                    \
    do {                                                                                                        \
        if (!(cond)) {                                                                                          \
            std::cerr << "CHECK_RET failed: " << #cond << ", ret=" << (ret) << ", err=" << aclGetRecentErrMsg() \
                      << std::endl;                                                                             \
            return (ret);                                                                                       \
        }                                                                                                       \
    } while (0)

namespace {

// ------------------------------------------------------------------ 参考实现
// cv::BORDER_REFLECT：fedcba|abcdefgh|hgfedcb，周期 2n（-1->0, -2->1, n->n-1）
int64_t Mirror(int64_t i, int64_t n)
{
    const int64_t period = 2 * n;
    if (period <= 0) {
        return 0;
    }
    int64_t j = i % period;
    if (j < 0) {
        j += period;
    }
    return (j < n) ? j : (period - 1 - j);
}

// (2r+1)^2 归一化盒式均值，边界 BORDER_REFLECT。
// 注意：反射后的窗口不是原图上的连续区间（会重复取镜像像素），所以必须先把边界展开成
// padded 矩阵再做积分图；早期版本直接对端点取 min/max，边界像素会算错。
std::vector<double> BoxMean(const std::vector<double>& x, int64_t H, int64_t W, int64_t r)
{
    const int64_t k = 2 * r + 1;
    const int64_t HP = H + 2 * r, WP = W + 2 * r;
    std::vector<double> integ(static_cast<size_t>((HP + 1) * (WP + 1)), 0.0);
    for (int64_t py = 0; py < HP; ++py) {
        const int64_t ry = Mirror(py - r, H);
        double rowSum = 0.0;
        for (int64_t px = 0; px < WP; ++px) {
            const int64_t rx = Mirror(px - r, W);
            rowSum += x[static_cast<size_t>(ry * W + rx)];
            integ[static_cast<size_t>((py + 1) * (WP + 1) +
                                      (px + 1))] = integ[static_cast<size_t>(py * (WP + 1) + (px + 1))] + rowSum;
        }
    }
    std::vector<double> out(static_cast<size_t>(H * W), 0.0);
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t xx = 0; xx < W; ++xx) {
            const double v = integ[static_cast<size_t>((y + k) * (WP + 1) + (xx + k))] -
                             integ[static_cast<size_t>(y * (WP + 1) + (xx + k))] -
                             integ[static_cast<size_t>((y + k) * (WP + 1) + xx)] +
                             integ[static_cast<size_t>(y * (WP + 1) + xx)];
            out[static_cast<size_t>(y * W + xx)] = v / static_cast<double>(k * k);
        }
    }
    return out;
}

// guide: C 通道平面 / src: Cs 通道平面 -> q: Cs 通道平面（C <= 3，闭式求逆）
void GuidedFilterRef(const std::vector<std::vector<double>>& g, const std::vector<std::vector<double>>& p, int64_t H,
                     int64_t W, int64_t r, double eps, std::vector<std::vector<double>>& out)
{
    const int C = static_cast<int>(g.size());
    const int Cs = static_cast<int>(p.size());
    const int64_t n = H * W;

    std::vector<std::vector<double>> mu(C);
    for (int c = 0; c < C; ++c) {
        mu[c] = BoxMean(g[c], H, W, r);
    }
    // Σ 平面表（C x C，对称）
    std::vector<std::vector<std::vector<double>>> S(C, std::vector<std::vector<double>>(C));
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < C; ++d) {
            std::vector<double> prod(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                prod[i] = g[c][i] * g[d][i];
            }
            auto b = BoxMean(prod, H, W, r);
            for (int64_t i = 0; i < n; ++i) {
                b[i] -= mu[c][i] * mu[d][i];
            }
            if (c == d) {
                for (int64_t i = 0; i < n; ++i) {
                    b[i] += eps;
                }
            }
            S[c][d] = b;
        }
    }

    std::vector<std::vector<double>> nu(Cs);
    for (int s = 0; s < Cs; ++s) {
        nu[s] = BoxMean(p[s], H, W, r);
    }
    std::vector<std::vector<double>> alpha(static_cast<size_t>(C * Cs), std::vector<double>(static_cast<size_t>(n)));
    std::vector<std::vector<double>> beta(Cs);
    for (int s = 0; s < Cs; ++s) {
        std::vector<std::vector<double>> om(C, std::vector<double>(static_cast<size_t>(n)));
        for (int c = 0; c < C; ++c) {
            std::vector<double> prod(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                prod[i] = g[c][i] * p[s][i];
            }
            om[c] = BoxMean(prod, H, W, r);
            for (int64_t i = 0; i < n; ++i) {
                om[c][i] -= mu[c][i] * nu[s][i];
            }
        }
        beta[s].assign(static_cast<size_t>(n), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            double a[3] = {0.0, 0.0, 0.0};
            if (C == 1) {
                const double det = S[0][0][i];
                a[0] = om[0][i] / (std::fabs(det) < 1e-12 ? 1.0 : det);
            } else if (C == 2) {
                const double det = S[0][0][i] * S[1][1][i] - S[0][1][i] * S[1][0][i];
                const double dd = (std::fabs(det) < 1e-12) ? 1.0 : det;
                a[0] = (S[1][1][i] * om[0][i] - S[0][1][i] * om[1][i]) / dd;
                a[1] = (-S[1][0][i] * om[0][i] + S[0][0][i] * om[1][i]) / dd;
            } else {
                const double s00 = S[0][0][i], s01 = S[0][1][i], s02 = S[0][2][i];
                const double s11 = S[1][1][i], s12 = S[1][2][i], s22 = S[2][2][i];
                const double c00 = s11 * s22 - s12 * s12;
                const double c01 = s02 * s12 - s01 * s22;
                const double c02 = s01 * s12 - s02 * s11;
                const double c11 = s00 * s22 - s02 * s02;
                const double c12 = s01 * s02 - s00 * s12;
                const double c22 = s00 * s11 - s01 * s01;
                const double det = s00 * c00 + s01 * c01 + s02 * c02;
                const double dd = (std::fabs(det) < 1e-12) ? 1.0 : det;
                a[0] = (c00 * om[0][i] + c01 * om[1][i] + c02 * om[2][i]) / dd;
                a[1] = (c01 * om[0][i] + c11 * om[1][i] + c12 * om[2][i]) / dd;
                a[2] = (c02 * om[0][i] + c12 * om[1][i] + c22 * om[2][i]) / dd;
            }
            for (int c = 0; c < C; ++c) {
                alpha[static_cast<size_t>(c * Cs + s)][i] = a[c];
            }
            double b = nu[s][i];
            for (int c = 0; c < C; ++c) {
                b -= a[c] * mu[c][i];
            }
            beta[s][i] = b;
        }
    }
    out.assign(static_cast<size_t>(Cs), std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int c = 0; c < C; ++c) {
        for (int s = 0; s < Cs; ++s) {
            const auto ba = BoxMean(alpha[static_cast<size_t>(c * Cs + s)], H, W, r);
            for (int64_t i = 0; i < n; ++i) {
                out[static_cast<size_t>(s)][i] += ba[i] * g[c][i];
            }
        }
    }
    for (int s = 0; s < Cs; ++s) {
        const auto bb = BoxMean(beta[s], H, W, r);
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(s)][i] += bb[i];
        }
    }
}

// ------------------------------------------------------------------ 工具
enum class DT { F32, F16, U8 };

size_t DtSize(DT t) { return (t == DT::F32) ? 4U : ((t == DT::F16) ? 2U : 1U); }
aclDataType ToAcl(DT t) { return (t == DT::F32) ? ACL_FLOAT : ((t == DT::F16) ? ACL_FLOAT16 : ACL_UINT8); }

uint16_t ToHalf(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    const int32_t e = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    const uint32_t mant = x & 0x7fffffU;
    if (((x >> 23) & 0xffU) == 0xffU) {
        return static_cast<uint16_t>(sign | 0x7c00U | (mant ? 0x0200U : 0U));
    }
    if (e >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (e <= 0) {
        return static_cast<uint16_t>(sign);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
}

float FromHalf(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    const uint32_t exp = (h >> 10) & 0x1fU;
    const uint32_t mant = h & 0x3ffU;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0F : 0.0F;
        }
        const float v = static_cast<float>(mant) * std::pow(2.0F, -24.0F);
        return sign ? -v : v;
    }
    if (exp == 31) {
        const float v = (mant == 0) ? INFINITY : NAN;
        return sign ? -v : v;
    }
    const uint32_t x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

struct Rng {
    uint64_t s = 12345;
    uint32_t Next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
    double Uni() { return static_cast<double>(Next()) / 2147483648.0; }
};

// 带强阶跃边缘 + 纹理的合成图（值域 [0, scale)）
void GenImage(std::vector<float>& img, int64_t H, int64_t W, int64_t C, double scale, Rng& rng, uint32_t seed)
{
    Rng r = rng;
    r.s = seed;
    img.assign(static_cast<size_t>(H * W * C), 0.0F);
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t xx = 0; xx < W; ++xx) {
            const double base = static_cast<double>((y * 3 + xx * 5) % 97);
            for (int64_t c = 0; c < C; ++c) {
                // 取模打乱通道间的仿射关系，避免引导通道共线导致 3x3 求逆病态
                const double phase = base + 37.0 * c;
                const double wrapped = phase - 97.0 * std::floor(phase / 97.0);
                double v = wrapped / 97.0;
                v = (xx < W / 2) ? (0.65 + 0.35 * v) : (0.15 * v);
                v += 0.02 * (r.Uni() - 0.5);
                img[static_cast<size_t>((y * W + xx) * C + c)] = static_cast<float>(v * scale);
            }
        }
    }
}

struct Case {
    const char* name;
    int64_t N, H, W;
    int64_t C, Cs;
    DT inDt, outDt;
    int64_t radius;
    double eps;
    bool flat = false;  // true: guide=0.5, src=0.25 常量图（隔离归一化/窗口总数）
    bool rampX = false; // true: guide=src=x（线性斜坡，内点 box(box(x))=x，可检出窗口错位）
};

} // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::vector<Case> cases = {
        {"f32 r2 1->1 [H,W]", 1, 17, 23, 1, 1, DT::F32, DT::F32, 2, 0.01},
        {"f32 r1 1->1", 1, 32, 40, 1, 1, DT::F32, DT::F32, 1, 0.01},
        {"f32 r4 3->3", 1, 24, 33, 3, 3, DT::F32, DT::F32, 4, 0.04},
        {"f32 r3 3->1", 1, 21, 29, 3, 1, DT::F32, DT::F32, 3, 0.02},
        {"f32 r2 1->3", 1, 19, 27, 1, 3, DT::F32, DT::F32, 2, 0.01},
        {"f32 batch2 r2 1->1", 2, 20, 28, 1, 1, DT::F32, DT::F32, 2, 0.01},
        {"f32 wide r8 1->1", 1, 16, 1000, 1, 1, DT::F32, DT::F32, 8, 0.01},
        {"f32 tall r2 1->1", 1, 300, 17, 1, 1, DT::F32, DT::F32, 2, 0.01},
        {"f16 r2 1->1", 1, 20, 26, 1, 1, DT::F16, DT::F16, 2, 0.01},
        {"f16 r2 3->3", 1, 18, 22, 3, 3, DT::F16, DT::F16, 2, 0.02},
        {"u8 r2 1->1 (u8 out)", 1, 20, 30, 1, 1, DT::U8, DT::U8, 2, 0.01},
        {"u8 r3 3->3 (u8 out)", 1, 18, 24, 3, 3, DT::U8, DT::U8, 3, 0.01},
        {"u8->f32 dDepth=5", 1, 20, 30, 1, 1, DT::U8, DT::F32, 2, 0.01},
        // eps 极大 => alpha ~ 0, beta = mean_p => 输出应等于「纯 box 均值」，用于隔离盒式滤波本身
        {"BOXONLY f32 r2 1->1", 1, 17, 23, 1, 1, DT::F32, DT::F32, 2, 1.0e6},
        {"BOXONLY f32 r4 3->3", 1, 24, 33, 3, 3, DT::F32, DT::F32, 4, 1.0e6},
        {"RAMPX 9x16 r2", 1, 9, 16, 1, 1, DT::F32, DT::F32, 2, 1.0e6, false, true},
        {"RAMPX 9x16 r4", 1, 9, 16, 1, 1, DT::F32, DT::F32, 4, 1.0e6, false, true},
        {"FLAT 9x8 r2 (1 tile)", 1, 9, 8, 1, 1, DT::F32, DT::F32, 2, 1.0e6, true},
        {"FLAT 9x9 r2 (2 tiles)", 1, 9, 9, 1, 1, DT::F32, DT::F32, 2, 1.0e6, true},
        {"FLAT 9x16 r2", 1, 9, 16, 1, 1, DT::F32, DT::F32, 2, 1.0e6, true},
        {"FLAT 9x11 r2", 1, 9, 11, 1, 1, DT::F32, DT::F32, 2, 1.0e6, true},
        {"FLAT f32 r4 3->3", 1, 12, 14, 3, 3, DT::F32, DT::F32, 4, 1.0e6, true},
        // 规模/性能用例
        {"scale f32 512x512 r4 1->1", 1, 512, 512, 1, 1, DT::F32, DT::F32, 4, 0.01},
        {"scale f32 720x1280 r4 1->1", 1, 720, 1280, 1, 1, DT::F32, DT::F32, 4, 0.01},
        {"scale f32 720x1280 r4 3->3", 1, 720, 1280, 3, 3, DT::F32, DT::F32, 4, 0.04},
        {"scale f32 1080x1920 r4 1->1", 1, 1080, 1920, 1, 1, DT::F32, DT::F32, 4, 0.01},
        {"scale f32 1080x1920 r4 3->3", 1, 1080, 1920, 3, 3, DT::F32, DT::F32, 4, 0.04},
    };

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    const bool bench = (getenv("GF_BENCH") != nullptr);
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    int32_t devId = 0;
    ret = aclrtSetDevice(devId);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    int pass = 0;
    int fail = 0;
    // GF_ONLY=<子串> 只跑命中的用例（profiling / 单点复测用）；GF_ITERS 控制 bench 轮数
    const char* onlyEnv = getenv("GF_ONLY");
    const std::string only = (onlyEnv != nullptr) ? std::string(onlyEnv) : std::string();
    const char* itersEnv = getenv("GF_ITERS");
    const int benchIters = (itersEnv != nullptr) ? std::atoi(itersEnv) : 20;

    for (const auto& c : cases) {
        if (!only.empty() && std::string(c.name).find(only) == std::string::npos) {
            continue;
        }
        const int64_t H = c.H, W = c.W, N = c.N;
        Rng rng;
        const double scale = (c.inDt == DT::U8) ? 255.0 : 1.0;
        std::vector<float> gHost, pHost;
        GenImage(gHost, N * H, W, c.C, scale, rng, 20261025ULL);
        GenImage(pHost, N * H, W, c.Cs, scale, rng, 777ULL);
        if (c.rampX) {
            for (int64_t yy = 0; yy < N * H; ++yy) {
                for (int64_t xx = 0; xx < W; ++xx) {
                    for (int64_t ch = 0; ch < c.C; ++ch) {
                        gHost[static_cast<size_t>((yy * W + xx) * c.C + ch)] = static_cast<float>(xx);
                    }
                    for (int64_t ch = 0; ch < c.Cs; ++ch) {
                        pHost[static_cast<size_t>((yy * W + xx) * c.Cs + ch)] = static_cast<float>(xx);
                    }
                }
            }
        }
        if (c.flat) {
            std::fill(gHost.begin(), gHost.end(), static_cast<float>(0.5 * scale));
            std::fill(pHost.begin(), pHost.end(), static_cast<float>(0.25 * scale));
        }

        std::vector<uint8_t> gIn(static_cast<size_t>(N * H * W * c.C) * DtSize(c.inDt));
        std::vector<uint8_t> pIn(static_cast<size_t>(N * H * W * c.Cs) * DtSize(c.inDt));
        auto pack = [&](const std::vector<float>& src, std::vector<uint8_t>& dst) {
            for (size_t i = 0; i < src.size(); ++i) {
                if (c.inDt == DT::F32) {
                    const float v = src[i];
                    memcpy(dst.data() + i * 4, &v, 4);
                } else if (c.inDt == DT::F16) {
                    const uint16_t h = ToHalf(src[i]);
                    memcpy(dst.data() + i * 2, &h, 2);
                } else {
                    float v = src[i];
                    v = (v < 0.0F) ? 0.0F : ((v > 255.0F) ? 255.0F : v);
                    dst[i] = static_cast<uint8_t>(v + 0.5F);
                }
            }
        };
        pack(gHost, gIn);
        pack(pHost, pIn);

        // 量化后的 float 视图（参考实现的输入）
        auto unpack = [&](const std::vector<uint8_t>& src, size_t cnt) {
            std::vector<float> r(cnt);
            for (size_t i = 0; i < cnt; ++i) {
                if (c.inDt == DT::F32) {
                    memcpy(&r[i], src.data() + i * 4, 4);
                } else if (c.inDt == DT::F16) {
                    uint16_t h;
                    memcpy(&h, src.data() + i * 2, 2);
                    r[i] = FromHalf(h);
                } else {
                    r[i] = static_cast<float>(src[i]);
                }
            }
            return r;
        };
        const std::vector<float> gQ = unpack(gIn, static_cast<size_t>(N * H * W * c.C));
        const std::vector<float> pQ = unpack(pIn, static_cast<size_t>(N * H * W * c.Cs));

        // ---- CPU 参考 ----
        std::vector<std::vector<double>> refOut;
        {
            std::vector<std::vector<double>> gg(static_cast<size_t>(c.C)), pp(static_cast<size_t>(c.Cs));
            for (int64_t img = 0; img < N; ++img) {
                for (int64_t ch = 0; ch < c.C; ++ch) {
                    gg[static_cast<size_t>(ch)].assign(static_cast<size_t>(H * W), 0.0);
                    for (int64_t i = 0; i < H * W; ++i) {
                        gg[static_cast<size_t>(ch)]
                          [static_cast<size_t>(i)] = gQ[static_cast<size_t>((img * H * W + i) * c.C + ch)];
                    }
                }
                for (int64_t ch = 0; ch < c.Cs; ++ch) {
                    pp[static_cast<size_t>(ch)].assign(static_cast<size_t>(H * W), 0.0);
                    for (int64_t i = 0; i < H * W; ++i) {
                        pp[static_cast<size_t>(ch)]
                          [static_cast<size_t>(i)] = pQ[static_cast<size_t>((img * H * W + i) * c.Cs + ch)];
                    }
                }
                std::vector<std::vector<double>> one;
                GuidedFilterRef(gg, pp, H, W, c.radius, c.eps, one);
                if (img == 0) {
                    refOut = one;
                } else {
                    for (size_t s = 0; s < one.size(); ++s) {
                        refOut[s].insert(refOut[s].end(), one[s].begin(), one[s].end());
                    }
                }
            }
        }

        // ---- 设备侧 ----
        int64_t gShape[4], pShape[4], oShape[4];
        int64_t gRank = 0;
        if (c.C == 1 && c.Cs == 1 && c.N == 1) {
            gRank = 2;
            gShape[0] = H;
            gShape[1] = W;
            pShape[0] = H;
            pShape[1] = W;
        } else if (c.N == 1) {
            gRank = 3;
            gShape[0] = H;
            gShape[1] = W;
            gShape[2] = c.C;
            pShape[0] = H;
            pShape[1] = W;
            pShape[2] = c.Cs;
        } else {
            gRank = 4;
            gShape[0] = N;
            gShape[1] = H;
            gShape[2] = W;
            gShape[3] = c.C;
            pShape[0] = N;
            pShape[1] = H;
            pShape[2] = W;
            pShape[3] = c.Cs;
        }
        for (int64_t i = 0; i < gRank; ++i) {
            oShape[i] = pShape[i];
        }

        void* gDev = nullptr;
        void* pDev = nullptr;
        void* oDev = nullptr;
        const size_t oBytes = static_cast<size_t>(N * H * W * c.Cs) * DtSize(c.outDt);
        aclrtMalloc(&gDev, gIn.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&pDev, pIn.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&oDev, oBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(gDev, gIn.size(), gIn.data(), gIn.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(pDev, pIn.size(), pIn.data(), pIn.size(), ACL_MEMCPY_HOST_TO_DEVICE);

        const int64_t dDepth = (c.outDt == DT::F32) ? 5 : ((c.outDt == DT::F16) ? 7 : 0);
        const int64_t depthArg = (c.outDt == c.inDt) ? -1 : dDepth;

        aclTensor* gT = aclCreateTensor(gShape, gRank, ToAcl(c.inDt), nullptr, 0, ACL_FORMAT_ND, gShape, gRank, gDev);
        aclTensor* pT = aclCreateTensor(pShape, gRank, ToAcl(c.inDt), nullptr, 0, ACL_FORMAT_ND, pShape, gRank, pDev);
        aclTensor* oT = aclCreateTensor(oShape, gRank, ToAcl(c.outDt), nullptr, 0, ACL_FORMAT_ND, oShape, gRank, oDev);

        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnGuidedFilterGetWorkspaceSize(gT, pT, c.radius, c.eps, depthArg, 1.0, oT, &wsSize, &exec);
        if (st != ACL_SUCCESS) {
            printf("[FAIL] %-24s GetWorkspaceSize failed, st=%d\n", c.name, static_cast<int>(st));
            ++fail;
            continue;
        }
        void* ws = nullptr;
        if (wsSize > 0) {
            aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        st = aclnnGuidedFilter(ws, wsSize, exec, stream);
        if (st == ACL_SUCCESS) {
            st = aclrtSynchronizeStream(stream);
        }
        if (st != ACL_SUCCESS) {
            printf("[FAIL] %-24s run failed, st=%d err=%s\n", c.name, static_cast<int>(st), aclGetRecentErrMsg());
            ++fail;
            continue;
        }

        std::vector<uint8_t> outHost(oBytes);
        std::vector<double> gotMap(static_cast<size_t>(N * H * W * c.Cs), 0.0);
        aclrtMemcpy(outHost.data(), oBytes, oDev, oBytes, ACL_MEMCPY_DEVICE_TO_HOST);

        double maxErr = 0.0;
        double maxErrBorder = 0.0;
        double maxErrInner = 0.0;
        double maxRef = 0.0;
        int64_t argY = -1, argX = -1, argC = -1;
        int64_t bad = 0;
        const int64_t total = N * H * W * c.Cs;
        for (int64_t i = 0; i < total; ++i) {
            // HWC 扁平下标 -> (通道 s, 跨图平面下标 j)
            const int64_t s = i % c.Cs;
            const int64_t j = i / c.Cs;
            const double ref = refOut[static_cast<size_t>(s)][static_cast<size_t>(j)];
            double got = 0.0;
            if (c.outDt == DT::F32) {
                float v;
                memcpy(&v, outHost.data() + i * 4, 4);
                got = v;
            } else if (c.outDt == DT::F16) {
                uint16_t h;
                memcpy(&h, outHost.data() + i * 2, 2);
                got = FromHalf(h);
            } else {
                got = static_cast<double>(outHost[static_cast<size_t>(i)]);
            }
            if (static_cast<size_t>(i) < gotMap.size()) {
                gotMap[static_cast<size_t>(i)] = got;
            }
            maxRef = std::max(maxRef, std::fabs(ref));
            const double ad = std::fabs(got - ref);
            if (ad > maxErr) {
                maxErr = ad;
                argY = (j % (H * W)) / W;
                argX = (j % (H * W)) % W;
                argC = s;
            }
            const int64_t yy = (j % (H * W)) / W, xx2 = (j % (H * W)) % W;
            const bool onBorder = (yy < c.radius || yy >= H - c.radius || xx2 < c.radius || xx2 >= W - c.radius);
            if (onBorder) {
                maxErrBorder = std::max(maxErrBorder, ad);
            } else {
                maxErrInner = std::max(maxErrInner, ad);
            }
            const double tol = (c.outDt == DT::U8) ? 1.0 :
                                                     (c.outDt == DT::F16 ? (2e-3 + 3e-3 * std::fabs(ref)) :
                                                                           (1e-5 + 1e-4 * std::fabs(ref)));
            if (getenv("GF_SAMPLE") != nullptr && i < 6) {
                printf("       sample[%lld] got=%.6f ref=%.6f\n", static_cast<long long>(i), got, ref);
            }
            if (std::fabs(got - ref) > tol) {
                ++bad;
                if (bad <= 3) {
                    printf("       diff @%lld ref=%.6f got=%.6f\n", static_cast<long long>(i), ref, got);
                }
            }
        }
        if (getenv("GF_MAP") != nullptr && N * H * W <= 200) {
            printf("MAP %s (ref=%.4f)\n", c.name, total > 0 ? refOut[0][0] : 0.0);
            for (int64_t yy = 0; yy < H; ++yy) {
                printf("  ");
                for (int64_t xx = 0; xx < W; ++xx) {
                    printf("%7.4f ", gotMap[static_cast<size_t>((yy * W + xx) * c.Cs)]);
                }
                printf("\n");
            }
        }
        maxRef = std::max(maxRef, 1.0);
        const bool ok = (bad == 0);
        if (ok) {
            ++pass;
        } else {
            ++fail;
        }
        printf("[%s] %-24s maxErr=%.3e inner=%.3e border=%.3e (rel %.2e) argmax@(y=%lld,x=%lld,c=%lld) "
               "bad=%lld/%lld\n",
               ok ? " OK " : "FAIL", c.name, maxErr, maxErrInner, maxErrBorder, maxErr / maxRef,
               static_cast<long long>(argY), static_cast<long long>(argX), static_cast<long long>(argC),
               static_cast<long long>(bad), static_cast<long long>(total));

        if (bench) {
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < benchIters; ++it) {
                uint64_t ws2 = 0;
                aclOpExecutor* e2 = nullptr;
                aclnnGuidedFilterGetWorkspaceSize(gT, pT, c.radius, c.eps, depthArg, 1.0, oT, &ws2, &e2);
                aclnnGuidedFilter(ws, ws2, e2, stream);
            }
            aclrtSynchronizeStream(stream);
            auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / benchIters;
            printf("       bench: %.3f ms/iter (%lld px, C=%lld Cs=%lld r=%lld)\n", ms,
                   static_cast<long long>(N * H * W), static_cast<long long>(c.C), static_cast<long long>(c.Cs),
                   static_cast<long long>(c.radius));
        }

        aclDestroyTensor(gT);
        aclDestroyTensor(pT);
        aclDestroyTensor(oT);
        if (ws != nullptr) {
            aclrtFree(ws);
        }
        aclrtFree(gDev);
        aclrtFree(pDev);
        aclrtFree(oDev);
    }

    printf("\n===== %d passed, %d failed =====\n", pass, fail);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return (fail == 0) ? 0 : 1;
}
