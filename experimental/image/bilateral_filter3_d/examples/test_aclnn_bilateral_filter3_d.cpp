/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You can obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file test_aclnn_bilateral_filter3_d.cpp
 * \brief BilateralFilter3D device E2E test: correctness vs a double-precision
 * host reference plus an optional aclrtEvent benchmark mode.
 *
 * Usage:
 *   ./test_aclnn_bilateral_filter3_d                 -- run all correctness cases
 *   ./test_aclnn_bilateral_filter3_d --bench D H W dtype(0/1) k ss sc iters
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bilateral_filter3_d.h"

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
// fp16 (IEEE 754 binary16) conversion with round-to-nearest-even, matching
// numpy astype(float16) semantics.
// ---------------------------------------------------------------------------
static uint16_t Fp32ToFp16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 0x1U;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = x & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) { // (roughly) flush to signed zero
            return static_cast<uint16_t>(sign << 15);
        }
        // subnormal fp16: shift mantissa so that implicit bit aligns
        mant |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t sub = mant >> shift;
        const uint32_t rem = mant & ((1U << shift) - 1U);
        const uint32_t half = 1U << (shift - 1U);
        uint32_t rounded = sub + ((rem > half || (rem == half && (sub & 1U))) ? 1U : 0U);
        return static_cast<uint16_t>((sign << 15) + rounded);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | (0x1fU << 10));
    }
    uint32_t half = mant & 0x1fffU;
    mant >>= 13;
    if (half > 0x1000U || (half == 0x1000U && (mant & 1U) != 0U)) {
        mant += 1U;
        if (mant == 0x400U) {
            mant = 0U;
            return static_cast<uint16_t>((sign << 15) | (((exp + 1) & 0x1fU) << 10) | mant);
        }
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

static float Fp16ToFp32(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 0x1U;
    const uint32_t exp = (h >> 10) & 0x1fU;
    const uint32_t mant = h & 0x3ffU;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        const uint32_t x = (sign << 31) | ((127U - 14U) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
    if (exp == 31) {
        const uint32_t x = (sign << 31) | (0xffU << 23) | (mant ? (1U << 22) : 0U);
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
    const uint32_t x = (sign << 31) | ((exp - 15U + 127U) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------------------
// Deterministic LCG shared with the CUDA reference harness (M6 comparison).
// ---------------------------------------------------------------------------
static uint32_t LcgNext(uint32_t& state)
{
    state = state * 1664525U + 1013904223U;
    return state;
}

static float LcgUniform(uint32_t& state) { return static_cast<float>(LcgNext(state) >> 8) * (1.0f / 16777216.0f); }

// ---------------------------------------------------------------------------
// Double-precision host golden, exact spec semantics (truncation+renorm).
// ---------------------------------------------------------------------------
static void GoldenBilateral3D(const std::vector<float>& x, const std::vector<int64_t>& shape, int64_t k,
                              double sigmaSpace, double sigmaColor, std::vector<float>& y)
{
    const int64_t d = shape[shape.size() - 3];
    const int64_t h = shape[shape.size() - 2];
    const int64_t w = shape[shape.size() - 1];
    const int64_t volSize = d * h * w;
    const int64_t total = static_cast<int64_t>(x.size());
    const int64_t nVol = total / volSize;
    const int64_t r = k / 2;
    const double inv2ss = 0.5 / (sigmaSpace * sigmaSpace);
    const double inv2sc = 0.5 / (sigmaColor * sigmaColor);
    y.assign(x.size(), 0.0f);
    for (int64_t v = 0; v < nVol; ++v) {
        const int64_t base = v * volSize;
        for (int64_t z = 0; z < d; ++z) {
            for (int64_t row = 0; row < h; ++row) {
                for (int64_t col = 0; col < w; ++col) {
                    const int64_t lin = base + z * h * w + row * w + col;
                    const double c = x[lin];
                    double num = 0.0;
                    double den = 0.0;
                    for (int64_t dz = -r; dz <= r; ++dz) {
                        const int64_t zz = z + dz;
                        if (zz < 0 || zz >= d) {
                            continue;
                        }
                        for (int64_t dy = -r; dy <= r; ++dy) {
                            const int64_t yy = row + dy;
                            if (yy < 0 || yy >= h) {
                                continue;
                            }
                            const int64_t rowBase = base + zz * h * w + yy * w;
                            for (int64_t dx = -r; dx <= r; ++dx) {
                                const int64_t xx = col + dx;
                                if (xx < 0 || xx >= w) {
                                    continue;
                                }
                                const double nb = x[rowBase + xx];
                                const double dc = nb - c;
                                const double wgt = std::exp(
                                    -(static_cast<double>(dz * dz + dy * dy + dx * dx) * inv2ss + dc * dc * inv2sc));
                                num += wgt * nb;
                                den += wgt;
                            }
                        }
                    }
                    y[lin] = static_cast<float>(num / den);
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
    return 0;
}

struct CaseRt {
    std::vector<float> hostIn;
    std::vector<float> golden;
    void* devX = nullptr;
    void* devY = nullptr;
    aclTensor* tx = nullptr;
    aclTensor* ty = nullptr;
};

static int PrepareCase(int32_t deviceId, const std::vector<int64_t>& shape, aclDataType dt, uint32_t seed, CaseRt& rt,
                       bool needGolden, int64_t k, double ss, double sc)
{
    size_t total = 1;
    for (int64_t d : shape) {
        total *= static_cast<size_t>(d);
    }
    const bool isFp16 = (dt == ACL_FLOAT16);
    const size_t bytes = total * (isFp16 ? sizeof(uint16_t) : sizeof(float));
    rt.hostIn.resize(total);
    uint32_t state = seed;
    for (size_t i = 0; i < total; ++i) {
        rt.hostIn[i] = -1.0f + 3.0f * LcgUniform(state);
    }
    CHECK_RET(aclrtMalloc(&rt.devX, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&rt.devY, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    if (isFp16) {
        std::vector<uint16_t> packed(total);
        for (size_t i = 0; i < total; ++i) {
            packed[i] = Fp32ToFp16(rt.hostIn[i]);
            rt.hostIn[i] = Fp16ToFp32(packed[i]); // golden consumes rounded values
        }
        CHECK_RET(aclrtMemcpy(rt.devX, bytes, packed.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                  return -1);
    } else {
        CHECK_RET(aclrtMemcpy(rt.devX, bytes, rt.hostIn.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                  return -1);
    }
    CHECK_RET(aclrtMemset(rt.devY, bytes, 0, bytes) == ACL_SUCCESS, return -1);
    CHECK_RET(MakeTensor(shape, dt, rt.devX, bytes, &rt.tx) == 0, return -1);
    CHECK_RET(MakeTensor(shape, dt, rt.devY, bytes, &rt.ty) == 0, return -1);
    if (needGolden) {
        GoldenBilateral3D(rt.hostIn, shape, k, ss, sc, rt.golden);
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

static int RunOnce(const CaseRt& rt, int64_t k, double ss, double sc, aclrtStream stream)
{
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnBilateralFilter3DGetWorkspaceSize(rt.tx, k, ss, sc, rt.ty, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed, ret=%d\n", (int)ret); return -1);
    void* ws = nullptr;
    if (wsSize > 0) {
        CHECK_RET(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }
    ret = aclnnBilateralFilter3D(ws, wsSize, executor, stream);
    if (ws != nullptr) {
        (void)aclrtFree(ws);
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBilateralFilter3D failed, ret=%d\n", (int)ret); return -1);
    return 0;
}

struct CaseDef {
    const char* name;
    std::vector<int64_t> shape;
    aclDataType dt;
    int64_t k;
    double ss;
    double sc;
    double tol;
};

static int RunCorrectness(aclrtStream stream)
{
    const std::vector<CaseDef> cases = {
        {"fp32_37cube_k5", {37, 37, 37}, ACL_FLOAT, 5, 1.0, 1.0, 5e-4},
        {"fp32_64cube_k3", {64, 64, 64}, ACL_FLOAT, 3, 1.0, 1.0, 5e-4},
        {"fp16_64cube_k5", {64, 64, 64}, ACL_FLOAT16, 5, 0.8, 0.5, 2e-3},
        {"fp32_rank4_k7", {2, 32, 50, 41}, ACL_FLOAT, 7, 1.0, 1.0, 5e-4},
        {"fp16_rank5_k9", {2, 3, 32, 50, 41}, ACL_FLOAT16, 9, 1.2, 2.0, 2e-3},
        {"fp32_thind_k5", {2, 9, 11}, ACL_FLOAT, 5, 1.5, 2.0, 5e-4},
        {"fp32_tinyhw_k9", {9, 4, 6}, ACL_FLOAT, 9, 1.0, 1.0, 5e-4},
        {"fp32_sc_tiny", {32, 32, 32}, ACL_FLOAT, 5, 1.0, 1e-7, 5e-5},
        {"fp32_sc_huge", {32, 32, 32}, ACL_FLOAT, 5, 1.0, 1e9, 5e-4},
        {"fp16_ss_tiny", {32, 37, 29}, ACL_FLOAT16, 5, 1e-7, 0.5, 5e-5},
    };
    int pass = 0;
    int fail = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const CaseDef& c = cases[ci];
        CaseRt rt;
        if (PrepareCase(0, c.shape, c.dt, 1000U + static_cast<uint32_t>(ci), rt, true, c.k, c.ss, c.sc) != 0) {
            LOG_PRINT("[FAIL] %s prepare error\n", c.name);
            ++fail;
            continue;
        }
        if (RunOnce(rt, c.k, c.ss, c.sc, stream) != 0) {
            LOG_PRINT("[FAIL] %s run error\n", c.name);
            ++fail;
            ReleaseCase(rt);
            continue;
        }
        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ReleaseCase(rt); ++fail; continue);
        const size_t total = rt.golden.size();
        const bool isFp16 = (c.dt == ACL_FLOAT16);
        const size_t bytes = total * (isFp16 ? sizeof(uint16_t) : sizeof(float));
        std::vector<uint8_t> raw(bytes);
        CHECK_RET(aclrtMemcpy(raw.data(), bytes, rt.devY, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                  ReleaseCase(rt);
                  ++fail; continue);
        double maxErr = 0.0;
        double maxRel = 0.0;
        int firstBad = -1;
        for (size_t i = 0; i < total; ++i) {
            float out;
            if (isFp16) {
                uint16_t h;
                std::memcpy(&h, raw.data() + i * 2, 2);
                out = Fp16ToFp32(h);
            } else {
                std::memcpy(&out, raw.data() + i * 4, 4);
            }
            const double diff = std::fabs(static_cast<double>(out) - rt.golden[i]);
            const double bound = c.tol * (1.0 + std::fabs(static_cast<double>(rt.golden[i])));
            if (!(diff <= bound) && firstBad < 0) { // NaN 也判失败(比较恒 false)
                firstBad = static_cast<int>(i);
            }
            maxErr = std::max(maxErr, diff);
            maxRel = std::max(maxRel, diff / (1.0 + std::fabs(static_cast<double>(rt.golden[i]))));
        }
        if (firstBad < 0) {
            LOG_PRINT("[PASS] %-18s max_err=%.3e max_rel=%.3e\n", c.name, maxErr, maxRel);
            ++pass;
        } else {
            double o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            if (!raw.empty()) {
                auto rd = [&](size_t i) {
                    if (c.dt == ACL_FLOAT) {
                        float v;
                        std::memcpy(&v, raw.data() + i * 4, 4);
                        return static_cast<double>(v);
                    }
                    const uint16_t h = *reinterpret_cast<uint16_t*>(raw.data() + i * 2);
                    return static_cast<double>(c.dt == ACL_FLOAT16 ? Fp16ToFp32(h) : Fp16ToFp32(h));
                };
                o0 = rd(0);
                o1 = rd(1);
                o2 = rd(2);
                o3 = rd(3);
            }
            LOG_PRINT("[FAIL] %-18s max_err=%.3e max_rel=%.3e first_bad=%ld gold[fb]=%.5f\n", c.name, maxErr, maxRel,
                      static_cast<long>(firstBad), rt.golden[static_cast<size_t>(firstBad)]);
            if (firstBad + 8 < static_cast<int64_t>(rt.golden.size())) {
                for (int64_t k2 = firstBad; k2 < firstBad + 8; ++k2) {
                    double ov = 0.0;
                    if (c.dt == ACL_FLOAT) {
                        float v;
                        std::memcpy(&v, raw.data() + static_cast<size_t>(k2) * 4, 4);
                        ov = v;
                    } else {
                        const uint16_t h16 = *reinterpret_cast<uint16_t*>(raw.data() + static_cast<size_t>(k2) * 2);
                        ov = Fp16ToFp32(h16);
                    }
                    LOG_PRINT("  [%ld] out=%.5f gold=%.5f\n", static_cast<long>(k2), ov,
                              rt.golden[static_cast<size_t>(k2)]);
                }
            }
            ++fail;
        }
        ReleaseCase(rt);
    }
    LOG_PRINT("=== SUMMARY: %d/%d PASS ===\n", pass, pass + fail);
    return fail == 0 ? 0 : 1;
}

static int RunBench(aclrtStream stream, int64_t d, int64_t h, int64_t w, int isFp16, int64_t k, double ss, double sc,
                    int iters)
{
    const aclDataType dt = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    CaseRt rt;
    CHECK_RET(PrepareCase(0, {d, h, w}, dt, 20260827U, rt, false, k, ss, sc) == 0, return -1);
    for (int i = 0; i < 3; ++i) {
        CHECK_RET(RunOnce(rt, k, ss, sc, stream) == 0, ReleaseCase(rt); return -1);
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ReleaseCase(rt); return -1);
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    (void)aclrtCreateEvent(&begin);
    (void)aclrtCreateEvent(&end);
    std::vector<float> ms(static_cast<size_t>(iters), 0.0f);
    for (int i = 0; i < iters; ++i) {
        (void)aclrtRecordEvent(begin, stream);
        CHECK_RET(RunOnce(rt, k, ss, sc, stream) == 0, ReleaseCase(rt); return -1);
        (void)aclrtRecordEvent(end, stream);
        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ReleaseCase(rt); return -1);
        float elapsed = 0.0f;
        (void)aclrtEventElapsedTime(&elapsed, begin, end);
        ms[static_cast<size_t>(i)] = elapsed;
    }
    (void)aclrtDestroyEvent(begin);
    (void)aclrtDestroyEvent(end);
    std::sort(ms.begin(), ms.end());
    const float median = ms[ms.size() / 2];
    const float mean = [&ms]() {
        double s = 0.0;
        for (float v : ms) {
            s += v;
        }
        return static_cast<float>(s / ms.size());
    }();
    const double gbytes = 2.0 * static_cast<double>(d) * h * w * (isFp16 ? 2 : 4) / 1e9;
    LOG_PRINT("BENCH D=%ld H=%ld W=%ld dtype=%s k=%ld ss=%.3g sc=%.3g iters=%d median=%.3fms mean=%.3fms "
              "min=%.3fms io_gb=%.3f eff_bw=%.2fGB/s\n",
              d, h, w, isFp16 ? "fp16" : "fp32", k, ss, sc, iters, median, mean, ms.front(), gbytes,
              gbytes / (median / 1000.0));
    ReleaseCase(rt);
    return 0;
}

// ---------------------------------------------------------------------------
// Diagnostic single-shape mode. Unlike the fixed case list this reports *where*
// the mismatches sit -- which columns and which rows -- because that is what
// separates a boundary-mask bug (errors confined to specific columns) from a row
// indexing bug (errors confined to specific rows).
// ---------------------------------------------------------------------------
static int RunDiag(aclrtStream stream, int64_t d, int64_t h, int64_t w, int isFp16, int64_t k, double ss, double sc,
                   double tol)
{
    const aclDataType dt = isFp16 ? ACL_FLOAT16 : ACL_FLOAT;
    CaseRt rt;
    CHECK_RET(PrepareCase(0, {d, h, w}, dt, 4242U, rt, true, k, ss, sc) == 0, return -1);
    CHECK_RET(RunOnce(rt, k, ss, sc, stream) == 0, ReleaseCase(rt); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, ReleaseCase(rt); return -1);

    const size_t total = rt.golden.size();
    const size_t bytes = total * (isFp16 ? sizeof(uint16_t) : sizeof(float));
    std::vector<uint8_t> raw(bytes);
    CHECK_RET(aclrtMemcpy(raw.data(), bytes, rt.devY, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, ReleaseCase(rt);
              return -1);

    std::vector<int64_t> badPerColumn(static_cast<size_t>(w), 0);
    std::vector<int64_t> badRows;
    size_t badCount = 0;
    double maxErr = 0.0;
    for (size_t i = 0; i < total; ++i) {
        float out;
        if (isFp16) {
            uint16_t packed;
            std::memcpy(&packed, raw.data() + i * 2, 2);
            out = Fp16ToFp32(packed);
        } else {
            std::memcpy(&out, raw.data() + i * 4, 4);
        }
        const double diff = std::fabs(static_cast<double>(out) - rt.golden[i]);
        const double bound = tol * (1.0 + std::fabs(static_cast<double>(rt.golden[i])));
        maxErr = std::max(maxErr, diff);
        if (!(diff <= bound)) { // NaN also lands here (comparison is always false)
            ++badCount;
            const int64_t col = static_cast<int64_t>(i) % w;
            const int64_t row = static_cast<int64_t>(i) / w;
            badPerColumn[static_cast<size_t>(col)]++;
            if (badRows.empty() || badRows.back() != row) {
                badRows.push_back(row);
            }
        }
    }

    const int64_t totalRows = static_cast<int64_t>(total) / w;
    LOG_PRINT("DIAG D=%ld H=%ld W=%ld %s k=%ld ss=%.3g sc=%.3g : bad=%ld/%ld rows=%ld max_err=%.3e %s\n", d, h, w,
              isFp16 ? "fp16" : "fp32", k, ss, sc, static_cast<long>(badCount), static_cast<long>(total),
              static_cast<long>(totalRows), maxErr, badCount == 0 ? "OK" : "MISMATCH");
    if (badCount > 0) {
        LOG_PRINT("  bad columns:");
        for (int64_t c = 0; c < w; ++c) {
            if (badPerColumn[static_cast<size_t>(c)] > 0) {
                LOG_PRINT(" %ld(n=%ld)", c, static_cast<long>(badPerColumn[static_cast<size_t>(c)]));
            }
        }
        LOG_PRINT("\n  bad rows: %ld of %ld distinct, first 12:", static_cast<long>(badRows.size()),
                  static_cast<long>(totalRows));
        for (size_t i = 0; i < badRows.size() && i < 12; ++i) {
            LOG_PRINT(" %ld", static_cast<long>(badRows[i]));
        }
        LOG_PRINT("\n");
    }
    ReleaseCase(rt);
    return badCount == 0 ? 0 : 1;
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
    if (argc >= 2 && std::string(argv[1]) == "--bench" && argc >= 10) {
        rc = RunBench(stream, std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]), std::atoi(argv[5]),
                      std::atoll(argv[6]), std::atof(argv[7]), std::atof(argv[8]), std::atoi(argv[9]));
    } else if (argc >= 2 && std::string(argv[1]) == "--diag" && argc >= 10) {
        // --diag D H W dtype(0=fp32,1=fp16) k sigma_space sigma_color tol
        rc = RunDiag(stream, std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]), std::atoi(argv[5]),
                     std::atoll(argv[6]), std::atof(argv[7]), std::atof(argv[8]), std::atof(argv[9]));
    } else {
        LOG_PRINT("=== BilateralFilter3D E2E verification (Ascend 910B device) ===\n");
        rc = RunCorrectness(stream);
    }
    (void)aclrtSynchronizeStream(stream);
    (void)aclrtDestroyStream(stream);
    (void)aclrtDestroyContext(ctx);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return rc;
}
