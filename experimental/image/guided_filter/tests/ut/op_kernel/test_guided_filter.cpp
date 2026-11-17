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
 * \file test_guided_filter.cpp
 * \brief GuidedFilter Kernel UT（ICPU/CPU 模拟执行内核源码，无需真机）
 *
 * 数据管线与同仓 kernel UT（pixel_shuffle / rgb2yuv422 / yuv444_to_yuv422）一致：
 *   SetUpTestCase 把 guided_filter_data 拷到运行目录 → 每个用例 cd 进该目录调
 *   gen_data.py 生成唯一可复现的输入与 golden → ICPU 跑核 → 结果写成 *output*.bin →
 *   compare_data.py 逐元素比对。数据目录用编译期 __FILE__ 推导（不依赖运行时 CWD）。
 *   参考实现是 numpy float64（见 gen_data.py），与内核的 C++ float32 实现互相独立，
 *   是真正的第二实现比对，而不是「内核算法的 C++ 复述」。
 *
 * 覆盖点：
 *   1) 通道组合 1->1 / 3->3 / 3->1 / 1->3（C、Cs 的耦合路径不同）；
 *   2) 半径 r = 1 / 2 / 4（r 是 4 的倍数时 CPU 模型对 Gather 源张量的尺寸校验会额外
 *      要求 8 列余量，见 MakeGeo 的注释）；大 eps、强边缘图；
 *   3) 单 tile 与多 tile：后者用手工 tileW 在小图上强制 nTx > 1，覆盖跨 tile 的
 *      halo 拼接（tile 左右边界要读邻 tile 写出的 alpha/beta）—— 真机上只有大图才
 *      走得到这条路径，而 ICPU 跑不动 1080p，所以用 override 复现；
 *   4) batch > 1：DecodeTask 的 img 维切分与 workspace 里逐图的 alpha/beta 段。
 *
 * 与 pixel_shuffle 的一处关键差异：本内核用无参硬件 SyncAll 做阶段 A/B 的跨核同步
 * （阶段 A 把 alpha/beta 写进 workspace，阶段 B 再读回来做第二次盒式滤波）。ICPU 把
 * block 顺序执行、屏障退化为空操作，所以 blockNum 只能是 1 —— 多 block 时 block0 会在
 * block1 尚未写出 alpha/beta 时就开始阶段 B，语义与真机不同。
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_kernel/guided_filter.cpp"
#include "guided_filter_tiling.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

namespace {
constexpr const char* kDtype = "float32"; // 与 CMakeLists 的 -DDTYPE_GUIDE=float ... 对应
const string kDataDir = "./guided_filter_data";
const string kGuideBin = kDataDir + "/" + kDtype + "_input_guided_filter_guide.bin";
const string kSrcBin = kDataDir + "/" + kDtype + "_input_guided_filter_src.bin";
const string kGoldenBin = kDataDir + "/" + kDtype + "_golden_guided_filter.bin";
const string kOutputBin = kDataDir + "/" + kDtype + "_output_guided_filter.bin";

// ICPU 下 SyncAll 退化为空操作、block 串行 ⇒ 只能单 block 跑（见文件头说明）
constexpr uint32_t kBlockNum = 1U;

bool FileExists(const string& path)
{
    ifstream f(path);
    return f.good();
}

// 数据目录定位：__FILE__ 在构建时是「构建目录相对路径」（形如 ../../experimental/...），
// 不能直接当路径用，因此按「__FILE__ 同目录 → 从 CWD 逐级向上找」的顺序取第一个真实存在的候选。
string SrcDataDir()
{
    const string dirName = "guided_filter_data";
    const string marker = "tests/ut/op_kernel/" + dirName; // 候选=目录，检查时再拼 /gen_data.py
    vector<string> cands;
    const string f = __FILE__;
    const size_t pos = f.find_last_of('/');
    if (pos != string::npos) {
        cands.push_back(f.substr(0, pos) + "/" + dirName);
    }
    string up;
    for (int i = 0; i <= 6; ++i) {
        cands.push_back(up + marker);                                       // CWD 向上（如从仓库根运行）
        cands.push_back(up + "experimental/image/guided_filter/" + marker); // 构建目录向上
        up += "../";
    }
    for (const auto& c : cands) {
        if (FileExists(c + "/gen_data.py")) {
            return c;
        }
    }
    return "./" + dirName; // 兜底：沿用同仓 kernel UT 的 CWD 相对写法
}

struct Case {
    int n;
    int h;
    int w;
    int C;
    int Cs;
    int r;
    float eps;
    int tileWOverride; // >0 时手工指定 tileW（用于小图强制 nTx > 1）
};

struct GeoP {
    int tileW;
    int haloW;
    int nTx;
    int nTy;
    int bandH;
};

// gen_data.py 的形状字符串：NHWC 口径 (N,H,W,C,Cs)
string ShapeStr(const Case& c)
{
    ostringstream os;
    os << "(" << c.n << "," << c.h << "," << c.w << "," << c.C << "," << c.Cs << ")";
    return os.str();
}

// haloW 的取法：除了满足内核自己的「移位副本读 v[i+k]，i < align8(T)，k <= 2r」，
// 还要覆盖 CPU 模型对 Gather 源张量尺寸的保守上界
//   align8((mapBatch-1)*haloW + align8(T) + (2r+1) + 8)
// 当 r 是 4 的倍数时 op_host 的 align8(T + 2r + 8) 恰好比这个上界少 8 列，模拟器会直接
// 报 "srcOffsetLocal too small" 并 SIGABRT。多留一列（+9）就能让两者对齐；真机上这条
// 约束严格来说不需要，但多 8 列 halo 在 UB 预算里完全放得下。
GeoP MakeGeo(const Case& c)
{
    auto makeHalo = [&](int tileW) { return GfGeo::Align8(GfGeo::Align8(tileW) + 2 * c.r + 9); };
    if (c.tileWOverride > 0) {
        const int tw = std::min(c.tileWOverride, c.w);
        return GeoP{tw, makeHalo(tw), (c.w + tw - 1) / tw, 1, c.h};
    }
    // 默认走与 op_host 完全相同的搜索（coreNum = 1 ⇒ nTy = 1、blockDim = 1）
    const GfGeo::Geo g = GfGeo::DeriveGeo(c.w, c.h, c.n, c.r, c.C, c.Cs, 4, 4, 4, 1);
    return GeoP{static_cast<int>(g.tileW), makeHalo(static_cast<int>(g.tileW)), static_cast<int>(g.nTx),
                static_cast<int>(g.nTy), static_cast<int>(g.bandH)};
}

// 与 op_host/guided_filter_tiling.cpp 逐字段对齐（ABI 见 op_kernel/guided_filter_tiling_data.h）
void FillTiling(GuidedFilterTilingData* t, const Case& c, const GeoP& geo)
{
    memset(t, 0, sizeof(*t));
    const uint32_t planeElems = static_cast<uint32_t>(
        GfGeo::Align8(static_cast<int32_t>(c.h) * static_cast<int32_t>(c.w)));
    const uint64_t alphaElems = static_cast<uint64_t>(c.C) * c.Cs * c.n * planeElems;
    const float win = static_cast<float>(2 * c.r + 1);

    t->batch = static_cast<uint32_t>(c.n);
    t->height = static_cast<uint32_t>(c.h);
    t->width = static_cast<uint32_t>(c.w);
    t->guideCh = static_cast<uint32_t>(c.C);
    t->srcCh = static_cast<uint32_t>(c.Cs);
    t->radius = static_cast<uint32_t>(c.r);
    t->tileW = static_cast<uint32_t>(geo.tileW);
    t->haloW = static_cast<uint32_t>(geo.haloW);
    t->nTx = static_cast<uint32_t>(geo.nTx);
    t->nTy = static_cast<uint32_t>(geo.nTy);
    t->bandH = static_cast<uint32_t>(geo.bandH);
    t->atomNum = static_cast<uint32_t>(GfGeo::AtomNum(c.C, c.Cs));
    t->coefNum = static_cast<uint32_t>(GfGeo::CoefNum(c.C, c.Cs));
    t->planeElems = planeElems;
    t->outDtype = 0U; // f32
    t->eps = c.eps;
    t->invWin = 1.0F / (win * win);
    t->alphaOff = 0ULL; // 相对 GetUserWorkspace，单位字节
    t->betaOff = alphaElems * sizeof(float);
}

void RunCase(const char* tag, const Case& c)
{
    // 1) 生成输入与 golden（cd 进数据目录；文件名与 compare_data.py 的 glob 约定一致）
    ostringstream gen;
    gen << "cd " << kDataDir << " && python3 gen_data.py '" << ShapeStr(c) << "' '" << kDtype << "' " << c.r << " "
        << setprecision(9) << c.eps << " > /dev/null 2>&1";
    ASSERT_EQ(system(gen.str().c_str()), 0) << tag << ": gen_data.py 执行失败（检查 python3/numpy 与数据目录）";

    const GeoP geo = MakeGeo(c);
    GuidedFilterTilingData td;
    FillTiling(&td, c, geo);

    const size_t gBytes = static_cast<size_t>(c.n) * c.h * c.w * c.C * sizeof(float);
    const size_t pBytes = static_cast<size_t>(c.n) * c.h * c.w * c.Cs * sizeof(float);
    const uint64_t alphaElems = static_cast<uint64_t>(c.C) * c.Cs * c.n * td.planeElems;
    const uint64_t betaElems = static_cast<uint64_t>(c.Cs) * c.n * td.planeElems;
    const size_t wsBytes = static_cast<size_t>((alphaElems + betaElems) * sizeof(float));

    // workspace 放在最前面分配：ICPU 模型的 GmAlloc 池有限，大块 workspace 若排在
    // 最后分配，容易和前面几块挤在同一段里（实测表现为 kernel 写出的 alpha/beta 读不
    // 回来）。先分配可以让它独占一段，行为与真机一致。
    uint8_t* wsDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wsBytes));
    uint8_t* gDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gBytes));
    uint8_t* pDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(pBytes));
    uint8_t* oDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(pBytes));
    uint8_t* tDev = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GuidedFilterTilingData)));
    ASSERT_TRUE(gDev != nullptr && pDev != nullptr && oDev != nullptr && wsDev != nullptr && tDev != nullptr)
        << tag << ": GmAlloc 失败";

    size_t got = 0;
    ASSERT_TRUE(ReadFile(kGuideBin, got, gDev, gBytes)) << tag << ": 读不到 " << kGuideBin;
    ASSERT_TRUE(ReadFile(kSrcBin, got, pDev, pBytes)) << tag << ": 读不到 " << kSrcBin;
    memset(oDev, 0, pBytes);
    memset(wsDev, 0, wsBytes);
    memcpy(tDev, &td, sizeof(GuidedFilterTilingData));

    // 2) ICPU 跑核（单 block，见文件头）
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(guided_filter<GUIDED_FILTER_TPL_SCH_MODE_0>, kBlockNum, gDev, pDev, oDev, wsDev, tDev);

    // 2.5) 诊断用：把 workspace 里的 alpha/beta 落盘（GF_UT_DBG=1 时），
    //      用来二分「NaN 出在阶段 A 还是阶段 B」。文件名不含 golden/output，不影响比对。
    const char* dbgEnv = getenv("GF_UT_DBG");
    if (dbgEnv != nullptr && dbgEnv[0] == '1') {
        const size_t aBytes = static_cast<size_t>(alphaElems) * sizeof(float);
        const size_t bBytes = static_cast<size_t>(betaElems) * sizeof(float);
        WriteFile(kDataDir + "/" + kDtype + "_dbg_alpha.bin", wsDev, aBytes);
        WriteFile(kDataDir + "/" + kDtype + "_dbg_beta.bin", wsDev + aBytes, bBytes);
        const float* af = reinterpret_cast<const float*>(wsDev);
        const float* bf = reinterpret_cast<const float*>(wsDev + aBytes);
        const float* of = reinterpret_cast<const float*>(oDev);
        auto stat = [](const char* name, const float* p, size_t n) {
            size_t nanCnt = 0, fin = 0;
            float lo = 0.0F, hi = 0.0F;
            for (size_t i = 0; i < n; ++i) {
                if (p[i] != p[i]) {
                    ++nanCnt;
                } else if (fin == 0 || p[i] < lo) {
                    lo = p[i];
                    if (fin == 0) {
                        hi = p[i];
                    }
                    ++fin;
                } else {
                    if (p[i] > hi) {
                        hi = p[i];
                    }
                    ++fin;
                }
            }
            printf("       [dbg] %-6s elems=%zu NaN=%zu finite=%zu min=%.6g max=%.6g\n", name, n, nanCnt, fin, lo, hi);
        };
        printf("       [dbg] H=%d W=%d C=%d Cs=%d planeElems=%u tileW=%d haloW=%d nTx=%d\n", c.h, c.w, c.C, c.Cs,
               td.planeElems, geo.tileW, geo.haloW, geo.nTx);
        stat("alpha", af, static_cast<size_t>(alphaElems));
        stat("beta", bf, static_cast<size_t>(betaElems));
        stat("output", of, static_cast<size_t>(pBytes / sizeof(float)));
        // 落到 CWD（不放进数据目录，避免被 TearDown 清掉），便于用 numpy 分析分布规律
        WriteFile("./gf_dbg_alpha.bin", wsDev, static_cast<size_t>(alphaElems) * sizeof(float));
        WriteFile("./gf_dbg_beta.bin", wsDev + static_cast<size_t>(alphaElems) * sizeof(float),
                  static_cast<size_t>(betaElems) * sizeof(float));
        WriteFile("./gf_dbg_out.bin", oDev, pBytes);
    }

    // 3) 写回结果，供 compare_data.py 消费（*output*.bin 与 *golden*.bin 配对）
    ASSERT_TRUE(WriteFile(kOutputBin, oDev, pBytes)) << tag << ": 写 " << kOutputBin << " 失败";

    // 注意不要重定向 compare_data.py 的输出：失败时那几行（超差个数 + 前后几个元素）
    // 是定位问题的唯一线索，pixel_shuffle 的写法把它吞掉了，这里保留。
    ostringstream cmp;
    cmp << "cd " << kDataDir << " && python3 compare_data.py '" << kDtype << "' 2>&1";
    EXPECT_EQ(system(cmp.str().c_str()), 0)
        << tag << ": 输出与 numpy golden 不一致"
        << "  [几何 tileW=" << geo.tileW << " haloW=" << geo.haloW << " nTx=" << geo.nTx << " nTy=" << geo.nTy
        << " blockNum=" << kBlockNum << "]";

    AscendC::GmFree(gDev);
    AscendC::GmFree(pDev);
    AscendC::GmFree(oDev);
    AscendC::GmFree(wsDev);
    AscendC::GmFree(tDev);
}
} // namespace

class guided_filter_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "guided_filter_test SetUp" << endl;
        ASSERT_TRUE(FileExists(dataPath + "/gen_data.py")) << "找不到数据目录: " << dataPath;
        const string cmd = "rm -rf ./guided_filter_data && cp -rf " + dataPath + " ./guided_filter_data";
        ASSERT_EQ(system(cmd.c_str()), 0) << "复制数据目录失败: " << dataPath;
        system("chmod -R 755 ./guided_filter_data/");
    }

    static void TearDownTestCase()
    {
        cout << "guided_filter_test TearDown" << endl;
        system("rm -rf ./guided_filter_data");
    }

private:
    static const std::string dataPath;
};

const std::string guided_filter_test::dataPath = SrcDataDir();

// ---------------- 基础通道组合（单 tile，几何由 DeriveGeo 决定） ----------------
// 1->1 r=4：最简退化情形（协方差矩阵 1x1，走 C==1 的闭式分支）
TEST_F(guided_filter_test, f32_1to1_r4) { RunCase("1->1 r4", Case{1, 24, 32, 1, 1, 4, 1e-2F, 0}); }

// 3->3 r=2：走 3x3 伴随矩阵求逆的完整分支
TEST_F(guided_filter_test, f32_3to3_r2) { RunCase("3->3 r2", Case{1, 16, 24, 3, 3, 2, 1e-2F, 0}); }

// 3->1：多引导通道、单输出通道（alpha 有 3 行、只用其中一组）
TEST_F(guided_filter_test, f32_3to1_r3) { RunCase("3->1 r3", Case{1, 14, 40, 3, 1, 3, 1e-2F, 16}); }

// 1->3：单引导通道、多输出通道（Cs > C，易暴露 a/b 通道错位）
TEST_F(guided_filter_test, f32_1to3_r1) { RunCase("1->3 r1", Case{1, 16, 20, 1, 3, 1, 1e-2F, 0}); }

// ---------------- 半径与 eps 分支 ----------------
// r=1：窗口行数 3，镜像边界占比重
TEST_F(guided_filter_test, f32_1to1_r1) { RunCase("1->1 r1", Case{1, 12, 20, 1, 1, 1, 1e-2F, 0}); }

// 大 eps：alpha -> 0，输出趋近 box(box(p))，压 eps 在伴随矩阵对角线上的加法位置
TEST_F(guided_filter_test, f32_3to3_large_eps) { RunCase("3->3 大 eps", Case{1, 14, 18, 3, 3, 2, 1.0F, 0}); }

// ---------------- 多 tile（跨 tile halo 拼接） ----------------
TEST_F(guided_filter_test, f32_1to1_multi_tile) { RunCase("1->1 多 tile", Case{1, 20, 60, 1, 1, 4, 1e-2F, 24}); }

TEST_F(guided_filter_test, f32_3to3_multi_tile) { RunCase("3->3 多 tile", Case{1, 15, 21, 3, 3, 2, 1e-2F, 11}); }

// ---------------- batch > 1 ----------------
TEST_F(guided_filter_test, f32_1to1_batch2) { RunCase("1->1 batch2", Case{2, 12, 16, 1, 1, 2, 1e-2F, 0}); }

// ---------------- 极小图 ----------------
// H*W 很小、W 小于一个 tile：压 tiling 边界与镜像周期
TEST_F(guided_filter_test, f32_tiny_9x16) { RunCase("极小图 9x16", Case{1, 9, 16, 1, 1, 2, 1e-2F, 0}); }
