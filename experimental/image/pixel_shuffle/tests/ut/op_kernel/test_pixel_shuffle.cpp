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
 * \file test_pixel_shuffle.cpp
 * \brief PixelShuffle Kernel UT（ICPU/CPU 模拟执行内核源码，无需真机）
 *
 * 数据管线与同仓 kernel UT（rgb2yuv422 / yuv444_to_yuv422）保持一致：
 *   SetUpTestCase 把 pixel_shuffle_data 拷到运行目录 → 每个用例 cd 进该目录调 gen_data.py
 *   生成唯一值输入与 golden → ICPU 跑核 → 把结果写成 *output*.bin → compare_data.py 比对。
 *   输入用逐元素递增的唯一值填充，置换/Gather 索引/W 分段任何错位都会在比对中暴露。
 *   数据目录用编译期 __FILE__ 推导（不依赖运行时 CWD）。
 *
 * 覆盖点（对应内核里的各条分支）：
 *   1) 4D 亚像素重排（rank=4）与 5D 重排（rank=5，planesPerGroup_ = D*r）；
 *   2) blocksPerGroup_ == 1 的「整组连续搬入」路径；
 *   3) W 方向分段路径（bPerRow_ > 1，含末段 wLast_ 的逐行 DMA 与行字节非 32B 对齐）；
 *   4) 组内分块时的 32B 对齐步长（chanStride_ / alignedBlockElems_）；
 *   5) 多 tile 与尾 tile（tileDataNum 驱动 blockPerTile_/tileNum_）；
 *   6) 多核块划分（同一形状用 1/4/8 个 block 跑，结果必须一致）。
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_kernel/pixel_shuffle.cpp"
#include "pixel_shuffle_tiling.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

namespace {
constexpr const char* kDtype = "float32"; // 与 CMakeLists 的 -DDTYPE_X=float 对应
const string kDataDir = "./pixel_shuffle_data";
const string kInputBin = kDataDir + "/" + kDtype + "_input_pixel_shuffle.bin";
const string kGoldenBin = kDataDir + "/" + kDtype + "_golden_pixel_shuffle.bin";
const string kOutputBin = kDataDir + "/" + kDtype + "_output_pixel_shuffle.bin";

bool FileExists(const string& path)
{
    ifstream f(path);
    return f.good();
}

// 数据目录定位：__FILE__ 在构建时是「构建目录相对路径」（形如 ../../experimental/...），
// 不能直接当路径用，因此按「__FILE__ 同目录 → 从 CWD 逐级向上找」的顺序取第一个真实存在的候选。
string SrcDataDir()
{
    const string dirName = "pixel_shuffle_data";
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
        cands.push_back(up + "experimental/image/pixel_shuffle/" + marker); // 构建目录向上
        up += "../";
    }
    for (const auto& c : cands) {
        if (FileExists(c + "/gen_data.py")) {
            return c;
        }
    }
    return "./" + dirName; // 兜底：沿用同仓 kernel UT 的 CWD 相对写法
}

struct Shape {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t r;
    int64_t rank;
    int64_t d; // rank=5 时有效
};

int64_t InElems(const Shape& s) { return s.n * s.c * ((s.rank == 5) ? s.d : 1) * s.h * s.w; }

// gen_data.py 的形状字符串：4D=(N,C,H,W)，5D=(N,C,D,H,W)
string ShapeStr(const Shape& s)
{
    ostringstream os;
    os << "(" << s.n << "," << s.c;
    if (s.rank == 5) {
        os << "," << s.d;
    }
    os << "," << s.h << "," << s.w << ")";
    return os.str();
}

void FillTiling(PixelShuffleTilingData* t, const Shape& s, uint32_t tileDataNum)
{
    memset(t, 0, sizeof(*t));
    t->N = s.n;
    t->C_in = s.c;
    t->D = (s.rank == 5) ? s.d : 1;
    t->H = s.h;
    t->W = s.w;
    t->r = s.r;
    t->rank = s.rank;
    t->totalLength = InElems(s);
    t->tileNum = 1;
    t->tileDataNum = tileDataNum;
}

void RunCase(const char* tag, const Shape& s, uint32_t tileDataNum, uint32_t blockNum)
{
    // 1) 生成输入与 golden（cd 进数据目录，文件名与 compare_data.py 的 glob 约定一致）
    ostringstream gen;
    gen << "cd " << kDataDir << " && python3 gen_data.py '" << ShapeStr(s) << "' '" << kDtype << "' " << s.r
        << " > /dev/null 2>&1";
    ASSERT_EQ(system(gen.str().c_str()), 0) << tag << ": gen_data.py 执行失败（检查 python3/numpy 与数据目录）";

    const size_t elems = static_cast<size_t>(InElems(s));
    size_t bytes = elems * sizeof(float);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(PixelShuffleTilingData)));
    ASSERT_TRUE(x != nullptr && y != nullptr && workspace != nullptr && tiling != nullptr);

    ReadFile(kInputBin, bytes, x, bytes);
    memset(y, 0, bytes);
    memset(workspace, 0, 1024);
    FillTiling(reinterpret_cast<PixelShuffleTilingData*>(tiling), s, tileDataNum);

    // 2) ICPU 跑核
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(pixel_shuffle<0>, blockNum, x, y, workspace, tiling);

    // 3) 写回结果，供 compare_data.py 消费（*output*.bin 与 *golden*.bin 配对）
    WriteFile(kOutputBin, y, bytes);

    ostringstream cmp;
    cmp << "cd " << kDataDir << " && python3 compare_data.py '" << kDtype << "' > /dev/null 2>&1";
    EXPECT_EQ(system(cmp.str().c_str()), 0) << tag << ": 输出与 golden 不一致（compare_data.py 报错）";

    // 4) 纯搬运算子再补一次逐字节精确比对（compare_data.py 用 1e-5 容差，这里更严）
    vector<float> want(elems, 0.0F);
    vector<float> got(elems, 0.0F);
    ReadFile(kGoldenBin, bytes, want.data(), bytes);
    memcpy(got.data(), y, bytes);
    size_t bad = elems;
    for (size_t i = 0; i < elems; ++i) {
        if (memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            bad = i;
            break;
        }
    }
    if (bad == elems) {
        SUCCEED();
    } else {
        ADD_FAILURE() << tag << ": 首个不符元素 idx=" << bad << " got=" << got[bad] << " want=" << want[bad];
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

class pixel_shuffle_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "pixel_shuffle_test SetUp" << endl;
        ASSERT_TRUE(FileExists(dataPath + "/gen_data.py")) << "找不到数据目录: " << dataPath;
        const string cmd = "rm -rf ./pixel_shuffle_data && cp -rf " + dataPath + " ./pixel_shuffle_data";
        ASSERT_EQ(system(cmd.c_str()), 0) << "复制数据目录失败: " << dataPath;
        system("chmod -R 755 ./pixel_shuffle_data/");
    }

    static void TearDownTestCase() { cout << "pixel_shuffle_test TearDown" << endl; }

private:
    // 源码侧数据目录（同仓 kernel UT 用 rootPath + "custom/<op>/..." 指向安装目录，
    // 这里改用 __FILE__ 推导，避免依赖运行时 CWD 与安装布局）
    static const std::string dataPath;
};

const std::string pixel_shuffle_test::dataPath = SrcDataDir();

// 4D 基本置换：r=2，(1,4,2,4) 时 blocksPerGroup_ == 1 ⇒ 走「整组连续搬入」路径
TEST_F(pixel_shuffle_test, four_dim_whole_group_contiguous)
{
    RunCase("4D 整组连续", Shape{1, 4, 2, 4, 2, 4, 1}, 1024U, 1U);
}

// 4D 常规：H 被切成多个行块 ⇒ 组内分块 + 对齐步长
TEST_F(pixel_shuffle_test, four_dim_regular_blocking) { RunCase("4D 常规分块", Shape{1, 4, 8, 8, 2, 4, 1}, 1024U, 1U); }

// 4D W 分段：W=2048 超过 fp32 单块列上限（1920）⇒ bPerRow_=2 且末段 wLast_=128
TEST_F(pixel_shuffle_test, four_dim_w_segmented) { RunCase("4D W 分段", Shape{1, 4, 8, 2048, 2, 4, 1}, 8192U, 1U); }

// 4D W 分段 + 非对齐末段（W 不是 wPerBlock_ 的整数倍，且行字节非 32B 对齐）
TEST_F(pixel_shuffle_test, four_dim_w_segmented_unaligned_tail)
{
    RunCase("4D W 分段非对齐", Shape{1, 4, 6, 2000, 2, 4, 1}, 4096U, 1U);
}

// 不同 r：r=3，(1,9,6,6)
TEST_F(pixel_shuffle_test, four_dim_r3) { RunCase("4D r=3", Shape{1, 9, 6, 6, 3, 4, 1}, 1024U, 1U); }

// 质数/不规则形状：H 取约数行块，尾行与尾段都不整齐
TEST_F(pixel_shuffle_test, four_dim_prime_shape) { RunCase("4D 质数形状", Shape{1, 4, 5, 7, 2, 4, 1}, 512U, 1U); }

// 5D 置换：planesPerGroup_ = D*r
TEST_F(pixel_shuffle_test, five_dim_basic) { RunCase("5D 基本", Shape{1, 8, 4, 4, 2, 5, 2}, 1024U, 1U); }

// 5D 且 D 与 r 不整除关系（D*r 平面边界）
TEST_F(pixel_shuffle_test, five_dim_depth_three) { RunCase("5D D=3", Shape{1, 8, 4, 4, 2, 5, 3}, 1024U, 1U); }

// 多 tile / 尾 tile：tileDataNum=1 时 blockPerTile_=1，tileNum_ 很大
TEST_F(pixel_shuffle_test, multi_tile_tail) { RunCase("多 tile 含尾 tile", Shape{2, 4, 8, 8, 2, 4, 1}, 1U, 1U); }

// 单 tile：tileDataNum 取大值，覆盖「无尾 tile」分支
TEST_F(pixel_shuffle_test, single_tile) { RunCase("单 tile", Shape{2, 4, 8, 8, 2, 4, 1}, 1U << 20, 1U); }

// 多核块划分：同一形状分别用 1/4/8 个 block 跑，结果必须完全一致
TEST_F(pixel_shuffle_test, multi_core_split)
{
    RunCase("多核 blockNum=1", Shape{2, 8, 16, 32, 2, 4, 1}, 4096U, 1U);
    RunCase("多核 blockNum=4", Shape{2, 8, 16, 32, 2, 4, 1}, 4096U, 4U);
    RunCase("多核 blockNum=8", Shape{2, 8, 16, 32, 2, 4, 1}, 4096U, 8U);
}

// 多核 + W 分段组合
TEST_F(pixel_shuffle_test, multi_core_with_w_segmented)
{
    RunCase("多核+W分段 b=4", Shape{2, 4, 8, 2048, 2, 4, 1}, 4096U, 4U);
}
