/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/resize_upsample_trilinear_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ResizeUpsampleTrilinearTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeUpsampleTrilinearTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeUpsampleTrilinearTiling TearDown" << std::endl; }
};

struct ResizeUpsampleTrilinearCompileInfo {
    uint16_t coreNum = 0;
    uint16_t socVersionType = 0;
};

TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_001)
{
    gert::StorageShape input_shape = {{1, 1, 4, 128, 128}, {1, 1, 4, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 12, 256, 256}, {1, 1, 12, 256, 256}};
    ResizeUpsampleTrilinearCompileInfo compileInfo = {48, 220};
    std::vector<int64_t> output_size = {12, 256, 256};
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 3000;

    string expectTilingData =
        "4539628425446424576 207209802411 0 160 256 256 12 128 128 4 1 0 0 0 0 0 0 0 734439407616 342 1468878815403 "
        "734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 "
        "1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 "
        "1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 "
        "2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 "
        "730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 "
        "0 68719476736 68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 "
        "412316860512 481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 "
        "755914244272 824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 "
        "133143986191 133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 "
        "476741369967 545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 "
        "820338753727 889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 0 68719476736 "
        "68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 412316860512 "
        "481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 755914244272 "
        "824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 133143986191 "
        "133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 476741369967 "
        "545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 820338753727 "
        "889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2199023255553 549755814144 2199023255680 42949672976 68719477248 4294967312 "
        "4294967297 1 0 145135534866432 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "1099511627777 2199023255808 1099511628288 42949673216 1099511627904 8589934608 8589934593 1 0 140737488355328 "
        "131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 51539607553 17179934720 51539607556 "
        "34359803904 1099511627792 4294967304 4294967312 8 4294967296 565148976676864 16384 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {35158016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_002)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 0, 0, 0}, {1, 1, 0, 0, 0}};

    // compile info
    ResizeUpsampleTrilinearCompileInfo compileInfo = {20, 200};
    std::vector<int64_t> output_size = {0, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 3000;

    string expectTilingData =
        "4539628425446424576 207209802411 0 160 256 256 12 128 128 4 1 0 0 0 0 0 0 0 734439407616 342 1468878815403 "
        "734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 "
        "1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 "
        "1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 "
        "2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 "
        "730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 "
        "0 68719476736 68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 "
        "412316860512 481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 "
        "755914244272 824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 "
        "133143986191 133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 "
        "476741369967 545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 "
        "820338753727 889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 0 68719476736 "
        "68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 412316860512 "
        "481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 755914244272 "
        "824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 133143986191 "
        "133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 476741369967 "
        "545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 820338753727 "
        "889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2199023255553 549755814144 2199023255680 42949672976 68719477248 4294967312 "
        "4294967297 1 0 145135534866432 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "1099511627777 2199023255808 1099511628288 42949673216 1099511627904 8589934608 8589934593 1 0 140737488355328 "
        "131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 51539607553 17179934720 51539607556 "
        "34359803904 1099511627792 4294967304 4294967312 8 4294967296 565148976676864 16384 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {35158016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_003)
{
    gert::StorageShape input_shape = {{1, 2, 60, 256, 512}, {1, 2, 60, 256, 512}};
    gert::StorageShape out_shape = {{1, 2, 4, 1, 2}, {1, 2, 4, 1, 2}};

    // compile info
    ResizeUpsampleTrilinearCompileInfo compileInfo = {20, 220};
    std::vector<int64_t> output_size = {4, 1, 2};

    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "4539628425446424576 207209802411 0 160 256 256 12 128 128 4 1 0 0 0 0 0 0 0 734439407616 342 1468878815403 "
        "734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 "
        "1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 734439407616 342 1468878815403 "
        "1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 "
        "2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 "
        "730144440831 2194728288597 1464583848106 730144440831 2194728288597 1464583848106 730144440831 2194728288597 "
        "0 68719476736 68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 "
        "412316860512 481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 "
        "755914244272 824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 "
        "133143986191 133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 "
        "476741369967 545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 "
        "820338753727 889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 0 68719476736 "
        "68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 412316860512 "
        "481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 755914244272 "
        "824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 64424509455 133143986191 "
        "133143986207 201863462959 270582939695 270582939711 339302416463 408021893199 408021893215 476741369967 "
        "545460846703 545460846719 614180323471 682899800207 682899800223 751619276975 820338753711 820338753727 "
        "889058230479 957777707215 957777707231 1026497183983 1095216660719 1095216660735 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 "
        "8589934592 3 12884901890 8589934592 3 12884901890 8589934592 3 12884901890 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 12884901890 8589934596 17179869187 "
        "12884901890 8589934596 17179869187 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 47244640267 "
        "47244640267 47244640267 47244640267 47244640267 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2199023255553 549755814144 2199023255680 42949672976 68719477248 4294967312 "
        "4294967297 1 0 145135534866432 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "1099511627777 2199023255808 1099511628288 42949673216 1099511627904 8589934608 8589934593 1 0 140737488355328 "
        "131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 51539607553 17179934720 51539607556 "
        "34359803904 1099511627792 4294967304 4294967312 8 4294967296 565148976676864 16384 4294967297 4294967297 "
        "4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {35158016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_004)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 2, 60, 256, 512}, {1, 2, 60, 256, 512}};
//     gert::StorageShape out_shape = {{1, 2, 4, 1, 2}, {1, 2, 4, 1, 2}};

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     struct UpsampleTrilinearCompileInfo {
//         uint16_t totalCoreNum = 20;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("ResizeUpsampleTrilinear");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {4, 1, 2};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("ResizeUpsampleTrilinear")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeAttrs(
//                           {{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)},
//                            {"scales_d", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_005)
{
    gert::StorageShape input_shape = {{1, 2, 60, 128, 128}, {1, 2, 60, 128, 128}};
    gert::StorageShape out_shape = {{1, 2, 90, 192, 192}, {1, 2, 90, 192, 192}};

    // compile info
    ResizeUpsampleTrilinearCompileInfo compileInfo = {20, 200};
    std::vector<int64_t> output_size = {90, 192, 192};

    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 3000;

    string expectTilingData =
        "4575657222468184747 86964699136 20 288 90 2 1 60 2 1 16384 0 9 0 180 1 1024 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {45373952};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_006)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 4, 128, 130}, {1, 1, 4, 128, 130}};
//     gert::StorageShape out_shape = {{1, 1, 12, 256, 260}, {1, 1, 12, 256, 260}};

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     struct UpsampleTrilinearCompileInfo {
//         uint16_t totalCoreNum = 8;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("ResizeUpsampleTrilinear");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {12, 256, 260};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("ResizeUpsampleTrilinear")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeAttrs(
//                           {{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                            {"scales_d", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 3000);
// }

// TEST_F(ResizeUpsampleTrilinearTiling, upsample_trilinear3d_tiling_007)
// {
//     dlog_setlevel(0, 0, 0);
//     gert::StorageShape input_shape = {{1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}};
//     gert::StorageShape out_shape = {{1, 1, 1, 2, 4}, {1, 1, 1, 2, 4}};

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}};
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     struct UpsampleTrilinearCompileInfo {
//         uint16_t totalCoreNum = 8;
//         uint16_t socVersionType = 220;
//     } compile_info;

//     std::string op_type("ResizeUpsampleTrilinear");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(8192);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     ASSERT_NE(param, nullptr);
//     std::vector<int64_t> output_size = {1, 2, 4};
//     auto holder = gert::TilingContextFaker()
//                       .SetOpType("ResizeUpsampleTrilinear")
//                       .NodeIoNum(1, 1)
//                       .IrInstanceNum({1})
//                       .InputShapes({&input_shape})
//                       .OutputShapes({&out_shape})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeAttrs(
//                           {{"output_size", ge::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(false)},
//                            {"scales_d", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_h", ge::AnyValue::CreateFrom<float>(0.0)},
//                            {"scales_w", ge::AnyValue::CreateFrom<float>(0.0)}})
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();

//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
// }
