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
#include "../../../op_host/upsample_bilinear2d_grad_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class UpsampleBilinear2dGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpsampleBilinear2dGrad Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpsampleBilinear2dGrad Tiling TearDown" << std::endl; }
};

TEST_F(UpsampleBilinear2dGradTiling, rejects_short_output_size)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {6};
    std::vector<int64_t> input_size = {2, 3, 2, 3};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{2, 3, 6, 7}, {2, 3, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 2, 3}, {2, 3, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(UpsampleBilinear2dGradTiling, rejects_short_input_size)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {6, 7};
    std::vector<int64_t> input_size = {1, 1, 2};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 6, 7}, {1, 1, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 2, 3}, {1, 1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(UpsampleBilinear2dGradTiling, rejects_output_shape_not_match_input_size)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {1, 13};
    std::vector<int64_t> input_size = {161, 161, 1, 161};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{161, 161, 1, 13}, {161, 161, 1, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{161, 161, 1, 13}, {161, 161, 1, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_001)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {256, 256};
    std::vector<int64_t> input_size = {1, 1, 128, 128};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 2 4539628425446424576 32768 3298534884096 1 1 256 256 1 1 128 128 34359738416 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 16 16 16 32 32 32 32 32 "
        "32 48 48 48 48 48 48 64 64 64 64 64 64 80 80 80 80 80 80 96 96 96 96 96 96 112 112 112 112 112 112 0 0 16 16 "
        "16 16 16 16 32 32 32 32 32 32 48 48 48 48 48 48 64 64 64 64 64 64 80 80 80 80 80 80 96 96 96 96 96 96 112 112 "
        "112 112 112 112 128 128 128 128 128 128 0 0 0 43 86 129 172 215 0 43 86 129 172 215 0 43 86 129 172 215 0 43 "
        "86 129 172 215 0 43 86 129 172 215 0 43 86 129 172 215 0 43 86 129 172 215 0 43 86 129 172 215 0 0 43 86 129 "
        "172 215 256 43 86 129 172 215 256 43 86 129 172 215 256 43 86 129 172 215 256 43 86 129 172 215 256 43 86 129 "
        "172 215 256 43 86 129 172 215 256 43 86 129 172 215 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 32 48 64 80 96 112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 32 48 64 80 96 112 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 22 22 22 22 22 22 22 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 4294967297 "
        "4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1099511627777 1099511627904 1099511628032 206158430224 "
        "68719476992 8589934624 4294967298 1 0 299067162755072 16384 4294967297 4294967297 8589934594 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 549755813889 1099511627904 68719476992 206158430336 549755813904 4294967344 4294967297 1 0 "
        "118747255799808 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {34111488};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_002)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {128, 128};
    std::vector<int64_t> input_size = {1, 1, 128, 128};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 2 4575657222473777152 16384 2199023256064 1 1 128 128 1 1 128 128 34359738368 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 16 32 48 64 80 96 112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 16 32 48 64 80 96 112 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 22 22 "
        "22 22 22 22 22 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814016 68719476864 "
        "137438953600 549755813904 4294967328 4294967297 1 0 79164837199872 8192 4294967297 4294967297 4294967297 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33718272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_003)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {128, 128};
    std::vector<int64_t> input_size = {1, 1, 256, 256};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(2.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(2.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 2 4611686019501129728 32768 1649267442048 1 1 128 128 1 1 256 256 68719476784 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 "
        "80 80 96 96 96 112 112 112 128 128 128 144 144 144 160 160 160 176 176 176 192 192 192 208 208 208 224 224 "
        "224 240 240 240 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 144 144 144 "
        "160 160 160 176 176 176 192 192 192 208 208 208 224 224 224 240 240 240 256 256 256 0 0 0 43 86 0 43 86 0 43 "
        "86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 "
        "0 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 "
        "128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86 86 86 86 86 86 86 86 86 86 86 86 86 86 86 86 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 103079215120 68719476864 4294967320 4294967297 1 0 "
        "59373627899904 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 "
        "549755814144 68719476864 103079215360 1099511627792 4294967320 4294967297 1 0 112150186033152 16384 "
        "4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33964032};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_004)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {12, 12};
    std::vector<int64_t> input_size = {1, 1, 1, 1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 2 4443551634805140139 12 17042430234496 1 1 12 12 1 1 1 1 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 51539607553 51539607553 51539607564 1065151889424 "
        "68719476752 4294967552 4294967297 1 0 140737488355328 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 4294967297 51539607553 68719476748 1065151889409 68719476752 4294967552 4294967297 1 0 "
        "140737488355328 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33586272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_005)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {1, 1};
    std::vector<int64_t> input_size = {1, 1, 20, 20};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 20, 20}, {1, 1, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 2 4728779609840025600 20 1099511628032 1 1 1 1 1 1 20 20 8589934594 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 20 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 20 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967316 4294967297 "
        "68719476752 68719476752 4294967312 4294967297 1 0 8796093022208 1024 4294967297 4294967297 4294967297 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 85899345921 4294967316 68719476737 68719476756 137438953488 4294967312 "
        "4294967297 1 0 13194139533312 2048 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33558688};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dGradTiling, upsample_bilinear2d_grad_tiling_006)
{
    UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {1, 1};
    std::vector<int64_t> input_size = {1, 1, 1, 1};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2dGrad",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
         {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
         {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)},
         {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
         {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo, socVersion, 48, 192 * 1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "16 2 1 0 1 1099511628032 1 1 1 1 1 1 1 1 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 4294967297 68719476752 68719476752 4294967312 "
        "4294967297 1 0 8796093022208 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 4294967297 "
        "4294967297 68719476737 68719476737 68719476752 4294967312 4294967297 1 0 8796093022208 1024 4294967297 "
        "4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33556488};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
