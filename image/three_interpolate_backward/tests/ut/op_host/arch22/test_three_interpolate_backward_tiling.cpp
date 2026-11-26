/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/three_interpolate_backward_tiling.h"

namespace {
constexpr int64_t kC0 = 16;

optiling::ThreeInterpolateBackwardCompileInfo g_compileInfo{48, 262144};

// 局部复现辅助的三态结果：区分“未调用到 tiling 函数”、“tiling 被调用
// 且被拒绝”、“tiling 被调用且成功”。负向用例断言 kRejected 以证明 tiling 确实被
// 调用并因判空返回 GRAPH_FAILED（排除 helper 在更早路径提前返回导致的空转通过）。
enum class TilingNoOutputsResult {
    kNotInvoked, // helper 未调用到 tiling 函数（上下文构造/注册表查找失败，用例应视为失败）
    kRejected,   // tiling 函数被调用且返回 GRAPH_FAILED
    kSuccess     // tiling 函数被调用且返回 GRAPH_SUCCESS
};

// 局部复现辅助：按共享 executor 的方式构造 TilingContext 并调用注册的
// tiling 函数，但允许省略 TilingData/Workspace（构造 GetRawTilingData()/
// GetWorkspaceSizes() 返回空指针的异常上下文）。仅本文件内使用，不改动共享测试组件；
// 属性仅支持 VT_INT（本算子 tiling 只读取 int64 属性 m）。
TilingNoOutputsResult ExecuteTilingNoOutputs(const gert::TilingContextPara& para, bool withTilingData,
                                             bool withWorkspace)
{
    auto contextFaker = gert::TilingContextFaker();
    contextFaker.NodeIoNum(para.inputTensorDesc_.size(), para.outputTensorDesc_.size());
    std::vector<gert::Tensor*> inputTensors;
    std::vector<gert::Tensor*> outputTensors;
    std::vector<std::unique_ptr<gert::Tensor>> tensorsKeepAlive;
    for (const auto& desc : para.inputTensorDesc_) {
        tensorsKeepAlive.push_back(std::make_unique<gert::Tensor>(
            desc.shape_, gert::StorageFormat(desc.format_, desc.storageFormat_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, desc.dtype_, nullptr));
        inputTensors.push_back(tensorsKeepAlive.back().get());
    }
    for (const auto& desc : para.outputTensorDesc_) {
        tensorsKeepAlive.push_back(std::make_unique<gert::Tensor>(
            desc.shape_, gert::StorageFormat(desc.format_, desc.storageFormat_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, desc.dtype_, nullptr));
        outputTensors.push_back(tensorsKeepAlive.back().get());
    }
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);
    for (const auto& attrInfo : para.attrs_) {
        if (attrInfo.attr_.type_ != Ops::Cv::AnyValue::ValueType::VT_INT) {
            return TilingNoOutputsResult::kNotInvoked;
        }
        contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<const int64_t*>(attrInfo.attr_.valuePtr_.get()));
    }
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    contextFaker.SetOpType(para.opName_);
    contextFaker.CompileInfo(para.compileInfo_);
    contextFaker.PlatformInfo(reinterpret_cast<char*>(&platformInfo));
    auto tilingData = gert::TilingData::CreateCap(para.tilingDataSize_);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    if (withTilingData) {
        contextFaker.TilingData(tilingData.get());
    }
    if (withWorkspace) {
        contextFaker.Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    }
    auto contextHolder = contextFaker.Build();
    auto tilingContext = contextHolder.GetContext();
    // 平台 SoC 信息与共享 executor 的设置一致，保证 IsRegbaseSocVersion 正确分流到 arch22 路径
    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", std::to_string(para.coreNum_)}, {"l2_size", "33554432"}, {"core_type_list", "AICore"}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    std::map<std::string, std::string> aicoreSpec = {
        {"ub_size", std::to_string(para.ubSize_)}, {"vec_calc_size", "128"}, {"vector_reg_width", "256"}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    std::map<std::string, std::string> vectorCoreSpec = {
        {"ubblock_size", "32"}, {"vec_calc_size", "128"}, {"vector_reg_width", "256"}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("VectorCoreSpec", vectorCoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    std::map<std::string, std::string> socversions = {{"NpuArch", "2201"}, {"Short_SoC_version", para.socVersion_}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl(para.opName_.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return TilingNoOutputsResult::kNotInvoked;
    }
    return (opImpl->tiling(tilingContext) == ge::GRAPH_SUCCESS) ? TilingNoOutputsResult::kSuccess :
                                                                  TilingNoOutputsResult::kRejected;
}

gert::TilingContextPara MakeContext(int64_t bs, int64_t c1, int64_t ms, int64_t ns)
{
    gert::StorageShape gradX({bs, c1, ns, 1, kC0}, {bs, c1, ns, 1, kC0});
    gert::StorageShape idx({bs, ns, 3}, {bs, ns, 3});
    gert::StorageShape weight({bs, ns, 3}, {bs, ns, 3});
    gert::StorageShape gradY({bs, c1, ms, 1, kC0}, {bs, c1, ms, 1, kC0});
    return gert::TilingContextPara("ThreeInterpolateBackward",
                                   {
                                       {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
                                       {idx, ge::DT_INT32, ge::FORMAT_ND},
                                       {weight, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
                                   },
                                   {
                                       gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(ms)),
                                   },
                                   &g_compileInfo, "Ascend910B", 48, 262144);
}
} // namespace

TEST(ThreeInterpolateBackwardTiling910B, accepts_positive_m)
{
    auto context = MakeContext(2, 1, 6, 4);
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_zero_m)
{
    auto context = MakeContext(1, 1, 0, 1);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_negative_m)
{
    auto context = MakeContext(1, 1, -1, 1);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_unsupported_idx_dtype)
{
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 4, 3}, {2, 4, 3});
    gert::StorageShape weight({2, 4, 3}, {2, 4, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_FLOAT, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &g_compileInfo, "Ascend910B", 48, 262144);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_ub_insufficient)
{
    optiling::ThreeInterpolateBackwardCompileInfo tinyUb{48, 64};
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 4, 3}, {2, 4, 3});
    gert::StorageShape weight({2, 4, 3}, {2, 4, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_INT32, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &tinyUb, "Ascend910B", 48, 64);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_idx_n_mismatch)
{
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 5, 3}, {2, 5, 3});
    gert::StorageShape weight({2, 5, 3}, {2, 5, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_INT32, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &g_compileInfo, "Ascend910B", 48, 262144);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

// 正向对照（评审要求）：走 ExecuteTilingNoOutputs 的完整路径（TilingData/Workspace
// 均设置），期望 tiling 被调用且成功。该用例证明 helper 手工构造的上下文可成功进入
// arch22 tiling 主流程；若失败，说明 helper 的上下文构造与共享 executor 存在偏差，
// 下方两个负向用例的 kRejected 结论不可信（存在更早校验点提前拒绝的空转风险）。
TEST(ThreeInterpolateBackwardTiling910B, no_outputs_helper_positive_control)
{
    auto context = MakeContext(2, 1, 6, 4);
    EXPECT_EQ(ExecuteTilingNoOutputs(context, true, true), TilingNoOutputsResult::kSuccess);
}

// GetWorkspaceSizes(1) 返回空指针（TilingContext 未设置 Workspace）时，
// tiling 须以 OP_CHECK_NULL_WITH_CONTEXT 判空并返回 GRAPH_FAILED，而非直接写
// currentWorkspace[0] 触发空指针崩溃。断言 kRejected 以证明 tiling 确实被调用并
// 因判空被拒绝（排除 helper 未调用到 tiling 的空转通过）。修复前该用例触发 SIGSEGV。
TEST(ThreeInterpolateBackwardTiling910B, null_workspace_rejected_not_crash)
{
    auto context = MakeContext(2, 1, 6, 4);
    EXPECT_EQ(ExecuteTilingNoOutputs(context, true, false), TilingNoOutputsResult::kRejected);
}

// GetRawTilingData() 返回空指针（TilingContext 未设置 TilingData）时，
// SaveTilingDataToBuffer 中的链式 GetRawTilingData()->GetData()/GetCapacity()/SetDataSize
// 须先判空并以 GRAPH_FAILED 返回，而非链式解引用崩溃。断言 kRejected 以证明 tiling
// 确实被调用并因判空被拒绝。修复前该用例触发 SIGSEGV。
TEST(ThreeInterpolateBackwardTiling910B, null_raw_tiling_data_rejected_not_crash)
{
    auto context = MakeContext(2, 1, 6, 4);
    EXPECT_EQ(ExecuteTilingNoOutputs(context, false, true), TilingNoOutputsResult::kRejected);
}
