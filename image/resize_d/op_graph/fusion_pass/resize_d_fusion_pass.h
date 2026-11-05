/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_d_fusion_pass.h
 * \brief resize_d fusion pass (ResizeD --> ResizeLinear / ResizeBicubicV2)
 */

#ifndef OPS_CV_RESIZE_D_FUSION_PASS_H_
#define OPS_CV_RESIZE_D_FUSION_PASS_H_

#include <memory>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) ResizeDFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& matchResult) override;

    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& matchResult) override;
};
} // namespace ops

#endif // OPS_CV_RESIZE_D_FUSION_PASS_H_
