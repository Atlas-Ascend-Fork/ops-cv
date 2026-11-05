/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPS_CV_DEV_TESTS_UT_COMMON_SOC_SPEC_MOCK_H
#define OPS_CV_DEV_TESTS_UT_COMMON_SOC_SPEC_MOCK_H

#include <cstdint>
#include <cstdio>
#include <cstring>

/*
 * Shared mock of the runtime soc spec query used by platform gates: the passes call
 * rtGetSocSpec (libruntime) in production; the UT binary provides its own definition so
 * platform simulation stays possible without a device. All op_graph suites link into a
 * single executable in the standard UT flow, so the mock state is an inline variable
 * (one ODR-merged instance across translation units) and the query a weak definition
 * emitted by every suite: the linker picks one copy reading the shared state, no
 * multiple-definition error. Suites run serially inside one process, so the shared
 * state needs no locking.
 */
inline uint32_t g_mockNpuArch = 2201U;

extern "C" int rtGetSocSpec(const char* label, const char* key, char* val, const unsigned int maxLen)
    __attribute__((weak));

extern "C" int rtGetSocSpec(const char* label, const char* key, char* val, const unsigned int maxLen)
{
    if (std::strcmp(key, "NpuArch") == 0) {
        (void)std::snprintf(val, maxLen, "%u", g_mockNpuArch);
        return 0;
    }
    return -1;
}

#endif // OPS_CV_DEV_TESTS_UT_COMMON_SOC_SPEC_MOCK_H
