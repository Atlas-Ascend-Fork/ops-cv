# Operator Cross-Platform Migration Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:07:38.812Z pushedAt=2026-11-18T09:59:29.666Z nl=2 -->

This guide describes the adaptation points and solutions for migrating operators between platforms. It uses the migration of operators from Atlas A2 to Ascend 950 as an example, compares the hardware architecture differences and the adaptation points involved, and provides related operator adaptation examples.

## 1. Hardware Architecture and Specification Comparison

### Atlas A2 Hardware Architecture

<div align="center">
  <img src="../figures/AtlasA2_hardware_architecture.png" width="900" alt="Atlas A2 hardware architecture" />
</div>

### Ascend 950 Hardware Architecture

<div align="center">
  <img src="../figures/Ascend950_hardware_architecture.png" width="900" alt="Ascend 950 hardware architecture" />
</div>

### Inter-Generation Specification Comparison

Products are usually divided into multiple models based on different use cases, processes, or hardware configurations. Each model may differ in performance, resource configuration, and other aspects. For ease of description and direct comparison, this section selects a representative configuration for parameter display and difference analysis. For other related adjustments, see the actual manual or official release.

<table>
  <tr>
    <th colspan="2" style="width: 25%;">Specification Item</th>
    <th style="width:37.5%;">Atlas A2 (Ascend910B2 as an example)</th>
    <th style="width:37.5%;">Ascend 950 (Ascend950PR Server tier as an example)</th>
  </tr>
  <tr>
    <td rowspan="4">AI Core</td>
    <td>Cube core count</td>
    <td>24</td>
    <td>32</td>
  </tr>
  <tr>
    <td>Vector core count</td>
    <td>48</td>
    <td>64</td>
  </tr>
  <tr>
    <td>Frequency (GHz)</td>
    <td>1.8</td>
    <td>1.65</td>
  </tr>
  <tr>
    <td>Cube compute capability specification</td>
    <td>Approximately 353.9 TFLOPS @BF16, FP16</td>
    <td>432T @BF16, FP16; 865T @FP8, HiF8, MXFP8; 1730T @MXFP4</td>
  </tr>
  <tr>
    <td>Vector</td>
    <td>Vector compute capability specification (FP16)</td>
    <td>Approximately 22.1 TFLOPS</td>
    <td>54T</td>
  </tr>
  <tr>
    <td rowspan="2">Memory</td>
    <td>Memory capacity (GB)</td>
    <td>64</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Memory bandwidth</td>
    <td>1.6TB/s</td>
    <td>1.6TB/s</td>
  </tr>
</table>

> Note:
>
> - Total FP16 compute capability = Cube + Vector: Ascend910B2 is approximately 376.0 TFLOPS (Cube: about 353.9 TFLOPS, Vector: about 22.1 TFLOPS, total: the sum of unrounded values); the Ascend950PR Server tier is 486 TFLOPS (432+54). For 950 specifications, see Table 3-1 in the *Ascend 950 NPU Architecture White Paper*. The Ascend910B2 theoretical peak is calculated at 1.8 GHz: Cube is `16×16×16×24×1.8×2/1000`, Vector FP16 is `128×48×1.8×2/1000`, both in TFLOPS, with each multiply-accumulate counted as 2 operations.
> - Ascend950PR PCIE tier: Cube 28 cores/Vector 56 cores, Memory 112GB @ 1.4TB/s, Cube BF16 378T, FP8 family 756T, MXFP4 1513T, Vector 47T; Ascend950DT: Cube 36/32/28 cores in three tiers, Memory 144/96GB @ 4TB/s. See Table 3-1 in the white paper for details.
> - INT8/TF32 compute capability is listed separately in Table 3-1 of the white paper and is not expanded here; Cube and Vector compute capability are listed separately.
> - The table above shows typical SKU examples. In operator code, the core count and buffer capacity should be obtained at runtime through the `PlatformAscendC` API (`GetCoreNumAic`/`GetCoreNumAiv`/`GetCoreMemSize`) to avoid hardcoding.

## II. Adaptation Points Introduced by Hardware Capability Changes

<table>
  <tr>
    <th style="width: 25%;">Hardware Unit</th>
    <th style="width:35%;">Hardware Capability Change</th>
    <th style="width:40%;">Typical Impact Scope</th>
  </tr>
  <tr>
    <td rowspan="5">Transfer unit</td>
    <td>Removed the L1-to-GM data path</td>
    <td>Kernels that rely on L1 writing directly back to GM must change to the L1→UB→GM or L0C/FIXPIPE→GM path; related DataCopy links, event synchronization, and buffer planning need adjustment</td>
  </tr>
  <tr>
    <td>Removed the GM-to-L0A and GM-to-L0B data paths</td>
    <td>The GM→L0A/L0B direct connection is no longer available and must be completed through GM→L1→L0A/L0B; the L1 tiling strategy and MTE1/2 pipeline need to be restructured</td>
  </tr>
  <tr>
    <td>ND DMA flexible data movement, supporting in-transit ND-&gt;NZ conversion</td>
    <td>ND2NZ/DN2NZ can be used to complete format conversion at the MTE2 stage, reducing intermediate buffers and format conversion overhead; focus on stride, alignment, and NZ shape mapping</td>
  </tr>
  <tr>
    <td>Supports efficient Cube-&gt;Vector internal data paths: L1&lt;-&gt;UB, L0C-&gt;UB, FIXP-&gt;UB</td>
    <td>Intermediate accumulation/activation/fusion (such as K-split accumulation and post-processing) can be performed on the UB side, reducing GM round trips; corresponding synchronization and pipeline partitioning need adjustment</td>
  </tr>
  <tr>
    <td>Introduced the collective communication accelerator CCU1.0</td>
    <td>For communication-computation fused operators, adjust HcclServerType in Eager mode; switch to the CCU series GE APIs in Graph mode</td>
  </tr>
  <tr>
    <td rowspan="3">Compute unit</td>
    <td>Vector adds the Regbase paradigm</td>
    <td>Memory access patterns, alignment methods, and register count assumptions that originally relied on Membase need to be re-reviewed; templates/tiling may need to be updated to the Regbase version</td>
  </tr>
  <tr>
    <td>Cube MMAD no longer supports INT4 matrix computation (int4b_t and the packed type int4x2_t remain valid built-in types and can be used for Vector computation and weight storage)</td>
    <td>Operators that rely on Cube MMAD for INT4 computation need to switch to a supported data type (such as int8) and update the quantization computation logic</td>
  </tr>
  <tr>
    <td>4:2 sparse matrix computation is not supported</td>
    <td>Kernels that originally relied on the 4:2 sparse feature for acceleration need to change to dense or other supported sparse strategies and update the performance expectation description</td>
  </tr>
  <tr>
    <td rowspan="1">Storage unit</td>
    <td>Local Buffer memory improvement: Cube L0C expanded to 256KB (128KB on A2); UB is 256KB physical per AIV, 248KB user-available (192KB on A2)</td>
    <td>Larger L0C/UB allows increasing the basic block and double buffering capacity, reducing K-split/tiling rounds; the L1/L0/UB ratio and tile size need to be re-evaluated, and all UB and other buffer capacity tiling must follow the return value of <code>GetCoreMemSize</code></td>
  </tr>
  <tr>
    <td rowspan="2">Other</td>
    <td>Performance optimization for multiple cores accessing the same Global Memory address simultaneously</td>
    <td>Templates of matrix multiplication related operators can be optimized</td>
  </tr>
  <tr>
    <td>SIMT</td>
    <td>After SIMT is introduced, thread-level parallelism can be used to process branches/irregular computation, but thread partitioning, shared memory, and synchronization semantics need adaptation; some Vector implementations can be migrated to SIMT versions</td>
  </tr>
</table>

## III. Recommended Migration Steps

1. Confirm whether the compute units (Cube/Vector) involved in the operator and the data types supported by the corresponding units differ between platforms.
2. Confirm whether the data movement units involved (ND->NZ, GM&lt;-&gt;Lx, collective communication, etc.) differ between platforms.
3. Modify item by item according to the hardware capability change points (Vector architecture, Cube supported data types, L1/L0/UB sizes, CCU communication, etc.).
4. Refer to the operator migration examples to adjust/supplement the Atlas A2/Ascend 950 branch logic.

## IV. Operator Migration Examples

### Cube Matrix Computation Operators

#### Global Memory Same-Address Access Conflict Optimization

Ascend 950 hardware adds the same-address request parallel processing feature, eliminating the need to avoid same-address access conflicts in various core partitioning scenarios. During migration, the core partitioning strategy designed on Atlas A2 for "misalignment conflict avoidance" can be simplified to a more regular sliding window template (such as row-group windows plus column-wise round-trip scanning), reducing invalid offsets and redundant address transformations. In practice, first retain the original tile size with functional equivalence as the goal, then gradually relax the core partitioning constraints, and use profiling data to observe key indicators such as MAC utilization, MTE2 utilization, and L2 hit rate to confirm whether the template adjustment brings stable benefits.

<div align="center">
  <img src="../figures/SWAT_sliding_window_template.png" width="900" alt="SWAT sliding window template" />
</div>

#### Tile Size Adjustment

The L0C size on Atlas A2 is 128KB, and Ascend 950 increases it to 256KB, meaning a larger accumulation result block can be held in a single pass. During migration, prioritize increasing the Tile block partitioning granularity or the single-round processing depth in the K direction to reduce tiling and K-split rounds, lowering loop control and movement overhead. Simultaneously rebalance the L1/L0/UB capacity budget to avoid the enlarged L0C squeezing the A/B/scale buffers and causing pipeline stalls.

### Vector Computation Operators

#### SIMT

The Ascend 950 series adds a SIMT unit. Compared to SIMD, SIMT has a major advantage in processing irregular discrete access. It is suitable for use cases with non-contiguous addresses, large memory access stride variation, and inconsistent branch paths, such as scatter/gather, index reordering, and sparse update.

During migration, prioritize identifying operator sub-processes that are "memory-access-dominated" and have "low vectorization efficiency". If the original SIMD implementation has many mask branches, a high proportion of invalid lanes, or requires complex address assembly, rewrite that part as a SIMT path. This usually reduces control overhead and improves effective memory access throughput.

In practice, focus on the following: first, thread task partitioning must match data sparsity to avoid extremely unbalanced thread loads; second, reduce pipeline idle caused by high-frequency random memory access, and complete index regularization and bucketing upstream as much as possible; third, decouple boundary processing from the main path to avoid introducing too many branches in hot loops. After migration, compare the "pure SIMD implementation" and the "SIMD+SIMT hybrid implementation", and select the optimal strategy based on data distribution rather than fixing a single path.

**Example: gather_v2 operator, SIMD vs. SIMT implementation comparison**

The gather_v2 operator performs gather based on the tail axis after axis merging. Therefore, the template selection is based on: when the tail axis is ≤ 2048, use the SIMT template; when the tail axis is > 2048, use the SIMD template. When the tail axis is small, multiple non-contiguous small block addresses need to be accessed discretely, so SIMT is more efficient. The following compares the core differences between the two implementations:

**1. Programming Model Differences**

The SIMD implementation uses the traditional vectorized programming model and requires explicit management of the UB buffer and pipeline queue:

```cpp
// SIMD: Use the queue mechanism to manage data buffers.
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
TBuf<QuePosition::VECCALC> indexBuf_;

// SIMD: Process row by row, with explicit transfer and synchronization.
for (int64_t j = 0; j < rows; j++) {
    INDICES_T index = GetIndex(yIdx, indiceEndIdx);  // Scalar read index.
    int64_t xIndex = index * tilingData_->innerSize;
    DataCopyPad(xLocal[j * colsAlign], xGm[offset], dataCopyExtParams, dataCopyPadExtParams); // Move in a batch of consecutive numbers.
}
inQueue_.EnQue<int8_t>(xLocal);  // Enqueue and wait for output.
```

SIMT adopts a thread-level parallel model, where each thread independently processes elements:

```cpp
// SIMT: Use thread-level parallelism without explicit buffer management.
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
    for (INDEX_SIZE_T index = threadIdx.x;
         index < currentCoreElements;
         index += blockDim.x) {  // Thread-strided parallelism.
        // Each thread independently computes a single-point index and accesses it.
        INDEX_SIZE_T gatherI = yIndex / innerSize;
        INDICES_T indicesValue = indices[gatherI];  // Directly access GM based on the single-point index gatherI.
        y[yIndex] = idxOutOfBound ? 0 : x[xIndex];  // Write back to GM directly.
    }
}
```

**2. Memory Access Pattern Differences**

| Feature | SIMD Implementation | SIMT Implementation |
|------|----------|----------|
| Data access | Explicitly move to UB through DataCopyPad. | Threads directly access GM through the `__gm__` pointer. |
| Buffer management | Requires AllocTensor/EnQue/DeQue/FreeTensor. | No explicit buffer; hardware manages it automatically. |
| Synchronization mechanism | Explicit event synchronization (HardEvent::MTE2_V, etc.). | Implicit synchronization between threads. |

**3. Applicable Use Case Differences**

SIMD is suitable for use cases that access large contiguous addresses, efficiently processing contiguous data through vectorized instructions.

SIMT is suitable for discrete access, with threads processing in parallel.

#### Regbase

The Ascend 950 series introduces the Regbase programming paradigm. Compared to the traditional Membase (Vector API) programming, Regbase is closer to the underlying hardware register operations and provides finer-grained vectorization control capability.

**Features**

- Use the low-level APIs under the `AscendC::Reg` namespace
- Directly operates on the register `RegTensor<T>` rather than explicitly managing the UB buffer queue.
- Implements flexible element-level mask control through `MaskReg`.

**Comparison with the Membase programming model**

| Feature | Membase (Traditional Vector API) | Regbase (Reg) |
|------|---------------------------|---------------------|
| Data carrier | `LocalTensor<T>` + Queue mechanism | `RegTensor<T>` register |
| Memory management | Explicit Alloc/EnQue/DeQue/Free | Automatic register allocation |
| Mask control | Function parameter control | `MaskReg` register control |
| Data movement | `DataCopy`/`DataCopyPad` | `Reg::DataCopy` + distribution mode |

**Code example**

```cpp
__simd_vf__ __aicore__ void GenIndexBuf(ubuf int32_t* helpAddr, int32_t colFactor)
{
    // Declare a register tensor.
    AscendC::Reg::RegTensor<int32_t> v0;
    AscendC::Reg::RegTensor<int32_t> v1;
    AscendC::Reg::RegTensor<int32_t> vd1;

    // Create a full mask.
    AscendC::Reg::MaskReg preg =
        AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    // Copy the scalar to the register.
    AscendC::Reg::Duplicate(v1, colFactor, preg);
    // Generate the sequence [0, 1, 2, ...].
    AscendC::Reg::Arange(v0, 0);
    // Vector operation.
    AscendC::Reg::Div(vd1, v0, v1, preg);
    AscendC::Reg::Mul(vd2, vd1, v1, preg);
    AscendC::Reg::Sub(vd3, v0, vd2, preg);
    // Write register data back to UB.
    AscendC::Reg::DataCopy(helpAddr, vd3, preg);
}
```

```cpp
// Dynamic mask: process incomplete tail data.
__simd_vf__ __aicore__ void GatherProcess(ubuf int8_t* curYAddr, uint16_t repeatTimes, uint16_t computeSize)
{
    Reg::RegTensor<int8_t> vregTemp;
    Reg::MaskReg preg;

    for (uint16_t r = 0; r < repeatTimes; r++) {
        // Update the mask based on the remaining element count.
        preg = Reg::UpdateMask<int8_t>(sreg);
        // Create an address offset register.
        Reg::AddrReg offset = Reg::CreateAddrReg<int8_t>(r, computeSize);
        Reg::DataCopy(vregTemp, curXAddr, offset);
        // Data storage with mask.
        Reg::DataCopy(curYAddr, vregTemp, offset, preg);
    }
}
```

```cpp
// Data aggregation.
__VEC_SCOPE_
{
    Reg::RegTensor<uint32_t> indicesReg;
    Reg::RegTensor<int32_t> vd0;

    for (uint16_t indices = 0; indices < indicesLoopNum; indices++) {
        // Load the index (E2B distribution mode: broadcast a scalar to a vector).
        Reg::DataCopy<uint32_t, Reg::LoadDist::DIST_E2B_B32>(indicesReg, indicesAddr);
        // Perform Gather data aggregation based on the index.
        Reg::DataCopyGather(vd0, curXAddr, indicesReg, preg);
        // Copy the data block to output.
        Reg::DataCopy<int32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(
            curYAddr, vd0, blockStride, preg);
    }
}
```

**Key Regbase API description**

| API Category | API Name | Function |
|---------|---------|----------|
| Register type | `RegTensor<T>` | Vector register tensor type. |
| Mask type | `MaskReg` | Mask register type. |
| Mask creation | `CreateMask<T, Pattern>()` | Creates a mask (ALL/HALF and other patterns). |
| Mask update | `UpdateMask<T>(count)` | Dynamically updates the mask based on the remaining element count. |
| Scalar operation | `Duplicate(reg, val, mask)` | Copies a scalar value to all elements of the register. |
| Sequence generation | `Arange(reg, start)` | Generates a contiguous sequence. |
| Arithmetic operation | `Add/Sub/Mul/Div(dst, src1, src2, mask)` | Vector arithmetic operation. |
| Scalar operation | `Adds/Muls(dst, src, scalar, mask)` | Vector-scalar operation. |
| Type conversion | `Cast<DT, ST>(dst, src, mask)` | Data type conversion. |
| Comparison operation | `Compare<T, CMPMODE>(mask, src1, src2, pred)` | Vector comparison that generates a mask. |
| Data load | `DataCopy<T, LoadDist>(reg, addr)` | Loads from UB to the register. |
| Data storage | `DataCopy<T>(addr, reg, mask)` | Stores from the register to UB. |
| Gather | `DataCopyGather(dst, base, indices, mask)` | Collects data based on indices. |
| Address offset | `CreateAddrReg<T>(loop, stride)` | Creates a loop address offset register. |

**Distribution mode (LoadDist) description**

| Mode | Description | Typical Use |
|------|------|----------|
| `DIST_NORM` | Normal contiguous load. | Contiguous data processing. |
| `DIST_UNPACK_B16` | 16-bit unpack load. | FP16/BF16 to FP32. |
| `DIST_BRC_B32/B16` | Broadcast load. | Scalar scale broadcast. |
| `DIST_E2B_B32` | Scalar-to-vector broadcast. | Index value broadcast. |

**Migration recommendation**

1. Use cases suitable for Regbase: scenarios requiring fine-grained register allocation control, complex mask logic, and Gather/Scatter memory access patterns.
2. Use cases where Membase is retained: simple contiguous data movement and computation, and double-buffering pipelines.
3. Hybrid use: combine the two paradigms in the same operator, using Regbase for core computation logic and Membase for data movement management.

### Cube-Vector fusion operator class

#### MTE Data Movement Path Changes

The Ascend 950 new architecture introduces direct connection paths between UB2L1 and L0C2UB to enable fast movement of matrix computation data, aiming to simplify CV fusion operator development and improve performance.
<div align="center">
  <img src="../figures/Ascend950_CV_passthrough_link.png" width="700" alt="Ascend 950 new CV direct connection path" />
</div>

**Matrix Move-In**

Enable the UB-to-L1 (UB2L1) direct connection path. Through the DataCopy API, the vector computation result of a fusion operator can be moved directly into L1.

**Matrix Move-Out**

Enable the L0C-to-UB (L0C2UB) direct connection path. Through the DataCopy API, the matrix computation result of a fusion operator can be moved directly into UB for subsequent vector computation.

For K-split or multi-stage fusion scenarios, change "move L0C back to GM and then read back to UB" to "L0C directly to UB for accumulation/post-processing", reducing GM round-trip bandwidth pressure and latency. During migration, complete intermediate result merging and activation/quantization pre-processing on the UB side, and explicitly sort out the event synchronization order among MTE1/MTE2/MTE3 and the compute units to ensure continuous cross-unit pipelines and avoid data visibility or synchronization timing issues introduced by the new pathways. For key API definitions, refer to:

```cpp
// 1. Addition: the move-in API adds UB2L1 Nd2Nz move-in, supporting the form where both Src and Dst are LocalTensor.
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const Nd2NzParams& intriParams);

// 2. Addition: the move-out API adds L0C2UB move-out, supporting direct move-out from L0C to UB, and the form where both Src and Dst are LocalTensor.
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams);
template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 {
    // ...
    uint8_t dualDstCtl = 0;
};

// 3. Capability enhancement: the inter-core synchronization API adds mode 3.
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

#### Inter-core Synchronization Semaphore Matching

`CrossCoreSetFlag` and `CrossCoreWaitFlag` are inter-core synchronization semaphore APIs, widely used for data dependency and collaborative control among multiple cores. In essence, they decouple and orderly advance the data processing stages between different AI Cores through "semaphores", and are commonly used in pipeline control, double buffering switching, and cross-core collaboration scenarios.

- `CrossCoreSetFlag`: after the current core (or thread) completes data processing of a certain stage, it actively sets the specified flag signal to inform the dependent party (usually other cores or downstream pipeline stages) that this stage is complete and subsequent processes can continue.
- `CrossCoreWaitFlag`: the current core (or thread) needs to wait for a flag signal to be set (that is, the dependent data or event is complete), and continues execution only after detecting the flag.

The essence of this semaphore set is to ensure a consistent synchronization sequence among multiple threads/multiple pipeline stages, preventing hardware exceptions such as data races or deadlocks caused by resources not being ready or dependencies not being completed. For the CrossCoreSetFlag and CrossCoreWaitFlag API descriptions, refer to the official documentation: [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

On Ascend 950, the quantities of `CrossCoreWaitFlag` and `CrossCoreSetFlag` must strictly match, and it is recommended to design them in pairs in the "produce first, then consume" order within the same synchronization semantics domain. On Atlas A2, if there are redundant `CrossCoreSetFlag` semaphores between operators, HWTS performs special processing to clear the counter. The Ascend 950 series no longer relies on this fallback mechanism to reduce hardware overhead, and requires one-to-one matching of inter-core synchronization semaphores within a single operator kernel; otherwise, a guaranteed hang occurs.

During migration, focus on the following issues: first, an abnormal branch returns early, causing only `Set` to execute without the corresponding `Wait` (or vice versa); second, multiple pipeline stages reuse the same `flagId` with overlapping lifecycles, causing "cross-stage crosstalk"; third, synchronization is conditionally triggered within a loop but the loop boundaries are not aligned, causing inconsistent iteration counts. These issues may be masked on Atlas A2 but will be directly exposed as blocking timeouts or deadlocks on Ascend 950. For operators with complex cross-core pipelines, first build a minimal dataset for single-stage verification, then gradually add double buffering and multiple stages to reduce the complexity of locating synchronization issues.

### Collective Communication Operator Class

Ascend 950 introduces the collective communication accelerator CCU1.0, which reduces memory access requirements and scheduling latency. To effectively utilize this feature, the cross-device communication approach of operators is changed from AICPU on A2 to CCU communication.

**Eager Mode**

In the second-phase API of the aclnn two-phase API, specify the collective communication type for the operator executor aclOpExecutor.

Take the [MatmulAllReduce](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce) operator migration adaptation as an example:
Set the NnopbaseSetHcclServerType enumeration value. A2 uses NNOPBASE_HCCL_SERVER_AICPU, and 950 uses NNOPBASE_HCCL_SERVER_TYPE_CCU.

```CPP
// ...
aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // ...
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    // ...
    return ACLNN_SUCCESS;
}
```

**Graph Mode**

1. In the CalcParamFunc callback API used for resource computation and application, which involves auxiliary stream related information, distinguish the collective communication type of the auxiliary stream for the GE context.
2. In the GenerateTask callback API used for setting custom tasks and parameter customization on the main/auxiliary streams, distinguish the two sets of GE KernelLaunch APIs, and call the creation and customization processes of AICPU communication or CCU communication respectively.

For the task type of creating communication tasks on the static graph GE side, A2 uses aicpu kfc server + kfc_stream; 950 uses ccu server + ccu_stream. Involved code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
ge::Status MatmulAllReduceCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    // A2
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}
// ...
```

The static graph GenTask call APIs differ, and the processes differ. Involved code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
// A2
ge::Status MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>>& tasks) {
    // ...
    // aicpu task
    ge::KernelLaunchInfo aicpu_task =
        ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
    // ...
}


// 950
ge::Status Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    // ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    // ...
}

ge::Status MatmulAllReduceGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    // A2
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}
// ...
```

## 5. FAQ and Performance Tuning Recommendations (FAQ/Performance Tips)

If an operator's performance decreases instead of improving on Ascend 950, prioritize checking the following:

1. Whether the Atlas A2 misaligned core partitioning template is still used.
2. Whether CCU communication is not enabled and AICPU is still used.
3. Whether tiling still follows the Atlas A2 L1/L0/UB partitioning strategy, causing the larger on-chip cache of Ascend 950 to be underutilized.
