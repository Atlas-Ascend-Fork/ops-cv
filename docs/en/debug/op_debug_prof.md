# Operator Debugging and Tuning

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:06:34.461Z pushedAt=2026-11-18T09:59:29.661Z nl=2 -->

## Debugging and Locating (AI Core Operator)

During operator runtime, if issues such as operator execution failure or precision exceptions occur, print the information of each stage, such as Kernel intermediate results, to analyze and locate the issue.

### 1. Method for Obtaining Logs on the Host

* **plog collection**

   After the program execution ends, view the logs in `$HOME/ascend/log` by default. The host log file storage path is as follows:

   ```sh
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   Enable the environment variable ASCEND_SLOG_PRINT_TO_STDOUT to print log logs directly to the screen (1: enable screen printing, 0: disable screen printing). The configuration example is as follows:

   ```sh
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   For log-related introduction, see [*Log Reference*](https://hiascend.com/en/document/redirect/CannCommunitylogref). For environment variable introduction, see [*Environment Variables*](https://hiascend.com/en/document/redirect/CannCommunityEnvRef).

* **aclnn exception error message collection**

   Obtain the exception information during the aclnn API call through the aclGetRecentErrMsg API ([*Runtime APIs*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi)). The usage is as follows:

   ```cpp
   const char* errMsg = aclGetRecentErrMsg();
   if (errMsg != nullptr) {
       printf("%s", errMsg);
   }
   ```

   The printed error message example is as follows:

   ```sh
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel Debugging

Common debugging methods are as follows:

* **printf**

  This API supports printing Scalar type data, such as integers, characters, and Boolean types. For detailed introduction, see "Operator Debugging APIs > printf" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print the block length computed by the current core.
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This API supports dumping the content of a specified Tensor and printing custom additional information, such as the current line number. For detailed introduction, see "Operator Debugging APIs > DumpTensor" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print the zLocal tensor information.
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

For locating issues in complex use cases, such as operator hang and GM/UB access out-of-bounds, use **single-step debugging**. For specific operations, see the [msDebug](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsdebug) operator debugging tool.

## Debugging and Locating (AI CPU Operator)

During operator runtime, if issues such as operator execution failure or precision exceptions occur, print the information of each stage, such as Kernel intermediate results, to analyze and locate the issue.

### 1. Method for Obtaining Logs on the Host

   See [Method for Obtaining Logs on the Host](#1-method-for-obtaining-logs-on-the-host) of the AI Core operator.

### 2. Kernel Debugging

Common debugging methods are as follows:

- **KERNEL\_LOG macro**

  Use the following macros to print log information during operator execution, including DEBUG, INFO, WARN, and ERROR level logs.

  ```Cpp
  KERNEL_LOG_DEBUG(fmt, …)      // fmt indicates the format control string.
  KERNEL_LOG_INFO(fmt, …)
  KERNEL_LOG_WARN(fmt, …)
  KERNEL_LOG_ERROR(fmt, …)      // Print ERROR level logs by default.
  ```

  To print non-ERROR level logs, configure the environment variable `ASCEND_GLOBAL_LOG_LEVEL` in advance. For specific usage, see [*Environment Variables*](https://hiascend.com/en/document/redirect/CannCommunityEnvRef).

  The print example is as follows:

  ```c++
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(0);

  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    // Print the error message.
    KERNEL_LOG_ERROR("Invalid argument");
    return kParamInvalid;
  }

  int64_t num_elements = input0->NumElements();
  // Print the number of input elements.
  KERNEL_LOG_INFO("Num of elements is %ld", num_elements);
  ```

## Performance Tuning

During operator runtime, if issues such as execution precision degradation or abnormal memory usage occur, use the [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof) performance analysis tool to analyze the metric data of each operator runtime stage (such as throughput, memory usage, and time consumption), determine the root cause of the issue, and optimize accordingly.

This chapter uses the `AddExample` custom operator as an example to introduce the commonly used on-board performance collection and pipeline diagram simulation methods in operator tuning.

**Use case differences:**

- **On-board performance collection**: applicable to running operators on real NPU hardware to quickly obtain the overall performance metrics of the operator (such as Kernel time consumption, Block count, and pipeline proportion), helping determine whether the operator has performance issues.
- **Pipeline diagram simulation**: applicable to developers without NPU hardware, or scenarios that require in-depth analysis of instruction-level pipeline bottlenecks within the operator and optimization of instruction scheduling. It provides more detailed instruction-level pipeline analysis than on-board collection.

### Method 1: On-board Performance Collection

   - **Prerequisites**

      After completing operator development and compilation, assuming the use of the aclnn API call method, the directory where the generated operator executable file (test_aclnn_add_example) is located is `examples/add_example/examples/build/bin/` in this project.

   - **Collect performance data**

      When collecting various pipeline metrics of the operator on-board runtime, enter the directory where the operator executable file is located and execute the following command:

      ```bash
      msprof op ./test_aclnn_add_example
      ```

      The collection results are in the `examples/add_example/examples/build/bin/OPPROF_*` directory in this project. After collection, the following information is printed:

      ``` text
      Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
      Op Type: vector
      Task Duration(us): 97.861954
      Block Dim: 8
      Mix Block Dim:
      Device Id: 0
      Pid: 2776181
      Current Freq: 1800
      Rated Freq: 1800
      ```

      Among them, Task Duration is the time consumption of the current operator Kernel, and Block Dim is the number of cores for the current operator execution.

      For detailed metrics of various operator pipelines, refer to the `ArithmeticUtilization` file under `OPPROF_*`, which contains the proportion of various pipelines. For specific introduction, see the "Performance Data File > msprof op > ArithmeticUtilization (time consumption and proportion of cube and vector type instructions)" chapter in [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof).

### Method 2: Pipeline Diagram Simulation Collection

   - **Prerequisites**

      After completing operator development and compilation, assuming the use of the aclnn API call method, the directory where the generated operator executable file (test_aclnn_add_example) is located is `examples/add_example/examples/build/bin/` in this project.

   - **For Ascend 950PR products, use the [NPU Simulator](./npu_sim.md) simulation tool to execute the simulation command and generate simulation data.**

      Execute the simulation command to generate simulation data

      ```bash
      npusim record ./test_aclnn_add_example -s Ascend950 --gen-report
      ```

      The simulation results are in the `examples/add_example/examples/build/bin/npusim_*/report/results/kernel_*/core_*` directory in this project. The pipeline-related files are:

      ```bash
      trace_core0.json
       ```

      Enter "chrome://tracing" in the Chrome browser and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For specific parameter introduction, see [Simulation Result Parsing](./npu_sim.md#simulation-result-parsing) in NPU Simulator.

   - **For Atlas A2 products and Atlas A3 products, use [msProf](https://www.hiascend.com/en/document/redirect/CannCommunityToolMsprof) to execute the simulation command and generate simulation data.**

      Before using the msProf tool for operator simulation tuning, execute the following command to configure the environment variables.

      ```bash
      export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
      ```

      Modify the preceding environment variables according to the actual installation path of the CANN software package and the AI processor model.

      Then enter the directory where the operator executable file is located and execute the following command:

      ```bash
      msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
      ```

      The collection results are in the `$PWD/pipeline_auto/OPPROF_**` directory in this project.
      Among them, the pipeline-related file path is `OPPROF**/simulator/visualize_data.bin`. Use the "Basic Operations > Import Data" chapter in [MindStudio Insight](https://www.hiascend.com/en/document/redirect/MindStudioInsight) to view how to import pipeline data.
