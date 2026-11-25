# Quick Start: Based on the ops-cv Repository

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:46:24.275Z pushedAt=2026-11-18T09:59:20.092Z nl=2 -->

## Precautions

This guide helps developers quickly get started with CANN and the `ops-cv` operator repository. To quickly understand the complete operator development process, use the **AddExample** operator as the practice object. The operator source code is located in `ops-cv/examples/add_example`. The operation process is as follows:

1. **[Prerequisites](../README.md)**: See the project README to complete environment preparation and source code download. This is not repeated here. For the quick start scenario, **CANNLab or Docker deployment is recommended**, which is simple to operate.

   > **Note**: The CANNLab or Docker environment provides the latest CANN package by default. If you need to experience the latest capabilities of the master branch, set up the environment manually.

2. **[Build and Run](#1-build-and-run)**: Build and install the custom operator package to quickly call the operator.

3. **[Operator Development](#2-operator-development)**: Modify the existing operator Kernel to experience the complete loop of development, compilation, and verification.

4. **[Operator Debugging](#3-operator-debugging)**: Learn the methods of operator printing and performance collection.

5. **[Operator Verification](#4-operator-verification)**: Learn how to modify the operator example sample to verify the functional correctness of the operator under different inputs.

## 1. Build and Run

The purpose of this phase is to **quickly experience the standard project process** and verify whether the environment can successfully build, package, install, and run the operator source code.

> This guide uses the single-operator compilation process as an example. It also supports building all operators in the operator library, offline compilation, and other tasks. For common issues during build, see [*Build from Source Code*](en/install/compile.md).

### 1. Enter the Project Source Code Directory

- CANNLab cloud development environment:

   The project source code supporting the latest CANN package is provided by default. The resources are generally in the `/mnt/workspace/gitCode` directory. Enter the target branch source code directory.

- Non-CANNLab cloud development environment:

  Obtain the source code and CANN version compatibility from the release-notes files in each folder of the [release repository](https://gitcode.com/cann/release-management) (the subpackage compatibility that can be independently upgraded in the CANN ops combination package). Run the following command to download the source code, and replace \$\{tag\_version\} with the target branch tag, for example, 9.0.0.

  ```bash
  git clone -b ${tag_version} https://gitcode.com/cann/ops-cv.git && cd ops-cv
  ```

> Note: To switch the source code branch version, refer to the following instructions.
>
> 1. Run `git branch` in the source code directory to query the current source code version.
> 2. Run `git checkout ${tag_version}` in the source code directory to switch to the target branch source code. Ensure that the source code and CANN version compatibility is met. If the branch source code has been cloned locally, run `git pull` to pull the latest source code.

### 2. Build the AddExample Operator

General command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

**Note**: Before build, ensure that the CANN environment variables are configured. Otherwise, the build may fail because `ASCEND_HOME_PATH` cannot be found. For the default path installation, run `source /usr/local/Ascend/cann/set_env.sh`.

Use the AddExample operator as an example. The build command is as follows:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
```

The ${soc_version} values corresponding to the product name are as follows. Pass parameters according to the actual scenario.

- Atlas A2 products: The value is ascend910b.
- Atlas A3 products: The value is ascend910_93.
- 950 products: The value is ascend950.

If the following information is displayed, the compilation is successful.

```output
Self-extractable archive "cann-ops-cv-custom-linux.${arch}.run" successfully created.
```

After successful build, the run package is stored in the build_out directory under the project root directory.

### 3. Install the AddExample Operator Package

```bash
./build_out/cann-ops-cv-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path. ```${ASCEND_HOME_PATH}``` indicates the CANN software installation directory.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variable to ensure that it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_cv/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run the Operator Sample

The general run command format is: `bash build.sh --run_example <operator name> <run mode> <package mode> --vendor_name=<custom operator package name>`.

Take AddExample as an example. It provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
```

> **Note**: When running the sample, ensure that the `--soc` parameter is consistent with the `--soc` value used when compiling the operator package. Otherwise, `error 161001` may be reported (for example, `aclnnXxxGetWorkspaceSize failed`). If this error occurs, return to [Build the AddExample Operator](#2-build-the-addexample-operator) to check the `--soc` value and then recompile and install.

- operator name: add_example
- run mode: eager
- package mode: cust
- --vendor_name (optional): vendor_name indicates the custom operator package name to build. The default name is custom

Expected output: The addition result of the `AddExample` operator is printed, indicating that the operator has been successfully deployed and executed correctly.

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## 2. Operator Development

The purpose of this phase is to attempt to **modify the kernel function code** of the successfully run AddExample operator.

### 1. Modify the Kernel Implementation

Find the core kernel implementation file of the AddExample operator `ops-cv/examples/add_example/op_kernel/add_example.h`, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, currentNum);
    AscendC::Mul(zLocal, xLocal, yLocal, currentNum);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Build and Verify

Repeat the steps in the [Build and Run](#1-build-and-run) section:

1. **Rebuild**:

    First return to the project root directory. The build command is as follows:

    ```bash
    bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
    ```

   > **Note**: Fill in `${soc_version}` based on the actual chip model. The value method is the same as that in [Build the AddExample Operator](#2-build-the-addexample-operator).

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-cv-*linux*.run
    ```

3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

4. **Success indicator**: The output result becomes the multiplication result. Each group of results should equal the product of the two inputs.

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## 3. Operator Debugging

In this phase, use AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and location.

### 1. Operator Print

If the operator encounters issues such as execution failure or precision exceptions, add printing for problem analysis and location.

Modify the code in `examples/add_example/op_kernel/add_example.h`.

- **printf**

  This API supports printing Scalar type data, such as integers, characters, and Boolean types. For detailed introduction, see "Operator Debugging APIs > printf" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```cpp
  blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
  ubLength_ = tilingData->ubFactor;
  // Print the current kernel computation Block length
  AscendC::PRINTF("Tiling blockLength is %ld\n", blockLength_);
  ```

- **DumpTensor**

  This API supports dumping the content of a specified Tensor and printing custom additional information, such as the current line number. For detailed introduction, see "Operator Debugging APIs > DumpTensor" in [*Ascend C APIs*](https://hiascend.com/en/document/redirect/CannCommunityAscendCApi).

  ```cpp
  AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
  AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
  AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
  AscendC::Add(zLocal, xLocal, yLocal, currentNum);
  AscendC::DumpTensor(zLocal, 0, 128);
  outputQueueZ.EnQue<T>(zLocal);
  ```

### 2. Performance Collection

After the operator function is verified to be correct, use the `msprof op` command to collect operator-level performance data.

1. **Generate an executable file**

    Call the example sample of the AddExample operator to generate the executable file (test_aclnn_add_example). The file is located in the `ops-cv/build` directory of the project.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

2. **Collect performance data**

    Enter the executable file directory of the AddExample operator `ops-cv/build/`, and run the following command:

    ```bash
    msprof op --application="./test_aclnn_add_example"
    ```

    After execution, the basic operator information (such as Op Name, Op Type, Task Duration, and Block Dim) and performance bottleneck prompts are printed directly.

The collection result is saved in the `OPPROF_*` folder under the `ops-cv/build/` directory of the project. After the command is executed, the performance data file is automatically parsed and exported. To further interpret each performance metric (such as pipeline ratio and bandwidth utilization), see [msProf Performance Data File Reference](https://gitcode.com/Ascend/msprof/blob/master/docs/en/user_guide/profile_data_file_references.md).

## 4. Operator Verification

This phase verifies the functional correctness of the AddExample operator in multiple scenarios by modifying the input data in the example sample of the AddExample operator.

### 1. Modify the Test Input

Find and edit `ops-cv/examples/add_example/examples/test_aclnn_add_example.cpp` of `AddExample`, and modify the shape and value of the input tensor.

**Modify input/output data**: Modify the shape information of the input and output, as well as the initialization data, and construct the corresponding input and output tensors.

```cpp
int main() {
    // ... Initialization code...

    // === ① Modify the input of selfX ===
    // Before modification: shape = {32, 4, 4, 4}, all values are 1
    // After modification: change the input shape to {8, 8, 8, 8}, and fill different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    auto selfXNumElements = GetShapeSize(selfXShape);
    std::vector<float> selfXHostData(selfXNumElements);
    // Use a loop to fill more distinguishable data, for example, an increasing sequence
    for (int64_t i = 0; i < selfXNumElements; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill cyclic values 0-9
    }
    // === ② Refer to selfX, modify selfY and out similarly, and ensure that the hostData length is consistent with the number of shape elements ===

    // ... Subsequent execution code...
}
```

### 2. Rebuild and Verify

1. Since only the example test code is modified, there is no need to rebuild the operator package.

2. Run the verification command again:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom --soc=${soc_version}
    ```

3. Observe whether the operator output result meets expectations.

## Conclusion

After the above process, you have basically completed the operator development process. If you want to further contribute new operators or learn more advanced development and debugging skills, see the [Advanced Tutorials](../README_en.md#tutorials) and [Contribution Guide](../README_en.md#related-information) in the project README.
