# build Parameters

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:08:01.414Z pushedAt=2026-11-18T09:59:29.674Z nl=2 -->

## Introduction

build.sh is the build script of this project. It is located in the project root directory by default. It automatically compiles, links, and configures the source code, and finally generates executable files, library files, or other target files that can be installed or run directly. Specifically, the script implements multiple functions by configuring different parameters, including building multiple target libraries (for example, libophost_cv.so), compiling operator packages, and running unit tests.

## Usage

1. **Configure environment variables**

   Refer to [Environment Deployment](./quick_install.md) to complete the basic environment setup.

   ```bash
   # Install to the default path, using the root user as an example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build command format**

   Take the command for compiling an operator package as an example. The format is as follows, where `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For the meaning of all parameters, see the Parameters section below. Select appropriate parameters based on the actual situation.

## Parameters

build.sh supports multiple functions. Run the following command to view all function parameters.

```bash
bash build.sh --help
```

| Name             | Required/Optional  | Description                                                                            |
|-----------------|--------|---------------------------------------------------------------------------------|
| -j${n}          | Optional     | Specifies the number of compilation threads. ${n} is the specific number of threads, and the default value is 8 (for example, -j8). If the number of threads exceeds the number of CPU cores, it is automatically adjusted to the number of CPU cores. |
| -v              | Optional     | Views the CMake compilation configuration information.                                                                  |
| -O${n}          | Optional     | Specifies the compilation optimization level. Supports O0/O1/O2/O3 (for example, -O3). ${n} is the optimization level identifier.                                      |
| -u              | Optional     | Enables the unit test (UT) compilation mode and compiles all UT targets.                                                        |
| --help,-h      | Optional     | Prints the script help information.                                                                     |
| --ops           | Optional     | Specifies the operators to be compiled, for example, grid_sample,iou_v2. Multiple operators are separated by English commas ",". In non-UT builds, it cannot be used together with --ophost, --opapi, or --opgraph. In UT mode, it can be used with -u and the preceding targets to filter tests by operator. |
| --soc           | Optional     | Specifies the NPU model. Only one NPU model is supported per compilation.                                                         |
| --jit           | Optional     | In the static graph scenario, when compiling the full package `cann-${soc_name}-ops-cv_${cann_version}_linux-${arch}.run`, the operator binary files do not need to be compiled (the graph runtime compiles them online). Configure this option to speed up compilation. |
| --static        | Optional     | After configuration, generates static library files, including libcann_cv_static.a and aclnn API header files. Used together with the --pkg parameter to generate a static library package.              |
| --vendor_name   | Optional     | Specifies the name of the custom operator package. The default value is custom.                                                         |
| --build-type    | Optional     | Specifies the build type. Supports Release/Debug (for example, --build-type=Debug). The default value is Release. Cannot be used together with --mssanitizer, --oom, or --dump_cce.        |
| --cov           | Optional     | Enables unit test code coverage statistics (must be used together with -u). Uses lcov to generate the coverage report.                                                                 |
| --noexec        | Optional     | Only compiles the unit test binary files and does not automatically run the compiled UT executable files.                                                  |
| --opkernel      | Optional     | Compiles the binary kernel.                                                                        |
| --pkg           | Optional     | Generates the installation package. Cannot be used together with -u (UT mode), --ophost, --opapi, or --opgraph.                               |
| --asan          | Optional     | Enables the host-side ASAN (AddressSanitizer) memory detection function.                                            |
| --valgrind      | Optional     | Uses valgrind to run unit tests for memory detection (must be used together with -u). After it is enabled, ASAN and --noexec are disabled.                                                                 |
| --make_clean    | Optional     | Performs basic cleanup (cleans compilation artifacts). The script exits after execution.                                                       |
| --ophost        | Optional     | Compiles the libophost_cv.so library. It cannot be used together with --pkg. In non-UT builds, it cannot be used together with --ops. When used with -u to run UTs, --ops filtering is allowed.                                          |
| --opapi         | Optional     | Compiles the libopapi_cv.so library. It cannot be used together with --pkg. In non-UT builds, it cannot be used together with --ops. When used with -u to run UTs, --ops filtering is allowed.                                           |
| --opgraph       | Optional     | Compiles the libopgraph_cv.so library. It cannot be used together with --pkg. In non-UT builds, it cannot be used together with --ops. When used with -u to run UTs, --ops filtering is allowed.                                         |
| --ophost_test   | Optional     | Compiles ophost-related unit tests. Equivalent to the combination of -u and --ophost.                                                |
| --opapi_test    | Optional     | Compiles opapi-related unit tests. Equivalent to the combination of -u and --opapi.                                                  |
| --opgraph_test  | Optional     | Reserved parameter. Developers do not need to pay attention to it for now.                                                                 |
| --opkernel_test | Optional     | Compiles opkernel-related unit tests. Equivalent to the combination of -u and --opkernel.                                            |
| --run_example   | Optional     | Compiles the samples of the specified operators and modes, and runs the compiled executable files.                                                       |
| --simulator     | Optional     | Enables the simulator mode to run the --run_example task. In simulator mode, the corresponding simulation library is linked based on soc_version.                                  |
| --genop         | Optional     | Creates the initial directory of an AI Core custom operator. A value is required: `--genop=<op_class>/<op_name>` |
| --genop_aicpu   | Optional     | Creates the initial directory of an AI CPU custom operator. A value is required: `--genop_aicpu=<op_class>/<op_name>` |
| --experimental  | Optional     | Compiles the user operators under the experimental directory.                                                         |
| --mssanitizer   | Optional     | Enables the kernel-side mssanitizer memory detection function. It cannot be used together with --oom.                                         |
| --oom           | Optional     | Enables the kernel-side oom memory detection function. It cannot be used together with --mssanitizer.                                         |
| --dump_cce      | Optional     | Enables the kernel-side function of dumping precompiled files.                                                      |
| --bisheng_flags | Optional     | Specifies the Bisheng compiler compilation parameters. Multiple compilation parameters are separated by English commas ",". It cannot be used together with --mssanitizer, --oom, or --dump_cce.     |
| --kernel_template_input  | Optional     | Specifies the template parameters for compiling the kernel. Multiple template parameters are separated by English commas ",". It is used together with --ops and only one operator can be specified.     |
| --cann_3rd_lib_path      | Optional     | The directory where third-party libraries are stored in offline compilation scenarios.                                                               |
