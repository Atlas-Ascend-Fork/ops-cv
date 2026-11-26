# Operator Call

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:12:46.237Z pushedAt=2026-11-18T09:59:29.706Z nl=2 -->

## Precautions

- Prerequisites: Before calling operators, complete environment preparation and source code download by referring to the project README. Details are not repeated here.
- Call scope: For the list of built-in operators that can be called, see [Operator List](../op_list.md). Custom operators (such as contributed operators in the experimental directory) are also supported.
- Call use cases: Select an appropriate operator call solution based on your actual use case.

  |Call Use Case|Scenario Description|Features|
  |--------|----|-----|
  | [Quick Operator Call](#quick-operator-call) | Suitable for quickly experiencing and verifying operator functions. | **No call project setup required.** Operator examples can be called directly based on the source build package and the project script build.sh. |
  | [Integrating Operators into Business Applications](#integrating-operators-into-business-applications) | Suitable for flexibly integrating operators into actual business applications. | **A call project must be set up.** Manually create call scripts/CMake projects to flexibly compile and run operators. |

- Call methods: The following operator call methods are currently provided. Select one as needed.

  |Call Method|Description|
  |--------|----|
  |[PyTorch API](#pytorch-api)|Registers the operator kernel with the native PyTorch framework to call operators in a way similar to native Torch APIs.|
  |[aclnn API](#aclnn-api)|Provides corresponding C language APIs (prefixed with aclnn) for operators. No IR definition is required, and operators are called directly through the APIs.|
  |[GE Graph Mode](#ge-graph-mode)|Calls operators through graph construction based on operator IR (Intermediate Representation) definitions.|

## Quick Operator Call

**To quickly experience or verify the functions of existing operators in the project, use the simplest call method in this chapter to run operator examples through build.sh.**

This method requires no call project setup (that is, no compile/run scripts need to be created), and is simple and easy to use.

> **Note**: For Ascend 950PR products, you can run operator samples through the Simulator tool. For details, see [Simulation Guide](../debug/op_debug_prof.md#method-2-pipeline-diagram-simulation-collection).

**Step 1**: Prepare the environment. Before calling operators, ensure that the CANN-toolkit package and the compiled operator package are installed in your environment.

**Step 2**: Run the examples of existing operators in the project.

- Run operator examples based on the **custom operator package**. The command is as follows:

  ```bash
  bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}] [--soc=${soc_version}] [--experimental]
  # Take the grid_sample operator example execution as an example
  # bash build.sh --run_example grid_sample eager cust --vendor_name=custom
  # Take the grid_sample operator experimental execution as an example
  # bash build.sh --experimental --run_example grid_sample eager cust --vendor_name=custom
  ```

  - \$\{op\}: Indicates the operator to be executed, in the lowercase underscore form of the operator name, such as grid_sample.
  - \$\{mode\}: Indicates the call method. Currently, eager (aclnn call) and graph (graph mode call) are supported.
  - \$\{pkg_mode\}: Indicates the package mode. Currently, only cust (custom operator package) is supported.
  - \$\{vendor\_name\} (optional): Must be consistent with the built custom operator package setting. The default name is custom.
  - \$\{soc_version\} (optional): Indicates the NPU model.
  - \$\{experimental\} (optional): Indicates running operators saved by users in the experimental contribution directory.

  Note: When \$\{mode\} is graph, do not specify \$\{pkg_mode\} and \$\{vendor\_name\}.

- Run the operator sample based on the **ops-cv package** by executing the following command:

  ```bash
  bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}]
  # Take the grid_sample operator example execution as an example
  # bash build.sh --run_example grid_sample eager
  ```

  - \$\{op\}: Indicates the operator to be executed, in the lowercase underscore form of the operator name, such as grid_sample.
  - \$\{mode\}: Indicates the call method. Currently, eager (aclnn call) and graph (graph mode call) are supported.
  - \$\{soc_version\} (optional): Indicates the NPU model.

- Run the operator sample based on the **ops-cv static library**:

    1. **Prerequisites**

        The ops-cv static library depends on the ops-legacy static library and the ops-math static library. Prepare the preceding static libraries, decompress them, and move all lib64 and include directories to the unified directory \$\{static_lib_path}.

        > Note: The ops-legacy static library ```cann-${soc_name}-ops-legacy-static_${cann_version}_linux-${arch}.tar.gz``` can be obtained by clicking the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master). The ops-cv static library and the ops-math static library are not provided as software packages yet. Generate them through local compilation.

    2. **Create run.sh**

        Create a run.sh file in the same directory as the `examples/test_aclnn_${op_name}.cpp` file of the operator to run.

        Take the grid_sample operator executing test_aclnn_grid_sample2_d.cpp and test_aclnn_grid_sample3_d as an example. The example is as follows:

        ```bash
        # Static library file path
        static_lib_path=""

        # Make the environment variable take effect
        if [ -n "$ASCEND_INSTALL_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
        elif [ -n "$ASCEND_HOME_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
        else
            _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
        fi

        source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

        # Compile the executable file
        g++ test_aclnn_grid_sample2_d.cpp \
        -I ${static_lib_path}/include \
        -L ${static_lib_path}/lib64 \
        -I ${_ASCEND_INSTALL_PATH}/include \
        -I ${_ASCEND_INSTALL_PATH}/include/aclnnop \
        -L ${_ASCEND_INSTALL_PATH}/lib64 \
        -Wl,--allow-multiple-definition \
        -Wl,--start-group -lcann_cv_static -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lgraph_base \
        -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform \
        -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog \
        -o test_aclnn_grid_sample2_d   # Replace with the actual operator executable file name

        g++ test_aclnn_grid_sample3_d.cpp \
        -I ${static_lib_path}/include \
        -L ${static_lib_path}/lib64 \
        -I ${_ASCEND_INSTALL_PATH}/include \
        -I ${_ASCEND_INSTALL_PATH}/include/aclnnop \
        -L ${_ASCEND_INSTALL_PATH}/lib64 \
        -Wl,--allow-multiple-definition \
        -Wl,--start-group -lcann_cv_static -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lgraph_base \
        -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform \
        -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog \
        -o test_aclnn_grid_sample3_d   # Replace with the actual operator executable file name

        # Run the program
        ./test_aclnn_grid_sample2_d
        ./test_aclnn_grid_sample3_d
        ```

        \$\{static_lib_path} indicates the unified static library placement path. \$\{ASCEND_INSTALL_PATH} is configured through an environment variable and indicates the CANN toolkit package installation path. Replace the final executable file name with the **actual operator executable file name**.

        Among which, lcann_cv_static, lcann_math_static, and lcann_legacy_static indicate the static library files that the operator depends on, obtained from the unified static library placement path \$\{static_lib_path};
        lgraph, lmetadef, and others indicate the underlying library files that the operator depends on, which can be obtained from the CANN toolkit package.

    3. **Run run.sh**

        ```bash
        bash run.sh
        ```

**Step 3**: Check the execution result.

After the operator sample is executed, the result is printed. Take the grid_sample operator execution as an example:

```sh
This environment does not have the ASAN library, no need enable ASAN
CMAKE_ARGS: -DENABLE_UT_EXEC=TRUE
----------------------------------------------------------------
Start to run examples,name:grid_sample mode:eager
Start compile and run examples file: ../image/grid_sample/examples/test_aclnn_grid_sample2_d.cpp
pkg_mode:cust vendor_name:custom
resultData[0] is: 0.250000
resultData[1] is: 2.250000
resultData[2] is: 2.000000
resultData[3] is: 8.500000
resultData[4] is: 20.500000
resultData[5] is: 12.000000
resultData[6] is: 8.250000
resultData[7] is: 18.250000
resultData[8] is: 10.000000
Start compile and run examples file:
../image/grid_sample/examples/test_aclnn_grid_sample3_d.cpp
pkg_mode:cust vendor_name:custom
resultData[0] is: 0.250000
resultData[1] is: 0.875000
resultData[2] is: 2.000000
resultData[3] is: 4.000000
```

## Integrating Operators into Business Applications

**To integrate operators into actual business applications, set up a call project by referring to this chapter. Compile and run operators through custom call scripts/CMake projects.**

This method requires manually setting up a call project. It offers high flexibility and strong portability.

Different call methods correspond to different build projects. Currently, PyTorch API, aclnn API, and GE graph mode call methods are supported. For call methods and practices, see the following content. Select one as needed.

### PyTorch API

This method registers the operator Kernel to the PyTorch native framework so that it can be directly called like a native Torch API.

For the specific call principle and process, see [examples/fast_kernel_launch_example](../../../examples/fast_kernel_launch_example/README.md).

### aclnn API

#### Call Process

To facilitate operator calls, the host side provides C language APIs (that is, APIs prefixed with aclnn) for operators. No operator IR (Intermediate Representation) definition is required. The aclnn API call process is as follows:

![Schematic diagram](../figures/aclnn_calling.png)

#### Compile and Run

> Note: During the operation, if the log prompts you to set environment variables, follow the prompts.

1. Prepare the environment. Before compiling and running, ensure that the CANN-toolkit package and the compiled operator package are installed in your environment.

2. Create a call script.

   In any directory of the environment, create a call cpp script with a custom name (for example, `${test_aclnn_op_name}.cpp`).

   For ease of understanding, take the `AddExample` operator as an example. The call script is as follows, for reference only. For the complete code, see [test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp).

   ```Cpp
   int main()
   {
       // 1. Call acl to initialize the device/stream
       int32_t deviceId = 0;
       aclrtStream stream;
       auto ret = Init(deviceId, &stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

       // 2. Construct the input and output. Custom construction is required based on the API definition.
       aclTensor* selfX = nullptr;
       void* selfXDeviceAddr = nullptr;
       std::vector<int64_t> selfXShape = {32, 4, 4, 4};
       std::vector<float> selfXHostData(2048, 1);
       ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       aclTensor* selfY = nullptr;
       void* selfYDeviceAddr = nullptr;
       std::vector<int64_t> selfYShape = {32, 4, 4, 4};
       std::vector<float> selfYHostData(2048, 1);
       ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclDataType::ACL_FLOAT, &selfY);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       aclTensor* out = nullptr;
       void* outDeviceAddr = nullptr;
       std::vector<int64_t> outShape = {32, 4, 4, 4};
       std::vector<float> outHostData(2048, 1);
       ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       // 3. Call the CANN operator library API. Modify it to the specific API name.
       uint64_t workspaceSize = 0;
       aclOpExecutor* executor;

       // 4. Call the first-phase API of aclnnAddExample
       ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

       // Apply for device memory based on the workspaceSize calculated by the first-phase API
       void* workspaceAddr = nullptr;
       if (workspaceSize > static_cast<uint64_t>(0)) {
           ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
           CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
       }

       // 5. Call the second-phase API of aclnnAddExample
       ret = aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);

       // 6. (Boilerplate) Synchronously wait for the task execution to finish
       ret = aclrtSynchronizeStream(stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

       // 7. Obtain the output value and copy the result from the device memory to the host side. Modify it based on the specific API definition.
       PrintOutResult(outShape, &outDeviceAddr);

       // 8. Release aclTensor. Modify it based on the specific API definition.
       aclDestroyTensor(selfX);
       aclDestroyTensor(selfY);
       aclDestroyTensor(out);

       // 9. Release device resources
       aclrtFree(selfXDeviceAddr);
       aclrtFree(selfYDeviceAddr);
       aclrtFree(outDeviceAddr);
       if (workspaceSize > static_cast<uint64_t>(0)) {
           aclrtFree(workspaceAddr);
       }
       aclrtDestroyStream(stream);
       aclrtResetDevice(deviceId);

       // 10. Deinitialize acl
       aclFinalize();
       return 0;
   }
   ```

3. Create a CMakeLists.txt file.

   Create a CMakeLists.txt file in the same directory as `${test_aclnn_op_name}.cpp`. Note that the build scripts differ when calling custom operators (such as those in the experimental directory) and standard project operators (built-in operators). The examples are as follows, for reference only. Modify them based on the actual situation.

   - **Call custom operators**: Depends on the custom operator package

     ```bash
     cmake_minimum_required(VERSION 3.14)
     # Set the project name
     project(ACLNN_EXAMPLE)

     # Set the C++ compilation standard
     add_compile_options(-std=c++11)

     # Set the compilation output directory to the bin folder in the current directory
     set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

     # Set compilation options for debug and release modes
     set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
     set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

     # Add an executable file (custom: replace with the actual *.cpp file of the operator being called)
     add_executable(${test_aclnn_op_name}
     ${test_aclnn_op_name}.cpp)

     # ASCEND_PATH (if the CANN package path is incorrect, please modify it according to the actual path)
     if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
         set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
     else()
         set(ASCEND_PATH "/usr/local/Ascend/cann")
     endif()

     # Obtain the custom operator package name. If multiple custom operator packages exist, only one of them is used.
     set(VENDORS_DIR "${ASCEND_PATH}/opp/vendors")
     file(GLOB CUSTOM_DIRS "${VENDORS_DIR}/*")
     foreach(CUSTOM_DIR ${CUSTOM_DIRS})
         if(IS_DIRECTORY ${CUSTOM_DIR})
             set(TARGET_SUBDIR ${CUSTOM_DIR})
         endif()
     endforeach()

     if(NOT DEFINED TARGET_SUBDIR)
         message(FATAL_ERROR "No custom operator package found in the path ${ASCEND_PATH}")
     endif()

     # Set the header file path
     set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
     include_directories(
         ${INCLUDE_BASE_DIR}
         ${TARGET_SUBDIR}/op_api/include
     )
     include_directories(
         ${INCLUDE_BASE_DIR}
     )

     # Link the required dynamic library (custom: replace with the actual operator executable file)
     target_link_libraries(${test_aclnn_op_name} PRIVATE
         ${ASCEND_PATH}/lib64/libascendcl.so
         ${ASCEND_PATH}/lib64/libnnopbase.so
         ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so      # Link the custom operator library file
     )
     target_link_options(${test_aclnn_op_name} PRIVATE
         "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
     )

     # Install the target file to the bin directory (custom: replace with the actual operator executable file)
     install(TARGETS ${test_aclnn_op_name} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
     ```

   - **Call standard operators (built-in operators)**: Depends on the op-cv full package

     ```bash
     cmake_minimum_required(VERSION 3.14)
     # Set the project name
     project(ACLNN_EXAMPLE)

     # Set the C++ compilation standard
     add_compile_options(-std=c++11)

     # Set the compilation output directory to the bin folder in the current directory
     set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

     # Set compilation options for debug and release modes
     set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
     set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

     # Add an executable file (custom: replace with the actual *.cpp file of the operator being called)
     add_executable(${test_aclnn_op_name}
     ${test_aclnn_op_name}.cpp)

     # ASCEND_PATH (if the CANN package path is incorrect, please modify it according to the actual path)
     if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
         set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
     else()
         set(ASCEND_PATH "/usr/local/Ascend/cann")
     endif()

     # Set the header file path
     set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
     include_directories(
         ${INCLUDE_BASE_DIR}
         ${ASCEND_PATH}/include/aclnnop
     )

     # Link the required dynamic library (custom: replace with the actual operator executable file)
     target_link_libraries(${test_aclnn_op_name} PRIVATE
         ${ASCEND_PATH}/lib64/libascendcl.so
         ${ASCEND_PATH}/lib64/libnnopbase.so
         ${ASCEND_PATH}/lib64/libopapi_cv.so            # Link the built-in operator library file
     )

     # Install the target file to the bin directory (custom: replace with the actual operator executable file)
     install(TARGETS ${test_aclnn_op_name} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
     ```

4. Create a run.sh file.

    Create a run.sh file in the same-level directory of `${test_aclnn_op_name}.cpp`. Take the `AddExample` operator as an example. The example is as follows. Modify it yourself based on the actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute the build command
    make
    cd bin
    ./${test_aclnn_op_name}             # Replace with the actual operator executable file name
    ```

5. Run the run.sh file.

   Run the following command in the directory where the run.sh file is located:

   ```bash
   bash run.sh
   ```

    By default, the executable file ${test_aclnn_op_name} is generated under the current execution path `/build/bin`. Take test_aclnn_add_example as an example of the running result:

   ```bash
   mean result[2046] is 2.000000
   mean result[2047] is 2.000000
   ```

### GE Graph Mode

#### Call Process

This method calls operators through graph construction based on operator GE IR (Intermediate Representation) definitions. The call process is as follows:

![Schematic diagram](../figures/geir_call_flow.png)

#### Compile and Run

> Note: During the operation, if the log prompts you to set environment variables, follow the prompts.

1. Prepare the environment. Before compiling and running, ensure that the CANN-toolkit package and the compiled operator package are installed in your environment.

2. Create a call script.

   In any directory of the environment, create a new call cpp script with a custom name (for example, `${test_geir_op_name}.cpp`).

   For ease of understanding, take the `AddExample` operator as an example. The call script is as follows, for reference only. For the full code, see [test_geir_add_example.cpp](../../../examples/add_example/examples/test_geir_add_example.cpp).

   ```CPP
   int main() {
       // 1. Create a graph object
       Graph graph(graphName);

       // 2. Initialize the global graph compilation options
       Status ret = ge::GEInitialize(globalOptions);

       // 3. Create an AddExample operator instance
       auto add1 = op::AddExample("add1");

       // 4. Define the graph input and output vectors
       std::vector<Operator> inputs{};
       std::vector<Operator> outputs{};

       // 5. Prepare the input data
       std::vector<int64_t> xShape = {32,4,4,4};
       // Handle variable assignment through macro expansion
       ADD_INPUT(1, x1, inDtype, xShape);
       ADD_INPUT(2, x2, inDtype, xShape);
       ADD_OUTPUT(1, y, inDtype, xShape);

       outputs.push_back(add1);

       // 6. Set the input operator and output operator of the graph object
       graph.SetInputs(inputs).SetOutputs(outputs);

       // 7. Create a session object
       ge::Session* session = new Session(buildOptions);

       // 8. Add the graph to the session
       ret = session->AddGraph(graphId, graph, graphOptions);

       // 9. Run the graph
       ret = session->RunGraph(graphId, input, output);

       // 10. Release resources
       GEFinalize();

       return 0;
   }
   ```

3. Create the CMakelists file.

   Create the CMakeLists.txt file in the same-level directory of `${test_geir_op_name}.cpp`. The GE graph engine automatically loads the library files of the installed operator package (whether a custom operator package or a standard built-in operator package) based on the configured environment variables, with no need to distinguish them. The example is as follows, for reference only. Please modify it according to the actual situation.

    ```bash
   cmake_minimum_required(VERSION 3.14)

   # Set the project name
   project(GE_IR_EXAMPLE)

   if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
       get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
   elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
   else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
   endif()

   set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")

   message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

   file(GLOB files CONFIGURE_DEPENDS
        ${test_geir_op_name}.cpp
   )

   # Add an executable file (replace with the actual operator executable file)
   add_executable(${test_geir_op_name} ${files})

   find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")

   # Link the required dynamic library
   target_link_libraries(${test_geir_op_name} PRIVATE
        ${GRAPH_LIBRARY_DIR}
        ${GE_RUNNER_LIBRARY_DIR}
        ${GRAPH_BASE_LIBRARY_DIR}
   )

   # Set the header file path
   target_include_directories(${test_geir_op_name} PRIVATE
        ${FWK_INCLUDE_DIR}/graph/
        ${FWK_INCLUDE_DIR}/ge/
        ${ASCEND_PATH}/opp/built-in/op_proto/inc/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_PATH}/compiler/include
   )
    ```

4. Create a run.sh script.

   Create the run.sh file in the same-level directory of `${test_geir_op_name}.cpp`. Take the `AddExample` operator as an example. The example is as follows. Please modify it according to the actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute the build command
    make
    ./${test_geir_op_name}                                     # Replace with the actual operator executable file name

5. Run the run.sh script.

    Execute the following command in the path where the run.sh file is located:

    ```bash
    bash run.sh
    ```

    By default, the executable file ${test_geir_op_name} is generated under the current execution path `/build/bin`. The running result is as follows:

    ```bash
    INFO - [XIR]: Finalize ir graph session success
    ```
