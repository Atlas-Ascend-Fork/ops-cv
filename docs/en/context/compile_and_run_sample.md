# Compile and Run Samples

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment is set up, including the driver, firmware, CANN software package, and ops package.
- For details about the operator API invocation process and compilation and running operations, see "Single Operator Invocation > Single Operator API Execution > Example Code for Calling aclnn APIs" in [*App Development (C&C++)*](https://hiascend.com/en/document/redirect/CannCommunityCppInferWizard).

## Preparation Before Compilation

This chapter uses the co-located development and runtime environment scenario as an example, where the machine with the AI Processor serves as both the development environment and the runtime environment. In this scenario, code development and code execution occur on the same machine. The **GridSample operator** is used here as an example. The invocation logic, workflow, and compilation scripts for other operators are largely the same as those for the GridSample operator. Modify the API invocation script (\*.cpp) and the compilation script (CMakeLists) as needed based on your actual situation.

- **Sample Code**

    The GridSample operator provides an input tensor and a corresponding grid. Based on the coordinate information provided at each position in the grid, it fills the pixel value at the corresponding position in the input into the position specified by the grid to obtain the final output. You can obtain the sample invocation code from "Calling Example" in [aclnnGridSampler2D](../../../image/grid_sample/docs/aclnnGridSampler2D.md), and name the code file **test\_grid\_sampler2\_d.cpp**.

- **CMakeLists File**

    The following is an example of the CMake file. Modify it based on the actual situation:

    ```CMake
    # Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set the project name
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set the compilation options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (for example, opapi_test) and specify the directory of the operator file *.cpp to be run
    add_executable(opapi_test
                   test_grid_sampler2_d.cpp)

    # Set ASCEND_PATH (CANN package directory; modify based on the actual path) and INCLUDE_BASE_DIR (header file directory)
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH "$ENV{ASCEND_CUSTOM_PATH}")
    elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set the library file path for linking
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libacl_rt.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_cv.so)

    # The executable is located in the bin directory under the build directory
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    Operators that integrate collective communication with MatMul computation and parallelism are collectively referred to as communication-computation fusion operators (MC2 operators for short), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, and others. When calling APIs of such operators, multi-threading and HCCL (Huawei Collective Communication Library) are typically involved. Therefore, the CMake file must additionally import the following content; otherwise, compilation will fail.

  ```CMake
  # Set the library file path for linking
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libacl_rt.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_cv.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library files that multithreading depends on
  ```

  Here, "find_package(Threads REQUIRED)" is a CMake command for locating the thread library, which automatically links the header files that the thread library depends on or the indirectly dependent library files.

## Compilation and Running

  1. Prepare the operator invocation code (\*.cpp) and the build script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the CANN runtime environment as the user and execute the following command to make the environment variables take effect.

        ```sh
        source ${INSTALL_DIR}/set_env.sh
        ```

      Where ${INSTALL_DIR} is the storage path of the CANN software after installation. Replace it with the actual path as needed.
  3. Compile and run.
        - Go to the directory where CMakeLists.txt is located and run the following command to create a build directory for storing the generated compilation files.

            ```sh
            mkdir -p build
            ```

        - Go to the build directory, run the cmake command to compile, and then run the make command to generate the executable file. `-DCMAKE_CXX_COMPILER=g++` specifies g++ as the C++ compiler, and `-DCMAKE_SKIP_RPATH=TRUE` disables RPATH embedding so the executable can run from any path.

          ```sh
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the executable file opapi_test is generated in the bin folder under the build directory.

        - Go to the bin directory and run the executable file opapi_test.

        ```sh
        cd bin
        ./opapi_test
        ```

        Taking the running result of the GridSample operator as an example, the result after running is as follows:

        ```output
        resultData[0] is: 0.250000
        resultData[1] is: 2.250000
        resultData[2] is: 2.000000
        resultData[3] is: 8.500000
        resultData[4] is: 20.500000
        resultData[5] is: 12.000000
        resultData[6] is: 8.250000
        resultData[7] is: 18.250000
        resultData[8] is: 10.000000
        ```

        If the execution result reports an error and the expected result does not appear, you can use the aclGetRecentErrMsg API to obtain the specific error information.
        The following is an example of obtaining exception information when calling aclnnGridSampler2DGetWorkspaceSize fails:

        ```cpp
        // input is nullptr
        ret = aclnnGridSampler2DGetWorkspaceSize(
        input, grid, interpolationMode, paddingMode, alignCorners, out, &  workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DGetWorkspaceSize   failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        The following is an example of obtaining error information for the null pointer issue described above:

        ```output
        aclnnGridSampler2DGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument input.
        ```
