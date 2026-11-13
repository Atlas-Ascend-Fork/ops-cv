# 编译与运行样例

## 前提说明

- 如需编译执行算子API，请确保基础环境已搭建完成，包括驱动、固件、CANN软件包、ops包等。
- 算子API的调用流程和编译运行操作详情请参见《[应用开发（C&C++）](https://hiascend.com/document/redirect/CannCommunityadevguide)》中“单算子调用>单算子API执行>调用aclnn接口示例代码”。

## 编译前准备

本章以开发和运行环境合设场景为例，即带AI处理器的机器既作为开发环境又作为运行环境。该场景下，代码开发和代码运行在同一台机器上。这里以**GridSample算子**为例，其他算子的调用逻辑、流程、编译脚本与GridSample算子大致一样，请根据实际情况自行修改API调用脚本（\*.cpp）和编译脚本（CMakeLists）。

- **示例代码**

   已知GridSample算子功能是提供一个输入Tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。您可以从[aclnnGridSampler2D](../../../image/grid_sample/docs/aclnnGridSampler2D.md)中“调用示例”获取示例代码，并将代码文件命名为“**test\_grid\_sampler2\_d.cpp**”。

- **CMakeLists文件**

    CMake文件示例如下，请根据实际情况修改：

    ```CMake
    # Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # 设置工程名
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # 设置编译选项
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
    add_executable(opapi_test
                   test_grid_sampler2_d.cpp)

    # 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # 设置链接的库文件路径
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libacl_rt.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_cv.so)

    # 可执行文件在CMakeLists文件所在目录的bin目录下
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    对于集合通信和MatMul计算融合、并行的算子，统称为通算融合算子（简称MC2算子），包括AllGatherMatmul、AlltoAllAllGatherBatchMatMul、BatchMatMulReduceScatterAlltoAll、MatmulAllReduce、MatmulAllReduceAddRmsNorm、MatmulReduceScatter等。调用该类算子API时，一般会涉及多线程和HCCL（Huawei Collective Communication Library，集合通信库），因此CMake文件需要额外导入如下内容，否则无法成功编译。

  ```CMake
  # 设置链接的库文件路径
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libacl_rt.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_cv.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # 集合通信库文件
                        ${CMAKE_THREAD_LIBS_INIT})           # 多线程依赖的库文件
  ```

  其中“find_package(Threads REQUIRED)”是CMake用于查找线程库的命令，可自动链接线程库依赖的头文件或间接依赖的库文件。

## 编译与运行

  1. 提前准备好算子的调用代码（\*.cpp）和编译脚本（CMakeLists.txt）。
  2. 配置环境变量。

    安装CANN软件后，使用CANN运行用户登录环境，执行如下命令使环境变量生效：

        ```sh
        source ${INSTALL_DIR}/set_env.sh
        ```

     其中${INSTALL_DIR}为CANN软件安装后文件存储路径，请根据实际情况替换。
  3. 编译并运行。
        - 进入CMakeLists.txt所在目录，执行如下命令，新建build目录存放生成的编译文件：

            ```sh
            mkdir -p build
            ```

        - 进入build目录，执行cmake命令编译，再执行make命令生成可执行文件。

          ```sh
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          编译成功后，会在build目录的bin文件夹下生成opapi\_test可执行文件。

        - 进入bin目录，运行可执行文件opapi\_test。

        ```sh
        cd bin
        ./opapi_test
        ```

        以GridSample算子的运行结果为例，运行后的结果示例如下：

        ```sh
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

        若执行结果报错，未出现预期结果，可以使用aclGetRecentErrMsg接口获取报错的具体信息。
        调用aclnnGridSampler2DGetWorkspaceSize报错获取异常信息示例如下：

        ```cpp
        // input is nullptr
        ret = aclnnGridSampler2DGetWorkspaceSize(
        input, grid, interpolationMode, paddingMode, alignCorners, out, &  workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DGetWorkspaceSize   failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        上述构造空指针问题获取报错信息示例如下：

        ```sh
        aclnnGridSampler2DGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument input.
        ```
