# Introduction

English | [简体中文](./README.md)

> Note:
>
>- Currently, most operators in the operator library run on the AI Core, and a few operators run on the AI CPU. By default, operators mentioned in this project refer to AI Core operators.
>- For the introduction to AI Core and AI CPU, refer to "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples for AI Core operators and AI CPU operators. Developers can refer to the corresponding implementations based on actual situations.

## Directory Description

```Cpp
├── examples
│   ├── add_example                # AI Core operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Retain the original file
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Operator information library, Tiling, and InferShape implementations
│   │   └── op_kernel              # Operator kernel directory
│   ├── add_example_aicpu          # AI CPU operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Retain the original file
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Operator information library and InferShape implementations
│   │   └── op_kernel_aicpu        # Operator kernel directory
│   ├── CMakeLists.txt             # Operator compilation configuration file. Retain the original file
│   └── README.md                  # Operator description document

```

## Operator Development Samples

|    Sample directory    |    Description    |    Operator development    |    Operator invocation    |
|---|------------------|---|---|
|  add_example|An operator that adds two tensors.  |  For the end-to-end operator development process, refer to the [AI Core Operator Development Guide](../docs/zh/develop/aicore_develop_guide.md)  |  For the invocation, refer to [README](add_example/README_en.md)  |
|  add_example_aicpu  |  An operator that adds two tensors.  |  For the end-to-end operator development process, refer to the [AI CPU Operator Development Guide](../docs/zh/develop/aicpu_develop_guide.md)  |  For the invocation, refer to [README](add_example_aicpu/README_en.md)  |
|  fast_kernel_launch_example  |  A sample for fast end-to-end operator development in the PyTorch scenario.  |  For the end-to-end operator development process, refer to the [PyTorch Operator Quick Development Guide](./fast_kernel_launch_example/README_en.md)  |  For the invocation, refer to [README](fast_kernel_launch_example/README_en.md)  |
