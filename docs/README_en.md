# Documentation Center

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:11:54.296Z pushedAt=2026-11-18T09:59:29.700Z nl=2 -->

## Directory Structure

The `docs` directory structure is described as follows:

```text
├── zh
│    ├── context                            # Common documents, such as terminology and basic concepts
│    ├── debug                              # Operator debugging guide document
│    │   ├── op_debug_prof.md
│    │   ├── ...
│    ├── develop                            # Operator development guide document
│    │   ├── aicore_develop_guide.md
│    │   ├── aicpu_develop_guide.md
│    │   ├── ...
│    ├── figures                            # Image directory
│    ├── install                            # Environment installation and compilation guide document
│    │   ├── build.md
│    │   ├── compile.md
│    │   ├── quick_install.md
│    │   └── ...
│    ├── invocation                         # Operator call guide document (including aclnn call, graph mode call, etc.)
│    │   ├── quick_op_invocation.md
│    │   ├── ...
│    ├── menu_aclnn_api.md                  # Full aclnn API index file
│    ├── op_api_list.md                     # aclnn API list
│    ├── op_list.md                         # Full operator list
├── CONTRIBUTING_DOCS.md                 # Document contribution description
├── QUICKSTART.md                        # Quick start
└── README.md
```

## Advanced Tutorials

### Guide Documentation

| Documentation                                                         | Description                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Build from Source Code](en/install/compile.md)                        | Introduces different source code build methods and verification methods in online/offline scenarios.          |
| [Operator Call](en/invocation/quick_op_invocation.md)         | Introduces the method of calling operator examples and different operator call methods (such as PyTorch/aclnn/graph). |
| [Standard Operator Development](en/develop/aicore_develop_guide.md)       | Introduces how to define operator prototypes and implement Tiling and Kernel based on the standard project. Such operators are called "standard operators".<br>Standard operators support aclnn and graph mode calls. |
| [Simple Operator Development](../examples/fast_kernel_launch_example/README.md) | Introduces how to implement fast_kernel_launch based on the simple project, that is, the `<<<>>>` method. Such operators are called "simple operators".<br>Simple operators support only PyTorch calls. |
| [Operator Debugging and Tuning](en/debug/op_debug_prof.md)                    | Introduces common operator function debugging and performance tuning methods (such as data collection and simulation pipeline). |

### API Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Operator List](en/op_list.md)                        | Introduces the list of all operators included in the project.                                 |
| [aclnn List](en/op_api_list.md)                   | Introduces the list of all operator aclnn APIs included in the project. To facilitate calling operators on the Host side, C language APIs, that is, APIs with the aclnn prefix, are provided. |

### Tool Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Simulator Simulation Tool](en/debug/npu_sim.md) | A SoC-level simulation tool for operator development scenarios, used to analyze the precision and performance data of AI tasks running on the AI simulator at each stage. |

### More Documentation

- Sample documents: Refer to the operator samples in the [cann-samples](https://gitcode.com/cann/cann-samples) repository.
- Practice documents: **Under construction**. Please visit the project wiki to perform a search.
- FAQ documents: **Under construction**. Please visit the project wiki to perform a search.

## Appendix

| Documentation                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](en/context/basic_concept.md) | Introduces basic concepts and terminology in the operator domain, such as quantization/sparsity, data types, and data formats. |
| [build Parameter Description](en/install/build.md)   | Introduces the functions and values of the build.sh parameters in this project, including source code compilation, operator call, and debugging. |
