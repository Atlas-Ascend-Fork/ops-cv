# Contribution Guide

This project welcomes all developers to experience and contribute to it. Before participating in community contributions, see [cann-community](https://gitcode.com/cann/community) to learn about the code of conduct, sign the CLA, and familiarize yourself with the contribution process of the source code repository.

## Preparations

Developers need to pay attention to the following matters when preparing local code and submitting PRs:

1. When submitting a PR, carefully fill in the service background, purpose, and solution of the PR according to the PR template.
2. If your modification involves adding features, adding interfaces, adding configuration parameters, or modifying the code flow (rather than a simple bug fix), discuss the solution through an Issue first to prevent the code from being rejected. If you are not sure whether the modification is a "simple bug fix", submit an Issue for discussion.

## Contribution Scenarios

Developer contribution scenarios mainly include:

- [Contributing a new operator](#1-contributing-a-new-operator)
- [Operator bug fixes](#2-operator-bug-fixes)
- [Operator optimization](#3-operator-optimization)
- [Documentation fixes](#4-documentation-fixes)
- [Helping resolve issues raised by others](#5-helping-resolve-issues-raised-by-others)

---

## 1. Contributing a New Operator

The operator development contribution process is as follows:

![Operator development contribution process](./docs/en/figures/op_dev_and_contribute_flow.png "Operator development and contribution flowchart")

If you have a brand-new operator and want to design and implement it based on the NPU, you are welcome to propose your ideas and design in an Issue. The complete contribution process is as follows:

### 1. Create a Requirement Issue

Create an Issue of the `Requirement|需求建议` type and describe the design of the new operator. The Issue generally needs to include the following content:

- **Background information**
- **Value/function**
- **Design solution**

For specific operation steps, refer to the [Issue Operation Guide](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md).

### 2. Requirement Review

After you create a requirement Issue, apply for a SIG topic and attend the ops-basic SIG regular meeting for review at the specified time.

- [Meeting time](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md#社区运作)
- [Topic application](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-basic)
- [Meeting link](https://meeting.osinfra.cn/cann/)

**Urgent requirement handling:**

If a requirement is urgent, you can apply for a temporary SIG meeting:

1. Fill in the topic application: [Topic application](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-basic)
2. Send an email to the [maintainer](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md). Use "application for a temporary SIG topic" plus the topic content as the suggested email subject.

**Requirement acceptance:**

If the requirement is accepted, [SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/sig-info.yaml) will assign a proper operator classification path (for example, `experimental/image`) to you. Submit the operator to the corresponding operator classification directory under `experimental`.

### 3. PR Submission

#### Minimum Deliverables of an Ecosystem Operator

```text
${op_class}                    # Operator classification
├── ${op_name}                 # Operator name
│   ├── ${op_name}.cpp         # Operator kernel implementation file
│   ├── tests                  # Test directory
│   │   └── test_${op_name}.py # Operator test file
│   ├── CMakeLists.txt         # Operator compilation configuration file
│   └── README.md              # Operator README document
```

#### Notes on PR Submission

**Preparations before submission:**

- It is recommended that you submit the PR after the requirement review is passed.
- Before submitting a PR, prepare the development environment, and carefully learn about the project-specific development specifications and copyright statement requirements (if open-source code snippets are referenced, refer to the snippet reference guide). Ensure that your contribution complies with the project standards and that you have signed the CLA. For specific operation steps, refer to the [PR Operation Guide](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md).

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator kernel implementation and operator test files | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README_en.md) |
| Documentation deliverables | The operator README document is required. Other documents are provided as needed | [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md) |
| Precision requirements | Newly contributed operators must meet the precision standard | [Open-Source Precision Standard for Ecosystem Operators](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |

**Compliance checks:**

- [ ] Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [ ] Whether the code can be compiled
- [ ] Whether the Markdown document syntax complies with the specifications

**Submission specifications:**

- **Contribution directory**: Submit the code to the specified directory `experimental/${op_class}` as advised by SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Submit a PR to the target branch using `git` commands. Check whether the PR title is clear and whether the PR description is standardized (indicating the change content and reason, and whether the related Issue is associated).

> **Note**: If you want to contribute a project-standard operator, its deliverables and development process are more complex than those of an ecosystem operator, including the kernel and Tiling implementations. For the detailed contribution guide, refer to the [Appendix](#appendix).

### 4. CI Gate

Trigger the open-source repository gate by commenting with the `compile` command, and make modifications based on the CI detection results. Currently, the CI gate includes the following checks:

- Code compilation
- Static check (if codecheck false positives are involved, submit them to SIG members for masking)
- UT tests
- Smoke tests

After the gate is passed, @ Committer in the associated Issue.

### 5. Committer Review

The Committer provides review comments after the review. Modify the code based on the comments and @ Committer after the modification.

### 6. Maintainer Merge

After the Committer review is passed, the `/lgtm` label is added. The Maintainer performs the final review. After confirming that everything is correct, the Maintainer adds the `/approve` label and merges the PR.

---

## 2. Operator Bug Fixes

If you find operator bugs in this project and want to fix them, you are welcome to create an Issue for feedback and tracking.

You can follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create an Issue of the `Bug-Report|缺陷反馈` type to describe the bug.

---

## 3. Operator Optimization

If you have ideas about generalization enhancement or performance optimization of some operator implementations in this project and want to implement the optimization points, you are welcome to contribute operator optimizations.

You can follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create an Issue of the `Requirement|需求建议` type to describe the optimization points and provide your design solution.

---

## 4. Documentation Fixes

If you find description errors in some operator documents in this project, you are welcome to create an Issue for feedback and fixes. For the documentation specifications, refer to the [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md).

You can follow the [Submit Issue/Handle Issue Tasks](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create an Issue of the `Documentation|文档反馈` type to point out the problems in the corresponding document.

---

## 5. Helping Resolve Issues Raised by Others

If you have a proper solution to a problem raised by others in the community, you are welcome to post comments in the Issue for communication, help others solve problems and pain points, and jointly improve usability.

If the Issue requires code modification, you can enter `/assign` or `/assign @yourself` in the Issue comment box to assign the Issue to yourself and track and help resolve the problem.

---

## Appendix

### Deliverables of a Project-Standard Operator

```text
${op_class}                           # Operator classification
├── ${op_name}                        # Operator name
│   ├── op_host                       # Operator definition and Tiling-related implementations
│   │   ├── ${op_name}_def.cpp        # Operator definition file
│   │   ├── ${op_name}_tiling.cpp     # Operator Tiling implementation file
│   │   ├── ${op_name}_tiling_${sub_case}.cpp # Optional. Tiling implementation of the sub-scenario. ${sub_case} indicates the sub-scenario (for example, arch35)
│   │   ├── ${op_name}_tiling_${sub_case}.h   # Optional. Header file of the Tiling implementation of the sub-scenario
│   │   └── CMakeLists.txt
│   ├── op_kernel                     # Operator kernel directory
│   │   ├── ${op_name}.cpp            # Kernel entry file, containing the main function and scheduling logic
│   │   ├── ${op_name}.h              # Kernel implementation file, defining the kernel header file, including function descriptions, structure definitions, and logic implementation
│   │   ├── ${op_name}_tiling_data.h  # TilingData file, storing configuration information related to the Tiling strategy
│   │   └── ${op_name}_tiling_key.h   # TilingKey file, defining the key of the Tiling strategy to identify different partition modes
│   ├── CMakeLists.txt                # Operator compilation configuration file. Retain the original file
│   ├── README.md                     # Operator description document
│   └── tests                         # Operator test files
│       └── ut                        # Operator UT test files
```

For the complete operator directory structure (including optional deliverables), refer to [Project Directory Structure](docs/zh/install/dir_structure.md#项目目录).

> **Description**: The Tiling implementation files that participate in compilation in the op_host directory must contain the `_tiling` identifier in the file names (for example, `${op_name}_tiling.cpp` and `${op_name}_tiling_${sub_case}.cpp`). Otherwise, the compilation system does not recognize the files. Follow this naming rule when splitting Tiling implementations for sub-scenarios (such as a specific architecture arch35).

### PR Merge Requirements

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator Tiling implementation in op_host, the operator kernel implementation in op_kernel, and operator UT test files | [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md) |
| Documentation deliverables | The operator README document is required. Other documents are provided as needed | [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md) |

**Compliance checks:**

- [ ] Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [ ] Whether the code complies with the basic programming specifications of standard operators
- [ ] Whether the code can be compiled
- [ ] Whether the Markdown document syntax complies with the specifications

**Submission specifications:**

- **Contribution directory**: Submit the code to the specified directory `experimental/${op_class}` as advised by SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Submit a PR to the target branch using `git` commands. Check whether the PR title is clear and whether the PR description is standardized (indicating the change content and reason, whether the related Issue is associated, and whether the CLA is signed).
