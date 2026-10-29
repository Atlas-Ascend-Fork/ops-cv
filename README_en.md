<div align="center">

# ops-cv

English | [简体中文](./README.md)

[![CANNcommunity](https://img.shields.io/badge/docs-Community_Operator_List-brightgreen.svg?style=flat)](https://www.hiascend.com/document/redirect/CannCommunityOplist)
[![repo](https://img.shields.io/badge/docs-Project_Documentation-green.svg?style=flat)](docs)
[![license](https://img.shields.io/badge/license-CANN_Open_2.0-blue.svg)](LICENSE)
[![contributing](https://img.shields.io/badge/Contribution_Guide-teal)](CONTRIBUTING_en.md)
[![SIG](https://img.shields.io/badge/SIG-ops_basic-yellow)](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/ops-cv)

</div>

---

## 🔥Latest News

- [2026/01] Added [QuickStart](docs/QUICKSTART.md), which guides beginners to get started with operator project deployment (Docker environments supported), operator development, and the contribution process from zero.
- [2025/12] Open-source operators support Ascend 950PR&950DT products and Kirin X90 processor series products. You can develop and debug operators using the [NPU Simulator](docs/zh/debug/npu_sim.md) simulation tool. Guide documents were optimized to focus on the [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md), clarify the minimum deliverables and key example code, and provide guidance for migrating operators in the Ascend/samples repository to this project. Plugin support for ONNX operators was added.
- [2025/11] The [three_interpolate_backward](image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward_en.md) operator adds the deterministic computation feature. [opgen](docs/zh/develop/aicore_develop_guide.md#工程创建) was added to support automatic generation of operator projects.
- [2025/10] The experimental directory was added, and the [Contribution Guide](CONTRIBUTING_en.md) was improved. Developers can debug and contribute custom operators.
- [2025/09] The ops-cv project was released for the first time. Open-source operators support Atlas A2 products and Atlas A3 products.

## 🚀Overview

ops-cv is a high-level operator library in the [CANN](https://www.hiascend.com/cann) (Compute Architecture for Neural Networks) operator library. It provides image processing, object detection, and other capabilities. It includes image and objdetect operators and covers common image processing operations. The position of the sub-library in the architecture is shown below.

<img src="docs/en/figures/architecture.png" alt="Architecture diagram"  width="700px" height="326px">

## 📌Version Compatibility

The source code of this project is released with CANN software versions. For the mapping between CANN software versions and the tags of this project, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure smooth custom development based on the source code, select the compatible CANN version and the Gitcode tag source code. Using the master branch may cause version mismatch risks.

## 🛠️Environment Preparation

[Environment deployment](docs/zh/install/quick_install.md) is the prerequisite for experiencing the capabilities of this project. Complete the installation of the NPU driver, CANN package, and so on first to ensure that the environment is normal.

## ⬇️Source Code Download

After the environment is ready, download the source code of the branch compatible with the CANN version. The command is as follows. Replace \$\{tag\_version\} with the branch tag name.

 > Note: If the source code of the compatible branch already exists in the environment, **you can skip this step**. For example, CANNLab provides the source code corresponding to the latest CANN version by default.

```bash
# General command: git clone -b ${tag_version} https://gitcode.com/cann/ops-cv.git
git clone -b ${tag_version} https://gitcode.com/cann/ops-cv.git
```

## 📖Tutorials

- [Quick Start](docs/QUICKSTART.md): Quickly experience the core basic capabilities of the project from zero, covering source code compilation, operator invocation, development, and debugging.
- [Advanced Tutorials](docs/README.md): For more information about project compilation and deployment, operator invocation, development, debugging, and tuning, refer to the documentation center for detailed guidance.

## 💬Related Information

- [Directory Structure](docs/zh/install/dir_structure.md)
- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

---

PS: The features and documentation of this project are continuously updated and improved. Stay tuned for the latest version.

- **Issue feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/ops-cv/issues).
- **Community interaction**: Join discussions through GitCode [Discussions](https://gitcode.com/cann/ops-cv/discussions).
- **Technical column**: Get technical articles through GitCode [Wiki](https://gitcode.com/cann/ops-cv/wiki).
