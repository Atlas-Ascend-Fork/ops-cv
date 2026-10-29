# Security Statement

## Recommended Running User

For security reasons, do not run any command as an administrator account such as root. Follow the principle of least privilege.

## File Permission Control

- It is recommended that you set the system umask value to 0027 or higher on the host (including the host machine) and in the container. This ensures that a newly created folder has a default maximum permission of 750 and a newly created file has a default maximum permission of 640.
- It is recommended that you take security measures such as permission control for sensitive content, including personal privacy data, commercial assets, source files, and various files saved during operator development. For example, when controlling the permissions of the installation directory of this project or the permissions of public input data files, refer to [A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios](#a-recommended-maximum-permissions-for-files-folders-in-different-scenarios) for the recommended permission settings.
- When operators run, operator compilation files may be cached in the `kernel_meta_*` folder under the running directory to speed up subsequent operator invocations. You can control the permissions of the generated files as needed.
- Perform proper permission control during installation and usage. For the settings, refer to the file permission reference in [A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios](#a-recommended-maximum-permissions-for-files-folders-in-different-scenarios).

## Build Security Statement

When you compile and install this project from source code, you need to perform the compilation yourself. The compilation process generates some intermediate files. After the compilation is complete, control the permissions of the intermediate files to ensure file security.

## Running Security Statement

- Write operator invocation scripts based on the resource status of the running environment. If an operator invocation script does not match the resource status, errors may occur and the process may exit unexpectedly. For example, the space used for generating input data or benchmark computation results exceeds the memory capacity limit, or the data saved locally by the script exceeds the disk space.
- When an operator runs abnormally, it exits the process and prints error information. Locate the specific cause of the error based on the error message, for example, setting the operator to run synchronously or viewing log files.
- When an operator is invoked through [PyTorch](https://gitcode.com/Ascend/pytorch), running errors may occur due to version mismatch. For details, refer to the [PyTorch security statement](https://gitcode.com/Ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network address statement contained in the code of this project is as follows:

|      Type      |                                           Open-source code address                                           |                            File name                             |             Public IP address/Public URL/Domain name/Email address/Compressed file address             |                   Purpose                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | N/A  | scripts/tools/third_lib_download.py | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Download the makeself source code from gitcode. It serves as a compilation dependency |
|  Dependency  | N/A  | scripts/tools/third_lib_download.py | [json-3.11.3.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz) | Download the json source code from OBS. It serves as a compilation dependency |
|  Dependency  | N/A  | scripts/tools/third_lib_download.py | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | Download the googletest source code from gitcode. It serves as a compilation dependency |
|  Dependency  | N/A  | scripts/tools/third_lib_download.py | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) | Download the eigen source code from gitcode. It serves as a compilation dependency |
|  Dependency  | N/A  | cmake/fetch_cann_cmake.cmake | [cmake-master-044.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-044.tar.gz) | Download the cann-cmake source code from OBS. It serves as a compilation dependency |

---

## Vulnerability Mechanism Statement

[Vulnerability management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Recommended Maximum Permissions for Files (Folders) in Different Scenarios

| Type               | Maximum recommended Linux permission |
|------------------| ---------------  |
| User home directory            |   750 (rwxr-x---)            |
| Program files (including script files, library files, and so on) |   550 (r-xr-x---)             |
| Program file directory           |   550 (r-xr-x---)            |
| Configuration file             |  640 (rw-r-----)             |
| Configuration file directory           |   750 (rwxr-x---)            |
| Log file (recording completed or archived) |  440 (r--r-----)             |
| Log file (being recorded)      |    640 (rw-r-----)           |
| Log file directory           |   750 (rwxr-x---)            |
| Debug file          |  640 (rw-r-----)         |
| Debug file directory        |   750 (rwxr-x---)  |
| Temporary file directory           |   750 (rwxr-x---)   |
| Maintenance and upgrade file directory         |   770 (rwxrwx---)    |
| Service data file           |   640 (rw-r-----)    |
| Service data file directory         |   750 (rwxr-x---)      |
| Directory of key components, private keys, certificates, and ciphertext files |  700 (rwx------)      |
| Key components, private keys, certificates, and encrypted ciphertext  | 600 (rw-------)      |
| Encryption/decryption interfaces and encryption/decryption scripts      |   500 (r-x------)        |
