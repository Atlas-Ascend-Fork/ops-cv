#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Library, binary, and package build functions.

build_static_lib() {
  echo $dotted_line
  echo "Start to build static lib."

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  local all_targets
  if ! all_targets=$(cmake --build . --target help); then
    echo "[ERROR] Failed to query CMake build targets." >&2
    exit 1
  fi
  rm -fr "${BUILD_PATH}/bin_tmp"
  mkdir -p "${BUILD_PATH}/bin_tmp"
  if grep -wq "ophost_cv_static" <<< "${all_targets}"; then
    cmake --build . --target ophost_cv_static -- ${VERBOSE} -j $THREAD_NUM
  fi

  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  if grep -wq "opapi_cv_static" <<< "${all_targets}"; then
    cmake --build . --target opapi_cv_static -- ${VERBOSE} -j $THREAD_NUM
  fi
  local jit_command=""
  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    jit_command="-j"
  fi
  for unit in "${UNITS[@]}"; do
    rm -fr "${BUILD_PATH}/bin_tmp/${unit}"
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" GenStaticOpResourceIni -s ${unit} -b "${BUILD_PATH}" ${jit_command}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" StaticCompile -s ${unit} -b "${BUILD_PATH}" -n=0 -a=${ARCH_INFO} ${jit_command}
  done
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  if grep -wq "cann_cv_static" <<< "${all_targets}"; then
    cmake --build . --target cann_cv_static -- ${VERBOSE} -j $THREAD_NUM
  fi
  echo "Build static lib success!"
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} -UENABLE_STATIC ..

  for lib in "${BUILD_LIBS[@]}"; do
    echo "Building target ${lib}"
    cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
  done

  echo $dotted_line
  echo "Build libs ${BUILD_LIBS[@]} success"
  echo $dotted_line
}

build_binary() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi

  echo $dotted_line
  echo "Start to build binary"

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake .. ${CMAKE_ARGS}

  echo "--------------- prepare build start ---------------"
  local all_targets
  if ! all_targets=$(cmake --build . --target help); then
    echo "[ERROR] Failed to query CMake build targets." >&2
    exit 1
  fi
  if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
    cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM || {
      echo "[ERROR] Failed to execute ascendc_impl_gen."
      exit 1;
    }
  else
    echo "[WARNING] Build target 'ascendc_impl_gen' not found in cmake targets, available targets: ${all_targets}"
  fi

  if echo "${all_targets}" | grep -wq "gen_bin_scripts"; then
    cmake --build . --target gen_bin_scripts -- ${VERBOSE} -j $THREAD_NUM || {
      echo "[ERROR] Failed to execute gen_bin_scripts."
      exit 1;
    }
  else
    echo "[WARNING] Build target 'gen_bin_scripts' not found in cmake targets, available targets: ${all_targets}"
  fi
  echo "--------------- prepare build end ---------------"

  echo "--------------- binary build start ---------------"
  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  for unit in "${UNITS[@]}"; do
    rm -rf "${BUILD_PATH}/binary/${unit}/bin/opc_cmd"
    if grep -wq "prepare_binary_compile_${unit}" <<< "${all_targets}"; then
      cmake --build . --target prepare_binary_compile_${unit} -- ${VERBOSE} -j 1 || {
        print_error "opc gen failed!" && exit 1
      }
    fi
    OPC_CMD_FILE="${BUILD_PATH}/binary/${unit}/bin/opc_cmd/opc_cmd.sh"
    [[ -f "$OPC_CMD_FILE" ]] && opc_list_num=$(wc -l < "$OPC_CMD_FILE") || opc_list_num=0
    CMAKE_ARGS="${CMAKE_ARGS} -DOPC_NUM_${unit}=${opc_list_num}"
  done
  cd "$BUILD_PATH" && cmake .. ${CMAKE_ARGS}

  local cur_path=$(pwd)
  mkdir -p "${cur_path}/op_impl/ai_core/tbe/op_tiling"
  if [ ! -L op_impl/ai_core/tbe/op_tiling/liboptiling.so ]; then
    if [ -e "${cur_path}/libophost_cv.so" ]; then
      ln -s "${cur_path}/libophost_cv.so" op_impl/ai_core/tbe/op_tiling/liboptiling.so
    else
      cmake --build . --target ophost_cv -- ${VERBOSE} -j $THREAD_NUM
      ln -s "${cur_path}/libophost_cv.so" op_impl/ai_core/tbe/op_tiling/liboptiling.so
    fi
  fi
  export ASCEND_CUSTOM_OPP_PATH="${cur_path}"
  if echo "${all_targets}" | grep -wq "binary"; then
    cmake --build . --target binary -- ${VERBOSE} -j $THREAD_NUM || {
      echo "[ERROR] Kernel compile failed!" && exit 1
    }
  else
    echo "[WARNING] Compile kernel 'binary' failed! Build target 'binary' not found in cmake targets. Available targets: ${all_targets}"
  fi
  if echo "${all_targets}" | grep -wq "gen_bin_info_config"; then
    cmake --build . --target gen_bin_info_config -- ${VERBOSE} -j $THREAD_NUM || { exit 1; }
  else
    echo "[WARNING] Generate 'gen_bin_info_config' failed! Build target 'gen_bin_info_config' not found in cmake targets. Available targets: ${all_targets}"
  fi
  echo "--------------- binary build end ---------------"

  echo "Build binary success"
  echo $dotted_line
}

build_package() {
  echo "--------------- build package start ---------------"
  clean_build_out

  local all_targets
  if ! all_targets=$(cmake --build . --target help); then
    echo "[ERROR] Failed to query CMake build targets." >&2
    exit 1
  fi
  if [[ "$ENABLE_BINARY" != "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    # gen impl python files
    if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
      cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM || { exit 1; }
    fi
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  if echo "${all_targets}" | grep -wq "build_es_cv"; then
    cmake --build . --target build_es_cv -- ${VERBOSE} -j $THREAD_NUM || { echo "[ERROR] target:build_es_cv compile failed!" && exit 1; }
  fi
  clean_rpm_deb_package
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM || {
    echo "[ERROR] target:package build failed!"
    exit 1
  }
  collect_rpm_deb_package
  echo "--------------- build package end ---------------"
}

find_rpm_deb_package() {
  if [[ "$PACKAGE_TYPE" == "run" ]]; then
    return 0
  fi

  find "${BUILD_PATH}" -type f -name "cann-ops-cv*.${PACKAGE_TYPE}" | sort
}

clean_rpm_deb_package() {
  if [[ "$PACKAGE_TYPE" == "run" ]]; then
    return 0
  fi

  local package_files=()
  while IFS= read -r package_file; do
    package_files+=("${package_file}")
  done < <(find_rpm_deb_package)

  if [[ ${#package_files[@]} -eq 0 ]]; then
    return 0
  fi

  for package_file in "${package_files[@]}"; do
    rm -f "${package_file}"
    echo "[INFO] Removed stale package artifact: ${package_file}"
  done
}

collect_rpm_deb_package() {
  if [[ "$PACKAGE_TYPE" == "run" ]]; then
    return 0
  fi

  local package_files=()
  while IFS= read -r package_file; do
    package_files+=("${package_file}")
  done < <(find_rpm_deb_package)

  for package_file in "${package_files[@]}"; do
    cp -f "${package_file}" "${BUILD_OUT_PATH}/"
    echo "[INFO] Package artifact copied to ${BUILD_OUT_PATH}/$(basename "${package_file}")"
  done
}

package_static() {
    # Check weather BUILD_OUT_PATH directory exists
    if [ ! -d "$BUILD_OUT_PATH" ]; then
        echo "Error: Directory $BUILD_OUT_PATH does not exist"
        return 1
    fi

    # Check weather *.run is exists and verify the file numbers
    local run_files=()
    shopt -s nullglob
    run_files=("$BUILD_OUT_PATH"/*.run)
    shopt -u nullglob
    if [ ${#run_files[@]} -eq 0 ]; then
        echo "Error: No .run files found in $BUILD_OUT_PATH directory"
        return 1
    fi
    if [ ${#run_files[@]} -gt 1 ]; then
        echo "Error: Multiple .run files found in $BUILD_OUT_PATH directory:"
        printf '%s\n' "${run_files[@]}"
        return 1
    fi

    # Get filename of *.run file and set new directory name
    local run_file=$(basename "${run_files[0]}")
    if [[ "$run_file" != *"ops-cv"* ]]; then
        echo "Error: Filename '$run_file' does not contain 'ops-cv'"
        return 1
    fi
    local new_name="${run_file/ops-cv/ops-cv-static}"
    new_name="${new_name%.run}"

    # Check weather $BUILD_PATH/static_library_files directory exists and not empty
    local static_files_dir="$BUILD_PATH/static_library_files"
    if [ ! -d "$static_files_dir" ]; then
        return 0
    fi
    if [ -z "$(ls -A "$static_files_dir")" ]; then
        echo "Error: Directory $static_files_dir is empty"
        return 1
    fi

    # Rename directory
    local new_dir_path="$BUILD_PATH/$new_name"
    if mv "$static_files_dir" "$new_dir_path"; then
        echo "Preparing for packaging: renamed $static_files_dir to $new_dir_path"
    else
        echo "Packaging preparation failed: directory rename failed ($static_files_dir -> $new_dir_path)"
        return 1
    fi

    # Create compressed package and restore directory name
    local new_filename="${new_name}.tar.gz"
    if tar -czf "$BUILD_OUT_PATH/$new_filename" -C "$BUILD_PATH" "$new_name"; then
        echo "[SUCCESS] Build static lib success!"
        echo "Successfully created compressed package: $BUILD_OUT_PATH/$new_filename"
        # Restore original directory name
        echo "Restoring original directory name: $new_dir_path -> $static_files_dir"
        mv "$new_dir_path" "$static_files_dir"
        return 0
    else
        echo "Error: Failed to create compressed package"
        # Attempt to restore original directory name
        mv "$new_dir_path" "$static_files_dir"
        return 1
    fi
}
