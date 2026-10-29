#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""GoodFeaturesToTrack 算子 vs OpenCV cv::goodFeaturesToTrack 的可复现对拍。

用例（输入与 examples/test_aclnn_good_features_to_track.cpp 中同名用例逐位一致）：
  - squares64  : fp32 64x64，4 个 8x8 亮方块（255 on 0），maxCorners=1000
  - checker256 : fp32 256x256 棋盘格 cell=8，maxCorners=20000

数据文件（golden/ 目录）：
  golden/ours_corners.json    算子真机输出角点（由 examples 的 CORNERS 日志导入）
  golden/opencv_golden.json   OpenCV 参考角点 + 环境/参数记录（--write-golden 生成）

判定口径：
  1) 几何真值（可闭式推导的角点）必须被算子与 OpenCV 双方完整、精确命中；
  2) OpenCV 找到的每个角点，算子输出中都必须有 1px 内的对应点（不允许漏检）；
  3) 真值以外的额外角点只做报告、不硬判：阈值 / 局部极大判据 / 并列响应取舍不同所致，
     其中落在边界窄带的部分对应 Sobel 边界延拓差异（见 README）。

用法：
  python3 run_crosscheck.py --import-dump /tmp/gft_ex.log   # 从 examples 日志导入算子输出
  python3 run_crosscheck.py                 # 需要 opencv-python：实时重算参考并比对
  python3 run_crosscheck.py --no-opencv     # 无 OpenCV：只与已提交的 golden 比对
  python3 run_crosscheck.py --write-golden  # 刷新 golden/opencv_golden.json
退出码：0 = 通过，1 = 不通过。
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(HERE, "golden")
OURS_PATH = os.path.join(DATA_DIR, "ours_corners.json")
GOLDEN_PATH = os.path.join(DATA_DIR, "opencv_golden.json")

# 与 examples 中 Params 默认值一致
OPENCV_PARAMS = {
    "maxCorners": 1000,
    "qualityLevel": 0.01,
    "minDistance": 1.0,
    "blockSize": 3,
    "useHarrisDetector": False,
    "k": 0.04,
}

CASES = {
    "fp32 方块 blockSize=3": {
        "shape": (64, 64),
        "params": dict(OPENCV_PARAMS),
    },
    "fp32 棋盘格 256x256": {
        "shape": (256, 256),
        "params": dict(OPENCV_PARAMS, maxCorners=20000),
    },
}


# ---------------------------------------------------------------------------
# 输入重建（与 examples 的 GenSyntheticImage / GenCheckerboard 逐位一致）
# ---------------------------------------------------------------------------
def gen_squares(h, w, squares, sq, on=255.0):
    img = np.zeros((h, w), np.float32)
    for y0, x0 in squares:
        y1 = min(y0 + sq, h)
        x1 = min(x0 + sq, w)
        img[y0:y1, x0:x1] = on
    return img


def gen_checkerboard(h, w, cell):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(
        (((yy // cell) + (xx // cell)) & 1) != 0, np.float32(255.0), np.float32(0.0)
    )


def build_case_image(name):
    if name == "fp32 方块 blockSize=3":
        return gen_squares(64, 64, [(4, 4), (4, 40), (40, 4), (40, 40)], 8)
    if name == "fp32 棋盘格 256x256":
        return gen_checkerboard(256, 256, 8)
    raise KeyError(name)


def case_truth(name):
    """几何真值：(x, y) 数组。方块图 = 4 个方块各自的 4 个外角；棋盘格 = 内部格点。"""
    if name == "fp32 方块 blockSize=3":
        pts = []
        for y0, x0 in [(4, 4), (4, 40), (40, 4), (40, 40)]:
            for dy in (0, 7):
                for dx in (0, 7):
                    pts.append((x0 + dx, y0 + dy))
        return np.array(sorted(pts), np.int64)
    if name == "fp32 棋盘格 256x256":
        # cell=8 的棋盘格，格点落在 8k-1 处；去掉贴边的一圈（x/y = 255 或 7 之外）
        ks = np.arange(1, 256 // 8)  # 1..31 -> 坐标 7,15,...,247
        pts = [(int(x), int(y)) for y in (ks * 8 - 1) for x in (ks * 8 - 1)]
        return np.array(sorted(pts), np.int64)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# OpenCV 参考
# ---------------------------------------------------------------------------
def opencv_ref(img, params):
    import cv2  # 延迟导入，--no-opencv 时不依赖

    cvp = cv2.goodFeaturesToTrack(
        img,
        maxCorners=params["maxCorners"],
        qualityLevel=params["qualityLevel"],
        minDistance=params["minDistance"],
        blockSize=params["blockSize"],
        useHarrisDetector=params["useHarrisDetector"],
        k=params["k"],
    )
    if cvp is None:
        return np.zeros((0, 2), np.float64)
    return cvp.reshape(-1, 2).astype(np.float64)


def opencv_env():
    env = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import cv2

        env["opencv"] = cv2.__version__
        info = [
            ln.strip() for ln in cv2.getBuildInformation().splitlines() if ln.strip()
        ]
        env["opencv_build"] = info[0] if info else ""
    except ImportError:
        env["opencv"] = None
    return env


# ---------------------------------------------------------------------------
# 算子输出：examples 的 CORNERS 日志 → JSON → 比对
# ---------------------------------------------------------------------------
def parse_corners_log(path):
    """解析日志中的 `CORNERS|<case>|n=<num>|x,y;x,y;...` 行（坐标序 (x, y)，对齐 OpenCV）。"""
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as fd:
        for line in fd:
            line = line.strip()
            idx = line.find("CORNERS|")
            if idx < 0:
                continue
            _, name, num_field, body = line[idx:].split("|", 3)
            pairs = [p for p in body.split(";") if p]
            pts = [[float(v) for v in p.split(",")] for p in pairs]
            arr = np.array(pts, np.float64) if pts else np.zeros((0, 2), np.float64)
            assert int(num_field.split("=")[1]) == len(arr), (
                f"{name}: n= 与坐标个数不一致"
            )
            out[name] = arr
    return out


def write_ours(path, ours_all):
    payload = {
        "note": "算子真机输出角点（examples 设 GFT_DUMP_CORNERS=1 得到），由 run_crosscheck.py --import-dump 生成",
        "cases": {
            name: {"n": int(len(pts)), "corners_xy": pts.astype(int).tolist()}
            for name, pts in ours_all.items()
            if name in CASES
        },
    }
    with open(path, "w", encoding="utf-8") as fd:
        json.dump(payload, fd, ensure_ascii=False, indent=1)
        fd.write("\n")


def load_ours(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fd:
        payload = json.load(fd)
    return {
        name: np.array(item["corners_xy"], np.float64)
        for name, item in payload.get("cases", {}).items()
    }


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------
def chebyshev(a, b):
    return np.abs(a.reshape(-1, 1, 2) - b.reshape(1, -1, 2)).max(axis=2)


def cover(target, source, tol=0):
    """target 中每个点是否能在 source 中找到（不要求一对一）Chebyshev 距离 <= tol 的点。"""
    if len(target) == 0:
        return np.zeros(0, bool)
    if len(source) == 0:
        return np.zeros(len(target), bool)
    return (chebyshev(target, source) <= tol).any(axis=1)


def in_border_band(pts, h, w, band=2):
    """是否位于受边界延拓影响的窄带内（y/x 距边界 < band 或 >= 边界-band）。"""
    if len(pts) == 0:
        return np.zeros(0, bool)
    x, y = pts[:, 0], pts[:, 1]
    return (x < band) | (y < band) | (x >= w - band) | (y >= h - band)


def report_case(name, ours, cvp, truth):
    h, w = CASES[name]["shape"]
    print(
        f"\n=== {name}  (h={h}, w={w}, maxCorners={CASES[name]['params']['maxCorners']}) ==="
    )
    print(
        f"  角点数：算子 = {len(ours)}   OpenCV = {len(cvp)}   几何真值 = {len(truth)}"
    )

    ok = True

    # 1) 几何真值必须被双方精确命中（可闭式推导的部分要求逐点相等）
    for tag, pts in (("算子", ours), ("OpenCV", cvp)):
        hit = cover(truth, pts, 0)
        miss = truth[np.logical_not(hit)]
        print(
            f"  真值覆盖[{tag}]：精确命中 {int(hit.sum())}/{len(truth)}"
            + (f"  漏 {len(miss)} 个" if len(miss) else "")
        )
        if len(miss):
            print(f"    漏检示例（最多 5 个）：{miss[:5].tolist()}")
            ok = False

    # 2) 算子不得漏掉 OpenCV 找到的位置（1px 容差）
    miss_cv = cvp[np.logical_not(cover(cvp, ours, 1))] if len(cvp) else cvp
    print(f"  OpenCV 角点在算子输出 1px 内的覆盖：{len(cvp) - len(miss_cv)}/{len(cvp)}")
    if len(miss_cv):
        print(f"    未覆盖示例：{miss_cv[:5].tolist()}")
        ok = False

    # 3) 真值以外的差异：只做报告（阈值/局部极大/tie 规则不同所致），标注是否落在边界窄带
    for tag, pts in (("算子", ours), ("OpenCV", cvp)):
        extra = pts[np.logical_not(cover(pts, truth, 0))] if len(pts) else pts
        band = in_border_band(extra, h, w, 2)
        msg = f"  真值外角点[{tag}]：{len(extra)} 个"
        if len(extra):
            msg += f"，其中边界窄带内 {int(band.sum())} 个（y 范围 {int(extra[:, 1].min())}..{int(extra[:, 1].max())}）"
        print(msg)

    # 4) 一对一匹配与距离分布
    for tol in (0, 1):
        o_hit = cover(ours, cvp, tol)
        c_hit = cover(cvp, ours, tol)
        print(
            f"  一对一匹配@{tol}px：共同 {int(o_hit.sum())}/{len(ours)}，"
            f"算子独有 {int((~o_hit).sum())}，OpenCV 独有 {int((~c_hit).sum())}"
        )
    if len(ours) and len(cvp):
        d = chebyshev(ours, cvp).min(axis=1)
        print(
            f"  算子每个点到最近 OpenCV 点的距离：最大={d.max():.1f}  中位={np.median(d):.1f}"
        )
    return ok


def main():
    ap = argparse.ArgumentParser(description="GFT vs OpenCV cross-check")
    ap.add_argument(
        "--no-opencv",
        action="store_true",
        help="不实时调用 OpenCV，只用已提交的 golden",
    )
    ap.add_argument(
        "--write-golden",
        action="store_true",
        help="刷新 golden/opencv_golden.json",
    )
    ap.add_argument(
        "--import-dump",
        default=None,
        metavar="LOG",
        help="从 examples 日志导入算子输出到 golden/ours_corners.json",
    )
    ap.add_argument("--ours", default=OURS_PATH, help="算子输出 JSON 路径")
    args = ap.parse_args()

    if args.import_dump:
        ours_all = parse_corners_log(args.import_dump)
        if not ours_all:
            print(f"日志里没有 CORNERS 行：{args.import_dump}")
            return 1
        write_ours(OURS_PATH, ours_all)
        print(
            f"已写出 {os.path.relpath(OURS_PATH, HERE)}："
            + ", ".join(f"{k} n={len(v)}" for k, v in ours_all.items() if k in CASES)
        )
        return 0

    ours_all = load_ours(args.ours)
    golden = {}
    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, encoding="utf-8") as fd:
            golden = json.load(fd)

    ok = True
    new_golden = {"env": opencv_env(), "cases": {}}
    for name in CASES:
        if name not in ours_all:
            print(f"[skip] dump 里没有用例 {name}")
            ok = False
            continue

        if args.no_opencv:
            if name not in golden.get("cases", {}):
                print(f"[skip] golden 里没有用例 {name}")
                ok = False
                continue
            cvp = np.array(golden["cases"][name]["corners_xy"], np.float64)
        else:
            cvp = opencv_ref(build_case_image(name), CASES[name]["params"])

        new_golden["cases"][name] = {
            "shape": list(CASES[name]["shape"]),
            "params": CASES[name]["params"],
            "corners_xy": cvp.astype(int).tolist(),
        }
        ok = report_case(name, ours_all[name], cvp, case_truth(name)) and ok

    if args.write_golden and not args.no_opencv:
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fd:
            json.dump(new_golden, fd, ensure_ascii=False, indent=1)
            fd.write("\n")
        print(
            f"\n已写出 {os.path.relpath(GOLDEN_PATH, HERE)}（env={new_golden['env']}）"
        )

    print("\n结果：" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
