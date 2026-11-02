#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK golden for ResizeBilinearV2 (2D bilinear forward) + aclnnUpsampleBilinear2d.

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度 CPU
实现. 采用 **Torch + TensorFlow 混合标杆**, 按 (align_corners,
half_pixel_centers, scales) 在调用框架**之前**做确定性分支选择, 不重新设计后端
选择规则, 不捕获 Torch/TF 异常静默回退。

两个入口:
  * ``resize_bilinear_v2``      —— kernel 路径 (Ascend IR, TF ResizeBilinear 语义)
  * ``aclnnUpsampleBilinear2d`` —— ACLNN 路径 (ATen upsample_bilinear2d 语义)

分支表 (kernel 前向, 逻辑 NCHW):
  A (ac=1, hp=0)             : torch.ops.aten.upsample_bilinear2d(x,[Ho,Wo],True,None,None)   ~1e-16
  B (ac=0, hp=1)             : torch.ops.aten.upsample_bilinear2d(x,[Ho,Wo],False,sh,sw)      ~1e-16
                               sh=sw=scale_attr (scales>0) 否则 None
  C (ac=0, hp=0) 无显式scales: tf.raw_ops.ResizeBilinear(NHWC,[Ho,Wo],ac=False,hp=False)      ~1e-7
  D 其余(如 ac=0/hp=0+scales, : 文件内 Torch fp64 张量实现(§2 坐标映射) —— 成熟框架无法
     ac=1/hp=1)                 表达的组合, 标准允许的确定性自实现分支
  ALL_COPY (in==out, 任意 hp)  : 直返输入, 对齐 kernel IsMatchAllCopy
                               (dtype 相等 + H/W 长度相等, 不判 half_pixel/scales)

torch align_corners=False 恒为 half_pixel, 故 (ac=0,hp=0) 无法用 torch 表达 → TF;
TF ResizeBilinear 只吃 size 不吃 scales, 故 (ac=0,hp=0)+scales → 分支 D。

布局 (kernel 入口, 读 ``input_formats[0]``):
  * NCHW   : 直接计算
  * NHWC   : (N,H,W,C) → NCHW 计算 → 搬回
``size`` 输入恒为逻辑 [H_out, W_out], 与存储布局无关。设备侧
``CheckFormatValid`` 只放行 NCHW/NHWC, 故本文件不实现其它存储格式。

dtype: 一律 float64 计算。输出 dtype 优先 ``output_dtypes[0]``, 其次 ``dtype``
属性(acl dtype 枚举), 最后回退输入 dtype —— ResizeBilinearV2 允许混合
dtype(fp16 in → fp32 out), 按输入 dtype 截断会把真值降精度。
"""

import sys

import numpy as np
import torch

__spec__ = {
    "resize_bilinear_v2": "ResizeBilinearV2TestSpec",
    "aclnnUpsampleBilinear2d": "AclnnUpsampleBilinear2dTestSpec",
    "torch.ops.aten.upsample_bilinear2d": "E2eUpsampleBilinear2dTestSpec",
}

_CHANNELS_LAST_FORMATS = ("NHWC",)

# acl dtype 枚举 → numpy dtype 名 (ttk/utilities/dtypes.py DATA_TYPE_INT_TO_STR 子集,
# 只列 ResizeBilinearV2 的 y 可取的浮点类型)
_ACL_DTYPE_TO_NAME = {
    0: "float32",
    1: "float16",
    11: "float64",
    27: "bfloat16",
}


def _np_dtype(name):
    """dtype 名 → numpy dtype; bfloat16 走 ml_dtypes 扩展类型。"""
    if name == "bfloat16":
        from ml_dtypes import bfloat16 as np_bfloat16

        return np.dtype(np_bfloat16)
    return np.dtype(name)


def _as_numpy_keep_dtype(x):
    """三方 provider 入参 → ``(numpy 计算体, 原逻辑 numpy dtype)``。

    远端三方执行把输入构造成 torch 张量后才交给 provider, 于是 bf16 用例拿到的
    是**真 bf16** 张量; 而 torch bf16 既没有 ``.numpy()``(实测
    ``TypeError: Got unsupported ScalarType BFloat16`` —— 整条标杆直接报错而
    非 SKIP), ml_dtypes 的 numpy bf16 也不能被 ``torch.from_numpy`` 接收。
    故 bf16 单独先在 torch 侧升到 fp32(位宽足够, 转换无损)再出 numpy, 并把原
    dtype 一并带出, 由 ``restore`` 回落。

    其余 dtype **原样直通, 不做任何精度变动** —— ``golden_mode=Promote``
    (cross_check) 下 kernel 输入可能已被抬到 fp64, 盲目降到 fp32 会掉精度。
    """
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            return t.to(torch.float32).numpy(), _np_dtype("bfloat16")
        return t.numpy(), _np_dtype(str(t.dtype).replace("torch.", ""))
    arr = np.asarray(x)
    return arr, arr.dtype


# ------------------------------------------------------- 框架共存 / 绑定辅助


# --------------------------- TensorFlow 进程隔离 (HPTG3 #150 F4 / #151 F-07)
# 子进程源码: 只 import tensorflow, 绝不 import torch / tbe, 因此没有符号冲突,
# 也就不需要 RTLD_DEEPBIND。按行 JSON 收请求, 张量走临时 .npy 落盘。
_TF_CHILD_SRC = r"""
import json
import sys

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
sys.stdout.write("READY\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        kw = {k: tf.constant(np.load(v)) for k, v in req["tensors"].items()}
        for name, shape in req["zeros"].items():
            kw[name] = tf.zeros([int(v) for v in shape], tf.float32)
        for name, vals in req["int32"].items():
            kw[name] = tf.constant([int(v) for v in vals], dtype=tf.int32)
        kw.update(req["attrs"])
        with tf.device("/CPU:0"):
            y = getattr(tf.raw_ops, req["op"])(**kw)
        np.save(req["out"], np.asarray(y))
        resp = {"ok": True}
    except Exception as exc:  # 原样回传给父进程, 不吞异常
        resp = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"""

_TF_ISO = {"proc": None, "dir": None, "atexit": False}
_TF_ISO_TIMEOUT_S = 600


def _tf_iso_stop():
    """终止常驻子进程(atexit 与错误恢复共用), 可重复调用。"""
    proc = _TF_ISO.get("proc")
    _TF_ISO["proc"] = None
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def _tf_iso_fail(msg):
    """按 SKIP 语义抛错, 并带上子进程 stderr 尾部(TF 的真实报错在那里)。"""
    import os

    tail = ""
    d = _TF_ISO.get("dir")
    if d:
        path = os.path.join(d, "tf_child.err")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                tail = fh.read().strip()[-400:]
    _tf_iso_stop()
    raise NotImplementedError(f"{msg}{(' | stderr: ' + tail) if tail else ''}")


def _tf_iso_readline(proc, timeout):
    """带超时地读一行子进程应答; 超时/EOF 都转成 SKIP, 不挂死。"""
    import select

    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        _tf_iso_fail(f"TF 隔离子进程 {timeout}s 内无响应")
    line = proc.stdout.readline()
    if not line:
        _tf_iso_fail(f"TF 隔离子进程异常退出 (rc={proc.poll()})")
    return line


def _tf_iso_spawn():
    """拉起常驻 TF 子进程并等 READY。stderr 重定向到文件而非管道 —— TF 启动
    日志很啰嗦, 走无人读取的管道会把缓冲区填满而死锁。"""
    import atexit
    import os
    import shutil
    import subprocess
    import tempfile

    d = _TF_ISO["dir"]
    if d is None:
        d = tempfile.mkdtemp(prefix="ttk_tf_iso_")
        _TF_ISO["dir"] = d
    if not _TF_ISO["atexit"]:
        atexit.register(_tf_iso_stop)
        atexit.register(shutil.rmtree, d, True)
        _TF_ISO["atexit"] = True
    src = os.path.join(d, "tf_child.py")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(_TF_CHILD_SRC)
    env = dict(os.environ)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = ""  # TF 无昇腾后端, 同时避免误占 GPU
    env.pop("LD_PRELOAD", None)  # 不把 worker 的打桩带进隔离子进程
    errf = open(os.path.join(d, "tf_child.err"), "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", src],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errf,
            text=True,
            cwd=d,
            env=env,
        )
    except OSError as exc:
        raise NotImplementedError(f"无法启动 TF 隔离子进程: {exc}") from exc
    _TF_ISO["proc"] = proc
    if _tf_iso_readline(proc, _TF_ISO_TIMEOUT_S).strip() != "READY":
        _tf_iso_fail("TF 隔离子进程未就绪")
    return proc


def _tf_isolated(op, tensors=None, zeros=None, int32=None, attrs=None):
    """在**独立子进程**里调用 ``tf.raw_ops.<op>``, 返回 numpy 结果。

    本 worker 进程**全程不 import tensorflow**。原因: 进程内已有 torch(kernel
    通路还额外持有 CANN 的 ``tbe``), 再 import tensorflow 会在 tf 的 ``.so``
    静态初始化阶段 SIGSEGV(本机实测 ``exit 139``, 连一行 Python 都跑不到) ——
    两边都把静态链入的 protobuf/absl 同名 vague-linkage 弱符号导到全局而被跨库
    合并, 但 torch 由 GCC 编、tf 由 clang 编, 模板实例化并不等价。

    此前是在**本进程内**用 ``RTLD_DEEPBIND`` 导入绕开, 那是任务私有的规避手段
    (HPTG3 #150 ``board/a5-geir/F4``、#151 ``F-07`` 针对的正是它), 且副作用真实:
    DEEPBIND 会让 ``LD_PRELOAD`` 打桩与 sanitizer 对该库失效。改为进程隔离后,
    冲突在根上消失, worker 的符号空间完全不被污染。

    子进程常驻复用(``import tensorflow`` 约数秒, 不能每个用例付一次)。
    ``zeros``/``int32`` 让形状类入参在子进程里现造, 不必跨进程搬运零张量。
    子进程不可用(未装 TF / 启动失败 / 超时)时抛 ``NotImplementedError`` 由 ttk
    记 SKIP —— 既不静默换一个参考, 也不假通过。
    """
    import json
    import os

    proc = _TF_ISO["proc"]
    if proc is None or proc.poll() is not None:
        proc = _tf_iso_spawn()
    d = _TF_ISO["dir"]
    out = os.path.join(d, "out.npy")
    if os.path.exists(out):
        os.unlink(out)
    req = {
        "op": op,
        "tensors": {},
        "zeros": {k: [int(v) for v in s] for k, s in (zeros or {}).items()},
        "int32": {k: [int(v) for v in s] for k, s in (int32 or {}).items()},
        "attrs": {
            k: bool(v) if isinstance(v, (bool, np.bool_)) else v
            for k, v in (attrs or {}).items()
        },
        "out": out,
    }
    for name, arr in (tensors or {}).items():
        path = os.path.join(d, f"in_{name}.npy")
        np.save(path, np.ascontiguousarray(arr))
        req["tensors"][name] = path
    try:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError) as exc:
        _tf_iso_fail(f"TF 隔离子进程管道已断: {exc}")
    resp = json.loads(_tf_iso_readline(proc, _TF_ISO_TIMEOUT_S))
    if not resp.get("ok"):
        raise RuntimeError(f"tf.raw_ops.{op} (隔离子进程) 失败: {resp.get('error')}")
    return np.load(out)


def _require_cpu_device(device, provider):
    """TF provider 的设备闸门: 非 CPU 请求一律拒绝, 不静默退回 CPU。

    ``tf.config.get_soft_device_placement()`` 实测为 True —— 请求
    ``/device:GPU:0`` 会被静默放到 ``/device:CPU:0`` 并正常返回结果。若不拦,
    "跑在目标设备上通过了"就是假的。本机 ``tf.config.list_physical_devices()``
    只有 CPU, 且 TF 无昇腾后端, 故任何非 CPU 请求都应显式失败。
    """
    dev = str(device or "cpu").strip().lower()
    if dev not in ("", "cpu", "/device:cpu:0", "cpu:0"):
        raise NotImplementedError(
            f"{provider} 标杆请求设备 {device!r}: TensorFlow 无昇腾后端, 且 "
            "soft device placement 会把非法设备静默放回 CPU —— 拒绝执行而不是"
            "在 CPU 上算完冒充目标设备通过"
        )
    return "cpu"


# ttk 的 XPU 绑定器按**被测 API 自己的参数名**匹配形参
# (``ttk.core_modules.common.xpu.bind_params``), 而 aclnnUpsampleBilinear2d 的
# C 头首参名与 ATen schema 首参名都是 ``self``。所以三方 provider 必须把该形参
# **逐字**声明为 ``self``、并把实例槽改名为 ``impl`` —— 写成 ``x`` 会在计算
# 开始前就抛 ``UnknownParamError: parameter 'x' ... has no value``(``x`` 既不在
# 输入也不在属性里), 整个三方标杆静默不执行。本仓已合入的
# ``image/resize_nearest_neighbor_v2`` / ``image/upsample_nearest3d`` 同此写法。
# 反向入口的首参名是 ``gradOut``/``grad_output``, 与实例槽不冲突, 照常声明。


# ---------------------------------------------------------------- 分支选择辅助
def _scale_attr(scales, idx):
    """scales[idx] as positive float, or None when absent/<=0."""
    if scales is None:
        return None
    try:
        v = float(scales[idx])
    except (IndexError, TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _has_explicit_scales(scales):
    return _scale_attr(scales, 0) is not None or _scale_attr(scales, 1) is not None


# ---------------------------------------------------------------- 布局 / dtype
def _first(seq):
    """取序列首元素; 字符串整体即首元素; 空/非序列 → None。"""
    if seq is None:
        return None
    if isinstance(seq, str):
        return seq
    try:
        return seq[0]
    except (TypeError, IndexError, KeyError):
        return None


def _fmt_of(kwargs, key):
    v = _first(kwargs.get(key))
    return str(v).strip().upper() if v else ""


def _h_axis(fmt):
    """物理视图里 H 所在的轴: NHWC → 1; NCHW → 2。"""
    return 1 if fmt in _CHANNELS_LAST_FORMATS else 2


def _w_axis(fmt):
    """物理视图里 W 所在的轴: NHWC → 2; NCHW → 3。"""
    return 2 if fmt in _CHANNELS_LAST_FORMATS else 3


def _to_logical_nchw(arr, fmt):
    """物理视图 → 逻辑 (N,C,H,W) float64; 返回 (nchw, restore_fn)。

    restore_fn(y_nchw_f64, out_np_dtype) → 物理视图的 numpy 数组。
    """
    if arr.ndim != 4:
        raise ValueError(f"ResizeBilinearV2 expects a 4-D input, got shape {arr.shape}")

    if fmt in _CHANNELS_LAST_FORMATS:
        nchw = np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)), dtype=np.float64)

        def restore(y, out_dtype):
            return np.ascontiguousarray(np.transpose(y, (0, 2, 3, 1))).astype(
                out_dtype, copy=False
            )

        return nchw, restore

    nchw = np.ascontiguousarray(arr, dtype=np.float64)

    def restore(y, out_dtype):
        return np.ascontiguousarray(y).astype(out_dtype, copy=False)

    return nchw, restore


def _resolve_out_dtype(x, dtype_attr, kwargs):
    """输出 dtype: output_dtypes[0] > dtype 属性(acl 枚举) > 输入 dtype。"""
    name = _first(kwargs.get("output_dtypes"))
    if name:
        return _np_dtype(str(name))
    if dtype_attr is not None:
        try:
            key = int(dtype_attr)
        except (TypeError, ValueError):
            key = None
        if key in _ACL_DTYPE_TO_NAME:
            return _np_dtype(_ACL_DTYPE_TO_NAME[key])
    return x.dtype


# ---------------------------------------------------------------- torch 分支 A/B
def _torch_forward(x, ho, wo, align_corners, scales):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    if align_corners:
        sh = sw = None  # align_corners 忽略 scales(与 CANN 一致)
    else:
        sh = _scale_attr(scales, 0)
        sw = _scale_attr(scales, 1)
    y = torch.ops.aten.upsample_bilinear2d(
        t, [int(ho), int(wo)], bool(align_corners), sh, sw
    )
    return y.numpy()


# ---------------------------------------------------------------- TF 分支 C
def _tf_forward(x, ho, wo):
    # TF ResizeBilinear/Grad 仅注册 float32(legacy TF 兼容语义即 float32 计算),
    # 与该分支对应的 kernel 语义一致; 结果升回 float64 容器保持统一输出契约。
    nhwc = np.ascontiguousarray(np.transpose(x.astype(np.float32), (0, 2, 3, 1)))
    y = _tf_isolated(
        "ResizeBilinear",
        tensors={"images": nhwc},
        int32={"size": [int(ho), int(wo)]},
        attrs={"align_corners": False, "half_pixel_centers": False},
    )
    return np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2))).astype(np.float64)


# ------------------------------------------------- 分支 D: Torch fp64 张量实现
def _compute_scale(src, dst, align_corners, scale_attr):
    if align_corners:
        return (src - 1.0) / (dst - 1.0) if dst > 1 else float(src) / float(dst)
    if scale_attr is not None and scale_attr > 0.0:
        return 1.0 / float(scale_attr)
    return float(src) / float(dst)


def _src_coord_t(dst_len, scale, half_pixel):
    """标量 _src_coord 的张量形式(逐字同规则, fp64):

    v = (i + 0.5) * scale - 0.5 if half_pixel else i * scale
    return v if v > 0 else 0
    """
    idx = torch.arange(int(dst_len), dtype=torch.float64)
    v = (idx + 0.5) * scale - 0.5 if half_pixel else idx * scale
    return torch.where(v > 0.0, v, torch.zeros_like(v))


def _axis_weight_t(orig, src_len):
    """标量 _axis_weight 的张量形式(逐字同 clamp 规则, fp64):

        left  = floor(orig);  left  = min(left,  src_len-1)
        right = left + 1;     right = min(right, src_len-1)
        delta = clamp(orig - left, 0, 1)

    left 先 clamp, right 与 delta 都基于 clamp 后的 left —— 与标量实现的
    求值顺序一致, fp64 下逐位可复现。
    """
    hi = int(src_len) - 1
    left = torch.clamp(torch.floor(orig), max=float(hi)).to(torch.int64)
    right = torch.clamp(left + 1, max=hi)
    delta = torch.clamp(orig - left.to(torch.float64), min=0.0, max=1.0)
    return left, right, delta


def _torch_tensor_forward(x, ho, wo, align_corners, half_pixel, scales):
    """分支 D 的 Torch fp64 张量实现 (替换原三重 Python 循环的 NumPy 版本)。

    权重与 clamp 规则与标量参考实现逐字一致, 仅把逐点循环改成
    ``index_select`` + 权重外积; fp64 下与旧实现逐位相同(见 issue-139 等价性脚本)。
    """
    n, c, hi, wi = (int(v) for v in x.shape)
    xf = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    sh = _compute_scale(hi, ho, align_corners, _scale_attr(scales, 0))
    sw = _compute_scale(wi, wo, align_corners, _scale_attr(scales, 1))

    ty, by, dy = _axis_weight_t(_src_coord_t(ho, sh, half_pixel), hi)
    lx, rx, dx = _axis_weight_t(_src_coord_t(wo, sw, half_pixel), wi)

    # H 方向取上下两行: (N,C,Ho,Wi)
    top = xf.index_select(2, ty)
    bot = xf.index_select(2, by)
    # W 方向取左右两列: (N,C,Ho,Wo)
    top_l = top.index_select(3, lx)
    top_r = top.index_select(3, rx)
    bot_l = bot.index_select(3, lx)
    bot_r = bot.index_select(3, rx)

    wx_l = (1.0 - dx).view(1, 1, 1, wo)
    wx_r = dx.view(1, 1, 1, wo)
    wy_t = (1.0 - dy).view(1, 1, ho, 1)
    wy_b = dy.view(1, 1, ho, 1)

    vt = top_l * wx_l + top_r * wx_r
    vb = bot_l * wx_l + bot_r * wx_r
    y = vt * wy_t + vb * wy_b
    assert tuple(y.shape) == (n, c, ho, wo)
    return y.numpy()


def _check_attrs(align_corners, half_pixel_centers):
    """算子原型禁止的属性组合必须**拒绝**, 不能给出一个看起来合理的结果。

    设备侧 ``ResizeBilinearV2AscendCTilingImpl::CheckFormatMatchDims``
    (``op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp``) 开头即:

        OP_CHECK_IF(alignCorners_ && halfPixelCenters_, ... return GRAPH_FAILED);
        "The values of attributes align_corners and half_pixel_centers
         cannot be true at the same time"

    即该组合在 tiling 阶段就失败、算子根本不出结果(与 ResizeBilinearV2Grad
    的 ``GetAttrInfo`` 同一口径)。golden 若按"两套坐标规则叠加"自行算一个值,
    用例就会拿一个**不存在的期望值**去比对设备的失败, 把原型校验这条真实约束
    掩盖掉。故此处与设备侧同步拒绝。
    """
    if align_corners and half_pixel_centers:
        raise ValueError(
            "align_corners and half_pixel_centers cannot be true at the same "
            "time (rejected by ResizeBilinearV2 tiling CheckFormatMatchDims); "
            "该组合属算子原型禁止的非法属性组合, golden 不提供参考值"
        )


def _bilinear_forward(x, ho, wo, align_corners, half_pixel, scales, all_copy):
    """ALL_COPY 闸门由调用方按 **原始 dtype** 判定后传入。

    设备侧 ``IsMatchAllCopy``(``resize_bilinear_v2_tiling_arch35.cpp:245-252``)
    先判 ``dtypeX_ != dtypeY_ -> false``, 再判 H/W 等长 —— **dtype 相等是必要
    条件**。本函数拿到的 ``x`` 已被 ``_to_logical_nchw`` 统一升到 fp64, 原始
    dtype 已丢失, 故不能在此就地判 dtype, 只能接收调用方在升型前算好的
    ``all_copy``。(混合 dtype 同尺寸误走拷贝的缺陷正源于此。)

    同尺寸但 ``all_copy=False`` 时**不得**落到分支 A/B/C: ATen 与 TF 对
    ``in == out`` 都有 early-exit 直接返回拷贝、**忽略显式 scales**(实测
    ``aten.upsample_bilinear2d(2x2 -> 2x2, scales=2.0)`` 返回原值, 而正确结果
    随 scales 变化), 故该象限统一走文件内 fp64 逐点实现 —— 它按
    ``scale``/``half_pixel`` 显式求坐标, 无 early-exit。
    """
    hi, wi = int(x.shape[2]), int(x.shape[3])
    if all_copy:
        return x.astype(np.float64)
    same_size = hi == ho and wi == wo
    # 确定性分支选择 (调用框架之前)
    if align_corners and not half_pixel:  # 分支 A
        if same_size:
            return _torch_tensor_forward(x, ho, wo, True, False, None)
        return _torch_forward(x, ho, wo, True, None)
    if (not align_corners) and half_pixel:  # 分支 B
        if same_size:
            return _torch_tensor_forward(x, ho, wo, False, True, scales)
        return _torch_forward(x, ho, wo, False, scales)
    # 剩下的只有 (ac=0, hp=0) —— (ac=1, hp=1) 已由 _check_attrs 在入口拒绝
    if _has_explicit_scales(scales) or same_size:  # 分支 D
        return _torch_tensor_forward(x, ho, wo, False, False, scales)
    return _tf_forward(x, ho, wo)  # 分支 C (legacy TF 兼容)


class ResizeBilinearV2TestSpec:
    """TestSpec golden for ResizeBilinearV2 (kernel 路径)."""

    @staticmethod
    def golden(
        x,
        size,
        *,
        align_corners=False,
        half_pixel_centers=False,
        dtype=None,
        scales=None,
        **kwargs,
    ):
        _check_attrs(bool(align_corners), bool(half_pixel_centers))
        x = np.ascontiguousarray(x)
        dst = np.asarray(size).reshape(-1)
        assert dst.size >= 2, "size input must carry [H_out, W_out]"
        ho, wo = int(dst[0]), int(dst[1])

        fmt = _fmt_of(kwargs, "input_formats")
        # ALL_COPY 闸门必须在 _to_logical_nchw 升 fp64 **之前**算: 设备侧
        # IsMatchAllCopy 要求 dtypeX_ == dtypeY_, 升型后原始 dtype 已不可见。
        out_dtype = _resolve_out_dtype(x, dtype, kwargs)
        all_copy = (
            x.dtype == out_dtype
            and int(x.shape[_h_axis(fmt)]) == ho
            and int(x.shape[_w_axis(fmt)]) == wo
        )
        nchw, restore = _to_logical_nchw(x, fmt)
        y = _bilinear_forward(
            nchw,
            ho,
            wo,
            bool(align_corners),
            bool(half_pixel_centers),
            scales,
            all_copy,
        )
        return [restore(y, out_dtype)]

    class ThirdPartyImpl:
        """``torch.ops.aten.upsample_bilinear2d`` 三方标杆。

        覆盖 (align_corners, half_pixel_centers) 中能被 ATen 单个
        ``align_corners`` 开关表达的两条对角:

          * (ac=1, hp=0) → align_corners=True
          * (ac=0, hp=1) → align_corners=False  (torch 的 False 恒为 half_pixel)

        **显式正 scales 已纳入合法域**: 入口从 ``F.interpolate`` 换成底层
        ``aten.upsample_bilinear2d(self, output_size, align_corners, scales_h,
        scales_w)``。``size``/``scale_factor`` 互斥与 ``recompute_scale_factor``
        重算属**包装层**限制, ATen 低阶算子**同时接受 output_size 与 scales**,
        且 scales 真参与坐标映射(实测 2x2 → 3x3 显式 scales=2.0 的结果与
        size-derived 不同, 且与文件内 fp64 参考逐位一致) —— 与 IR 的
        "scale>0 → 坐标尺度取 1/scale, 输出尺寸由 size 独立给定" 一致。

        ``align_corners=True`` 时 ATen 忽略 scales —— kernel / aclnn / ATen
        三方一致, 直通不特判。

        **其余两组合由同文件的 ``TfThirdPartyImpl`` 覆盖**:
        ``(ac=0, hp=0)`` 是 legacy TF 坐标 ``src = dst * in/out``, torch 无此
        语义, 但 ``tf.raw_ops.ResizeBilinear`` 原生就有 ``align_corners`` 与
        ``half_pixel_centers`` 两个独立开关(与 kernel 同构), 故改由 ``tf``
        provider 承担, 不再记为"框架无法表达"。``(ac=1, hp=1)`` 被 TF 自身
        拒绝(InvalidArgumentError), 保留为缺口。

        **仍然拒绝的象限(有据, 非回避)**: 同尺寸 + 显式正 scales —— ATen 对
        ``in == out`` early-exit 返回拷贝并忽略 scales(实测 2x2→2x2
        ``scales=2.0`` 原值返回, 而按坐标映射应随 scale 变化), 复现同一
        early-exit 的标杆无交叉校验价值; 该象限由文件内 fp64 golden 覆盖。

        布局: 远端三方执行只对**类形态**实现注入 ``input_formats``
        (``ttk/remote/server/executor.py:_compose_kwargs``), 故这里读
        ``input_formats``; 缺省按 NCHW 解读, 显式 NHWC 才做归一/还原。
        """

        def __init__(
            self,
            *,
            align_corners=False,
            half_pixel_centers=False,
            scales=None,
            device=None,
            **kwargs,
        ):
            # 阶段 1: 参数准备 / 可用性判定 (不触碰数据, 不迁移设备)
            ac, hp = bool(align_corners), bool(half_pixel_centers)
            _check_attrs(ac, hp)  # (1,1) 是原型禁止组合, 先于框架能力判定拒绝
            if ac:
                self.align_corners = True
            elif hp:
                self.align_corners = False
            else:
                raise NotImplementedError(
                    "(align_corners=0, half_pixel_centers=0) 无法用 ATen 的单个 "
                    "align_corners 开关表达: 它是 legacy TF 坐标 src=dst*in/out, "
                    "改由本文件的 tf provider 覆盖"
                )
            # ac=True 时 ATen 与 kernel/aclnn 一致忽略 scales
            self.scale_h = None if self.align_corners else _scale_attr(scales, 0)
            self.scale_w = None if self.align_corners else _scale_attr(scales, 1)
            # device 必须**显式声明为形参**才会被注入: ttk 的
            # bind_params 只对 _FRAMEWORK_RESERVED={"device"} 这个**形参名**
            # 注入设备, 它不在 inputs/attrs 里, 因此 `kwargs.get("device")`
            # 恒为空 —— 实测会静默退回 cpu, 配置的设备根本不生效。
            self.device = device or "cpu"
            self.fmt = _fmt_of(kwargs, "input_formats")
            self.kwargs = kwargs

        def __call__(self, x, size, **kwargs):
            # 阶段 2/3: 布局归一 + 设备迁移 + 计算
            dst = np.asarray(size).reshape(-1)
            ho, wo = int(dst[0]), int(dst[1])
            arr, in_dtype = _as_numpy_keep_dtype(x)
            nchw, restore = _to_logical_nchw(np.ascontiguousarray(arr), self.fmt)
            if (self.scale_h is not None or self.scale_w is not None) and (
                int(nchw.shape[2]) == ho and int(nchw.shape[3]) == wo
            ):
                raise NotImplementedError(
                    "同尺寸 + 显式正 scales: ATen upsample_bilinear2d 对 in == out "
                    "early-exit 返回拷贝并忽略 scales, 无法作为该象限的独立标杆; "
                    "由文件内 fp64 golden 覆盖"
                )
            t = torch.from_numpy(nchw).to(self.device).contiguous()
            y = torch.ops.aten.upsample_bilinear2d(
                t, [ho, wo], self.align_corners, self.scale_h, self.scale_w
            )
            return [restore(y.detach().cpu().numpy(), in_dtype)]

    class TfThirdPartyImpl:
        """``tf.raw_ops.ResizeBilinear`` 三方标杆 —— 覆盖 torch 表达不了的组合。

        TF 原生暴露 ``align_corners`` 与 ``half_pixel_centers`` **两个独立
        开关**, 与 kernel 的属性一一同构, 故四组合里 ``(0,0)`` / ``(0,1)`` /
        ``(1,0)`` 都可直接表达(实测 2x2 → 3x3: ``(0,0)`` 给
        ``[1, 1.667, 2]`` 的 legacy TF 坐标, ``(0,1)`` 与 ``(1,0)`` 各给
        ``[1, 1.5, 2]``)。``(1,1)`` 被 **TF 自身**以 InvalidArgumentError 拒绝
        (两套坐标规则不能叠加), 保留为缺口。

        **显式正 scales**: TF 的输出尺寸由 ``size`` 给定、坐标尺度恒按
        ``in/out`` 内部推导, 没有独立的 scales 入参。``(ac=0, hp=0)`` 下
        ``src = dst * in/size``, 而 IR 的 ``scale``(out/in) 对应的坐标尺度是
        ``1/scale`` (见文件内 ``_compute_scale``), 故令 ``size = in * scale``
        即得 ``src = dst * in/(in*scale) = dst * (1/scale)``, 与分支 D 逐位
        一致; 再把前 ``out`` 行/列**切片**取出 —— 坐标公式逐 dst 独立, 不依赖
        总输出长度, 故切片不改变前 ``out`` 个值。
        要求 ``in * scale`` 为正整数且 ``>= out``; 否则该构造不成立, 拒绝并说明。

        dtype: TF 的 ResizeBilinear 只注册 float32(legacy TF 兼容语义即 fp32
        计算), 与该分支的 kernel 语义一致。
        """

        def __init__(
            self,
            *,
            align_corners=False,
            half_pixel_centers=False,
            scales=None,
            device=None,
            **kwargs,
        ):
            ac, hp = bool(align_corners), bool(half_pixel_centers)
            # (1,1) 由 _check_attrs 按算子原型拒绝 —— tf.raw_ops.ResizeBilinear
            # 自身同样以 InvalidArgumentError 拒绝该组合, 两边结论一致。
            _check_attrs(ac, hp)
            self.align_corners, self.half_pixel = ac, hp
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinear")
            self.scale_h = _scale_attr(scales, 0)
            self.scale_w = _scale_attr(scales, 1)
            self.fmt = _fmt_of(kwargs, "input_formats")
            self.kwargs = kwargs

        def __call__(self, x, size, **kwargs):
            dst = np.asarray(size).reshape(-1)
            ho, wo = int(dst[0]), int(dst[1])
            arr, in_dtype = _as_numpy_keep_dtype(x)
            nchw, restore = _to_logical_nchw(np.ascontiguousarray(arr), self.fmt)
            hi, wi = int(nchw.shape[2]), int(nchw.shape[3])

            # 显式正 scales: 仅 (ac=0, hp=0) 可用放大后切片构造
            grow_h, grow_w = ho, wo
            explicit = self.scale_h is not None or self.scale_w is not None
            if explicit:
                if self.align_corners or self.half_pixel:
                    raise NotImplementedError(
                        "显式正 scales 仅在 (ac=0, hp=0) 下可由 TF 构造(放大后切片); "
                        f"当前 (ac={self.align_corners}, hp={self.half_pixel}) 下 TF "
                        "无独立 scales 入参, 由 ATen provider 或文件内 golden 覆盖"
                    )
                for axis, (src, out, sc) in enumerate(
                    ((hi, ho, self.scale_h), (wi, wo, self.scale_w))
                ):
                    if sc is None:
                        continue
                    grown = src * float(sc)
                    if abs(grown - round(grown)) > 1e-12 or round(grown) < out:
                        raise NotImplementedError(
                            f"axis {axis}: in*scale = {src}*{sc} = {grown} 不是 >= 输出"
                            f"尺寸 {out} 的正整数, TF 放大后切片的构造不成立; "
                            "由文件内 fp64 golden 覆盖"
                        )
                    if axis == 0:
                        grow_h = int(round(grown))
                    else:
                        grow_w = int(round(grown))

            nhwc = np.ascontiguousarray(
                np.transpose(nchw.astype(np.float32), (0, 2, 3, 1))
            )
            y = _tf_isolated(
                "ResizeBilinear",
                tensors={"images": nhwc},
                int32={"size": [grow_h, grow_w]},
                attrs={
                    "align_corners": bool(self.align_corners),
                    "half_pixel_centers": bool(self.half_pixel),
                },
            )
            y = y[:, :ho, :wo, :]
            out_nchw = np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2))).astype(
                np.float64
            )
            return [restore(out_nchw, in_dtype)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ---------------------------------------------------------------- ACLNN 入口
def _opt_scale(v):
    """aclnn OUT/IN scale → ATen Optional[float]; <=0 表示未指定。

    直通不反转: ACLNN 边界恒为 OUT/IN, ATen ``compute_scales_value`` 同样按
    ``scale > 0 → 1/scale`` 解读, 架构差异在 ``GetWorkspaceSize`` 内部消化。
    """
    if v is None:
        return None
    s = float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)
    return s if s > 0.0 else None


def _aclnn_format(kwargs):
    """aclnn / e2e 路径的首输入 format。两条通路的键名**不同**, 必须都读:

      * ``input_formats`` —— 远端三方执行注入(``executor._compose_kwargs``,
        仅类形态实现), 即 ``third_party`` provider 能看到的唯一 format 来源;
      * ``tensor_formats`` —— 本地 golden 调用注入
        (``ttk/core_modules/aclnn/golden.py``), provider 侧**拿不到**。

    只读 ``tensor_formats`` 会让 provider 恒拿到 ND → NHWC 用例被当成 NCHW
    静默算错(标杆与被测都"通过"却算的不是同一件事)。
    """
    for key in ("input_formats", "tensor_formats"):
        v = _first(kwargs.get(key))
        if v:
            return str(v).strip().upper()
    return "ND"


# ----------------------------------------------- third_party 共享数学核 (2D)
#
# ACLNN / E2E 两个入口的 third_party 共用下面两个核, 只是形参名不同
# (aclnn 是 camel ``outputSize``/``alignCorners``/``scalesH``/``scalesW``,
# E2E 是 ATen snake ``output_size``/``align_corners``/``scales_h``/
# ``scales_w``) —— 需求允许共享数学核, 但**声明必须各按自己的签名**, 否则
# ttk 的 bind_params 会以 UnknownParamError 拒绝(参数名不在 inputs ∪ attrs)。
#
# ``device`` 必须**显式声明为形参**才会被注入: bind_params 只对
# _FRAMEWORK_RESERVED={"device"} 这个形参名注入, 它不在 inputs/attrs 里,
# 所以 ``kwargs.get("device")`` 恒为空(实测会静默退回 cpu)。
#
# 注意: aclnn/e2e 入口只有**单个** ``alignCorners`` 开关(half_pixel 由它反推,
# op_api 层硬编码 ``hpc = !ac``), 不是 kernel 的两个独立属性, 所以这里不需要
# kernel provider 的四组合判定。


def _tp_bilinear_aten(x, oh, ow, align_corners, s_h, s_w, device, channels_last):
    """ATen 低阶 2D 双线性。与文件内 golden **同源**(都走 aten.upsample_bilinear2d)。

    交叉校验价值: 覆盖**参数适配与绑定路径**(camel/snake 形参名、scale 哨兵、
    osize 切片、NHWC 归一、dtype 处理), 数学与 golden 同源, 不构成独立数值
    标杆 —— 独立标杆由同入口的 ``tf`` provider 承担(见 ``_tp_bilinear_tf``)。
    """
    t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    t = t.to(device or "cpu")
    orig = t.dtype
    if channels_last:
        if t.dim() != 4:
            raise ValueError(f"NHWC expects a 4-D input, got {tuple(t.shape)}")
        t = torch.movedim(t, -1, 1)
    if orig in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    y = torch.ops.aten.upsample_bilinear2d(
        t.contiguous(), [int(oh), int(ow)], bool(align_corners), s_h, s_w
    )
    if channels_last:
        y = torch.movedim(y, 1, -1)
    return y.contiguous().to(orig)


def _tp_bilinear_tf(x, oh, ow, align_corners, channels_last, out_dtype=None):
    """**独立**标杆: ``tf.raw_ops.ResizeBilinear`` (TF 自己的 kernel)。

    aclnn/e2e 的单开关与 TF 的两开关映射(与本文件 kernel provider 同构):
      * align_corners=False -> TF(align_corners=False, half_pixel_centers=True)
      * align_corners=True  -> TF(align_corners=True,  half_pixel_centers=False)
    即 kernel 四组合里的两条对角, 另两组合在 aclnn/e2e 侧**不可达**(op_api 层
    硬编码 ``hpc = !ac``), 故此处无需判定。

    实测(torch 2.7.1 / tf 2.19.0, fp32): 2x2 -> 3x3 两种 ac 下 TF 与
    ``aten.upsample_bilinear2d`` 都给 ``[1, 1.5, 2]`` 一致坐标; 本文件 kernel
    入口的 ``TfThirdPartyImpl`` 已在 (0,1)/(1,0) 两组合上实跑通过。

    **显式正 scales 不支持**: TF 的坐标尺度恒按 ``in/size`` 内部推导, 没有独立
    scales 入参; kernel provider 的"放大后切片"构造只在 legacy ``(ac=0, hp=0)``
    坐标下成立, 在 half_pixel / align_corners 下不成立(坐标含 +0.5 偏移或
    ``(in-1)/(out-1)`` 端点约束, 都依赖总输出长度), 故该象限由 ATen provider
    与文件内 fp64 golden 覆盖。

    dtype: TF 的 ResizeBilinear 只注册 float32 且恒返回 float32。

    输入可能是**真 bf16** 的 torch 张量(远端经 ``_to_vendor_tensor`` 构造),
    而 torch bf16 没有 ``.numpy()``(实测 ``TypeError: Got unsupported
    ScalarType BFloat16``, 会让整条 tf 标杆报错而非 SKIP), 因此先在 torch 侧
    升到 fp32 再出 numpy, 最后按输入的 torch dtype 返回。
    """
    if torch.is_tensor(x):
        orig_dtype = x.dtype
        arr = x.detach().cpu().to(torch.float32).numpy()
        src_dtype = None
    else:
        orig_dtype = None
        arr = np.asarray(x)
        src_dtype = arr.dtype
    nhwc = arr if channels_last else np.transpose(arr, (0, 2, 3, 1))
    y = _tf_isolated(
        "ResizeBilinear",
        tensors={"images": np.ascontiguousarray(nhwc.astype(np.float32))},
        int32={"size": [int(oh), int(ow)]},
        attrs={
            "align_corners": bool(align_corners),
            "half_pixel_centers": not bool(align_corners),
        },
    )
    out = y if channels_last else np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2)))
    if orig_dtype is not None:
        return torch.from_numpy(np.ascontiguousarray(out)).to(orig_dtype)
    return out.astype(out_dtype if out_dtype is not None else src_dtype, copy=False)


class AclnnUpsampleBilinear2dTestSpec:
    """aclnnUpsampleBilinear2d(self, outputSize, alignCorners, scalesH, scalesW, out).

    单分支 ``aten::upsample_bilinear2d`` —— aclnn 的坐标语义被 ``alignCorners``
    完全确定(True → align_corners, False → half_pixel), ATen 原生覆盖, 不引入
    TF/自实现分支, 也不抄 kernel 的 ALL_COPY 捷径(ATen 同尺寸实测逐位恒等)。

    * scales: ``<=0 → None``(ATen 退回 in/out); ``alignCorners=True`` 时统一传
      None —— ATen / kernel / aclnn 三方一致忽略 scales。
    * dtype: 入参抬 fp64 计算(ATen CPU 原生支持 double, 取最高精度标杆),
      结果回原 dtype。``--golden-mode`` 默认不抬精度, 抬升必须在入口内部完成。
    * 布局: NHWC 用 ``movedim`` 归一到 NCHW 再搬回; ``outputSize`` 恒为逻辑
      [H,W], 与存储布局无关。
    """

    @staticmethod
    def golden(x, outputSize, alignCorners, scalesH, scalesW, out=None, **kwargs):
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        orig_dtype = t.dtype
        channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS
        if channels_last:
            t = torch.movedim(t, -1, 1)
        t = t.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
        assert len(osize) >= 2, "outputSize must carry [H_out, W_out]"
        ac = bool(alignCorners)
        sh = None if ac else _opt_scale(scalesH)
        sw = None if ac else _opt_scale(scalesW)

        y = torch.ops.aten.upsample_bilinear2d(t, osize[:2], ac, sh, sw)

        if channels_last:
            y = torch.movedim(y, 1, -1)
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 属性按 **aclnn C 头**命名(camel ``scalesH``/``scalesW``)。

        数学核与文件内 golden 同源(见 ``_tp_bilinear_aten``), 覆盖参数适配/绑定
        路径; 独立数值标杆见同类的 ``TfThirdPartyImpl``。

        首参按 C 头逐字声明为 ``self``, 实例槽改名 ``impl``(见文件上方说明)。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"
            self.channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        def __call__(impl, self, outputSize, alignCorners, scalesH, scalesW, **kwargs):
            osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
            ac = bool(alignCorners)
            return [
                _tp_bilinear_aten(
                    self,
                    osize[0],
                    osize[1],
                    ac,
                    None if ac else _opt_scale(scalesH),
                    None if ac else _opt_scale(scalesW),
                    impl.device,
                    impl.channels_last,
                )
            ]

    class TfThirdPartyImpl:
        """**独立**标杆: TF ResizeBilinear (见 ``_tp_bilinear_tf``)。

        显式正 scales 不可构造, 拒绝并说明(有据, 非回避)。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinear")
            self.channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        def __call__(impl, self, outputSize, alignCorners, scalesH, scalesW, **kwargs):
            osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
            ac = bool(alignCorners)
            if not ac and (
                _opt_scale(scalesH) is not None or _opt_scale(scalesW) is not None
            ):
                raise NotImplementedError(
                    "显式正 scales: TF 无独立 scales 入参, half_pixel 坐标含 +0.5 偏移"
                    "且依赖总输出长度, 放大后切片的构造不成立; 该象限由 ATen provider "
                    "与文件内 fp64 golden 覆盖"
                )
            return [_tp_bilinear_tf(self, osize[0], osize[1], ac, impl.channels_last)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ------------------------------------------------------- Torch E2E (框架) 入口
class E2eUpsampleBilinear2dTestSpec:
    """``torch.ops.aten.upsample_bilinear2d`` 的框架 E2E 入口 (NCHW)。

    被测对象是**框架调用链**: ttk 把输入搬到 ``npu:0`` 后调用
    ``torch.ops.aten.upsample_bilinear2d``, 经 torch_npu 适配层落到
    ``aclnnUpsampleBilinear2d`` → 设备 kernel ``ResizeBilinearV2``。
    本 golden 只在 **CPU** 上用 fp64 算参考值, 与被测路径完全分离。

    **重载**: 绑定 ``.default``
    ``aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool
    align_corners, float? scales_h=None, float? scales_w=None) -> Tensor``。

    * **不注册 ``.vec``**: ``.vec`` 是 ``CompositeImplicitAutograd`` 且
      PrivateUse1 无 kernel, 会被分解, **根本不落目标 kernel**。
    * **形参名**: ATen 是 snake ``scales_h``/``scales_w``, aclnn 是 camel
      ``scalesH``/``scalesW`` —— csv ``attributes`` 必须按 ATen schema 写
      snake_case。框架按**位置**绑定, 形参名仅作可读性。
    * **scales 哨兵**: csv 写 ``None`` 表示未指定(aclnn 侧的 ``0.0`` 哨兵在用例
      转换时已折叠为 ``None``, 实测两者结果逐位相同), 正值表示显式指定。
      ``align_corners=True`` 时 ATen 的 ``area_pixel_compute_scale`` 直接忽略
      scales(aclnn 入口 docstring 已实证), 故此处**直通不特判** —— 被测侧同样
      忽略, 特判反而会掩盖不一致。
    * **布局**: e2e csv 一律 **NCHW 逻辑维**。NHWC 逻辑维用例**不转换进 e2e**:
      ATen 恒从 dim 2,3 读 H,W, 照搬会静默改语义(缺口 **G-E8**);
      channels_last 内存格式另由独立脚本取证, 不经 ttk。
    * **dtype**: 入参抬 fp64 计算后回原 dtype —— ``--golden-mode Promote`` 对
      e2e 路径未实现, 抬升必须在入口内部完成。与本文件 aclnn 入口同口径。
    * **支持矩阵**(需求明文要求, 不把四组合都塞给 ``align_corners``):
      kernel 的 ``(align_corners, half_pixel_centers)`` 四组合里, Torch 只能
      表达两条对角 —— ``(False,True)`` / ``(True,False)``;
      ``(False,False)`` 在 op_api 层被硬编码 ``hpc = !ac`` 挡住, **框架不可达**,
      作为缺口 **G-E4** 保留, 不借 ``align_corners`` 冒充。
    * tolerance 沿用 kernel/aclnn 入口同值: 同一数学语义、同一后端精度, 无放宽理由。
    """

    @staticmethod
    def golden(
        x,
        output_size,
        align_corners,
        scales_h=None,
        scales_w=None,
        out=None,
        **kwargs,
    ):
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        orig_dtype = t.dtype
        assert t.dim() == 4, (
            f"aten.upsample_bilinear2d expects a 4-D NCHW tensor, got {tuple(t.shape)}"
        )
        t = t.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
        assert len(osize) >= 2, "output_size must carry [H_out, W_out]"

        y = torch.ops.aten.upsample_bilinear2d(
            t,
            osize[:2],
            bool(align_corners),
            _opt_scale(scales_h),
            _opt_scale(scales_w),
        )
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **ATen schema** 命名(snake ``scales_h``/``scales_w``)。

        数学核与文件内 golden 同源(见 ``_tp_bilinear_aten``), 覆盖参数适配/绑定
        路径; 独立数值标杆见同类的 ``TfThirdPartyImpl``。

        e2e csv 一律 NCHW 逻辑维(见类 docstring 的缺口 **G-E8**), 故
        ``channels_last`` 固定为 False。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(
            impl,
            self,
            output_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
            ac = bool(align_corners)
            return [
                _tp_bilinear_aten(
                    self,
                    osize[0],
                    osize[1],
                    ac,
                    None if ac else _opt_scale(scales_h),
                    None if ac else _opt_scale(scales_w),
                    impl.device,
                    False,
                )
            ]

    class TfThirdPartyImpl:
        """**独立**标杆: TF ResizeBilinear (见 ``_tp_bilinear_tf``)。

        显式正 scales 不可构造, 拒绝并说明(有据, 非回避)。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinear")

        def __call__(
            impl,
            self,
            output_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
            ac = bool(align_corners)
            if not ac and (
                _opt_scale(scales_h) is not None or _opt_scale(scales_w) is not None
            ):
                raise NotImplementedError(
                    "显式正 scales: TF 无独立 scales 入参, half_pixel 坐标含 +0.5 偏移"
                    "且依赖总输出长度, 放大后切片的构造不成立; 该象限由 ATen provider "
                    "与文件内 fp64 golden 覆盖"
                )
            return [_tp_bilinear_tf(self, osize[0], osize[1], ac, False)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
