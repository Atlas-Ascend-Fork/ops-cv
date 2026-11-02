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

"""TTK golden for ResizeBilinearV2Grad (2D bilinear backward, NCHW / NHWC).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度 CPU
实现. 按 (align_corners, half_pixel_centers, scales) 在调用框架**之前**做确定性
分支选择, 不捕获框架异常静默回退.

反向是前向 gather 的转置(scatter): 对每个上游位置 (oh,ow), 用与前向一致的 4 邻
点 / 双线性权重把 **实际上游梯度 grads[oh,ow]**(非 ones_like) 散射回源网格,
与 kernel SimtCompute (op_kernel/arch35/resize_bilinear_v2_grad_simt.h) 逐行一致。

语义真值源是 kernel 实现(非 proto 注释):
  * scale 取值 (tiling SetScales / docs/aclnnUpsampleBilinear2dBackward.md):
      align_corners=1            -> (src-1)/(dst-1), dst>1; 否则 src/dst; 忽略 scales
      align_corners=0 & attr>0   -> 1/attr                 (scales 仅在 ac=0 生效)
      align_corners=0            -> src/dst
  * 源坐标 orig = (idx+off)*scale - off, off = half_pixel ? 0.5 : 0, 再 clamp 到 >=0
  * left = min(floor(orig), len-1); right = left<len-1 ? left+1 : left
    delta = clamp(orig-left, 0, 1)
  * align_corners=1 且 half_pixel_centers=1 被 tiling GetAttrInfo 拒绝 -> 本 golden
    同样 raise, 不产出“看似正常”的结果。

默认后端路由 (ref_backend="auto"), **默认路径全部按 float64 计算**:

| ac | hp | scales       | 后端                                      | 精度    |
|----|----|--------------|-------------------------------------------|---------|
| 1  | 0  | 忽略         | torch aten.upsample_bilinear2d_backward   | float64 |
| 0  | 1  | 无等长轴冲突 | torch 同上(sh/sw = scales>0 时的 attr)    | float64 |
| 0  | 1  | 等长轴带显式 | NumPy kernel 转写(torch 会丢 scale)       | float64 |
| 0  | 0  | 任意         | NumPy kernel 转写                         | float64 |
| 1  | 1  | --           | raise ValueError (kernel 无该分支)        | --      |

**等长轴 + 显式 scale 必须绕开 torch**: torch CPU 的
upsample_bilinear2d_backward 对"输入输出等长"的空间轴走短路实现(aten
UpSample.h compute_index_ranges_weights / UpSampleMoreKernel.cpp), 直接用
同位置索引与 1/0 权重, 完全不看该轴传入的 scale; 而 kernel 没有这个短路,
ac=0 下仍按 orig = (idx+off)*(1/attr) - off 做真实散射。故"某一轴等长且该轴
带显式 scale>0"时(见 _torch_drops_scale)即便 hp=1 也必须交给 NumPy 转写,
否则会拿到未缩放的梯度 —— 这是 ALL_COPY 之外第二条会漏算的同尺寸路径,
单轴等长(仅 H 或仅 W)同样命中。

ALL_COPY(grads.dtype == original_image.dtype 且 H/W 均相等): 直返 grads, 严格对齐
kernel IsMatchAllCopy (tiling arch35 310-312 行: lenSrcH==lenDesH && lenSrcW==lenDesW
&& gradsDtype==yDtype), 该分支**不检查** ac/hp/scales。混合 dtype(grads=float32,
original_image=float16/bfloat16)不命中 ALL_COPY, kernel 走 SIMT 真实散射, 故 golden
也必须真实计算 —— 这是本文件修复的核心点。

torch align_corners=False 恒为 half_pixel, 故 (ac=0, hp=0) 无法用 torch 表达, 由与
kernel 逐行一致的 NumPy float64 转写承担。

精度例外(显式声明): ref_backend="tf" 调用 tf.raw_ops.ResizeBilinearGrad, TF 仅注册
float32, 即该后端**实际为 float32 计算**, 不是 float64; 它也不接受 scales 参数。该
后端仅用于 legacy TF 兼容性交叉核对(opt-in), 不是默认高精度标杆; 传入显式 scales 时
直接 raise 而非静默忽略参数。实测 (ac=0,hp=0, 7 组形状) NumPy float64 与 TF float32
最大相对偏差 4.1e-7, 符合单精度量级。

布局: 从 TTK 注入的 input_formats kwarg 读取(NCHW/NHWC/ND, 缺省与 ND 均按 NCHW
解释); tiling 判的是**存储格式**(GetFormat().GetStorageFormat()), 故 input_formats
优先于 input_ori_formats。内部统一转 NCHW 计算, 结果按原布局转回。输出 shape/dtype
恒等于 original_image。

dtype 校验按 TTK 注入的 golden_mode kwarg 分流: --golden-mode Promote 会在调用本
golden 之前把输入数组与 input_dtypes/output_dtypes 元数据一起按 DTYPE_PROMOTE_MAP
升精度(float16/bfloat16 -> float32, float32 -> float64, 见 ttk _promote_dtype),
升精度后的数组已无法还原设备 dtype; 而模式本身由
_collect_dynamic_golden_kwargs 以 golden_mode 下发(本 golden 声明了 **kwargs,
accepts_var_keyword 为真 => kwargs 不裁剪), 故 Promote 下按"升精度像"校验、其余
模式下按设备组合校验, 不必放宽成两者的并集。只在拿不到该 kwarg 时(不下发 mode
的旧版 TTK)才退回并集, 以免把合法用例判为非法, 也不为过校验把计算降精度。详见
_check_dtypes 处的推导与 Promote 下已知的 float16/bfloat16 信息损失。

本文件同时承载另外两个参考入口。它们的坐标语义由**单个** alignCorners 完全确定
(op_api 层硬编码 hpc = !ac), 故不存在 kernel 的四组合判定:

  * ``aclnnUpsampleBilinear2dBackward`` -> AclnnUpsampleBilinear2dBackwardTestSpec
  * ``torch.ops.aten.upsample_bilinear2d_backward``
    -> E2eUpsampleBilinear2dBackwardTestSpec

三个入口的 ``third_party`` 各按自己的签名声明形参(kernel 用 IR 名
grads/original_image, aclnn 用 C 头 camel 名, e2e 用 ATen snake 名): ttk 的
``bind_params`` **按名绑定**, 名字不符即 UnknownParamError, 整条三方标杆静默不跑。

本算子**不支持 NC1HWC0/5HD**(tiling CheckFormatValid 只放行 NCHW/NHWC), 故全文件
无 5HD 分支。
"""

import math
import sys

import numpy as np
import torch

__spec__ = {
    "resize_bilinear_v2_grad": "ResizeBilinearV2GradTestSpec",
    "aclnnUpsampleBilinear2dBackward": "AclnnUpsampleBilinear2dBackwardTestSpec",
    "torch.ops.aten.upsample_bilinear2d_backward": (
        "E2eUpsampleBilinear2dBackwardTestSpec"
    ),
}

_SUPPORTED_LAYOUTS = ("NCHW", "NHWC", "ND")
_CHANNELS_LAST_FORMATS = ("NHWC",)
_SCALES_NUM = 2  # 对齐 tiling SCALES_NUM


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

    ``tf.config.get_soft_device_placement()`` 实测为 True —— 请求一个不存在的
    设备会被静默放到 CPU 并正常返回结果, 于是"跑在目标设备上通过了"就是假的。
    TF 无昇腾后端, 故任何非 CPU 请求都应显式失败。
    """
    dev = str(device or "cpu").strip().lower()
    if dev not in ("", "cpu", "/device:cpu:0", "cpu:0"):
        raise NotImplementedError(
            f"{provider} 标杆请求设备 {device!r}: TensorFlow 无昇腾后端, 且 soft "
            "device placement 会把非法设备静默放回 CPU —— 拒绝执行而不是在 CPU "
            "上算完冒充目标设备通过"
        )
    return "cpu"


def _require_cpu_torch(device, provider):
    """ATen 标杆的设备闸门: 只在 CPU 上构造, 非 CPU 请求显式拒绝。

    两条理由都不是回避:

      * 本标杆的精度口径是 **float64**, 昇腾无 float64 —— 放到 NPU 上只能悄悄
        降成 float32, 那就不再是高精度参照;
      * ``torch_npu`` 会把 ``aten.upsample_bilinear2d_backward`` 路由回**被测**
        的 ACLNN 反向接口, 在 NPU 上跑它等于拿被测物当标杆, 不构成交叉校验。

    ttk 的三方执行默认就是 CPU 角色服务端(``xpu_server._resolve_devices``:
    ``devices=cpu`` -> ``use_device=False`` -> ``device_str="cpu"``), 故该闸门
    不会把正常通路挡掉。设备侧行为由 kernel/aclnn/e2e 三条被测通路自身覆盖。
    """
    dev = str(device or "cpu").strip().lower()
    if dev not in ("", "cpu", "cpu:0"):
        raise NotImplementedError(
            f"{provider} 标杆请求设备 {device!r}: 本标杆按 float64 取参照而昇腾无 "
            "float64, 且 torch_npu 会把该 ATen 调用路由回被测的 ACLNN 反向接口 "
            "—— 拒绝执行而不是降精度/拿被测物当标杆"
        )
    return "cpu"


def _as_torch(arr):
    """任意 provider 入参 -> CPU ``torch.Tensor``, **保持 dtype**。

    ttk 远端三方执行按 provider 还原张量(``executor._to_vendor_tensor``):
    ``torch`` provider 收到 ``torch.Tensor``(bfloat16 由 int16 位重铸),
    ``tf`` provider 收到 ``tf.Tensor``, 本地直调时才是 numpy。

    bfloat16 **不能**经 numpy 中转: ``torch.Tensor.numpy()`` 直接报
    ``Got unsupported ScalarType BFloat16``, 而 numpy 侧的 bf16 是 ml_dtypes 扩展
    标量类型, ``torch.from_numpy`` 不接收。故此处一律在框架内换手; bf16 是
    float32 截断高 16 位, float32 往返无损, 可安全借道。
    """
    if torch.is_tensor(arr):
        return arr.detach().cpu()
    if not isinstance(arr, np.ndarray) and hasattr(arr, "numpy"):
        arr = arr.numpy()  # tf.Tensor -> numpy
    arr = np.asarray(arr)
    if arr.dtype.name in ("bfloat16", "bf16"):
        return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(arr))


def _np_dtype(name):
    """dtype 名 -> numpy dtype; bfloat16 走 ml_dtypes 扩展类型。"""
    if name == "bfloat16":
        from ml_dtypes import bfloat16 as np_bfloat16

        return np.dtype(np_bfloat16)
    return np.dtype(name)


def _as_numpy_keep_dtype(x):
    """三方 provider 入参 -> ``(numpy 计算体, 原逻辑 numpy dtype)``。

    TF 侧的标杆改走隔离子进程后(见 ``_tf_isolated``), 本进程不再持有
    ``tf.Tensor``, 换手统一在 numpy 完成。远端三方执行把输入构造成 torch 张量
    才交给 provider, 于是 bf16 用例拿到的是**真 bf16** 张量; torch bf16 既没有
    ``.numpy()``(实测 ``TypeError: Got unsupported ScalarType BFloat16``, 会让
    整条标杆报错而非 SKIP), ml_dtypes 的 numpy bf16 也不被 ``torch.from_numpy``
    接收。故 bf16 单独先在 torch 侧升到 float32(位宽足够, 转换无损)再出 numpy,
    并把原 dtype 带出由调用方回落 —— 这也正是 ttk ``executor._to_numpy_pair``
    对 ``tf`` provider 的 bf16 落盘口径(numpy bf16 -> int16 位)。

    其余 dtype **原样直通, 不做精度变动**: ``golden_mode=Promote``(cross_check)
    下输入可能已被抬到 fp64, 盲目降到 fp32 会掉精度。
    """
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            return t.to(torch.float32).numpy(), _np_dtype("bfloat16")
        return t.numpy(), _np_dtype(str(t.dtype).replace("torch.", ""))
    arr = np.asarray(x)
    return arr, arr.dtype


def _t_to_nchw(t, layout):
    """torch 侧的 ``_to_nchw``: NHWC -> NCHW(numpy 版用于 golden, 此版用于 provider)。"""
    return torch.movedim(t, -1, 1) if layout == "NHWC" else t


def _t_from_nchw(t, layout):
    return torch.movedim(t, 1, -1) if layout == "NHWC" else t


# ---------------------------------------------------------------- 布局辅助
def _resolve_layout(kwargs):
    """Read the tensor layout from the TTK-injected format kwargs.

    TTK passes input_formats / input_ori_formats into the golden
    (ops-test-kit ttk/core_modules/kernel/golden.py). input_formats wins:
    tiling reads the STORAGE format (GetFormat().GetStorageFormat(), see
    GetTensorInfo) and input_ori_formats is only the original declaration,
    which TTK defaults to input_formats and then to "ND" -- preferring it would
    route an NHWC storage case through NCHW whenever a case declares both.
    Both grads and original_image must carry the same layout (tiling
    CheckFormatValid), so the first entry decides. Absent / "ND" is interpreted
    as NCHW, matching the historical NCHW-only cases of this operator.
    """
    formats = kwargs.get("input_formats") or kwargs.get("input_ori_formats")
    if not formats:
        return "NCHW"
    if isinstance(formats, str):
        entries = [formats]
    else:
        try:
            entries = [f for f in formats if f is not None]
        except TypeError:
            return "NCHW"
    if not entries:
        return "NCHW"
    names = [str(f).strip().upper() for f in entries[:2]]
    for name in names:
        if name not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"ResizeBilinearV2Grad supports NCHW/NHWC only, got format {name!r} "
                "(tiling CheckFormatValid rejects everything else)."
            )
    if len(names) == _SCALES_NUM and names[0] != names[1]:
        raise ValueError(
            "The formats of grads and original_image must be the same, got "
            f"{names[0]!r} and {names[1]!r}."
        )
    return "NCHW" if names[0] == "ND" else names[0]


def _to_nchw(arr, layout):
    return (
        np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
        if layout == "NHWC"
        else arr
    )


def _from_nchw(arr, layout):
    return (
        np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))
        if layout == "NHWC"
        else arr
    )


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


def _torch_drops_scale(hi, wi, ho, wo, align_corners, scales):
    """True when torch would silently ignore an explicit scale on an equal axis.

    torch's CPU upsample_bilinear2d_backward short-circuits any spatial axis
    whose input and output lengths are equal: it uses the identity index with
    weights 1/0 and never consults the passed scale (aten UpSample.h
    compute_index_ranges_weights, UpSampleMoreKernel.cpp backward). The kernel
    has no such short-circuit -- with scales=[a, b] and align_corners=false it
    still maps orig = (idx + off) * (1 / attr) - off, so an equal-length axis
    carrying attr != 1 really does scatter. Routing such a case to torch
    returns the unscaled gradient, which is why ALL_COPY-style wrong results
    survived on the half_pixel path even after the dtype guard was added.

    align_corners ignores scales entirely (tiling SetScales), and an axis with
    no explicit scale degenerates to src/dst == 1 on equal lengths, i.e. the
    identity torch already computes -- neither case diverges.
    """
    if align_corners:
        return False
    return (hi == ho and _scale_attr(scales, 0) is not None) or (
        wi == wo and _scale_attr(scales, 1) is not None
    )


# ---------------------------------------------------------------- torch 分支 A/B
def _torch_grad(grads, hi, wi, align_corners, scales):
    n, c, ho, wo = grads.shape
    g = torch.from_numpy(np.ascontiguousarray(grads, dtype=np.float64))
    if align_corners:
        sh = sw = None  # align_corners 忽略 scales(与 kernel SetScales 一致)
    else:
        sh = _scale_attr(scales, 0)
        sw = _scale_attr(scales, 1)
    y = torch.ops.aten.upsample_bilinear2d_backward(
        g,
        [int(ho), int(wo)],
        [int(n), int(c), int(hi), int(wi)],
        bool(align_corners),
        sh,
        sw,
    )
    return y.numpy()


# ---------------------------------------------------------------- TF 交叉核对后端
def _tf_grad(grads, hi, wi, align_corners, half_pixel, scales):
    """Legacy-TF cross-check backend. float32 compute (documented exception)."""
    if _has_explicit_scales(scales):
        raise ValueError(
            "tf.raw_ops.ResizeBilinearGrad has no scales argument; the TF backend "
            "cannot express explicit scales. Use ref_backend='numpy' (or 'auto')."
        )
    n, c = int(grads.shape[0]), int(grads.shape[1])
    g_nhwc = np.ascontiguousarray(np.transpose(grads.astype(np.float32), (0, 2, 3, 1)))
    # original_image 仅提供形状/dtype, 内容不参与梯度; 零张量在子进程里现造
    y = _tf_isolated(
        "ResizeBilinearGrad",
        tensors={"grads": g_nhwc},
        zeros={"original_image": [n, int(hi), int(wi), c]},
        attrs={
            "align_corners": bool(align_corners),
            "half_pixel_centers": bool(half_pixel),
        },
    )
    return np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2))).astype(np.float64)


# ---------------------------------------------------------------- NumPy kernel 转写
def _compute_scale(src, dst, align_corners, scale_attr):
    if align_corners:
        return (src - 1.0) / (dst - 1.0) if dst > 1 else float(src) / float(dst)
    if scale_attr is not None and scale_attr > 0.0:
        return 1.0 / float(scale_attr)
    return float(src) / float(dst)


def _src_coord(dst_idx, scale, half_pixel):
    v = (dst_idx + 0.5) * scale - 0.5 if half_pixel else dst_idx * scale
    return v if v > 0.0 else 0.0


def _axis_weight(orig, src_len):
    left = int(math.floor(orig))
    if left > src_len - 1:
        left = src_len - 1
    right = left + 1
    if right > src_len - 1:
        right = src_len - 1
    delta = orig - left
    if delta < 0.0:
        delta = 0.0
    elif delta > 1.0:
        delta = 1.0
    return left, right, delta


def _numpy_grad(grads, hi, wi, align_corners, half_pixel, scales):
    n, c, ho, wo = grads.shape
    gf = grads.astype(np.float64)
    sh = _compute_scale(hi, ho, align_corners, _scale_attr(scales, 0))
    sw = _compute_scale(wi, wo, align_corners, _scale_attr(scales, 1))
    wmap = [_axis_weight(_src_coord(ow, sw, half_pixel), wi) for ow in range(wo)]
    y = np.zeros((n, c, hi, wi), np.float64)
    for oh in range(ho):
        ty, by, dy = _axis_weight(_src_coord(oh, sh, half_pixel), hi)
        wy_top = 1.0 - dy
        wy_bot = dy
        for ow in range(wo):
            lx, rx, dx = wmap[ow]
            wx_left = 1.0 - dx
            wx_right = dx
            g = gf[:, :, oh, ow]
            y[:, :, ty, lx] += g * (wy_top * wx_left)
            y[:, :, ty, rx] += g * (wy_top * wx_right)
            y[:, :, by, lx] += g * (wy_bot * wx_left)
            y[:, :, by, rx] += g * (wy_bot * wx_right)
    return y


# ---------------------------------------------------------------- 校验
def _check_attrs(align_corners, half_pixel, scales):
    if align_corners and half_pixel:
        raise ValueError(
            "align_corners=True together with half_pixel_centers=True is rejected by "
            "ResizeBilinearV2Grad (tiling GetAttrInfo returns GRAPH_FAILED); this "
            "combination has no valid golden and belongs to the exception tests."
        )
    if scales is not None:
        try:
            length = len(scales)
        except TypeError as exc:
            raise ValueError(
                f"scales must be a 2-element sequence, got {scales!r}."
            ) from exc
        if length != _SCALES_NUM:
            raise ValueError(
                f"scales must hold exactly {_SCALES_NUM} elements, got {length}."
            )


# 设备侧合法 dtype 组合(tiling CheckDtypeValid): grads=float32 时
# y ∈ {float32, float16, bfloat16}; grads 非 float32 时必须 grads == y。
_DEVICE_DTYPE_PAIRS = (
    ("float32", "float32"),
    ("float32", "float16"),
    ("float32", "bfloat16"),
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
)
# 上表逐项按 DTYPE_PROMOTE_MAP 升精度后的像(float16/bfloat16 -> float32、
# float32 -> float64): float16/bfloat16 都映射到 float32, 故 5 条设备组合塌缩为
# 3 条。
_PROMOTED_DTYPE_PAIRS = (
    ("float64", "float64"),  # <- (float32, float32)
    ("float64", "float32"),  # <- (float32, float16) / (float32, bfloat16)
    ("float32", "float32"),  # <- (float16, float16) / (bfloat16, bfloat16)
)
# 不下发 golden_mode 的旧版 TTK 只能按"设备组合 ∪ 其升精度像"的并集放行: 两种
# 解释下 ALL_COPY(判两者 dtype 是否相等)的结论一致 —— 唯一的异 dtype 合法组合
# float32+float16/bfloat16 升精度为 float64+float32 后仍不相等, 同 dtype 组合升
# 精度后仍相等 —— 故并集不会改变本 golden 的分支选择, 只是放宽了校验。
_UNION_DTYPE_PAIRS = _DEVICE_DTYPE_PAIRS + tuple(
    pair for pair in _PROMOTED_DTYPE_PAIRS if pair not in _DEVICE_DTYPE_PAIRS
)


def _allowed_dtype_pairs(golden_mode):
    """Legal (grads, y) dtype pairs for the mode TTK says it is calling with.

    --golden-mode Promote promotes the input arrays (and the input_dtypes /
    output_dtypes kwargs) by DTYPE_PROMOTE_MAP before calling the golden
    (ttk/core_modules/kernel/golden.py _promote_dtype), so under Promote the
    arrays carry the promoted image of the device combination, not the device
    combination itself. The mode is injected as the golden_mode kwarg by
    _collect_dynamic_golden_kwargs, so each mode can be checked against
    exactly one table. Only a TTK that predates that kwarg forces the union.
    """
    if golden_mode is None:
        return _UNION_DTYPE_PAIRS
    if str(golden_mode) == "Promote":
        return _PROMOTED_DTYPE_PAIRS
    return _DEVICE_DTYPE_PAIRS


def _check_dtypes(grads, original_image, golden_mode=None):
    pair = (str(grads.dtype), str(original_image.dtype))
    allowed = _allowed_dtype_pairs(golden_mode)
    if pair in allowed:
        return
    mode_msg = (
        "a TTK revision that does not pass golden_mode"
        if golden_mode is None
        else f"golden_mode={golden_mode}"
    )
    if pair[0] == pair[1]:
        raise ValueError(
            f"The dtypes of grads and y are {pair[0]} under {mode_msg}, which is not "
            "a legal combination for ResizeBilinearV2Grad (tiling CheckDtypeValid "
            f"accepts {list(_DEVICE_DTYPE_PAIRS)}; under Promote the golden receives "
            f"their promoted image instead). Accepted here: {list(allowed)}."
        )
    if pair[0] not in ("float32", "float64"):
        raise ValueError(
            "The dtypes of grads and y must be the same when the dtype of grads is "
            f"not float32, got {pair[0]} and {pair[1]} (tiling CheckDtypeValid; "
            f"float64 grads only arise under golden_mode=Promote, here {mode_msg})."
        )
    raise ValueError(
        "The dtype of y must be float32, float16 or bfloat16 when grads is float32 "
        f"(float64 grads with float32 y under golden_mode=Promote), got {pair[0]} "
        f"and {pair[1]} under {mode_msg}. Accepted combinations: {list(allowed)}."
    )


def _check_shapes(grads, original_image):
    if grads.ndim != 4 or original_image.ndim != 4:
        raise ValueError(
            "The shapes of grads and original_image must be 4D, got "
            f"{grads.shape} and {original_image.shape}."
        )


def _check_axes(grads_nchw, image_nchw):
    gn, gc, gh, gw = grads_nchw.shape
    yn, yc, yh, yw = image_nchw.shape
    if gn != yn or gc != yc:
        raise ValueError(
            "The N/C dimensions of grads and original_image must be the same, got "
            f"N={gn}/{yn}, C={gc}/{yc}."
        )
    if min(gh, gw, yh, yw) <= 0:
        raise ValueError(
            "The H/W dimensions of grads and original_image must be greater than 0, "
            f"got grads HW=({gh},{gw}) and original_image HW=({yh},{yw})."
        )


# ---------------------------------------------------------------- 后端路由
def _bilinear_grad(grads, hi, wi, align_corners, half_pixel, scales, ref_backend):
    """Dispatch one legal (ac, hp, scales) combination to a deterministic backend."""
    ho, wo = int(grads.shape[2]), int(grads.shape[3])
    drops_scale = _torch_drops_scale(hi, wi, ho, wo, align_corners, scales)

    if ref_backend == "numpy":
        return _numpy_grad(grads, hi, wi, align_corners, half_pixel, scales)
    if ref_backend == "tf":
        return _tf_grad(grads, hi, wi, align_corners, half_pixel, scales)
    if ref_backend == "torch":
        if (not align_corners) and (not half_pixel):
            raise ValueError(
                "torch align_corners=False always implies half_pixel; the torch "
                "backend cannot express (align_corners=False, half_pixel_centers="
                "False). Use ref_backend='numpy' (or 'auto')."
            )
        if drops_scale:
            raise ValueError(
                "torch short-circuits a spatial axis whose input and output "
                "lengths are equal and ignores the scale on that axis, while the "
                f"kernel still scatters: grads HW=({ho},{wo}), y HW=({hi},{wi}), "
                f"scales={scales!r}. The torch backend cannot express this "
                "combination. Use ref_backend='numpy' (or 'auto')."
            )
        return _torch_grad(grads, hi, wi, align_corners, scales)
    if ref_backend != "auto":
        raise ValueError(
            f"ref_backend={ref_backend!r} is not supported; expected one of "
            "'auto', 'numpy', 'torch', 'tf'."
        )
    # auto: 确定性分支选择(调用框架之前), 全部 float64
    if align_corners:  # 分支 A (ac=1, hp=0), scales 被忽略
        return _torch_grad(grads, hi, wi, True, None)
    if half_pixel and not drops_scale:  # 分支 B (ac=0, hp=1), scales 透传 torch
        return _torch_grad(grads, hi, wi, False, scales)
    # 分支 C: torch 无法表达 -> NumPy float64 转写
    #   * (ac=0, hp=0) torch 的 align_corners=False 恒为 half_pixel;
    #   * 等长轴携带显式 scale 时 torch 会短路该轴、丢掉 scale(见
    #     _torch_drops_scale), 此时 hp=1 也必须交给 NumPy。
    return _numpy_grad(grads, hi, wi, False, half_pixel, scales)


class ResizeBilinearV2GradTestSpec:
    """TestSpec golden for ResizeBilinearV2Grad."""

    @staticmethod
    def golden(
        grads,
        original_image,
        *,
        align_corners=False,
        half_pixel_centers=False,
        scales=None,
        ref_backend="auto",
        **kwargs,
    ):
        grads = np.ascontiguousarray(grads)
        original_image = np.ascontiguousarray(original_image)
        align_corners = bool(align_corners)
        half_pixel_centers = bool(half_pixel_centers)

        _check_shapes(grads, original_image)
        _check_attrs(align_corners, half_pixel_centers, scales)
        _check_dtypes(grads, original_image, kwargs.get("golden_mode"))

        layout = _resolve_layout(kwargs)
        grads_nchw = _to_nchw(grads, layout)
        image_nchw = _to_nchw(original_image, layout)
        _check_axes(grads_nchw, image_nchw)

        hi, wi = int(image_nchw.shape[2]), int(image_nchw.shape[3])
        ho, wo = int(grads_nchw.shape[2]), int(grads_nchw.shape[3])

        # ALL_COPY: kernel IsMatchAllCopy 判 gradsDtype==yDtype + H/W 长度相等
        # (resize_bilinear_v2_grad_tiling_arch35.cpp:310-312), 不检查 ac/hp/scales。
        # dtype 不等时 kernel 走 SIMT 真实散射, 这里必须跟着真实计算。
        if grads.dtype == original_image.dtype and hi == ho and wi == wo:
            y = grads.astype(original_image.dtype, copy=True)
            return [y]

        y_nchw = _bilinear_grad(
            grads_nchw,
            hi,
            wi,
            align_corners,
            half_pixel_centers,
            scales,
            str(ref_backend).strip().lower(),
        )
        y = _from_nchw(y_nchw, layout)
        return [y.astype(original_image.dtype, copy=False)]

    class ThirdPartyImpl:
        """``torch.ops.aten.upsample_bilinear2d_backward`` 三方标杆(Kernel 入口)。

        覆盖 kernel 四组合里能被 ATen **单个** ``align_corners`` 开关表达的两条
        对角:

          * (ac=1, hp=0) -> align_corners=True
          * (ac=0, hp=1) -> align_corners=False (torch 的 False 恒为 half_pixel)

        ``(1,1)`` 先由 ``_check_attrs`` 按**算子原型**拒绝(tiling GetAttrInfo
        第 122 行 ``alignCorners_ && halfPixelCenters_`` 直接 GRAPH_FAILED), 早于
        任何框架能力判定 —— 该组合没有合法标杆, 不得折进自实现分支产出"看似正常"
        的参考值。``(0,0)`` 是 legacy TF 坐标 ``src = dst * in/out``, torch 无此
        语义, 改由同类的 ``TfThirdPartyImpl`` 承担(TF 原生有两个独立开关)。

        入口用底层 ``aten.upsample_bilinear2d_backward``(前向 gather 的精确转置)
        而非 ``F.interpolate`` + autograd: 少一层包装, 且 ``scales`` 可直通并真实
        参与坐标映射 —— ``size``/``scale_factor`` 互斥与 ``recompute_scale_factor``
        重算都是**包装层**限制, 低阶反向算子同时接受 output_size/input_size 与
        scales。``align_corners=True`` 时 ATen 与 kernel/aclnn 一致忽略 scales。

        **仍然拒绝的象限(有据, 非回避)**: 等长轴 + 该轴显式正 scale —— ATen 对
        ``in == out`` 的空间轴 early-exit 用同位索引与 1/0 权重、完全不看 scale
        (与 ``_torch_drops_scale`` 同一成因), 而 kernel 仍按
        ``orig = (idx+off)*(1/attr) - off`` 真实散射; 复现同一 early-exit 的标杆
        没有交叉校验价值, 该象限由文件内 fp64 golden 覆盖。

        布局读 ttk 注入的 ``input_formats``(经 ``_resolve_layout``, 白名单
        NCHW/NHWC/ND), 非法布局直接抛错; 本算子不支持 NC1HWC0/5HD。
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
            # 阶段 1: 参数准备 / 可用性判定(不触碰数据, 不迁移设备)
            ac, hp = bool(align_corners), bool(half_pixel_centers)
            _check_attrs(ac, hp, scales)  # (1,1) 原型禁止, 先于框架能力判定
            if not (ac or hp):
                raise NotImplementedError(
                    "(align_corners=False, half_pixel_centers=False) 无法用 ATen 的"
                    "单个 align_corners 开关表达: 这是 legacy TF 坐标 "
                    "src = dst * in/out, 而 torch 的 align_corners=False 恒为 "
                    "half_pixel —— 该组合改由本文件的 tf provider 覆盖"
                )
            self.align_corners = ac
            self.scale_h = None if ac else _scale_attr(scales, 0)
            self.scale_w = None if ac else _scale_attr(scales, 1)
            # device 必须**显式声明为形参**才会被注入: ttk 的 bind_params 只对
            # _FRAMEWORK_RESERVED={"device"} 这个**形参名**注入设备, 它不在
            # inputs/attrs 池里 —— 写成 kwargs.get("device") 恒为空, 实测会静默
            # 退回 cpu, 配置的设备根本不生效。
            self.device = _require_cpu_torch(
                device, "aten.upsample_bilinear2d_backward"
            )
            self.layout = _resolve_layout(kwargs)

        def __call__(self, grads, original_image, **kwargs):
            # 阶段 2/3: 布局归一 + 计算。全程在 torch 侧换手 —— kernel 通路的
            # bfloat16 用例(wb06)入参是 torch bf16 张量, 借 numpy 会直接报
            # unsupported ScalarType。
            g = _t_to_nchw(_as_torch(grads), self.layout).contiguous()
            o = _t_to_nchw(_as_torch(original_image), self.layout)
            out_dtype = _as_torch(original_image).dtype
            hi, wi = int(o.shape[2]), int(o.shape[3])
            ho, wo = int(g.shape[2]), int(g.shape[3])
            # 与 _torch_drops_scale 同一判据, 但此处 scale 已拆成两个标量字段
            # (ac=True 时已置 None), 故直接比对而不复用该函数。
            if (hi == ho and self.scale_h is not None) or (
                wi == wo and self.scale_w is not None
            ):
                raise NotImplementedError(
                    "等长空间轴 + 该轴显式正 scale: ATen upsample_bilinear2d_backward "
                    "对 in == out 的轴 early-exit 用同位索引与 1/0 权重并忽略 scale, "
                    f"kernel 仍真实散射(grads HW=({ho},{wo}), y HW=({hi},{wi}), "
                    f"scales=({self.scale_h}, {self.scale_w})); 该象限由文件内 fp64 "
                    "golden 覆盖"
                )
            y = torch.ops.aten.upsample_bilinear2d_backward(
                g.to(self.device).to(torch.float64),
                [ho, wo],
                [int(v) for v in o.shape],
                self.align_corners,
                self.scale_h,
                self.scale_w,
            )
            out = _t_from_nchw(y.detach().cpu(), self.layout).contiguous()
            # 输出 dtype 取 original_image 的声明 dtype —— 与同文件 golden 一致
            # (混合 dtype 用例 wb15 的 y 精度由第二个输入决定, 不是 grads)。
            return [out.to(out_dtype)]

    class TfThirdPartyImpl:
        """``tf.raw_ops.ResizeBilinearGrad`` 三方标杆 —— 覆盖 torch 表达不了的组合。

        TF 原生暴露 ``align_corners`` 与 ``half_pixel_centers`` **两个独立开关**,
        与 kernel 属性一一同构, 故 ``(0,0)``/``(0,1)``/``(1,0)`` 都能直接表达。
        ``(1,1)`` 由 ``_check_attrs`` 按算子原型拒绝(TF 自身也拒绝), 保留为缺口。

        TF 的形参 ``grads``/``original_image`` 与本算子 IR 的两个输入一一对应
        (``original_image`` 只提供形状/dtype, 内容不参与梯度)。

        **显式正 scales 不支持**: TF 没有独立 scales 入参, 坐标尺度恒按 in/out
        推导。反向是散射语义, 前向那套"放大后切片"的构造不成立(被切掉的梯度不会
        归零, 而是会改变落点), 故一律拒绝并说明, 由 ATen provider 与文件内 golden
        覆盖。

        精度口径: ResizeBilinearGrad 仅注册 float32, 即该 provider **实际按
        float32 计算**, 不是 float64 —— 它是独立实现的交叉核对标杆, 不是高精度
        标杆。
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
            _check_attrs(ac, hp, scales)  # (1,1) 原型禁止, 先于框架能力判定
            if _has_explicit_scales(scales):
                raise NotImplementedError(
                    "显式正 scales: tf.raw_ops.ResizeBilinearGrad 无独立 scales "
                    "入参, 坐标尺度恒按 in/out 推导; 反向是散射语义, 前向的"
                    "'放大后切片'构造不成立, 该象限由 ATen provider 与文件内 "
                    "fp64 golden 覆盖"
                )
            self.align_corners, self.half_pixel = ac, hp
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinearGrad")
            self.layout = _resolve_layout(kwargs)

        def __call__(self, grads, original_image, **kwargs):
            # 全程 numpy 换手: TF 只在隔离子进程里跑(``_tf_isolated``), 本进程
            # 不 import tensorflow, 也不持有 tf.Tensor。
            g, _ = _as_numpy_keep_dtype(grads)
            o, out_dtype = _as_numpy_keep_dtype(original_image)
            # NHWC 存储 -> 逻辑 NCHW -> TF 要求的 NHWC。两次搬运不能省: layout
            # 为 NCHW 时才需要转成 NHWC, 为 NHWC 时本就是 TF 的布局。
            g_nhwc = g if self.layout == "NHWC" else np.transpose(g, (0, 2, 3, 1))
            o_shape = [int(v) for v in o.shape]
            if self.layout != "NHWC":
                o_shape = [o_shape[i] for i in (0, 2, 3, 1)]
            y = _tf_isolated(
                "ResizeBilinearGrad",
                tensors={"grads": np.ascontiguousarray(g_nhwc.astype(np.float32))},
                zeros={"original_image": o_shape},
                attrs={
                    "align_corners": bool(self.align_corners),
                    "half_pixel_centers": bool(self.half_pixel),
                },
            )
            out = y if self.layout == "NHWC" else np.transpose(y, (0, 3, 1, 2))
            # 回落按 original_image 的 dtype —— 混合 dtype 用例 wb15 的输出精度
            # 由第二个输入决定, 不是 grads。
            return [np.ascontiguousarray(out).astype(out_dtype, copy=False)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ---------------------------------------------------------------- ACLNN / E2E
def _opt_scale(v):
    """aclnn OUT/IN scale -> ATen Optional[float]; <=0 表示未指定。

    直通不反转: ACLNN 边界恒为 OUT/IN, ATen ``compute_scales_value`` 同样按
    ``scale > 0 -> 1/scale`` 解读, 架构差异在 ``GetWorkspaceSize`` 内部消化。

    ``aclnnUpsampleBilinear2dBackward`` 的"未指定"哨兵是 **0.0**
    (``aclnn_upsample_bilinear_2d_backward_v2.cpp`` 的
    ``std::abs(scalesH) > 1e-9 || std::abs(scalesW) > 1e-9`` 才做一致性校验),
    前向的 ``aclnnUpsampleBilinear2d`` 还会用 ``CheckScalesValid`` 显式拒绝负数,
    故本入口的 csv 用 0.0 而不是 -1.0。
    """
    if v is None:
        return None
    s = float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)
    return s if s > 0.0 else None


def _aclnn_format(kwargs):
    """aclnn / e2e 路径的首输入 format。两条通路的键名**不同**, 必须都读:

      * ``input_formats`` —— 远端三方执行注入(``executor._compose_kwargs``, 仅类
        形态实现), 即 ``third_party`` provider 能看到的唯一 format 来源;
      * ``tensor_formats`` —— 本地 golden 调用注入
        (``ttk/core_modules/aclnn/golden.py``), provider 侧**拿不到**。

    只读 ``tensor_formats`` 会让 provider 侧恒按缺省 ND 解读, NHWC 用例被静默按
    NCHW 算完当成通过。
    """
    for key in ("input_formats", "tensor_formats"):
        fmts = kwargs.get(key)
        if not fmts:
            continue
        v = fmts if isinstance(fmts, str) else (fmts[0] if len(fmts) > 0 else None)
        if v:
            return str(v).strip().upper()
    return "ND"


# ---------------------------------- third_party 共享数学核 (2D 反向, aclnn/e2e)
#
# ACLNN 与 E2E 两个入口的 third_party 共用下面两个核, 只是**形参名**不同
# (aclnn 是 C 头 camel gradOut/outputSize/inputSize/alignCorners/scalesH/scalesW,
# e2e 是 ATen snake grad_output/output_size/input_size/align_corners/
# scales_h/scales_w) —— 数学核可以共享, 但**声明必须各按自己的签名**, 否则 ttk 的
# bind_params 会以 UnknownParamError 拒绝(参数名不在 inputs ∪ attrs 内)。


def _tp_bilinear_grad_aten(
    grads, input_size, align_corners, s_h, s_w, device, channels_last
):
    """ATen 低阶 2D 双线性反向。与文件内 golden 的 torch 分支同源。

    交叉校验价值在**参数适配与绑定路径**(camel/snake 形参名、scale 哨兵、NHWC
    归一、``inputSize`` 的 {N,H,W,C}->{N,C,H,W} 重排、dtype 处理), 数学与 golden
    同源, 故**不构成独立数值标杆** —— 独立标杆由同入口的 ``tf`` provider 承担
    (见 ``_tp_bilinear_grad_tf``)。
    """
    g = _as_torch(grads)
    orig = g.dtype
    isz = [int(v) for v in input_size]
    if channels_last:
        if g.dim() != 4:
            raise ValueError(f"NHWC expects a 4-D gradOut, got {tuple(g.shape)}")
        g = torch.movedim(g, -1, 1)
        # {N,H,W,C} -> {N,C,H,W}
        isz = [isz[0], isz[3], isz[1], isz[2]]
    g = g.to(device or "cpu").to(torch.float64).contiguous()
    y = torch.ops.aten.upsample_bilinear2d_backward(
        g,
        [int(g.shape[2]), int(g.shape[3])],
        isz,
        bool(align_corners),
        s_h,
        s_w,
    )
    if channels_last:
        y = torch.movedim(y, 1, -1)
    return y.contiguous().to(orig)


def _tp_bilinear_grad_tf(grads, input_size, align_corners, channels_last):
    """**独立**标杆: ``tf.raw_ops.ResizeBilinearGrad``(TF 自己的 kernel)。

    aclnn/e2e 的单开关与 TF 两开关的映射(与本文件 kernel provider 同构):

      * align_corners=False -> TF(align_corners=False, half_pixel_centers=True)
      * align_corners=True  -> TF(align_corners=True,  half_pixel_centers=False)

    即 kernel 四组合里的两条对角; 另两组合在 aclnn/e2e 侧**不可达**(op_api 层
    硬编码 ``hpc = !ac``), 故此处无需判定 —— 四组合的覆盖由本文件 kernel 入口的
    ``TfThirdPartyImpl`` 承担。

    ``original_image`` 只用形状/dtype(TF 据此分配输出), 零张量在子进程里现造。
    精度口径: ResizeBilinearGrad 仅注册 float32, 即**实际按 float32 计算**。

    TF 在隔离子进程里执行(``_tf_isolated``), 本进程不 import tensorflow; 换手
    统一走 numpy(``_as_numpy_keep_dtype``), bf16 借 float32 往返无损, 出参的
    numpy bf16 正是 ttk ``executor._to_numpy_pair`` 对 ``tf`` provider 的落盘
    口径(numpy bf16 -> int16 位)。
    """
    g, out_dtype = _as_numpy_keep_dtype(grads)
    isz = [int(v) for v in input_size]
    if channels_last:
        g_nhwc = g
        orig_nhwc = [isz[0], isz[1], isz[2], isz[3]]
    else:
        g_nhwc = np.transpose(g, (0, 2, 3, 1))
        # {N,C,H,W} -> {N,H,W,C}
        orig_nhwc = [isz[0], isz[2], isz[3], isz[1]]
    y = _tf_isolated(
        "ResizeBilinearGrad",
        tensors={"grads": np.ascontiguousarray(g_nhwc.astype(np.float32))},
        zeros={"original_image": orig_nhwc},
        attrs={
            "align_corners": bool(align_corners),
            "half_pixel_centers": not bool(align_corners),
        },
    )
    out = y if channels_last else np.transpose(y, (0, 3, 1, 2))
    return np.ascontiguousarray(out).astype(out_dtype, copy=False)


class AclnnUpsampleBilinear2dBackwardTestSpec:
    """``aclnnUpsampleBilinear2dBackward(gradOut, outputSize, inputSize,
    alignCorners, scalesH, scalesW, out)``。

    单分支 ``aten.upsample_bilinear2d_backward`` —— aclnn 的坐标语义被
    ``alignCorners`` 完全确定(op_api 硬编码 ``hpc = !ac``), ATen 原生覆盖, 既不
    引入自实现分支, 也不照搬 kernel 的 ALL_COPY 捷径(那是 kernel tiling 的调度
    优化, 不是接口语义)。

    * scales: ``<=0 -> None``; ``alignCorners=True`` 时统一传 None(三方一致忽略)。
    * dtype: 入参抬 float64 计算, 结果回原 dtype。
    * 布局: NHWC 下 ``gradOut`` 用 ``movedim`` 归一到 NCHW, 且 ``inputSize``
      **必须**从 {N,H,W,C} 重排为 {N,C,H,W} 再喂 ATen(不重排会报
      ``output.size(1) != grad_output.size(1)``), 返回后搬回 NHWC;
      ``outputSize`` 恒为逻辑 [H,W]。
    * **落点已实测**: 本入口经 ``aclnnUpsampleBilinear2dBackwardGetWorkspaceSize``
      -> ``aclnnUpsampleBilinear2dBackward`` -> ``ResizeBilinearV2Grad_AiCore``
      (Ascend950PR_9579 / CANN 9.0.0-beta.2, profiling 实测)。同文件的 Torch E2E
      入口落在 ``...BackwardV2`` 上, **是另一条通路** —— 两者结论不可互相顶替。

    **已知缺口 G-A1(设备侧非确定性 × 结构性零元素)**: H 轴做 7->3 这类下采样的
    用例(``bl2dgrad_nhwc_fp32_7to3``)在本机 Ascend950PR_9579 上**间歇性**不过
    ``stat_rel_err`` —— 实测 2/3 次 FAIL; 把同一逻辑形状改按 NCHW 声明后同样
    间歇失败(4 次里 1 次), 即**与布局无关, 由 shape/参数决定**, 不是本文件 NHWC
    归一的问题。

    成因: 该参数下 H 轴存在结构性全零输出行(1120 个元素里 321 个真值为 0)。设备
    侧原子累加顺序逐次不同, 会在"本该为 0"的位置留下约 2 ulp 的残留(实测
    2.384e-7), 而参考值是 fp64 累加后的 1.33e-15; ``stat_rel_err`` 的相消噪声
    地板只有 1 ulp@1.0 (``NOISE_FLOOR["float32"] = 2**-23``), 于是该单点
    ``mare ≈ 2`` 就把整条用例判掉。全张量最大绝对误差 3.34e-6, 相对幅值 8.27 为
    4.0e-7 —— 数值本身是吻合的。

    **不放宽容差、不删用例、不改判据**: 这是被测侧的非确定性, 应当被看见。
    """

    @staticmethod
    def golden(
        gradOut,
        outputSize,
        inputSize,
        alignCorners,
        scalesH,
        scalesW,
        out=None,
        **kwargs,
    ):
        g = _as_torch(gradOut)
        orig_dtype = g.dtype
        channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS
        if channels_last:
            g = torch.movedim(g, -1, 1)
        g = g.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
        isize = [int(v) for v in np.asarray(inputSize).reshape(-1)]
        if len(osize) < 2:
            raise ValueError(f"outputSize must carry [H_out, W_out], got {osize}")
        if len(isize) != 4:
            raise ValueError(f"inputSize must carry 4 dims, got {isize}")
        if channels_last:
            # {N,H,W,C} -> {N,C,H,W}
            isize = [isize[0], isize[3], isize[1], isize[2]]

        ac = bool(alignCorners)
        sh = None if ac else _opt_scale(scalesH)
        sw = None if ac else _opt_scale(scalesW)

        y = torch.ops.aten.upsample_bilinear2d_backward(g, osize[:2], isize, ac, sh, sw)

        if channels_last:
            y = torch.movedim(y, 1, -1)
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **aclnn C 头**命名(camel ``scalesH``/``scalesW``)。

        首个输入在 C 头里名为 ``gradOut``(不是 ``self``), 故实例槽可照常命名
        ``self``, 无需 ACLNN 前向入口那套 ``(impl, self, ...)`` 改名。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_torch(
                device, "aten.upsample_bilinear2d_backward"
            )
            self.channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        def __call__(
            self,
            gradOut,
            outputSize,
            inputSize,
            alignCorners,
            scalesH,
            scalesW,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(inputSize).reshape(-1)]
            ac = bool(alignCorners)
            return [
                _tp_bilinear_grad_aten(
                    gradOut,
                    isz,
                    ac,
                    None if ac else _opt_scale(scalesH),
                    None if ac else _opt_scale(scalesW),
                    self.device,
                    self.channels_last,
                )
            ]

    class TfThirdPartyImpl:
        """**独立**标杆: TF ResizeBilinearGrad(见 ``_tp_bilinear_grad_tf``)。

        显式正 scales 不可构造, 拒绝并说明(有据, 非回避)。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinearGrad")
            self.channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        def __call__(
            self,
            gradOut,
            outputSize,
            inputSize,
            alignCorners,
            scalesH,
            scalesW,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(inputSize).reshape(-1)]
            ac = bool(alignCorners)
            if not ac and (
                _opt_scale(scalesH) is not None or _opt_scale(scalesW) is not None
            ):
                raise NotImplementedError(
                    "显式正 scales: TF 无独立 scales 入参, 坐标尺度恒按 in/out "
                    "推导; 该象限由 ATen provider 与文件内 fp64 golden 覆盖"
                )
            return [_tp_bilinear_grad_tf(gradOut, isz, ac, self.channels_last)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class E2eUpsampleBilinear2dBackwardTestSpec:
    """``torch.ops.aten.upsample_bilinear2d_backward`` 的框架 E2E 入口 (NCHW)。

    被测对象是**框架调用链**: ttk 把上游梯度搬到 ``npu:0`` 后调用
    ``torch.ops.aten.upsample_bilinear2d_backward``, 经 torch_npu 适配层落到
    ACLNN 反向接口, 再到设备 kernel。本 golden 只在 **CPU** 上用 float64 算参考
    值, 与被测路径完全分离。

    **重载**: 绑定 ``.default``
    ``aten::upsample_bilinear2d_backward(Tensor grad_output, SymInt[2]
    output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None,
    float? scales_w=None) -> Tensor``。

    * **实际落点已实测, 且不是老接口**: 本机 Ascend950PR_9579 / CANN 9.0.0-beta.2
      上 18 条用例的 profiling **全部**落在 ``aclnnUpsampleBilinear2dBackwardV2``
      (op-plugin 对 bilinear 反向在 V1/V2 间做优先级选择)。因此本入口的通过**不能**
      用来声称 ``aclnnUpsampleBilinear2dBackward``(V1)已被验证 —— V1 由同文件的
      ``AclnnUpsampleBilinear2dBackwardTestSpec`` 直接覆盖(实测经
      ``ResizeBilinearV2Grad_AiCore``)。换 CANN/op-plugin 版本后落点可能变,
      仍须逐例以 profiling 复核, 不得照抄本结论。
    * **不绑 ``.grad_input``**(走 CPU fallback 风险路径; ttk ``match_overload``
      在 params 不含 ``grad_input`` 时选 ``.default``, 故 csv 不得出现该键),
      **不注册 ``.vec``**(composite, 会被分解)。
    * ``input_size`` 为 4 元 ``{N,C,H_in,W_in}``: e2e csv 一律 NCHW 逻辑维, 故
      **无需** {N,H,W,C}->{N,C,H,W} 重排(那是 aclnn 的 NHWC 用例才需要);
      csv ``tensor_view_shapes`` 第二元必须与 ``input_size`` 一致。
    * **布局**: NHWC 逻辑维用例**不进 e2e**(ATen 恒从 dim 2,3 读 H,W, 照搬会
      静默改语义), channels_last 另行取证。
    * **支持矩阵**: 同前向 —— Torch 只能表达 ``(ac,hpc)`` 的两条对角,
      ``(False,False)`` 与 ``(True,True)`` 在 Torch E2E 参数域不可达, 保留为缺口,
      不靠改写 ``half_pixel_centers`` 凑 PASS。
    * **不做逐位相等断言**: 反向默认非确定性, 只做 ``stat_rel_err`` 容差比对;
      上游梯度用真实非均匀随机值, 常量会让散射错误互相掩盖。
    """

    @staticmethod
    def golden(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h=None,
        scales_w=None,
        out=None,
        **kwargs,
    ):
        g = _as_torch(grad_output)
        orig_dtype = g.dtype
        if g.dim() != 4:
            raise ValueError(
                "aten.upsample_bilinear2d_backward expects a 4-D NCHW grad_output, "
                f"got {tuple(g.shape)}"
            )
        g = g.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
        isize = [int(v) for v in np.asarray(input_size).reshape(-1)]
        if len(osize) < 2:
            raise ValueError(f"output_size must carry [H_out, W_out], got {osize}")
        if len(isize) != 4:
            raise ValueError(
                f"input_size must carry 4 dims [N, C, H_in, W_in], got {isize}"
            )

        y = torch.ops.aten.upsample_bilinear2d_backward(
            g,
            osize[:2],
            isize,
            bool(align_corners),
            _opt_scale(scales_h),
            _opt_scale(scales_w),
        )
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **ATen schema** 命名(snake ``scales_h``/``scales_w``)。

        e2e csv 一律 NCHW 逻辑维, 故 ``channels_last`` 固定为 False。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_torch(
                device, "aten.upsample_bilinear2d_backward"
            )

        def __call__(
            self,
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(input_size).reshape(-1)]
            ac = bool(align_corners)
            return [
                _tp_bilinear_grad_aten(
                    grad_output,
                    isz,
                    ac,
                    None if ac else _opt_scale(scales_h),
                    None if ac else _opt_scale(scales_w),
                    self.device,
                    False,
                )
            ]

    class TfThirdPartyImpl:
        """**独立**标杆: TF ResizeBilinearGrad(见 ``_tp_bilinear_grad_tf``)。"""

        def __init__(self, *, device=None, **kwargs):
            self.device = _require_cpu_device(device, "tf.raw_ops.ResizeBilinearGrad")

        def __call__(
            self,
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(input_size).reshape(-1)]
            ac = bool(align_corners)
            if not ac and (
                _opt_scale(scales_h) is not None or _opt_scale(scales_w) is not None
            ):
                raise NotImplementedError(
                    "显式正 scales: TF 无独立 scales 入参, 坐标尺度恒按 in/out "
                    "推导; 该象限由 ATen provider 与文件内 fp64 golden 覆盖"
                )
            return [_tp_bilinear_grad_tf(grad_output, isz, ac, False)]

    third_party = {"torch": ThirdPartyImpl, "tf": TfThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
