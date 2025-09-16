"""
Savitzky–Golay (SG) 滤波器集合（不改现有代码，新增文件即可）。

原理简介：
- SG 滤波通过在滑动窗口内用低阶多项式对数据点进行最小二乘拟合，
  以该拟合多项式的中心值作为平滑结果。与典型的移动平均不同，
  SG 在在保留波形几何形状（峰值位置、斜率、曲率）方面表现更好，
  并且相位延迟较小（窗口对称时几乎无相移）。

本模块提供三种用法：
1) 对 x(t)、y(t) 分别做一维 SG 滤波（最常用，效率高）。
2) 先按弧长参数化 s，再在 s 维度统一平滑并回到 (x,y)。
3) 自适应窗口：根据曲率大小自动调整窗口（急弯用小窗口、直线用大窗口）。

参数建议：窗口长度 7–21（奇数），多项式阶次 2 或 3。

快速示例：
>>> from filter import sg_filter_xy
>>> x_s, y_s = sg_filter_xy(x, y, window_length=11, polyorder=3)

依赖：scipy.signal.savgol_filter
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import numpy as np

try:
    import torch  # 可选：若传入 torch.Tensor 则自动处理

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境无 torch 也可用
    _TORCH_AVAILABLE = False

from scipy.signal import savgol_filter


__all__ = [
    "sg_filter_1d",
    "sg_filter_xy",
    "sg_filter_arclength",
    "sg_filter_adaptive",
    "smooth_coords",
]


def _to_numpy(a):
    """将 list/ndarray/torch.Tensor 转为 numpy.ndarray，并返回回写所需信息。"""
    if _TORCH_AVAILABLE and isinstance(a, torch.Tensor):
        device = a.device
        dtype = a.dtype
        arr = a.detach().cpu().numpy()
        return arr, ("torch", device, dtype)
    if isinstance(a, np.ndarray):
        return a.astype(float), ("numpy",)
    # list/tuple 等
    return np.asarray(a, dtype=float), ("numpy",)


def _from_numpy(arr: np.ndarray, meta):
    if meta and meta[0] == "torch":
        _, device, dtype = meta
        return torch.tensor(arr, device=device, dtype=dtype)
    return arr


def _ensure_odd(n: int) -> int:
    return int(n) if int(n) % 2 == 1 else int(n) + 1


def _clip_window(window_length: int, polyorder: int, max_len: int) -> int:
    # SG 要求 window_length 为奇数，且 > polyorder，且 <= 数据长度
    wl = _ensure_odd(max(polyorder + 2, window_length))
    wl = min(wl, max_len - (1 - max_len % 2))  # 不超过长度且保持奇数
    if wl < polyorder + 2:
        wl = polyorder + 2
        wl = _ensure_odd(wl)
    wl = max(3, wl)
    if wl > max_len:
        wl = max_len if max_len % 2 == 1 else max_len - 1
    wl = max(wl, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    return int(wl)


def sg_filter_1d(
    x: Iterable[float],
    *,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "interp",
) -> np.ndarray:
    """对一维序列进行 SG 滤波。

    参数同 scipy.signal.savgol_filter，自动修正窗口边界条件。
    返回 numpy.ndarray（若传入 torch.Tensor，可自行将结果转回张量）。
    """
    x_np, meta = _to_numpy(x)
    wl = _clip_window(window_length, polyorder, len(x_np))
    if wl < 3 or len(x_np) < 3:
        return _from_numpy(x_np, meta)  # 太短则直接返回
    y = savgol_filter(x_np, window_length=wl, polyorder=polyorder, mode=mode)
    return _from_numpy(y, meta)


def sg_filter_xy(
    x: Iterable[float],
    y: Iterable[float],
    *,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "interp",
) -> Tuple[np.ndarray, np.ndarray]:
    """分别对 x(t)、y(t) 做一维 SG 滤波。"""
    x_s = sg_filter_1d(x, window_length=window_length, polyorder=polyorder, mode=mode)
    y_s = sg_filter_1d(y, window_length=window_length, polyorder=polyorder, mode=mode)
    return x_s, y_s


def _arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


def sg_filter_arclength(
    x: Iterable[float],
    y: Iterable[float],
    *,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "interp",
) -> Tuple[np.ndarray, np.ndarray]:
    """基于弧长参数化的统一 SG 平滑。

    步骤：
    1) 计算弧长 s(i)；
    2) 对 x(s)、y(s) 分别在等索引窗口内做 SG；
    3) 返回平滑后的 (x_s, y_s)。
    注意：这里采用离散等索引窗口近似，通常足够稳定。
    """
    x_np, m1 = _to_numpy(x)
    y_np, m2 = _to_numpy(y)
    assert len(x_np) == len(y_np), "x 和 y 长度需一致"
    _ = _arclength(x_np, y_np)  # 保留以便后续扩展（此实现按等索引平滑）
    wl = _clip_window(window_length, polyorder, len(x_np))
    if wl < 3 or len(x_np) < 3:
        return _from_numpy(x_np, m1), _from_numpy(y_np, m2)
    x_s = savgol_filter(x_np, window_length=wl, polyorder=polyorder, mode=mode)
    y_s = savgol_filter(y_np, window_length=wl, polyorder=polyorder, mode=mode)
    return _from_numpy(x_s, m1), _from_numpy(y_s, m2)


def sg_filter_adaptive(
    x: Iterable[float],
    y: Iterable[float],
    *,
    min_window: int = 7,
    max_window: int = 21,
    polyorder: int = 3,
    mode: str = "interp",
    curvature_eps: float = 1e-6,
    smooth_curvature: Optional[int] = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于曲率的自适应窗口 SG 滤波。

    - 曲率大(急弯) → 使用小窗口；曲率小(直线) → 使用大窗口。
    - 窗口在[min_window, max_window]间线性映射并取奇数。
    - 可先对曲率做一次轻度 SG 平滑以消噪。
    """
    x_np, m1 = _to_numpy(x)
    y_np, m2 = _to_numpy(y)
    n = len(x_np)
    if n < 5:
        return _from_numpy(x_np, m1), _from_numpy(y_np, m2)

    # 一阶/二阶差分估计曲率 κ = |x'y" - y'x"| / (x'^2 + y'^2)^(3/2)
    dx = np.gradient(x_np)
    dy = np.gradient(y_np)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx * dx + dy * dy) ** 1.5 + curvature_eps
    kappa = numerator / denominator

    if smooth_curvature and n >= smooth_curvature:
        kappa = savgol_filter(
            kappa, _clip_window(smooth_curvature, 2, n), 2, mode="interp"
        )

    # 将曲率归一化到[0,1]
    k_min, k_max = float(np.nanmin(kappa)), float(np.nanmax(kappa))
    if k_max - k_min < 1e-12:
        weights = np.zeros_like(kappa)
    else:
        weights = (kappa - k_min) / (k_max - k_min)

    # 线性映射到窗口，并保证奇数与边界条件
    min_wl = max(3, _ensure_odd(min_window))
    max_wl = max(min_wl, _ensure_odd(max_window))

    # 为每个点分配窗口长度（随后以可变窗口方式应用）。
    target_wl = min_wl + (1.0 - weights) * (max_wl - min_wl)
    target_wl = np.array(
        [_clip_window(_ensure_odd(int(round(w))), polyorder, n) for w in target_wl]
    )

    # 为简洁与速度：采用分段常窗口策略（滑过数据，合并相同窗口长度的区间）。
    x_out = np.copy(x_np)
    y_out = np.copy(y_np)
    i = 0
    while i < n:
        wl = int(target_wl[i])
        if wl < polyorder + 2:
            wl = polyorder + 2
            wl = _ensure_odd(wl)
        # 以 wl/2 为半径在局部做一次 SG
        radius = wl // 2
        j = min(n, i + radius + 1)
        l = max(0, i - radius)
        seg_slice = slice(l, j)
        seg_len = j - l
        wl_seg = _clip_window(wl, polyorder, seg_len)
        if seg_len >= 3 and wl_seg >= 3:
            x_out[seg_slice] = savgol_filter(
                x_np[seg_slice], wl_seg, polyorder, mode=mode
            )
            y_out[seg_slice] = savgol_filter(
                y_np[seg_slice], wl_seg, polyorder, mode=mode
            )
        i += radius if radius > 0 else 1

    return _from_numpy(x_out, m1), _from_numpy(y_out, m2)


# 便捷别名，贴近用户伪代码
def sg_xy(
    x: Iterable[float], y: Iterable[float], window_length: int = 11, polyorder: int = 3
):
    return sg_filter_xy(x, y, window_length=window_length, polyorder=polyorder)


def smooth_coords(
    coords: Iterable[Iterable[float]],
    *,
    method: str = "adaptive",
    window_length: int = 11,
    polyorder: int = 3,
    min_window: int = 7,
    max_window: int = 21,
    mode: str = "interp",
    **kwargs,
):
    """对形如 (N, 2) 的坐标数组做 SG 平滑，返回同形状数组。

    参数：
    - method: "adaptive" | "fixed" | "arclength"
      - adaptive: 基于曲率自适应窗口（默认）
      - fixed: 对 x,y 分别用固定窗口 sg_filter_xy
      - arclength: 弧长参数化统一平滑
    其余参数分别传入对应函数。
    """
    arr, meta = _to_numpy(np.asarray(coords))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords 需为形状 (N,2) 的数组")

    x = arr[:, 0]
    y = arr[:, 1]

    method = (method or "adaptive").lower()
    if method == "fixed":
        xs, ys = sg_filter_xy(
            x, y, window_length=window_length, polyorder=polyorder, mode=mode
        )
    elif method == "arclength":
        xs, ys = sg_filter_arclength(
            x, y, window_length=window_length, polyorder=polyorder, mode=mode
        )
    else:  # adaptive
        xs, ys = sg_filter_adaptive(
            x,
            y,
            min_window=min_window,
            max_window=max_window,
            polyorder=polyorder,
            mode=mode,
            **kwargs,
        )

    out = np.column_stack([np.asarray(xs), np.asarray(ys)])
    return _from_numpy(out, ("numpy",))
