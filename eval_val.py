#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate PIN / PIN-Seq UAV models  (默认权重 results/PINN+SERESMLP+LSTM.pth)
输出：RMSE / MAE / AvgMaxErr   （单位：米）
"""

from __future__ import annotations
import argparse, os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# ───── 项目内部依赖 ────────────────────────────────────────────
import src.config as config
from src.trajectory_dataset import TrajectoryDataset
from stage_1_dataset_gen import load_dataset
from src.detecting_region_info_generator import generate_detecting_region_infos
from src.uav_model import PinUavModel, PinUavSeqModel

# ═════════ 1. 数据集 ══════════════════════════════════════════
def load_val_dataset(seq_len: int) -> TrajectoryDataset:
    ds = load_dataset(config.stage_1_dataset_test_path)
    return TrajectoryDataset(
        ds["features"], ds["labels"], ds["extra_infos"], ds["trajectory_lengths"],
        sequence_length=seq_len,
    )

# ═════════ 2. 模型 ════════════════════════════════════════════
def build_model(name: str, dev: torch.device) -> torch.nn.Module:
    region = generate_detecting_region_infos(
        config.num_detecting_regions, seed=config.region_seed
    )[0]
    T = torch.tensor(region.transmittor_position, dtype=torch.float32, device=dev)
    R = torch.tensor(
        [region.receiver_position_1, region.receiver_position_2, region.receiver_position_3],
        dtype=torch.float32, device=dev
    )
    return (PinUavModel if name == "pin" else PinUavSeqModel)(T, R).to(dev)

# ═════════ 3. 权重加载（自动剥前缀） ═══════════════════════════
def _strip(state: Dict[str, torch.Tensor], pref: str) -> Dict[str, torch.Tensor]:
    n = len(pref)
    return {k[n:]: v for k, v in state.items() if k.startswith(pref)}

def load_checkpoint(model: torch.nn.Module, path: str, dev: torch.device) -> None:
    raw = torch.load(path, map_location=dev)

    # ① 提取 state_dict
    if isinstance(raw, torch.nn.Module):
        state = raw.state_dict()
    elif isinstance(raw, dict):
        state = raw.get("base_model",
                 raw.get("state_dict", raw))
    else:
        raise RuntimeError("无法识别 checkpoint 格式")

    # ② 尝试严格加载；失败则剥前缀再试
    for cand in (state,
                 _strip(state, "base_model."),
                 _strip(state, "module.")):
        try:
            model.load_state_dict(cand, strict=True)
            print(f"[Info] 权重加载成功 ({os.path.basename(path)})")
            return
        except RuntimeError:
            pass
    raise RuntimeError("模型结构与权重不匹配，请检查类定义")

# ═════════ 4. 极坐标→XY ═══════════════════════════════════════
def polar_to_xy(r, s, c, *, norm, scale):
    if norm:
        r = r * scale
    return torch.stack((r * c, r * s), -1)

# ═════════ 5. 评测 ════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, seq_len, *, norm, scale, dev) -> Dict[str, Any]:
    model.eval()
    errors: List[float] = []
    traj_ids: List[int] = []

    ds: TrajectoryDataset = loader.dataset  # type: ignore
    traj_lengths = ds.trajectory_lengths

    # sample → trajectory id
    if seq_len == 1:
        for tid, L in enumerate(traj_lengths):
            traj_ids.extend([tid] * L)
    else:
        cum, bounds = 0, []
        for L in traj_lengths:
            bounds.append((cum, cum + L - 1)); cum += L
        for st in ds.valid_indices:
            tgt = st + seq_len - 1
            traj_ids.append(next(tid for tid, (lo, hi) in enumerate(bounds) if lo <= tgt <= hi))

    for batch in loader:
        feats, lab, _ = (batch if seq_len == 1 else (batch[0], batch[1], batch[2]))
        pred = (model(feats.to(dev)) if seq_len == 1 else model(feats.to(dev))[:, -1])
        lab = lab.to(dev)

        # to XY
        if pred.shape[-1] == 3:
            pred = polar_to_xy(pred[:, 0], pred[:, 1], pred[:, 2], norm=norm, scale=scale)
        if lab.shape[-1] == 3:
            lab = polar_to_xy(lab[:, 0], lab[:, 1], lab[:, 2], norm=norm, scale=scale)

        # flatten() 保证即使只有 1 个元素也返回列表
        errors.extend(torch.linalg.norm(pred - lab, dim=-1).cpu().flatten().tolist())

    e = np.asarray(errors, dtype=np.float64)
    per_max = {}
    for v, tid in zip(e, traj_ids):
        per_max[tid] = max(per_max.get(tid, 0.0), v)

    return {"samples": int(e.size),
            "traj": len(traj_lengths),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "mae":  float(e.mean()),
            "avgmax": float(np.mean(list(per_max.values())))}

# ═════════ 6. CLI ═════════════════════════════════════════════
def main():
    default_ckpt = "results/PINN+SERESMLP+LSTM.pth"

    parser = argparse.ArgumentParser("UAV evaluator (默认权重已设置)")
    parser.add_argument("--model", choices=["pin", "pinseq"], default="pinseq")
    parser.add_argument("--ckpt", default=default_ckpt, help="checkpoint 路径")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dev = torch.device(args.device)
    seq_len = config.sequence_length if args.model == "pinseq" else 1

    dataset = load_val_dataset(seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(args.model, dev)
    load_checkpoint(model, args.ckpt, dev)

    norm = getattr(config, "normalize_distance", False)
    scale = getattr(config, "distance_scale", 1.0)

    m = evaluate(model, loader, seq_len, norm=norm, scale=scale, dev=dev)

    print("\n=== Validation Result ===")
    print(f"Samples      : {m['samples']}")
    print(f"Trajectories : {m['traj']}")
    print(f"RMSE   (m)   : {m['rmse']:.6f}")
    print(f"MAE    (m)   : {m['mae']:.6f}")
    print(f"AvgMaxErr (m): {m['avgmax']:.6f}")

if __name__ == "__main__":
    main()