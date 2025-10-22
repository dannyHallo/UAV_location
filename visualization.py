#!/usr/bin/env python3
# ============================================================
#  Trajectory Visualizer & Robust-PINN Predictor  (Tkinter GUI)
# ------------------------------------------------------------
#  • 支持 PinUavModel / PinUavSeqModel
#  • 自动剥离 state_dict 前缀： base_model.|model.|module.
#  • strict=False 宽容加载，避免 missing / unexpected key 报错
# ============================================================

import os, sys, warnings, pathlib, math
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

# ─────────── 第三方依赖检查 ───────────
try:
    import scipy  # 用于 smooth_coords
except ModuleNotFoundError as e:
    tk.messagebox.showerror("Dependency Error", f"Required library missing: {e}")
    sys.exit(1)

# suppress the FutureWarning about torch.load untrusted models
warnings.filterwarnings(
    "ignore",
    message=".*You are using torch.load with weights_only=False.*",
)

# ─────────── Matplotlib 内嵌 ───────────
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

# ─────────── 工程内模块 ───────────
sys.path.append(os.path.abspath("./"))  # 确保可以 import src.*

import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos
from src.trajectory_generator import (
    generate_trajectories_in_region,
    generate_lines_from_trajectory,
)
from filter import smooth_coords

try:
    import torch
    from src.uav_model import PinUavModel, PinUavSeqModel
    import src.doppler_info as doppler_info_module
    from src.get_phi_info import get_angle_phi
    from src.w_and_doppler_generator import generate_w_and_doppler
    from src.features_and_labels_generator import get_features, get_labels

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    IMPORT_ERROR_MSG = (
        f"A required library for prediction is missing: {e}\n\n"
        "Please ensure PyTorch and all project dependencies are installed. "
        "Prediction will be disabled."
    )


# ============================================================
#                 utils: strip_state_dict_prefix
# ============================================================
def strip_prefix_from_state_dict(
    state_dict, prefixes=("base_model.", "model.", "module.")
):
    """
    把 state_dict 里以 prefixes 任一字符串开头的前缀去掉。
    """
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                break
        out[k] = v
    return out


# ============================================================
#                         Tk GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trajectory Visualizer & Predictor")
        self.geometry("1020x770")

        # ─────────── 状态变量 ───────────
        self.trajectories = []
        self.outer_vertices, self.inner_vertices = [], []
        self.detecting_region_info = None
        self.predictions, self.true_labels = None, None
        self.model, self.device = None, None
        self.plot_in_local_frame = True
        self.status_var = tk.StringVar(value="Welcome.")

        # ─────────── 顶栏：生成参数 ───────────
        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Label(ctrl, text="Seed").pack(side=tk.LEFT)
        self.seed_var = tk.StringVar(value=config.visualize_line_seed)
        tk.Entry(ctrl, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Lines").pack(side=tk.LEFT)
        self.num_lines_var = tk.StringVar(value="5")
        tk.Entry(ctrl, textvariable=self.num_lines_var, width=8).pack(
            side=tk.LEFT, padx=4
        )

        tk.Button(ctrl, text="Generate", width=10, command=self.run_generation).pack(
            side=tk.LEFT, padx=10
        )

        # ─────────── 轨迹选择 & 预测按钮 ───────────
        visor = tk.Frame(self)
        visor.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Label(visor, text="Visualize").pack(side=tk.LEFT)
        self.trajectory_selector = ttk.Combobox(visor, state="readonly", width=18)
        self.trajectory_selector.pack(side=tk.LEFT)
        self.trajectory_selector.bind("<<ComboboxSelected>>", self.on_traj_select)

        self.predict_btn = tk.Button(
            visor,
            text="Predict",
            width=10,
            state=tk.DISABLED,
            command=self.run_prediction,
        )
        self.predict_btn.pack(side=tk.LEFT, padx=20)

        # ─────────── 模型加载 ───────────
        mdl = tk.Frame(self)
        mdl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Button(mdl, text="Load Model", width=12, command=self.load_model).pack(
            side=tk.LEFT
        )

        tk.Label(mdl, text="SeqLen").pack(side=tk.LEFT, padx=6)
        self.sequence_length_var = tk.StringVar(value=str(config.sequence_length))
        tk.Entry(mdl, textvariable=self.sequence_length_var, width=5).pack(side=tk.LEFT)

        self.model_path_var = tk.StringVar(value="No model loaded")
        tk.Label(mdl, textvariable=self.model_path_var).pack(side=tk.LEFT, padx=10)

        # ─────────── Matplotlib 画布 ───────────
        plot_frame = tk.Frame(self, relief=tk.SUNKEN, bd=1)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ─────────── 状态栏 ───────────
        status = tk.Frame(self, relief=tk.SUNKEN, bd=1)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(status, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        if not TORCH_AVAILABLE:
            messagebox.showwarning("Dependency Error", IMPORT_ERROR_MSG)
        self.run_generation()

    # ========================================================
    #                 生成 / 加载 / 预测
    # ========================================================
    def run_generation(self):
        try:
            seed = int(self.seed_var.get())
            num_lines = int(self.num_lines_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Seed & Lines must be integers.")
            return

        self.predictions = self.true_labels = None

        region_infos = generate_detecting_region_infos(1, seed=config.region_seed)
        if not region_infos:
            messagebox.showwarning(
                "Generation Failed", f"Could not generate region for seed {seed}"
            )
            return
        self.detecting_region_info = region_infos[0]

        self.trajectories, self.outer_vertices, self.inner_vertices = (
            generate_trajectories_in_region(
                self.detecting_region_info, num_lines, seed=seed, inner_scale_factor=0.7
            )
        )

        if self.trajectories:
            self.trajectory_selector["values"] = [t.name for t in self.trajectories]
            self.trajectory_selector.current(0)
        else:
            self.trajectory_selector["values"] = []
            self.trajectory_selector.set("")
        self.visualize_selected_trajectory()

    # ---------- Model loader ----------
    def load_model(self):
        if not TORCH_AVAILABLE:
            self.status_var.set("⚠️ PyTorch not installed.")
            return
        try:
            seq_len = int(self.sequence_length_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Sequence Length must be integer")
            return

        fp = filedialog.askopenfilename(
            title="Select PyTorch Model",
            filetypes=[("PyTorch Model", "*.pth"), ("All", "*.*")],
            initialdir="./models",
        )
        if not fp:
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # T & R from region (fallback to default)
            if self.detecting_region_info:
                T = torch.tensor(
                    self.detecting_region_info.transmittor_position,
                    dtype=torch.float32,
                    device=self.device,
                )
                R = torch.stack(
                    [
                        torch.tensor(
                            self.detecting_region_info.receiver_position_1,
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.tensor(
                            self.detecting_region_info.receiver_position_2,
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.tensor(
                            self.detecting_region_info.receiver_position_3,
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                T = torch.tensor([0.0, 0.0], device=self.device)
                R = torch.tensor(
                    [[100.0, 0.0], [50.0, 86.6], [-50.0, 86.6]], device=self.device
                )

            ModelCls = PinUavSeqModel if seq_len > 1 else PinUavModel
            self.model = ModelCls(T, R).to(self.device)

            raw_ckpt = torch.load(fp, map_location=self.device, weights_only=True)
            ckpt = strip_prefix_from_state_dict(raw_ckpt)
            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)

            self.model.eval()
            self.model_path_var.set(f"Loaded: {os.path.basename(fp)}")
            self.predict_btn.config(state=tk.NORMAL)
            self.status_var.set(
                f"✅ Model loaded ({len(missing)} missing, {len(unexpected)} extra)"
            )
        except Exception as e:
            self.model = None
            self.predict_btn.config(state=tk.DISABLED)
            tb = traceback.format_exc()
            print(tb)
            self.status_var.set(f"❌ Model load error: {e}")

    # ---------- Prediction ----------
    def run_prediction(self):
        if not self.model:
            self.status_var.set("⚠️ Load model first.")
            return
        idx = self.trajectory_selector.current()
        if idx == -1:
            self.status_var.set("⚠️ Select a trajectory.")
            return
        try:
            seq_len = int(self.sequence_length_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Sequence Length must be int")
            return

        traj = self.trajectories[idx]
        feats, labels = self._prepare_data_for_model(
            traj, self.detecting_region_info, seq_len
        )
        if feats is None:
            self.status_var.set("ℹ️ Trajectory too short.")
            return

        feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
        if seq_len == 1:  # (N,6)
            pass
        else:  # (N_seq, seq_len, 6)
            pass
        try:
            with torch.no_grad():
                cart = self.model(feats_t)
                if cart.dim() == 3:
                    x, y = cart[:, :, 0], cart[:, :, 1]
                else:
                    x, y = cart[:, 0], cart[:, 1]
                r = torch.sqrt(x**2 + y**2 + 1e-8)
                sinT = y / r
                cosT = x / r
                polar = torch.stack([r, sinT, cosT], dim=-1)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            self.status_var.set(f"❌ Prediction error: {e}")
            return

        self.predictions = polar.cpu().numpy()
        self.true_labels = labels
        self.status_var.set("✅ Prediction complete.")
        self.visualize_selected_trajectory()

    # ========================================================
    #                 Data preparation
    # ========================================================
    def _prepare_data_for_model(self, trajectory, region, seq_len):
        TIME_INTERVAL = 0.01
        FC = 6e9
        C = 3e8
        dop_info = doppler_info_module.DopplerInfo(
            c=C, fc=FC, time_interval=TIME_INTERVAL
        )
        ca, cb = generate_lines_from_trajectory(trajectory, TIME_INTERVAL)

        if len(ca) < seq_len:
            return None, None

        phis = np.array([get_angle_phi(region, a, b) for a, b in zip(ca, cb)])
        w, dop = generate_w_and_doppler(region, dop_info, ca, cb)
        feats = get_features(phis, w, dop)
        labels = get_labels(region, cb)

        if seq_len <= 1:
            return feats, labels
        # sliding window
        n_seq = len(feats) - seq_len + 1
        seq_feats = np.stack([feats[i : i + seq_len] for i in range(n_seq)], axis=0)
        seq_labels = labels[seq_len - 1 :]
        return seq_feats, seq_labels

    # ========================================================
    #                 可视化
    # ========================================================
    def on_traj_select(self, _=None):
        self.predictions = self.true_labels = None
        self.visualize_selected_trajectory()

    def visualize_selected_trajectory(self):
        self.ax.clear()

        def to_local(pts):
            return (
                self.detecting_region_info.transform_points_to_tx_rx1(pts)
                if self.plot_in_local_frame and self.detecting_region_info
                else pts
            )

        # draw boundaries
        if self.outer_vertices:
            self.ax.add_patch(
                Polygon(
                    to_local(self.outer_vertices),
                    fill=False,
                    edgecolor="k",
                    lw=2,
                    label="Outer",
                )
            )
        if self.inner_vertices:
            self.ax.add_patch(
                Polygon(
                    to_local(self.inner_vertices),
                    fill=False,
                    edgecolor="g",
                    ls="--",
                    lw=1.5,
                    label="GenArea",
                )
            )

        idx = self.trajectory_selector.current()
        if idx != -1 and self.trajectories:
            traj = self.trajectories[idx]
            if self.predictions is None:
                if traj.trajectory:
                    pts = to_local(np.array(traj.trajectory))
                    self.ax.plot(pts[:, 0], pts[:, 1], "b-", lw=2, label=traj.name)
                if traj.key_points:
                    kp = to_local(np.array(traj.key_points))
                    self.ax.scatter(
                        kp[:, 0],
                        kp[:, 1],
                        s=70,
                        c="purple",
                        edgecolors="k",
                        zorder=5,
                        label="KeyPts",
                    )
            else:

                def polar2cart(arr):
                    rho, s, c = arr[:, 0], arr[:, 1], arr[:, 2]
                    return np.stack([rho * c, rho * s], axis=1)

                # 统一预测极坐标形状到 (N,3)
                pred_polar = self.predictions
                try:
                    shp = getattr(pred_polar, "shape", None)
                    if (
                        isinstance(pred_polar, np.ndarray)
                        and pred_polar.ndim == 3
                        and pred_polar.shape[-1] == 3
                    ):
                        # 若存在多余的中间维（如序列维），取最后一帧
                        pred_polar = pred_polar[:, -1, :]
                        print(f"[viz] collapsed pred_polar to shape={pred_polar.shape}")
                except Exception:
                    pass

                true = polar2cart(self.true_labels)
                pred = polar2cart(pred_polar)
                # 调试输出预测形状
                try:
                    print(
                        f"[viz] true shape={getattr(self.true_labels,'shape',None)}, pred shape={getattr(pred,'shape',None)}"
                    )
                except Exception:
                    pass
                # 形状与长度检查后再平滑
                if (
                    isinstance(pred, np.ndarray)
                    and pred.ndim == 2
                    and pred.shape[1] == 2
                    and len(pred) >= 3
                ):
                    pred = smooth_coords(
                        pred,
                        method="adaptive",
                        min_window=50,
                        max_window=100,
                        polyorder=3,
                    )
                else:
                    self.status_var.set(
                        "ℹ️ Pred too short or bad shape; skip smoothing."
                    )

                self.ax.plot(true[:, 0], true[:, 1], "bo-", ms=4, label="True")
                self.ax.plot(pred[:, 0], pred[:, 1], "r--x", ms=4, label="Pred")
                for t, p in zip(true, pred):
                    self.ax.plot([t[0], p[0]], [t[1], p[1]], color="gray", lw=0.6)

        # limits
        if self.outer_vertices:
            ov = to_local(self.outer_vertices)
            xs, ys = zip(*ov)
            pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys))
            self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
            self.ax.set_ylim(min(ys) - pad, max(ys) + pad)

        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X (TX→RX1)")
        self.ax.set_ylabel("Y (CCW 90°)")
        self.ax.set_title(f"Trajectory Visualization (Seed {self.seed_var.get()})")
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        self.fig.tight_layout(rect=[0, 0, 0.82, 1])
        self.canvas.draw()


# ============================================================
#                        main
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
