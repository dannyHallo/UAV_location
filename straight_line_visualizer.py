"""
Straight-line Visualizer (follow visualization.py flow)

Flow (same as visualization.py):
- Click Generate → run_generation()
  - generate_detecting_region_infos(): 1 region
  - generate straight-line trajectories (num_lines)
  - save to self.trajectories and refresh combobox
- Selecting an item only changes index and re-draws (no re-generation)
- To get a new batch, click Generate again

Difference vs visualization.py: trajectories are straight-line samples
inside an inner-scaled quadrilateral; prediction pipeline (features/labels)
is the same (per small segment pair line_a/line_b).
"""


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import warnings
import sys
import os
import numpy as np

import src.config as config

warnings.filterwarnings(
    "ignore",
    message=".*You are using torch.load with weights_only=False.*",
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

from src.detecting_region_info_generator import generate_detecting_region_infos

# Prediction deps (optional)
try:
    import torch
    from src.uav_model import UavModel
    import src.doppler_info as doppler_info_module
    from src.get_phi_info import get_angle_phi
    from src.w_and_doppler_generator import generate_w_and_doppler
    from src.features_and_labels_generator import get_features, get_labels

    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    IMPORT_ERROR_MSG = (
        f"Prediction dependencies missing: {e}. Please use (pykan) with deps installed."
    )

# Straight-line generator
from straight_line_trajectory_generator import (
    StraightLineTrajectoryGenerator as SLGen,
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trajectory Visualizer and Predictor (Straight Lines)")
        self.geometry("950x750")

        # State (mirror visualization.py)
        self.trajectories = []  # list[np.ndarray], each [N,2]
        self.outer_vertices = []
        self.inner_vertices = []
        self.detecting_region_info = None

        self.model = None
        self.device = None
        self.predictions = None
        self.true_labels = None

        self.status_var = tk.StringVar(value="Welcome.")

        # Top controls
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(top, text="Seed:").pack(side=tk.LEFT, padx=(0, 4))
        self.seed_var = tk.StringVar(value=str(config.visualize_line_seed))
        tk.Entry(top, textvariable=self.seed_var, width=8).pack(side=tk.LEFT)

        tk.Label(top, text="Num Lines:").pack(side=tk.LEFT, padx=(10, 4))
        self.num_lines_var = tk.StringVar(value="5")
        tk.Entry(top, textvariable=self.num_lines_var, width=8).pack(side=tk.LEFT)

        tk.Label(top, text="Inner Scale:").pack(side=tk.LEFT, padx=(10, 4))
        self.scale_var = tk.StringVar(value="0.7")
        tk.Entry(top, textvariable=self.scale_var, width=8).pack(side=tk.LEFT)

        tk.Label(top, text="Min Len (m):").pack(side=tk.LEFT, padx=(10, 4))
        self.min_len_var = tk.StringVar(value="50")
        tk.Entry(top, textvariable=self.min_len_var, width=8).pack(side=tk.LEFT)

        tk.Button(top, text="Generate", command=self.run_generation).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Second row: selection + predict
        row2 = tk.Frame(self)
        row2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(row2, text="Select:").pack(side=tk.LEFT)
        self.trajectory_selector = ttk.Combobox(row2, state="readonly", width=18)
        self.trajectory_selector.pack(side=tk.LEFT, padx=(5, 10))
        self.trajectory_selector.bind("<<ComboboxSelected>>", self.on_trajectory_select)

        self.predict_btn = tk.Button(
            row2, text="Predict", command=self.run_prediction, state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Third row: model load
        row3 = tk.Frame(self)
        row3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        tk.Button(row3, text="Load Model", command=self.load_model).pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value="No model loaded.")
        tk.Label(row3, textvariable=self.model_path_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Status bar
        status = tk.Frame(self, relief=tk.SUNKEN, bd=1)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(status, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        # Plot
        frame = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------- Generation ----------------
    def run_generation(self):
        try:
            seed = int(self.seed_var.get())
            num_lines = int(self.num_lines_var.get())
            inner_scale = float(self.scale_var.get())
            min_len = float(self.min_len_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Seed/Num Lines/Scale/Length must be numbers."
            )
            return

        if not (0.0 < inner_scale < 1.0):
            messagebox.showerror("Invalid Input", "Inner Scale must be in (0,1).")
            return
        if min_len <= 0:
            messagebox.showerror("Invalid Input", "Min Length must be positive.")
            return

        self.predictions = None
        self.true_labels = None

        infos = generate_detecting_region_infos(
            num_configurations=1, seed=config.region_seed
        )
        if not infos:
            messagebox.showwarning("Generation Failed", "Could not generate region.")
            return
        self.detecting_region_info = infos[0]

        self.outer_vertices = [
            self.detecting_region_info.transmittor_position,
            self.detecting_region_info.receiver_position_1,
            self.detecting_region_info.receiver_position_2,
            self.detecting_region_info.receiver_position_3,
        ]
        centroid = np.mean(self.outer_vertices, axis=0)
        self.inner_vertices = [
            centroid + (np.array(v) - centroid) * inner_scale
            for v in self.outer_vertices
        ]

        gen = SLGen(time_interval=config.time_interval)
        # 直接生成 line pairs，再还原整条 P 用于显示与预测
        la_list, lb_list = gen.generate_line_pairs(
            quad_vertices=self.inner_vertices,
            num_lines=num_lines,
            seed=seed,
            inner_scale=1.0,
            min_length=min_len,
        )

        self.trajectories = []
        for a, b in zip(la_list, lb_list):
            if len(a) and len(b):
                P = np.vstack([a, b[-1:]])
                self.trajectories.append(P)

        if self.trajectories:
            names = [f"Line {i+1}" for i in range(len(self.trajectories))]
            self.trajectory_selector["values"] = names
            self.trajectory_selector.current(0)
        else:
            self.trajectory_selector["values"] = []
            self.trajectory_selector.set("")
            messagebox.showinfo("Info", "No straight lines generated.")

        self.visualize_selected_trajectory()

    # ---------------- Model load ----------------
    def load_model(self):
        if not TORCH_AVAILABLE:
            self.status_var.set("⚠️ Cannot load model: PyTorch not installed.")
            return
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*")),
            initialdir="./models",
        )
        if not path:
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UavModel().to(self.device)
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt)
            model.eval()
            self.model = model
            self.model_path_var.set(f"Loaded: {os.path.basename(path)}")
            self.predict_btn.config(state=tk.NORMAL)
            self.status_var.set(f"✅ Model loaded on {self.device}")
        except Exception as e:
            self.model = None
            self.device = None
            self.predict_btn.config(state=tk.DISABLED)
            self.model_path_var.set("Failed to load model.")
            self.status_var.set(f"❌ Model load error: {e}")

    # ---------------- Selection ----------------
    def on_trajectory_select(self, event=None):
        self.predictions = None
        self.true_labels = None
        self.visualize_selected_trajectory()

    # ---------------- Prediction ----------------
    def run_prediction(self):
        if not self.model:
            self.status_var.set("⚠️ Please load a model first.")
            return
        idx = self.trajectory_selector.current()
        if idx == -1 or not self.trajectories:
            self.status_var.set("⚠️ Please generate and select a trajectory.")
            return
        line = self.trajectories[idx]
        feats, labs = self._prepare_data_for_model_from_line(
            line, self.detecting_region_info
        )
        if feats is None:
            self.status_var.set("ℹ️ Line too short, no prediction.")
            self.visualize_selected_trajectory()
            return
        try:
            x = torch.tensor(feats, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y = self.model(x)
            self.predictions = y.cpu().numpy()
            self.true_labels = labs
            self.status_var.set("✅ Prediction complete.")
        except Exception as e:
            self.predictions = None
            self.true_labels = None
            self.status_var.set(f"❌ Prediction error: {e}")
        self.visualize_selected_trajectory()

    def _prepare_data_for_model_from_line(self, line_points: np.ndarray, info):
        if line_points is None or len(line_points) < 2:
            return None, None
        coords_a = line_points[:-1]
        coords_b = line_points[1:]
        phis1234 = np.array(
            [get_angle_phi(info, a, b) for a, b in zip(coords_a, coords_b)]
        )
        dop = doppler_info_module.DopplerInfo(
            c=config.c, fc=config.fc, time_interval=config.time_interval
        )
        w, doppler = generate_w_and_doppler(info, dop, coords_a, coords_b, phis1234)
        features = get_features(phis1234, w, doppler)
        labels = get_labels(info, coords_b)
        return features, labels

    # ---------------- Visualization ----------------
    def visualize_selected_trajectory(self):
        self.ax.clear()

        # boundaries
        if self.outer_vertices:
            self.ax.add_patch(
                Polygon(
                    self.outer_vertices, fill=False, edgecolor="k", lw=2, label="Outer"
                )
            )
        if self.inner_vertices:
            self.ax.add_patch(
                Polygon(
                    self.inner_vertices,
                    fill=False,
                    edgecolor="g",
                    ls="--",
                    lw=2,
                    label="Inner",
                )
            )

        colors = ["red", "blue", "green", "orange"]
        labels = ["TX (T)", "RX1 (R1)", "RX2 (R2)", "RX3 (R3)"]
        for v, c, lb in zip(self.outer_vertices, colors, labels):
            self.ax.scatter(v[0], v[1], color=c, zorder=5)
            self.ax.text(
                v[0] + 3,
                v[1] + 3,
                lb,
                fontsize=9,
                fontweight="bold",
                color=c,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        # draw all and highlight selected as small segments
        idx = self.trajectory_selector.current()
        for i, P in enumerate(self.trajectories):
            if len(P) < 2:
                continue
            if i == idx:
                a, b = P[:-1], P[1:]
                self.ax.scatter(P[0, 0], P[0, 1], color="#1f77b4", s=35, marker="o")
                self.ax.scatter(P[-1, 0], P[-1, 1], color="#1f77b4", s=35, marker="s")
                for p, q in zip(a, b):
                    seg = np.vstack([p, q])
                    self.ax.plot(seg[:, 0], seg[:, 1], "-", color="#1f77b4", lw=2.5)
            else:
                self.ax.plot(P[:, 0], P[:, 1], "-", color="#c7c7c7", lw=1.3, alpha=0.8)

        # overlay prediction
        if self.predictions is not None and self.true_labels is not None and idx != -1:

            def polar_to_xy(data):
                rho, sin_t, cos_t = data[:, 0], data[:, 1], data[:, 2]
                tx, ty = self.detecting_region_info.transmittor_position
                return np.vstack((tx + rho * cos_t, ty + rho * sin_t)).T

            true_xy = polar_to_xy(self.true_labels)
            pred_xy = polar_to_xy(self.predictions)
            self.ax.plot(
                true_xy[:, 0], true_xy[:, 1], "b-o", ms=4, label="True seg", zorder=3
            )
            self.ax.plot(
                pred_xy[:, 0], pred_xy[:, 1], "r--x", ms=4, label="Pred", zorder=3
            )
            for t, p in zip(true_xy, pred_xy):
                self.ax.plot(
                    [t[0], p[0]],
                    [t[1], p[1]],
                    color="gray",
                    alpha=0.5,
                    lw=0.8,
                    zorder=2,
                )

        if self.outer_vertices:
            xs = [v[0] for v in self.outer_vertices]
            ys = [v[1] for v in self.outer_vertices]
            pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.15
            self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
            self.ax.set_ylim(min(ys) - pad, max(ys) + pad)

        self.ax.set_title("Straight-line Trajectories")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.tight_layout(rect=[0, 0, 0.82, 1])
        self.canvas.draw()


if __name__ == "__main__":
    try:
        import scipy  # keep same dependency check as visualization.py
    except ImportError as e:
        messagebox.showerror(
            "Dependency Error", f"Required library missing: 'scipy'\n{e}"
        )
        sys.exit(1)
    app = App()
    app.mainloop()
