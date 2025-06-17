import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import warnings
import sys
import os
import numpy as np
import src.config as config

# suppress the FutureWarning about torch.load untrusted models
warnings.filterwarnings(
    "ignore",
    message=".*You are using torch.load with weights_only=False.*",
)

# Ensure the 'src' directory is in the Python path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Matplotlib imports for embedding in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

# Import necessary functions and classes from the provided source files
from src.detecting_region_info_generator import generate_detecting_region_infos
from src.trajectory_generator import (
    generate_trajectories_in_region,
    _generate_lines_from_trajectory,
)

# --- Imports for Model Prediction ---
try:
    import torch
    from src.uav_model import UavModel
    import src.doppler_info as doppler_info_module
    from src.get_phi_info import get_angle_phi
    from src.w_and_doppler_generator import generate_w_and_doppler
    from src.features_and_labels_generator import get_features, get_labels

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    IMPORT_ERROR_MSG = f"A required library for prediction is missing: {e}\n\nPlease ensure PyTorch and all project dependencies are installed. Prediction will be disabled."


class App(tk.Tk):
    """
    An interactive Tkinter application to generate, visualize, and predict trajectories.
    """

    def __init__(self):
        super().__init__()
        self.title("Trajectory Visualizer and Predictor")
        self.geometry("950x750")

        # --- State variables ---
        self.trajectories = []
        self.outer_vertices = []
        self.inner_vertices = []
        self.detecting_region_info = None
        self.model = None
        self.device = None
        self.predictions = None
        self.true_labels = None

        # status bar variable
        self.status_var = tk.StringVar(value="Welcome.")

        # --- Top frame for generation controls ---
        gen_control_frame = tk.Frame(self)
        gen_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(gen_control_frame, text="Seed:", font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        # self.seed_var = tk.StringVar(value="42")
        self.seed_var = tk.StringVar(value=config.test_line_seed)
        self.seed_entry = tk.Entry(
            gen_control_frame, textvariable=self.seed_var, width=8
        )
        self.seed_entry.pack(side=tk.LEFT)

        tk.Label(gen_control_frame, text="Num Lines:", font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=(10, 5)
        )
        self.num_lines_var = tk.StringVar(value="5")
        self.num_lines_entry = tk.Entry(
            gen_control_frame, textvariable=self.num_lines_var, width=8
        )
        self.num_lines_entry.pack(side=tk.LEFT)

        self.generate_button = tk.Button(
            gen_control_frame,
            text="Generate",
            command=self.run_generation,
            font=("Helvetica", 10, "bold"),
        )
        self.generate_button.pack(side=tk.LEFT, padx=(10, 0))

        # --- Second frame for visualization and prediction controls ---
        vis_control_frame = tk.Frame(self)
        vis_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(vis_control_frame, text="Visualize:", font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.trajectory_selector = ttk.Combobox(
            vis_control_frame, state="readonly", width=15
        )
        self.trajectory_selector.pack(side=tk.LEFT)
        self.trajectory_selector.bind("<<ComboboxSelected>>", self.on_trajectory_select)

        self.predict_button = tk.Button(
            vis_control_frame,
            text="Predict",
            command=self.run_prediction,
            font=("Helvetica", 10, "bold"),
            state=tk.DISABLED,
        )
        self.predict_button.pack(side=tk.LEFT, padx=(20, 0))

        # --- Third frame for model loading ---
        model_frame = tk.Frame(self)
        model_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.load_model_button = tk.Button(
            model_frame,
            text="Load Model",
            command=self.load_model,
            font=("Helvetica", 10, "bold"),
        )
        self.load_model_button.pack(side=tk.LEFT)

        self.model_path_var = tk.StringVar(value="No model loaded.")
        tk.Label(
            model_frame, textvariable=self.model_path_var, font=("Helvetica", 9)
        ).pack(side=tk.LEFT, padx=(10, 0))

        # --- Status bar at the bottom ---
        status_frame = tk.Frame(self, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        # --- Main frame for the Matplotlib plot ---
        plot_frame = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Check for dependencies and perform initial generation
        if not TORCH_AVAILABLE:
            messagebox.showwarning("Dependency Error", IMPORT_ERROR_MSG)
        self.run_generation()

    def run_generation(self):
        """Generates a new detecting region and a set of trajectories."""
        try:
            seed = int(self.seed_var.get())
            num_lines = int(self.num_lines_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Seed and Number of Lines must be integers."
            )
            return

        # Clear previous prediction data
        self.predictions = None
        self.true_labels = None

        # 1. Generate the base detecting region
        region_infos = generate_detecting_region_infos(num_configurations=1, seed=config.region_seed)
        if not region_infos:
            messagebox.showwarning(
                "Generation Failed", f"Could not generate a valid region for seed {seed}."
            )
            return
        self.detecting_region_info = region_infos[0]

        # 2. Generate trajectories within the region
        self.trajectories, self.outer_vertices, self.inner_vertices = (
            generate_trajectories_in_region(
                self.detecting_region_info,
                num_lines,
                seed=seed,
                inner_scale_factor=0.7,
            )
        )

        # 3. Update the dropdown menu
        if self.trajectories:
            self.trajectory_selector["values"] = [t.name for t in self.trajectories]
            self.trajectory_selector.current(0)
        else:
            self.trajectory_selector["values"] = []
            self.trajectory_selector.set("")
            messagebox.showinfo(
                "Info", "No valid trajectories could be generated with these settings."
            )

        # 4. Trigger visualization of the first trajectory
        self.visualize_selected_trajectory()

    def load_model(self):
        """Opens a dialog to load a PyTorch model state dictionary."""
        if not TORCH_AVAILABLE:
            self.status_var.set("⚠️ Cannot load model: PyTorch not installed.")
            return

        filepath = filedialog.askopenfilename(
            title="Select a PyTorch Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*")),
            initialdir="./models",
        )
        if not filepath:
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UavModel().to(self.device)
            # safe‐load weights only, to reduce untrusted‐pickle attack surface
            try:
                checkpoint = torch.load(
                    filepath, map_location=self.device, weights_only=True
                )
            except TypeError:
                # older torch versions don’t support weights_only
                checkpoint = torch.load(filepath, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()

            self.model = model
            self.model_path_var.set(f"Loaded: {os.path.basename(filepath)}")
            self.predict_button.config(state=tk.NORMAL)
            self.status_var.set(f"✅ Model loaded on {self.device}")
        except Exception as e:
            self.model = None
            self.device = None
            self.model_path_var.set("Failed to load model.")
            self.predict_button.config(state=tk.DISABLED)
            self.status_var.set(f"❌ Model load error: {e}")

    def on_trajectory_select(self, event=None):
        """Callback to visualize the newly selected trajectory and clear old predictions."""
        self.predictions = None
        self.true_labels = None
        self.visualize_selected_trajectory()

    def run_prediction(self):
        """Prepares data for the selected trajectory and runs the loaded model."""
        if not self.model:
            self.status_var.set("⚠️ Please load a model first.")
            return

        selected_index = self.trajectory_selector.current()
        if selected_index == -1:
            self.status_var.set("⚠️ Please generate and select a trajectory.")
            return

        selected_trajectory = self.trajectories[selected_index]

        # Prepare data for the model
        features, labels = self._prepare_data_for_model(
            selected_trajectory, self.detecting_region_info
        )

        if features is None:
            self.status_var.set("ℹ️ Trajectory too short, no prediction.")
            self.predictions = None
            self.true_labels = None
        else:
            try:
                features_tensor = torch.tensor(features, dtype=torch.float32).to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model(features_tensor)

                self.predictions = outputs.cpu().numpy()
                self.true_labels = labels
                self.status_var.set(f"✅ Prediction complete for {selected_trajectory.name}")
            except Exception as e:
                self.status_var.set(f"❌ Prediction error: {e}")
                self.predictions = None
                self.true_labels = None

        # Re-draw the plot with the new prediction data
        self.visualize_selected_trajectory()

    def _prepare_data_for_model(self, trajectory, detecting_region_info):
        """Constructs features and labels from a trajectory for model input."""
        # Constants from src/config.py
        TIME_INTERVAL = 0.01
        FC = 6e9
        C = 3e8

        doppler_info = doppler_info_module.DopplerInfo(
            c=C, fc=FC, time_interval=TIME_INTERVAL
        )
        coords_a, coords_b = _generate_lines_from_trajectory(
            trajectory, TIME_INTERVAL
        )

        if len(coords_a) < 1:
            return None, None

        phis1234 = np.array(
            [get_angle_phi(detecting_region_info, ca, cb) for ca, cb in zip(coords_a, coords_b)]
        )
        w, doppler = generate_w_and_doppler(
            detecting_region_info, doppler_info, coords_a, coords_b, phis1234
        )
        features = get_features(phis1234, w, doppler)
        labels = get_labels(detecting_region_info, coords_b)

        return features, labels

    def visualize_selected_trajectory(self):
        """Clears the canvas and draws boundaries, trajectories, and predictions."""
        self.ax.clear()

        # 1. Draw boundaries
        if self.outer_vertices:
            self.ax.add_patch(
                Polygon(
                    self.outer_vertices,
                    fill=False,
                    edgecolor="k",
                    lw=2,
                    label="Outer Boundary",
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
                    label="Generation Area",
                )
            )

        # 2. Get selected trajectory
        selected_index = self.trajectory_selector.current()
        if selected_index == -1 or not self.trajectories:
            self.ax.legend()
            self.canvas.draw()
            return

        trajectory = self.trajectories[selected_index]

        # 3. Plotting Logic
        if self.predictions is not None and self.true_labels is not None:
            # Plot with predictions
            if trajectory.trajectory:
                path_points = np.array(trajectory.trajectory)
                self.ax.plot(
                    path_points[:, 0],
                    path_points[:, 1],
                    "-",
                    color="lightgray",
                    lw=1.5,
                    label="Full True Path",
                    zorder=1,
                )

            def polar_to_cartesian(data):
                rho, sin_theta, cos_theta = data[:, 0], data[:, 1], data[:, 2]
                ref_x, ref_y = self.detecting_region_info.transmittor_position
                return np.vstack((ref_x + rho * cos_theta, ref_y + rho * sin_theta)).T

            true_coords = polar_to_cartesian(self.true_labels)
            pred_coords = polar_to_cartesian(self.predictions)

            self.ax.plot(
                true_coords[:, 0],
                true_coords[:, 1],
                "b-o",
                markersize=4,
                label="True Segment",
                zorder=3,
            )
            self.ax.plot(
                pred_coords[:, 0],
                pred_coords[:, 1],
                "r--x",
                markersize=4,
                label="Predicted Path",
                zorder=3,
            )

            for t_coord, p_coord in zip(true_coords, pred_coords):
                self.ax.plot(
                    [t_coord[0], p_coord[0]],
                    [t_coord[1], p_coord[1]],
                    color="gray",
                    alpha=0.5,
                    lw=0.8,
                    zorder=2,
                )
        else:
            # Plot without predictions
            if trajectory.trajectory:
                path_points = np.array(trajectory.trajectory)
                self.ax.plot(
                    path_points[:, 0], path_points[:, 1], "b-", lw=2, label=trajectory.name
                )
            if trajectory.key_points:
                key_points = np.array(trajectory.key_points)
                self.ax.scatter(
                    key_points[:, 0],
                    key_points[:, 1],
                    color="purple",
                    s=80,
                    marker="o",
                    edgecolors="black",
                    zorder=5,
                    label="Key Points",
                )

        # 4. Final plot adjustments
        if self.outer_vertices:
            all_x = [v[0] for v in self.outer_vertices]
            all_y = [v[1] for v in self.outer_vertices]
            padding = max((max(all_x) - min(all_x)), (max(all_y) - min(all_y))) * 0.15
            self.ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
            self.ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

        self.ax.set_title(f"Trajectory Visualization (Region Seed: {self.seed_var.get()})")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.tight_layout(rect=[0, 0, 0.82, 1])
        self.canvas.draw()


if __name__ == "__main__":
    try:
        import scipy
    except ImportError as e:
        messagebox.showerror(
            "Dependency Error",
            f"Required library missing: 'scipy'.\n\nDetails: {e}",
        )
        sys.exit(1)

    app = App()
    app.mainloop()