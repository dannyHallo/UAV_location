import tkinter as tk
from tkinter import messagebox
import sys
import os

# Ensure the 'src' directory is in the Python path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Matplotlib imports for embedding in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import necessary functions and classes from the provided source files
from src.detecting_region_info_generator import generate_detecting_region_infos
from src.detecting_region_info import DetectingRegionInfo


class App(tk.Tk):
    """
    An interactive Tkinter application to generate and visualize a 'Detecting Region'.

    This application provides a simple GUI with an input for a seed value and a button.
    Upon clicking the button, it generates a quadrilateral detecting region and calls
    the object's own draw_figure method to render it on the embedded canvas.
    """

    def __init__(self):
        super().__init__()
        self.title("Detecting Region Visualizer")
        self.geometry("800x650")

        # --- Top frame for user controls ---
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Seed:", font=("Helvetica", 10)).pack(side=tk.LEFT)

        self.seed_var = tk.StringVar(value="42")
        self.seed_entry = tk.Entry(control_frame, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side=tk.LEFT, padx=5)

        self.generate_button = tk.Button(
            control_frame,
            text="Generate and Draw",
            command=self.draw_region,
            font=("Helvetica", 10, "bold"),
        )
        self.generate_button.pack(side=tk.LEFT, padx=5)

        # --- Main frame for the Matplotlib plot ---
        plot_frame = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a Matplotlib figure and axes for the plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create a canvas to embed the figure in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Perform an initial draw on startup
        self.draw_region()

    def draw_region(self):
        """
        Generates a detecting region based on the seed and draws it on the canvas
        by calling the refactored draw_figure method from the DetectingRegionInfo class.
        """
        try:
            seed_value = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Seed must be an integer.")
            return

        # --- 1. Generate DetectingRegionInfo ---
        region_infos = generate_detecting_region_infos(
            num_configurations=1, seed=seed_value
        )

        if not region_infos:
            messagebox.showwarning(
                "Generation Failed",
                f"Could not generate a valid region for seed {seed_value}. Please try another seed.",
            )
            return

        detecting_region_info = region_infos[0]

        # --- 2. Clear the old plot and draw the new one ---
        self.ax.clear()

        # Call the encapsulated drawing method, passing the application's axes object.
        # The drawing logic is now correctly handled by the DetectingRegionInfo class.
        title = f"Detecting Region (Seed: {seed_value})"
        detecting_region_info.draw_figure(ax=self.ax, title=title)

        # Ensure the layout is clean after drawing
        self.fig.tight_layout()

        # --- 3. Refresh the canvas to show the new plot ---
        self.canvas.draw()


if __name__ == "__main__":
    # A simple check to ensure required libraries are available before launching.
    try:
        import scipy
        import numpy
    except ImportError as e:
        messagebox.showerror(
            "Dependency Error",
            f"A required library is missing. Please ensure 'scipy' and 'numpy' are installed.\n\nDetails: {e}",
        )
        sys.exit(1)

    app = App()
    app.mainloop()
