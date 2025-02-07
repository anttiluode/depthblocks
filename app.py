#!/usr/bin/env python
"""
Depth Blocks App (Tk + Matplotlib)

Features:
 - Webcam or Video input
 - Resolution choice
 - Fullscreen toggle
 - Downsample / Block Size / Blur Sigma / Zoom sliders
 - Black background toggle
 - Play/Pause and Loop for video
 - **New**: "Hide/Show Controls" button that toggles the entire control panel
"""

import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend so we can embed in Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


print("Loading MiDaS (small) model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


class DepthBlocksApp:
    def __init__(self, master):
        self.master = master
        # Note mention of Hide/Show Controls
        self.master.title("Depth Blocks App (Hide/Show Controls at top)")

        # Basic capture state
        self.capture_source = "Webcam"
        self.webcam_index = 0
        self.video_path = None
        self.cap = None
        self.running = True

        # Default resolution
        self.res_width = 640
        self.res_height = 480

        # Video playback state
        self.video_paused = False
        self.loop_var = tk.BooleanVar(value=False)
        self.last_frame_bgr = None

        # We'll keep track if controls are shown
        self.controls_shown = True

        # Build the control panel in a frame
        self.ctrl_frame = ttk.Frame(self.master)
        self.ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.setup_control_panel()  # populates self.ctrl_frame

        # Add a small separate "Hide/Show Controls" button at top or bottom
        hide_btn_frame = ttk.Frame(self.master)
        hide_btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))
        self.hide_btn = ttk.Button(hide_btn_frame, text="Hide Controls", command=self.toggle_controls)
        self.hide_btn.pack()

        # Build the matplotlib figure
        self.fig = plt.Figure(figsize=(6,5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Live Depth Blocks")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Depth")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Initialize capture
        self.init_capture()

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100,
                                 blit=False, repeat=True)

        # Bind close event, ESC for fullscreen
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.bind("<Escape>", self.exit_fullscreen)

    def setup_control_panel(self):
        row_idx = 0
        # Downsample
        ttk.Label(self.ctrl_frame, text="Downsample (step):").grid(row=row_idx, column=0, sticky=tk.E)
        self.downsample_var = tk.IntVar(value=30)
        ttk.Scale(self.ctrl_frame, from_=1, to=50,
                  variable=self.downsample_var,
                  orient=tk.HORIZONTAL).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Block size
        ttk.Label(self.ctrl_frame, text="Block Size:").grid(row=row_idx, column=0, sticky=tk.E)
        self.block_size_var = tk.DoubleVar(value=2.0)
        ttk.Scale(self.ctrl_frame, from_=0.5, to=10.0,
                  variable=self.block_size_var,
                  orient=tk.HORIZONTAL).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Blur sigma
        ttk.Label(self.ctrl_frame, text="Blur Sigma:").grid(row=row_idx, column=0, sticky=tk.E)
        self.blur_sigma_var = tk.DoubleVar(value=0.0)
        ttk.Scale(self.ctrl_frame, from_=0.0, to=5.0,
                  variable=self.blur_sigma_var,
                  orient=tk.HORIZONTAL).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Zoom => tk.Scale to allow resolution=0.01
        ttk.Label(self.ctrl_frame, text="Zoom:").grid(row=row_idx, column=0, sticky=tk.E)
        self.zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = tk.Scale(self.ctrl_frame, from_=0.5, to=3.0,
                              resolution=0.01, variable=self.zoom_var,
                              orient=tk.HORIZONTAL, length=150)
        zoom_scale.grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Black BG
        self.bg_black_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctrl_frame, text="Black BG", variable=self.bg_black_var
                       ).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Capture Source
        ttk.Label(self.ctrl_frame, text="Capture Source:").grid(row=row_idx, column=0, sticky=tk.E)
        self.source_var = tk.StringVar(value="Webcam")
        source_combo = ttk.Combobox(
            self.ctrl_frame, textvariable=self.source_var,
            values=["Webcam", "Video"], state="readonly"
        )
        source_combo.grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        source_combo.bind("<<ComboboxSelected>>", self.on_source_choice)
        row_idx += 1

        # Webcam index
        ttk.Label(self.ctrl_frame, text="Webcam Index:").grid(row=row_idx, column=0, sticky=tk.E)
        self.webcam_index_var = tk.IntVar(value=0)
        ttk.Entry(self.ctrl_frame, textvariable=self.webcam_index_var, width=5
                 ).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Video path
        ttk.Label(self.ctrl_frame, text="Video File:").grid(row=row_idx, column=0, sticky=tk.E)
        self.video_path_label = ttk.Label(self.ctrl_frame, text="No file selected", width=20)
        self.video_path_label.grid(row=row_idx, column=1, sticky=tk.W, padx=5)

        self.browse_button = ttk.Button(self.ctrl_frame, text="Browse...", command=self.browse_video)
        self.browse_button.grid(row=row_idx, column=2, sticky=tk.W, padx=5)
        if self.source_var.get() == "Webcam":
            self.browse_button.state(["disabled"])
        row_idx += 1

        # "Set Source" button
        self.set_source_button = ttk.Button(self.ctrl_frame, text="Set Source", command=self.set_capture_source)
        self.set_source_button.grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Resolution
        ttk.Label(self.ctrl_frame, text="Resolution:").grid(row=row_idx, column=0, sticky=tk.E)
        self.res_var = tk.StringVar(value="640x480")
        self.res_combo = ttk.Combobox(
            self.ctrl_frame, textvariable=self.res_var,
            values=["320x240","640x480","1280x720","1920x1080"], state="readonly"
        )
        self.res_combo.grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        self.res_combo.bind("<<ComboboxSelected>>", self.on_res_choice)
        row_idx += 1

        # Fullscreen
        self.fullscreen_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctrl_frame, text="Fullscreen",
                        variable=self.fullscreen_var,
                        command=self.toggle_fullscreen
                       ).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

        # Play/Pause + Loop
        self.play_pause_button = ttk.Button(self.ctrl_frame, text="Pause", command=self.toggle_play_pause)
        self.play_pause_button.grid(row=row_idx, column=0, sticky=tk.E, padx=5)

        ttk.Checkbutton(self.ctrl_frame, text="Loop", variable=self.loop_var
                       ).grid(row=row_idx, column=1, sticky=tk.W, padx=5)
        row_idx += 1

    def toggle_controls(self):
        """Hide/Show the entire control frame."""
        if self.controls_shown:
            self.ctrl_frame.pack_forget()  # remove from layout
            self.hide_btn.config(text="Show Controls")
            self.controls_shown = False
        else:
            self.ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            self.hide_btn.config(text="Hide Controls")
            self.controls_shown = True

    # -- The rest is unchanged from your working code, except for the addition of toggle_controls.
    def on_source_choice(self, event=None):
        self.capture_source = self.source_var.get()
        if self.capture_source == "Webcam":
            self.browse_button.state(["disabled"])
        else:
            self.browse_button.state(["!disabled"])

    def browse_video(self):
        fn = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                       ("All Files","*.*")]
        )
        if fn:
            self.video_path = fn
            self.video_path_label.config(text=fn)
            self.source_var.set("Video")
            self.capture_source = "Video"

    def set_source_source_vars(self):
        self.capture_source = self.source_var.get()
        if self.capture_source == "Webcam":
            self.video_path = None
            self.webcam_index = self.webcam_index_var.get()

    def set_capture_source(self):
        self.set_source_source_vars()
        self.init_capture()

    def on_res_choice(self, event=None):
        val = self.res_var.get()
        parts = val.split('x')
        if len(parts)==2:
            w,h = parts
            self.res_width = int(w)
            self.res_height = int(h)
        else:
            self.res_width = 640
            self.res_height = 480

    def toggle_fullscreen(self):
        if self.fullscreen_var.get():
            self.master.attributes("-fullscreen", True)
        else:
            self.master.attributes("-fullscreen", False)

    def exit_fullscreen(self, event=None):
        if self.fullscreen_var.get():
            self.fullscreen_var.set(False)
            self.master.attributes("-fullscreen", False)

    def toggle_play_pause(self):
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.play_pause_button.config(text="Play")
        else:
            self.play_pause_button.config(text="Pause")

    def init_capture(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        if self.capture_source=="Webcam":
            idx = self.webcam_index
            print(f"Opening webcam index={idx}")
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open webcam index {idx}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_height)
        else:
            print(f"Opening video file={self.video_path}")
            if not self.video_path:
                raise RuntimeError("No video file selected!")
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.video_path}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_height)

        self.video_paused = False
        self.play_pause_button.config(text="Pause")
        self.last_frame_bgr = None

    def on_close(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.master.destroy()

    def get_depth_map(self, frame_rgb):
        inp = transform(frame_rgb).to(device)
        with torch.no_grad():
            depth = midas(inp)
            depth = F.interpolate(depth.unsqueeze(1),
                                  size=frame_rgb.shape[:2],
                                  mode="bicubic",
                                  align_corners=False).squeeze()
        depth = depth.cpu().numpy()
        dmin, dmax = depth.min(), depth.max()
        if dmax> dmin:
            depth = (depth - dmin)/(dmax - dmin)
        else:
            depth = np.zeros_like(depth)
        return depth

    def update_plot(self, frame_idx):
        if not self.running:
            return

        if self.capture_source=="Video" and self.video_paused:
            if self.last_frame_bgr is None:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    return
                self.last_frame_bgr = frame_bgr.copy()
            else:
                frame_bgr = self.last_frame_bgr
        else:
            ret, frame_bgr = self.cap.read()
            if not ret:
                if self.capture_source=="Video" and self.loop_var.get():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, frame_bgr2 = self.cap.read()
                    if ret2:
                        frame_bgr = frame_bgr2
                    else:
                        return
                else:
                    return
            self.last_frame_bgr = frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (320,240))

        depth = self.get_depth_map(frame_rgb)
        sigma = self.blur_sigma_var.get()
        if sigma>0:
            depth = gaussian_filter(depth, sigma=sigma)

        if self.bg_black_var.get():
            self.ax.set_facecolor("black")
            self.fig.patch.set_facecolor("black")
        else:
            self.ax.set_facecolor("white")
            self.fig.patch.set_facecolor("white")

        self.ax.clear()
        self.ax.set_title("Live Depth Blocks")

        H,W = depth.shape
        step = self.downsample_var.get()
        block_sz = self.block_size_var.get()

        xs,ys,zs,cs = [],[],[],[]
        for y in range(0,H,step):
            for x in range(0,W,step):
                zval = depth[y,x]*50
                r,g,b = frame_rgb[y,x]/255.0
                xs.append(x)
                ys.append(y)
                zs.append(zval)
                cs.append((r,g,b))

        self.ax.scatter(xs, ys, zs, c=cs, marker='s', s=block_sz**2, alpha=1.0)

        zoom = self.zoom_var.get()
        width_range = W/zoom
        height_range = H/zoom
        cx = W/2
        cy = H/2
        left = cx - width_range/2
        right = cx + width_range/2
        top_ = cy - height_range/2
        bottom_ = cy + height_range/2

        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom_, top_)
        self.ax.set_zlim(0,50)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("DepthScaled")

        self.canvas.draw_idle()

def main():
    root = tk.Tk()
    app = DepthBlocksApp(root)
    root.mainloop()

if __name__=="__main__":
    main()
