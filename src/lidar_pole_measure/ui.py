from pathlib import Path
import ctypes
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import winreg
from PIL import Image, ImageTk

from .config import DetectionConfig
from .detection import ALGO_VERSION
from .main import process_folder
from .ml_utils import train_pole_model_from_csv


def _is_windows_dark_mode() -> bool:
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0
    except OSError:
        return False


def _apply_theme(root: tk.Tk, style: ttk.Style, mode: str) -> None:
    system_dark = _is_windows_dark_mode()
    use_dark = (mode == "auto" and system_dark) or mode == "dark"
    if not use_dark:
        root.configure(bg="#f2f2f2")
        style.theme_use("vista")
        style.configure("App.TFrame", background="#f2f2f2")
        style.configure("Header.TLabel", background="#f2f2f2", foreground="#222222", font=("Segoe UI", 14, "bold"))
        style.configure("Text.TLabel", background="#f2f2f2", foreground="#222222", font=("Segoe UI", 10))
        style.configure("Run.TButton", foreground="#ffffff", background="#0b74de", font=("Segoe UI", 10, "bold"))
        style.configure("Clear.TButton", foreground="#ffffff", background="#cc4c4c", font=("Segoe UI", 10, "bold"))
        style.configure("Color.Horizontal.TProgressbar", troughcolor="#d9d9d9", background="#0b74de")
        return

    root.configure(bg="#1f1f1f")
    style.theme_use("clam")
    style.configure("App.TFrame", background="#1f1f1f")
    style.configure("Header.TLabel", background="#1f1f1f", foreground="#f0f0f0", font=("Segoe UI", 14, "bold"))
    style.configure("Text.TLabel", background="#1f1f1f", foreground="#f0f0f0", font=("Segoe UI", 10))
    style.configure("TCheckbutton", background="#1f1f1f", foreground="#f0f0f0")
    style.configure("TCombobox", fieldbackground="#2d2d2d", foreground="#f0f0f0")
    style.configure("Run.TButton", foreground="#ffffff", background="#2f9e44", font=("Segoe UI", 10, "bold"))
    style.configure("Clear.TButton", foreground="#ffffff", background="#d9480f", font=("Segoe UI", 10, "bold"))
    style.configure("Color.Horizontal.TProgressbar", troughcolor="#2d2d2d", background="#4dabf7")


def run_ui() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LidarPoleMeasure.App")
    except Exception:
        pass

    root = tk.Tk()
    root.title("Lidar Pole Measure")
    root.geometry("980x760")
    root.minsize(760, 560)
    root.resizable(True, True)
    style = ttk.Style(root)

    _apply_theme(root, style, "auto")

    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    project_root = base_dir
    assets_dir = project_root / "assets"
    icon_candidates_ico = [
        assets_dir / "pole_icon.ico",
        assets_dir / "app_icon.ico",
        assets_dir / "icon.ico",
    ]
    icon_candidates_png = [
        assets_dir / "pole_icon.png",
        assets_dir / "app_icon.png",
        assets_dir / "icon.png",
    ]
    if assets_dir.exists():
        # Also support hidden/unknown extensions by filename stem.
        for stem in ("pole_icon", "app_icon", "icon"):
            for p in assets_dir.iterdir():
                if p.is_file() and p.stem.lower() == stem and p not in icon_candidates_ico and p not in icon_candidates_png:
                    icon_candidates_png.append(p)
    for ico_icon_path in icon_candidates_ico:
        if ico_icon_path.exists():
            try:
                root.iconbitmap(default=str(ico_icon_path))
                break
            except tk.TclError:
                continue
    else:
        for png_icon_path in icon_candidates_png:
            if png_icon_path.exists():
                try:
                    # PIL can decode image by content even when extension is unusual/missing.
                    pil_img = Image.open(png_icon_path).convert("RGBA")
                    icon_img = ImageTk.PhotoImage(pil_img)
                    root.iconphoto(True, icon_img)
                    root._icon_img = icon_img
                    break
                except Exception:
                    continue

    input_var = tk.StringVar(value="")
    output_var = tk.StringVar(value="")
    recursive_var = tk.BooleanVar(value=False)
    force_4326_var = tk.BooleanVar(value=True)
    unit_override_var = tk.StringVar(value="Force: meters")
    write_geojson_var = tk.BooleanVar(value=True)
    write_gpkg_var = tk.BooleanVar(value=True)
    write_shp_var = tk.BooleanVar(value=True)
    write_csv_var = tk.BooleanVar(value=True)
    min_pole_conf_var = tk.StringVar(value="Medium (recommended for large LAZ)")
    ml_model_var = tk.StringVar(value="")
    ml_threshold_var = tk.StringVar(value="0.55")
    train_csv_var = tk.StringVar(value="")
    train_label_var = tk.StringVar(value="pole_label")
    train_model_out_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Ready")
    progress_var = tk.DoubleVar(value=0.0)
    run_state = {"busy": False}

    def browse_input() -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            input_var.set(path)

    def browse_output() -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            output_var.set(path)

    def browse_ml_model() -> None:
        path = filedialog.askopenfilename(
            title="Select ML model",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if path:
            ml_model_var.set(path)

    def browse_train_csv() -> None:
        path = filedialog.askopenfilename(
            title="Select labeled CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            train_csv_var.set(path)

    def browse_train_out_model() -> None:
        path = filedialog.asksaveasfilename(
            title="Save model as",
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            initialfile="pole_model.joblib",
        )
        if path:
            train_model_out_var.set(path)

    def train_model() -> None:
        try:
            csv_path = Path(train_csv_var.get().strip())
            out_path_txt = train_model_out_var.get().strip()
            if not out_path_txt:
                out_path_txt = str((Path(output_var.get().strip()) if output_var.get().strip() else Path.cwd()) / "pole_model.joblib")
                train_model_out_var.set(out_path_txt)
            out_path = Path(out_path_txt)
            label_col = train_label_var.get().strip() or "pole_label"
            result = train_pole_model_from_csv(csv_path, out_path, label_column=label_col)
            ml_model_var.set(str(out_path))
            log_line(f"ML model trained on {result['rows_used']} rows -> {out_path}")
            messagebox.showinfo("ML Training Completed", f"Model saved:\n{out_path}\nRows used: {result['rows_used']}")
        except Exception as exc:
            messagebox.showerror("ML Training Error", str(exc))

    def clear_form() -> None:
        input_var.set("")
        output_var.set("")
        mode_notebook.select(0)
        recursive_var.set(False)
        force_4326_var.set(True)
        unit_override_var.set("Force: meters")
        write_geojson_var.set(True)
        write_gpkg_var.set(True)
        write_shp_var.set(True)
        write_csv_var.set(True)
        min_pole_conf_var.set("Medium (recommended for large LAZ)")
        ml_model_var.set("")
        ml_threshold_var.set("0.55")
        train_csv_var.set("")
        train_label_var.set("pole_label")
        train_model_out_var.set("")
        status_var.set("Cleared")
        progress_var.set(0.0)
        log_box.configure(state="normal")
        log_box.delete("1.0", tk.END)
        log_box.configure(state="disabled")

    def _on_progress_ui(done: int, total: int, filename: str) -> None:
        if total <= 0:
            progress_var.set(0.0)
            status_var.set("No files")
        else:
            progress_var.set((done / total) * 100.0)
            status_var.set(f"Processing {done}/{total}: {filename}")
        root.update_idletasks()

    def on_progress(done: int, total: int, filename: str) -> None:
        root.after(0, lambda: _on_progress_ui(done, total, filename))

    def _log_line_ui(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert(tk.END, text + "\n")
        log_box.see(tk.END)
        log_box.configure(state="disabled")
        root.update_idletasks()

    def log_line(text: str) -> None:
        root.after(0, lambda: _log_line_ui(text))

    def _finish_run_success(result: dict, mode_name: str) -> None:
        run_state["busy"] = False
        progress_var.set(100.0)
        status_var.set("Completed")
        outputs = []
        if result["geojson"] is not None:
            outputs.append(f"GeoJSON:\n{result['geojson']}")
        if result["gpkg"] is not None:
            outputs.append(f"GPKG:\n{result['gpkg']}")
        if result["shp"] is not None:
            outputs.append(f"SHP:\n{result['shp']}")
        if result["csv"] is not None:
            outputs.append(f"CSV:\n{result['csv']}")
        failed_txt = ""
        failed_files = result.get("failed_files", [])
        if failed_files:
            failed_txt = "\n\nFiles failed:\n" + "\n".join([f"- {name}" for name, _ in failed_files[:20]])
        messagebox.showinfo(
            "Completed",
            (
                f"Mode: {mode_name}\n"
                f"Processed files: {result['files']}\n"
                f"Poles detected: {result['count']}\n\n"
                "Output attributes include height/cross-arm in feet: "
                "height_ft, crossarm_ft\n\n"
                f"Output CRS: {result['output_crs']}\n\n"
                + "\n\n".join(outputs)
                + failed_txt
            ),
        )

    def _finish_run_error(error_text: str) -> None:
        run_state["busy"] = False
        status_var.set("Failed")
        messagebox.showerror("Error", error_text)

    def run_process(mode: str) -> None:
        try:
            if run_state["busy"]:
                messagebox.showinfo("Please wait", "Processing is already running.")
                return
            input_txt = input_var.get().strip()
            output_txt = output_var.get().strip()
            input_path = Path(input_txt)
            output_path = Path(output_txt)
            if not input_path.exists():
                messagebox.showerror("Input Folder Missing", f"Input path does not exist:\n{input_path}")
                return
            if not output_txt:
                messagebox.showerror("Output Folder Missing", "Please select an output folder.")
                return
            is_bulk_mode = mode == "bulk"
            cfg = DetectionConfig()
            if is_bulk_mode:
                # Large-file mode: return all detected poles per LAZ.
                cfg.keep_only_tallest_per_file = False
                cfg.min_samples = 16
                cfg.min_points_per_cluster = 45
            if not any([write_geojson_var.get(), write_gpkg_var.get(), write_shp_var.get(), write_csv_var.get()]):
                messagebox.showwarning("No Output Selected", "Select at least one output format.")
                return
            run_state["busy"] = True
            status_var.set("Starting...")
            progress_var.set(0.0)
            root.update_idletasks()

            unit_map = {
                "Auto-detect from CRS, per file (default)": "auto",
                "Force: meters": "meters",
                "Force: US survey feet": "us_survey_feet",
                "Force: international feet": "international_feet",
            }
            mode_name = "Large LAZ Auto-Detect" if is_bulk_mode else "Clipped Pole (single-pole) Mode"
            min_conf = {
                "Low (show almost all)": "low",
                "Medium (recommended for large LAZ)": "medium",
                "High (strict)": "high",
            }[min_pole_conf_var.get()]
            ml_model = Path(ml_model_var.get().strip()) if (is_bulk_mode and ml_model_var.get().strip()) else None
            ml_thr = float(ml_threshold_var.get().strip() or "0.55") if is_bulk_mode else 0.55

            def worker() -> None:
                try:
                    result = process_folder(
                        input_dir=input_path,
                        output_dir=output_path,
                        cfg=cfg,
                        recursive=recursive_var.get(),
                        unit_override=unit_map[unit_override_var.get()],
                        force_output_4326=force_4326_var.get(),
                        write_geojson_output=write_geojson_var.get(),
                        write_gpkg_output=write_gpkg_var.get(),
                        write_shp_output=write_shp_var.get(),
                        write_csv_output=write_csv_var.get(),
                        min_pole_conf=min_conf,
                        ml_model_path=ml_model,
                        ml_threshold=ml_thr,
                        progress_callback=on_progress,
                        log_callback=log_line,
                    )
                    log_line(f"Mode: {mode_name}")
                    log_line(f"Algorithm version: {ALGO_VERSION}")
                    if result["files"] == 0:
                        def _warn_no_files() -> None:
                            run_state["busy"] = False
                            messagebox.showwarning(
                                "No Files Found",
                                f"No .las/.laz files found in:\n{input_path}\n\nTry enabling recursive search.",
                            )

                        root.after(0, _warn_no_files)
                        return
                    root.after(0, lambda: _finish_run_success(result, mode_name))
                except Exception as exc:
                    root.after(0, lambda: _finish_run_error(str(exc)))

            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            run_state["busy"] = False
            status_var.set("Failed")
            messagebox.showerror("Error", str(exc))

    frame = ttk.Frame(root, style="App.TFrame", padding=14)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(7, weight=1)

    header = ttk.Label(frame, text="LiDAR Pole Measure", style="Header.TLabel")
    header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
    subtitle = ttk.Label(
        frame,
        text=f"Detect poles, measure height/cross-arm in feet, and export GeoJSON/GPKG/SHP/CSV. ({ALGO_VERSION})",
        style="Text.TLabel",
    )
    subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

    pad = {"padx": 8, "pady": 6}
    mode_notebook = ttk.Notebook(frame)
    mode_notebook.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))
    tab_clip = ttk.Frame(mode_notebook, style="App.TFrame")
    tab_bulk = ttk.Frame(mode_notebook, style="App.TFrame")
    mode_notebook.add(tab_clip, text="Clipped Pole Mode")
    mode_notebook.add(tab_bulk, text="Large LAZ Auto-Detect")
    tab_ml = ttk.Frame(mode_notebook, style="App.TFrame")
    mode_notebook.add(tab_ml, text="Machine Learning")
    ttk.Label(
        tab_clip,
        text="Use this when each LAZ/LAS contains a single clipped pole. Keeps only primary pole per file.",
        style="Text.TLabel",
    ).grid(row=0, column=0, sticky="w", padx=10, pady=8)
    ttk.Label(
        tab_bulk,
        text="Use this for big LAZ with many poles. App detects and exports all poles in each file.",
        style="Text.TLabel",
    ).grid(row=0, column=0, sticky="w", padx=10, pady=8)
    tab_clip.columnconfigure(1, weight=1)
    tab_bulk.columnconfigure(1, weight=1)

    # --- Clipped pole tab controls ---
    cpad = {"padx": 8, "pady": 5}
    ttk.Label(tab_clip, text="Input LAS/LAZ Folder", style="Text.TLabel").grid(row=1, column=0, sticky="w", **cpad)
    ttk.Entry(tab_clip, textvariable=input_var).grid(row=1, column=1, sticky="we", **cpad)
    ttk.Button(tab_clip, text="Browse", command=browse_input, width=12).grid(row=1, column=2, **cpad)
    ttk.Label(tab_clip, text="Output Folder", style="Text.TLabel").grid(row=2, column=0, sticky="w", **cpad)
    ttk.Entry(tab_clip, textvariable=output_var).grid(row=2, column=1, sticky="we", **cpad)
    ttk.Button(tab_clip, text="Browse", command=browse_output, width=12).grid(row=2, column=2, **cpad)
    ttk.Label(tab_clip, text="File coordinate unit override", style="Text.TLabel").grid(row=3, column=0, sticky="w", **cpad)
    unit_combo_clip = ttk.Combobox(
        tab_clip,
        textvariable=unit_override_var,
        width=46,
        state="readonly",
        values=[
            "Auto-detect from CRS, per file (default)",
            "Force: meters",
            "Force: US survey feet",
            "Force: international feet",
        ],
    )
    unit_combo_clip.grid(row=3, column=1, sticky="w", **cpad)
    unit_combo_clip.set("Force: meters")
    ttk.Checkbutton(tab_clip, text="Search recursively", variable=recursive_var).grid(row=4, column=1, sticky="w", **cpad)
    ttk.Checkbutton(tab_clip, text="Force output CRS to EPSG:4326", variable=force_4326_var).grid(row=5, column=1, sticky="w", **cpad)
    ttk.Label(tab_clip, text="Output formats", style="Text.TLabel").grid(row=6, column=0, sticky="w", **cpad)
    ttk.Checkbutton(tab_clip, text="GeoJSON", variable=write_geojson_var).grid(row=6, column=1, sticky="w", padx=8, pady=2)
    ttk.Checkbutton(tab_clip, text="GPKG", variable=write_gpkg_var).grid(row=6, column=1, padx=(100, 0), pady=2, sticky="w")
    ttk.Checkbutton(tab_clip, text="SHP", variable=write_shp_var).grid(row=6, column=1, padx=(180, 0), pady=2, sticky="w")
    ttk.Checkbutton(tab_clip, text="CSV", variable=write_csv_var).grid(row=6, column=1, padx=(250, 0), pady=2, sticky="w")
    clip_actions = ttk.Frame(tab_clip, style="App.TFrame")
    clip_actions.grid(row=7, column=0, columnspan=3, sticky="we", padx=8, pady=(8, 6))
    clip_actions.columnconfigure(0, weight=1)
    ttk.Button(
        clip_actions, text="Run Clipped Pole Detection", command=lambda: run_process("clip"), width=26, style="Run.TButton"
    ).grid(row=0, column=1, padx=(0, 10), sticky="e")
    ttk.Button(clip_actions, text="Clear", command=clear_form, width=12, style="Clear.TButton").grid(row=0, column=2, sticky="e")

    # --- Large LAZ tab controls ---
    ttk.Label(tab_bulk, text="Input LAS/LAZ Folder", style="Text.TLabel").grid(row=1, column=0, sticky="w", **cpad)
    ttk.Entry(tab_bulk, textvariable=input_var).grid(row=1, column=1, sticky="we", **cpad)
    ttk.Button(tab_bulk, text="Browse", command=browse_input, width=12).grid(row=1, column=2, **cpad)
    ttk.Label(tab_bulk, text="Output Folder", style="Text.TLabel").grid(row=2, column=0, sticky="w", **cpad)
    ttk.Entry(tab_bulk, textvariable=output_var).grid(row=2, column=1, sticky="we", **cpad)
    ttk.Button(tab_bulk, text="Browse", command=browse_output, width=12).grid(row=2, column=2, **cpad)
    ttk.Label(tab_bulk, text="File coordinate unit override", style="Text.TLabel").grid(row=3, column=0, sticky="w", **cpad)
    unit_combo_bulk = ttk.Combobox(
        tab_bulk,
        textvariable=unit_override_var,
        width=46,
        state="readonly",
        values=[
            "Auto-detect from CRS, per file (default)",
            "Force: meters",
            "Force: US survey feet",
            "Force: international feet",
        ],
    )
    unit_combo_bulk.grid(row=3, column=1, sticky="w", **cpad)
    unit_combo_bulk.set("Force: meters")
    ttk.Checkbutton(tab_bulk, text="Search recursively", variable=recursive_var).grid(row=4, column=1, sticky="w", **cpad)
    ttk.Checkbutton(tab_bulk, text="Force output CRS to EPSG:4326", variable=force_4326_var).grid(row=5, column=1, sticky="w", **cpad)
    ttk.Label(tab_bulk, text="Minimum pole confidence", style="Text.TLabel").grid(row=6, column=0, sticky="w", **cpad)
    pole_conf_combo = ttk.Combobox(
        tab_bulk,
        textvariable=min_pole_conf_var,
        width=46,
        state="readonly",
        values=["Low (show almost all)", "Medium (recommended for large LAZ)", "High (strict)"],
    )
    pole_conf_combo.grid(row=6, column=1, sticky="w", **cpad)
    pole_conf_combo.set("Medium (recommended for large LAZ)")
    ttk.Label(tab_bulk, text="ML pole model (optional)", style="Text.TLabel").grid(row=7, column=0, sticky="w", **cpad)
    ml_row = ttk.Frame(tab_bulk, style="App.TFrame")
    ml_row.grid(row=7, column=1, sticky="w", padx=8, pady=2)
    ttk.Entry(ml_row, textvariable=ml_model_var, width=28).grid(row=0, column=0, padx=(0, 6))
    ttk.Button(ml_row, text="Model", command=browse_ml_model, width=8).grid(row=0, column=1)
    ttk.Label(ml_row, text="Thr", style="Text.TLabel").grid(row=0, column=2, padx=(8, 4))
    ttk.Entry(ml_row, textvariable=ml_threshold_var, width=6).grid(row=0, column=3)
    ttk.Label(tab_bulk, text="Output formats", style="Text.TLabel").grid(row=8, column=0, sticky="w", **cpad)
    ttk.Checkbutton(tab_bulk, text="GeoJSON", variable=write_geojson_var).grid(row=8, column=1, sticky="w", padx=8, pady=2)
    ttk.Checkbutton(tab_bulk, text="GPKG", variable=write_gpkg_var).grid(row=8, column=1, padx=(100, 0), pady=2, sticky="w")
    ttk.Checkbutton(tab_bulk, text="SHP", variable=write_shp_var).grid(row=8, column=1, padx=(180, 0), pady=2, sticky="w")
    ttk.Checkbutton(tab_bulk, text="CSV", variable=write_csv_var).grid(row=8, column=1, padx=(250, 0), pady=2, sticky="w")
    bulk_actions = ttk.Frame(tab_bulk, style="App.TFrame")
    bulk_actions.grid(row=9, column=0, columnspan=3, sticky="we", padx=8, pady=(8, 6))
    bulk_actions.columnconfigure(0, weight=1)
    ttk.Button(
        bulk_actions, text="Run Large LAZ Detection", command=lambda: run_process("bulk"), width=24, style="Run.TButton"
    ).grid(row=0, column=1, padx=(0, 10), sticky="e")
    ttk.Button(bulk_actions, text="Clear", command=clear_form, width=12, style="Clear.TButton").grid(row=0, column=2, sticky="e")
    ttk.Label(
        tab_ml,
        text="Train a pole classifier from labeled CSV and use it to filter false positives in Large LAZ mode.",
        style="Text.TLabel",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 6))
    tab_ml.columnconfigure(1, weight=1)
    ttk.Label(tab_ml, text="Labeled CSV", style="Text.TLabel").grid(row=1, column=0, sticky="w", padx=10, pady=4)
    ttk.Entry(tab_ml, textvariable=train_csv_var, width=64).grid(row=1, column=1, sticky="we", padx=8, pady=4)
    ttk.Button(tab_ml, text="Browse", command=browse_train_csv, width=10).grid(row=1, column=2, padx=8, pady=4)
    ttk.Label(tab_ml, text="Label column", style="Text.TLabel").grid(row=2, column=0, sticky="w", padx=10, pady=4)
    ttk.Entry(tab_ml, textvariable=train_label_var, width=24).grid(row=2, column=1, sticky="w", padx=8, pady=4)
    ttk.Label(tab_ml, text="Output model (.joblib)", style="Text.TLabel").grid(row=3, column=0, sticky="w", padx=10, pady=4)
    ttk.Entry(tab_ml, textvariable=train_model_out_var, width=64).grid(row=3, column=1, sticky="we", padx=8, pady=4)
    ttk.Button(tab_ml, text="Browse", command=browse_train_out_model, width=10).grid(row=3, column=2, padx=8, pady=4)
    ttk.Button(tab_ml, text="Train Model", command=train_model, width=18, style="Run.TButton").grid(
        row=4, column=1, sticky="w", padx=8, pady=(8, 8)
    )

    progress = ttk.Progressbar(frame, variable=progress_var, maximum=100, style="Color.Horizontal.TProgressbar")
    progress.grid(row=4, column=0, columnspan=3, sticky="we", padx=8, pady=(8, 4))
    ttk.Label(frame, textvariable=status_var, style="Text.TLabel").grid(
        row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8)
    )
    ttk.Label(frame, text="Logs", style="Text.TLabel").grid(row=6, column=0, sticky="w", **pad)
    log_box = ScrolledText(frame, height=11, state="disabled", wrap="word")
    log_box.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))

    root.mainloop()
