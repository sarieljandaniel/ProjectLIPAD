"""Project LiPAD — desktop structural health analytics application."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import customtkinter as ctk
import pandas as pd
from tkinter import filedialog

from ui.components import body_label, newsprint_button
from ui.pages.analysis import render_analysis
from ui.pages.home import render_home
from ui.pages.inspection import render_inspection
from ui.pages.reports import render_reports
from ui.sidebar import build_sidebar
from ui.telemetry import TelemetryListener, TelemetryPacket
from ui.theme import ThemeMode, get_tokens

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")


class LipadApp(ctk.CTk):
    def __init__(self, weights_path: str, dist_ref: float = 1168.4):
        super().__init__()
        self.title("Project LiPAD — Structural Health Analytics")
        self.geometry("1280x860")
        self.minsize(1024, 720)

        self.theme_mode = ThemeMode.LIGHT
        self.tokens = get_tokens(self.theme_mode)
        self.current_tab = "home"

        self.selected_inspection = ctk.StringVar(value="Crack")
        self.selected_corrosion_type = ctk.StringVar(value="Wet")
        self.uploaded_video_path = None
        self.last_run_status = ctk.StringVar(value="Idle")
        self.last_output_csv_path = None
        self.gsd_value = ctk.StringVar(value="0.5436")
        self.dist_ref = dist_ref
        self.frame_stride_value = ctk.StringVar(value="1")
        self.inference_width_value = ctk.StringVar(value="0")
        self.video_files: list[str] = []
        self.selected_video_index = None
        self.engine_process = None
        self.engine_lock = threading.Lock()
        self.engine_stop_requested = False
        self.last_annotated_video_path = None
        self.telemetry_log = None
        self.current_lidar_distance = dist_ref
        self._telemetry_packets: list[TelemetryPacket] = []

        self._shell = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self._shell.pack(fill="both", expand=True)
        self._shell.grid_columnconfigure(0, weight=0, minsize=220)
        self._shell.grid_columnconfigure(2, weight=1)
        self._shell.grid_rowconfigure(0, weight=1)

        self._sidebar_slot = ctk.CTkFrame(
            self._shell, width=220, fg_color=self.tokens.bg, corner_radius=0
        )
        self._sidebar_slot.grid(row=0, column=0, sticky="ns")
        self._sidebar_slot.grid_propagate(False)

        self._divider = ctk.CTkFrame(self._shell, width=4, fg_color=self.tokens.border, corner_radius=0)
        self._divider.grid(row=0, column=1, sticky="ns")

        self.container = ctk.CTkFrame(self._shell, fg_color=self.tokens.bg, corner_radius=0)
        self.container.grid(row=0, column=2, sticky="nsew", padx=24, pady=24)

        self._telemetry = TelemetryListener(on_packet=self._queue_telemetry_packet)
        self._telemetry.start()

        self._apply_root_theme()
        self._build_shell()
        self.select_tab("home")

    def _apply_root_theme(self) -> None:
        self.tokens = get_tokens(self.theme_mode)
        self.configure(fg_color=self.tokens.bg)
        if hasattr(self, "container"):
            self.container.configure(fg_color=self.tokens.bg)
        if hasattr(self, "_divider"):
            self._divider.configure(fg_color=self.tokens.border)
        if hasattr(self, "_sidebar_slot"):
            self._sidebar_slot.configure(fg_color=self.tokens.bg)

    def _build_shell(self) -> None:
        for w in self._sidebar_slot.winfo_children():
            w.destroy()
        build_sidebar(self, self._sidebar_slot, self.tokens)

    def toggle_theme(self) -> None:
        self.theme_mode = ThemeMode.DARK if self.theme_mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        ctk.set_appearance_mode("Dark" if self.theme_mode == ThemeMode.DARK else "Light")
        self._apply_root_theme()
        self._build_shell()
        self.select_tab(self.current_tab)

    def _queue_telemetry_packet(self, packet: TelemetryPacket) -> None:
        self.after(0, lambda p=packet: self._on_telemetry_packet(p))

    def _on_telemetry_packet(self, packet: TelemetryPacket) -> None:
        self._telemetry_packets.append(packet)
        if len(self._telemetry_packets) > 200:
            self._telemetry_packets = self._telemetry_packets[-200:]
        if packet.distance_mm is not None:
            self.current_lidar_distance = packet.distance_mm
        if self.telemetry_log and self.telemetry_log.winfo_exists():
            self.telemetry_log.configure(state="normal")
            line = f"[{packet.timestamp}] FROM {packet.source_ip} → {packet.message}"
            if packet.distance_mm is not None:
                line += f"  {packet.distance_mm:.1f} mm"
            self.telemetry_log.insert("end", line + "\n")
            self.telemetry_log.see("end")
            self.telemetry_log.configure(state="disabled")
        self._refresh_home_metrics(packet)

    def _refresh_home_metrics(self, packet: TelemetryPacket | None = None) -> None:
        if packet is None and self._telemetry_packets:
            packet = self._telemetry_packets[-1]
        if hasattr(self, "_link_status_value") and self._link_status_value.winfo_exists():
            status = "Live" if self._telemetry.connected else "Standby"
            self._link_status_value.configure(text=status)
        if hasattr(self, "_distance_value") and self._distance_value.winfo_exists():
            d = self.current_lidar_distance
            self._distance_value.configure(text=f"{d:.1f} mm" if d is not None else "— mm")
        if hasattr(self, "_last_packet_value") and self._last_packet_value.winfo_exists():
            if packet:
                self._last_packet_value.configure(text=f"{packet.source_ip} · {packet.timestamp}")
        if hasattr(self, "_telemetry_header_status") and self._telemetry_header_status.winfo_exists():
            self._telemetry_header_status.configure(
                text="RECEIVING" if self._telemetry.connected else "STANDBY"
            )

    def _repo_root(self) -> str:
        return os.path.abspath(os.path.dirname(__file__))

    def _data_dir(self) -> str:
        return os.path.join(self._repo_root(), "data")

    def _morph_results_path(self) -> str:
        return os.path.join(self._data_dir(), "MorphologicalResults.csv")

    def _ui_results_path(self) -> str:
        return os.path.join(self._data_dir(), "results.csv")

    def _engine_script_path(self) -> str:
        return os.path.join(self._repo_root(), "Project_LIPAD_AI", "lipad_runtime_engine.py")

    def _default_weights_path(self) -> str:
        return os.path.join(self._repo_root(), "models", "best.pt")

    def _set_status(self, text: str) -> None:
        self.last_run_status.set(text)
        if hasattr(self, "status_lbl") and self.status_lbl.winfo_exists():
            self.status_lbl.configure(text=text)
        if hasattr(self, "selected_video_lbl") and self.selected_video_lbl.winfo_exists():
            self.selected_video_lbl.configure(text=self._selected_video_label_text())
        if hasattr(self, "_engine_status_lbl") and self._engine_status_lbl.winfo_exists():
            self._engine_status_lbl.configure(text=text)

    def _selected_video_label_text(self) -> str:
        if self.selected_video_index is None:
            return "Selected: (none)"
        if self.selected_video_index < 0 or self.selected_video_index >= len(self.video_files):
            return "Selected: (none)"
        return f"Selected: {os.path.basename(self.video_files[self.selected_video_index])}"

    def select_tab(self, name: str) -> None:
        self.current_tab = name
        for widget in self.container.winfo_children():
            widget.destroy()

        content = ctk.CTkScrollableFrame(
            self.container,
            fg_color=self.tokens.bg,
            corner_radius=0,
            scrollbar_button_color=self.tokens.rule,
            scrollbar_button_hover_color=self.tokens.muted,
        )
        content.pack(fill="both", expand=True)

        if name == "home":
            render_home(self, content, self.tokens)
        elif name == "inspection_manager":
            render_inspection(self, content, self.tokens)
        elif name == "analysis_overview":
            render_analysis(self, content, self.tokens)
        elif name == "reports":
            render_reports(self, content, self.tokens)

        self._build_shell()

    def set_inspection(self, target: str) -> None:
        self.selected_inspection.set(target)
        self.select_tab("inspection_manager")

    def _render_inspection_suboptions(self, tokens) -> None:
        if not hasattr(self, "sub_options_frame") or not self.sub_options_frame.winfo_exists():
            return
        for widget in self.sub_options_frame.winfo_children():
            widget.destroy()
        target = self.selected_inspection.get()
        body_label(self.sub_options_frame, tokens, f"Selected: {target}").pack(anchor="w", pady=(8, 0))
        if target == "Corrosion":
            meta = ctk.CTkFrame(self.sub_options_frame, fg_color="transparent")
            meta.pack(anchor="w", pady=(8, 0))
            from ui.components import meta_label
            meta_label(meta, tokens, "Environment").pack(anchor="w")
            row = ctk.CTkFrame(self.sub_options_frame, fg_color="transparent")
            row.pack(anchor="w", pady=8)
            newsprint_button(
                row, tokens, "Wet (marine)",
                command=lambda: self._set_corrosion_env("Wet"),
                variant="primary" if self.selected_corrosion_type.get() == "Wet" else "secondary",
            ).pack(side="left", padx=(0, 8))
            newsprint_button(
                row, tokens, "Dry (oxidation)",
                command=lambda: self._set_corrosion_env("Dry"),
                variant="primary" if self.selected_corrosion_type.get() == "Dry" else "secondary",
            ).pack(side="left")
            body_label(
                self.sub_options_frame, tokens, f"Environment: {self.selected_corrosion_type.get()}", mono=True
            ).pack(anchor="w")

    def _set_corrosion_env(self, env: str) -> None:
        self.selected_corrosion_type.set(env)
        self.select_tab("inspection_manager")

    def upload_video(self) -> None:
        file_paths = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")])
        if not file_paths:
            return
        for p in file_paths:
            if p and p not in self.video_files:
                self.video_files.append(p)
        with self.engine_lock:
            running = self.engine_process is not None
        if not running and self.selected_video_index is None and self.video_files:
            self.selected_video_index = 0
            self.uploaded_video_path = self.video_files[0]
        self._refresh_video_list()
        self._set_status("Videos added. Select one to analyze.")

    def _refresh_video_list(self) -> None:
        if not hasattr(self, "video_list") or not self.video_list.winfo_exists():
            return
        for iid in self.video_list.get_children():
            self.video_list.delete(iid)
        for idx, path in enumerate(self.video_files):
            self.video_list.insert("", "end", iid=str(idx), values=[path])
        if self.selected_video_index is not None and 0 <= self.selected_video_index < len(self.video_files):
            self.video_list.selection_set(str(self.selected_video_index))
            self.video_list.see(str(self.selected_video_index))
        if hasattr(self, "selected_video_lbl") and self.selected_video_lbl.winfo_exists():
            self.selected_video_lbl.configure(text=self._selected_video_label_text())

    def select_video_from_list(self, update_only: bool = False) -> None:
        if not hasattr(self, "video_list") or not self.video_list.winfo_exists():
            return
        sel = self.video_list.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except Exception:
            return
        if idx < 0 or idx >= len(self.video_files):
            return
        self.selected_video_index = idx
        self.uploaded_video_path = self.video_files[idx]
        if not update_only:
            self._set_status(f"Selected: {os.path.basename(self.uploaded_video_path)}")
        elif hasattr(self, "selected_video_lbl") and self.selected_video_lbl.winfo_exists():
            self.selected_video_lbl.configure(text=self._selected_video_label_text())

    def remove_selected_video(self) -> None:
        if self.selected_video_index is None:
            self._set_status("Nothing selected to remove.")
            return
        if self.engine_process is not None:
            self._set_status("Stop analysis before removing videos.")
            return
        if 0 <= self.selected_video_index < len(self.video_files):
            removed = self.video_files.pop(self.selected_video_index)
            self._set_status(f"Removed: {os.path.basename(removed)}")
        if not self.video_files:
            self.selected_video_index = None
            self.uploaded_video_path = None
        else:
            self.selected_video_index = min(self.selected_video_index, len(self.video_files) - 1)
            self.uploaded_video_path = self.video_files[self.selected_video_index]
        self._refresh_video_list()

    def open_selected_video_folder(self) -> None:
        if not self.uploaded_video_path:
            self._set_status("No video selected.")
            return
        try:
            os.startfile(os.path.dirname(self.uploaded_video_path))
        except Exception as e:
            self._set_status(f"Failed to open folder: {e}")

    def run_engine_from_ui(self) -> None:
        if not self.uploaded_video_path:
            self._set_status("No MP4 selected. Add/select a video first.")
            return
        with self.engine_lock:
            if self.engine_process is not None:
                self._set_status("Engine already running. Stop it first.")
                return
            self.engine_stop_requested = False

        os.makedirs(self._data_dir(), exist_ok=True)
        annotated_dir = os.path.join(self._data_dir(), "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.uploaded_video_path))[0]
        annotated_video = os.path.join(annotated_dir, f"{base}_annotated.mp4")
        self.last_annotated_video_path = annotated_video

        try:
            gsd = float(self.gsd_value.get())
        except Exception:
            gsd = 0.5436
        try:
            stride = max(1, int(float(self.frame_stride_value.get())))
        except Exception:
            stride = 1
        try:
            inf_w = max(0, int(float(self.inference_width_value.get())))
        except Exception:
            inf_w = 0

        inspection = self.selected_inspection.get() or "Crack"
        corrosion_env = self.selected_corrosion_type.get() or "Wet"
        cmd = [
            sys.executable, self._engine_script_path(),
            "--video", self.uploaded_video_path,
            "--gsd", str(gsd),
            "--output_csv", self._morph_results_path(),
            "--results_csv", self._ui_results_path(),
            "--no_preview",
            "--frame_stride", str(stride),
            "--inference_width", str(inf_w),
            "--output_video", annotated_video,
            "--inspection_type", inspection,
            "--corrosion_env", corrosion_env,
        ]
        if inspection.lower() == "crack":
            weights_path = self._default_weights_path()
            if not os.path.exists(weights_path):
                self._set_status(f"Weights missing: {weights_path}")
                return
            cmd.extend(["--weights", weights_path])

        def _runner() -> None:
            self.after(0, lambda: self._set_status("Running engine..."))
            try:
                with self.engine_lock:
                    self.engine_process = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
                    )
                    proc = self.engine_process
                code = proc.wait()
                with self.engine_lock:
                    stopped = self.engine_stop_requested
                    self.engine_process = None
                    self.engine_stop_requested = False
                if code != 0:
                    self.after(0, lambda: self._set_status("Analysis stopped." if stopped else "Engine failed."))
                    return
                self.last_output_csv_path = self._morph_results_path()
                self.after(0, lambda: self._set_status("Engine complete. Open Analysis Overview."))
            except Exception as ex:
                with self.engine_lock:
                    self.engine_process = None
                self.after(0, lambda: self._set_status(f"Engine error: {ex}"))

        threading.Thread(target=_runner, daemon=True).start()

    def stop_engine_from_ui(self) -> None:
        with self.engine_lock:
            proc = self.engine_process
            if proc is not None:
                self.engine_stop_requested = True
        if proc is None:
            self._set_status("No analysis running.")
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        self._set_status("Stopping analysis...")

    def export_report_csv(self) -> None:
        src = self._morph_results_path()
        if not os.path.exists(src):
            src = self._ui_results_path()
        if not os.path.exists(src):
            if hasattr(self, "reports_status_lbl"):
                self.reports_status_lbl.configure(text="No report available yet.")
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="LiPAD_Report.csv", filetypes=[("CSV files", "*.csv")]
        )
        if not dst:
            return
        try:
            pd.read_csv(src).to_csv(dst, index=False)
            if hasattr(self, "reports_status_lbl"):
                self.reports_status_lbl.configure(text=f"Exported to: {dst}")
        except Exception as e:
            if hasattr(self, "reports_status_lbl"):
                self.reports_status_lbl.configure(text=f"Export failed: {e}")

    def clear_results(self) -> None:
        cleared = 0
        for p in [self._morph_results_path(), self._ui_results_path()]:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    cleared += 1
            except OSError:
                pass
        self.last_output_csv_path = None
        self._set_status(f"Cleared {cleared} result file(s).")
        self.select_tab("analysis_overview")

    def watch_last_video(self) -> None:
        candidate = self.last_annotated_video_path or self.uploaded_video_path
        if not candidate or not os.path.exists(candidate):
            self._set_status("No video available to open yet.")
            return
        try:
            os.startfile(candidate)
        except Exception as e:
            self._set_status(f"Failed to open video: {e}")

    def destroy(self) -> None:
        self._telemetry.stop()
        super().destroy()


if __name__ == "__main__":
    root = os.path.abspath(os.path.dirname(__file__))
    weights_file = os.path.join(root, "models", "best.pt")
    app = LipadApp(weights_path=weights_file)
    app.mainloop()
