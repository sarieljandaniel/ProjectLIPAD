"""Project LiPAD — desktop app using quantized YOLO model."""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import threading
import time

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ui.prerequisites import (
    bootstrap_if_missing,
    ensure_local_ffmpeg_on_path,
    install_prerequisites,
    request_prerequisite_install as run_prerequisite_installer,
    summarize_prerequisites,
)

ensure_local_ffmpeg_on_path(_REPO_ROOT)
if __name__ == "__main__":
    bootstrap_if_missing(_REPO_ROOT)

import customtkinter as ctk
import pandas as pd
from PIL import Image
from tkinter import filedialog, messagebox

from ui.components import body_label, newsprint_button
from ui.pages.analysis import render_analysis
from ui.pages.home import render_home
from ui.pages.inspection import render_inspection
from ui.pages.reports import render_reports
from ui.pi_link import show_pi_link_dialog
from ui.sidebar import build_sidebar
from ui.telemetry import TelemetryListener, TelemetryPacket
from ui.theme import ThemeMode, get_tokens

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")


class LipadQuantizedApp(ctk.CTk):
    def __init__(
        self,
        weights_path: str = r"C:\Users\lenovo\Project_LIPAD_v3\Corrosion\PROJECT_LIPAD\models\best.onnx", # Updated default
        dist_ref: float = 1168.4,
    ):
        super().__init__()
        self.weights_path = weights_path

        self.title("Project LiPAD — Quantized YOLO Execution")
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

        self.live_pc_ip = ctk.StringVar(value=self._detect_local_ipv4())
        self.live_listen_host = ctk.StringVar(value="0.0.0.0")
        self.live_listen_port = ctk.StringVar(value="5000")
        self.live_protocol = ctk.StringVar(value="tcp")
        self.live_width = ctk.StringVar(value="1280")
        self.live_height = ctk.StringVar(value="720")
        self.live_bitrate = ctk.StringVar(value="3000000")
        self.live_firewall_status = ctk.StringVar(
            value="Not configured — allow the selected live-stream port before connecting the Pi."
        )
        self.prereq_status = ctk.StringVar(value=summarize_prerequisites(self._repo_root()))
        self._prereq_installing = False
        self.pi_ssh_host = ctk.StringVar(value="lipad.local")
        self.pi_ssh_user = ctk.StringVar(value="lipad")
        self.pi_ssh_password = ctk.StringVar(value="109791")
        self.live_auto_ssh = ctk.BooleanVar(value=True)
        self._live_running = False
        self._pi_ssh_process = None
        self._pi_ssh_client = None
        self._pi_ssh_channel = None
        self._live_preview_job = None
        self._live_preview_imgtk = None
        self._live_preview_mtime = None

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
        if hasattr(self, "_live_status_value") and self._live_status_value.winfo_exists():
            if self._live_running:
                live_text = "Streaming"
            elif self.engine_process is not None:
                live_text = "Analyzing"
            else:
                live_text = self.last_run_status.get() or "Idle"
            self._live_status_value.configure(text=live_text)
        if hasattr(self, "_last_packet_value") and self._last_packet_value.winfo_exists():
            if packet:
                self._last_packet_value.configure(text=f"{packet.source_ip} · {packet.timestamp}")
        if hasattr(self, "_telemetry_header_status") and self._telemetry_header_status.winfo_exists():
            self._telemetry_header_status.configure(
                text="RECEIVING" if self._telemetry.connected else "STANDBY"
            )

    def _repo_root(self) -> str:
        return os.path.abspath(os.path.dirname(__file__))

    @staticmethod
    def _detect_local_ipv4() -> str:
        """Return the IPv4 address chosen by Windows for normal LAN traffic."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                # UDP connect selects an interface locally without transmitting a packet.
                probe.connect(("192.0.2.1", 1))
                address = probe.getsockname()[0]
                if address and not address.startswith("127."):
                    return address
        except OSError:
            pass
        return "192.168.1.47"

    def _data_dir(self) -> str:
        return os.path.join(self._repo_root(), "data")

    def _morph_results_path(self) -> str:
        return os.path.join(self._data_dir(), "MorphologicalResults.csv")

    def _ui_results_path(self) -> str:
        return os.path.join(self._data_dir(), "results.csv")

    def _engine_script_path(self) -> str:
        return os.path.join(self._repo_root(), "Project_LIPAD_AI", "lipad_runtime_engine_quantized.py")

    def _quantized_weights_dir(self) -> str:
        return r"C:\Users\Admin\PROJECT_LIPAD\Corrosion\PROJECT_LIPAD\models"

    def _default_quantized_weights(self) -> str:
        return r"C:\Users\lenovo\Project_LIPAD_v3\Corrosion\PROJECT_LIPAD\models\best.onnx"

    def _set_status(self, text: str) -> None:
        self.last_run_status.set(text)
        if hasattr(self, "status_lbl") and self.status_lbl.winfo_exists():
            self.status_lbl.configure(text=text)
        if hasattr(self, "selected_video_lbl") and self.selected_video_lbl.winfo_exists():
            self.selected_video_lbl.configure(text=self._selected_video_label_text())
        if hasattr(self, "_engine_status_lbl") and self._engine_status_lbl.winfo_exists():
            self._engine_status_lbl.configure(text=text)
        if hasattr(self, "_file_status_lbl") and self._file_status_lbl.winfo_exists():
            self._file_status_lbl.configure(text=text)
        self._refresh_live_preview_chrome(text)
        if hasattr(self, "_live_status_value") and self._live_status_value.winfo_exists():
            self._refresh_home_metrics()

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
        if name == "inspection_manager":
            self._refresh_rpicam_command_box()
            if self._live_running:
                self._schedule_live_preview()
        elif name == "analysis_overview" and self._live_running:
            self._schedule_live_preview()

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

    def _live_preview_path(self) -> str:
        return os.path.join(self._data_dir(), "live_preview.jpg")

    def _live_raw_preview_path(self) -> str:
        return os.path.join(self._data_dir(), "live_raw_preview.jpg")

    def _parse_live_settings(self) -> tuple[str, str, int, int, int, int, str]:
        host = (self.live_listen_host.get() or "0.0.0.0").strip()
        pc_ip = (self.live_pc_ip.get() or "192.168.1.47").strip()
        try:
            port = int(float(self.live_listen_port.get()))
        except Exception:
            port = 5000
        try:
            width = int(float(self.live_width.get()))
        except Exception:
            width = 1280
        try:
            height = int(float(self.live_height.get()))
        except Exception:
            height = 720
        try:
            bitrate = int(float(self.live_bitrate.get()))
        except Exception:
            bitrate = 3_000_000
        protocol = (self.live_protocol.get() or "tcp").strip().lower()
        if protocol not in {"tcp", "udp"}:
            protocol = "tcp"
        return host, pc_ip, port, width, height, bitrate, protocol

    def rpicam_command_text(self) -> str:
        _host, pc_ip, port, width, height, bitrate, protocol = self._parse_live_settings()
        return (
            "rpicam-vid -t 0 "
            f"--width {width} --height {height} "
            f"--bitrate {bitrate} --inline "
            "--codec libav --libav-format mpegts "
            f"-o {protocol}://{pc_ip}:{port}"
        )

    @staticmethod
    def _live_firewall_rule_name(protocol: str, port: int) -> str:
        return f"Project LiPAD live stream ({protocol.upper()} {port})"

    def request_live_firewall_access(self) -> None:
        """Ask for consent, then add a scoped inbound Windows Firewall rule via UAC."""
        _host, _pc_ip, port, _width, _height, _bitrate, protocol = self._parse_live_settings()
        if not 1 <= port <= 65535:
            self.live_firewall_status.set("Invalid live-stream port. Enter a value from 1 to 65535.")
            return
        if os.name != "nt":
            self.live_firewall_status.set("Firewall setup is available only on Windows.")
            return

        rule_name = self._live_firewall_rule_name(protocol, port)
        permitted = messagebox.askyesno(
            "Allow live-stream port",
            "Project LiPAD needs an inbound Windows Firewall rule so the Raspberry Pi can "
            f"send its camera feed to this PC on {protocol.upper()} port {port}.\n\n"
            "Selecting Yes opens the Windows administrator permission prompt. The rule is "
            "limited to this port, but includes Public networks so it works on the current Wi-Fi connection.",
            icon="question",
            parent=self,
        )
        if not permitted:
            self.live_firewall_status.set("Firewall permission was not granted.")
            return

        self.live_firewall_status.set(
            f"Requesting administrator permission for {protocol.upper()} port {port}…"
        )

        def _add_rule() -> None:
            script = (
                "$ErrorActionPreference = 'Stop'; "
                f"$name = '{rule_name}'; "
                "Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | "
                "Remove-NetFirewallRule; "
                f"New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow "
                f"-Protocol {protocol.upper()} -LocalPort {port} -Profile Domain,Private,Public | Out-Null"
            )
            encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
            launcher = (
                "$process = Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -PassThru "
                "-ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', "
                f"'-EncodedCommand', '{encoded}'); exit $process.ExitCode"
            )
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", launcher],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if result.returncode == 0:
                    message = (
                        f"Firewall access enabled for {protocol.upper()} port {port} "
                        "on Domain, Private, and Public networks."
                    )
                else:
                    message = (
                        "Firewall permission was declined or the rule could not be created. "
                        "Allow it manually, then try again."
                    )
            except (OSError, subprocess.TimeoutExpired):
                message = "Could not request firewall permission. Allow the port manually, then try again."
            self.after(0, lambda: self.live_firewall_status.set(message))

        threading.Thread(target=_add_rule, daemon=True, name="live-firewall-rule").start()

    def request_prerequisite_install(self) -> None:
        """Ask for consent, then pip-install libraries and download FFmpeg if needed."""
        if self._prereq_installing:
            self.prereq_status.set("Library install is already running.")
            return
        permitted = run_prerequisite_installer(
            self._repo_root(),
            parent=self,
            force=True,
            install=False,
        )
        if not permitted:
            self.prereq_status.set("Library install permission was not granted.")
            return
        self._prereq_installing = True
        self.prereq_status.set("Installing required libraries…")

        def _on_status(message: str) -> None:
            self.after(0, lambda m=message: self.prereq_status.set(m))

        def _run() -> None:
            try:
                result = install_prerequisites(self._repo_root(), on_status=_on_status)
                ensure_local_ffmpeg_on_path(self._repo_root())
                self.after(0, lambda m=result: self.prereq_status.set(m))
            except Exception as exc:
                self.after(0, lambda e=exc: self.prereq_status.set(f"Library install failed: {e}"))
            finally:
                self._prereq_installing = False

        threading.Thread(target=_run, daemon=True, name="lipad-prereq-install").start()

    def _refresh_rpicam_command_box(self) -> None:
        if not hasattr(self, "rpicam_cmd_box") or not self.rpicam_cmd_box.winfo_exists():
            return
        cmd = self.rpicam_command_text()
        self.rpicam_cmd_box.configure(state="normal")
        self.rpicam_cmd_box.delete("0.0", "end")
        self.rpicam_cmd_box.insert("0.0", cmd)
        self.rpicam_cmd_box.configure(state="disabled")

    def _engine_common_args(self, gsd: float, stride: int, inf_w: int, inspection: str, corrosion_env: str) -> list[str]:
        return [
            "--gsd", str(gsd),
            "--output_csv", self._morph_results_path(),
            "--results_csv", self._ui_results_path(),
            "--no_preview",
            "--frame_stride", str(stride),
            "--inference_width", str(inf_w),
            "--inspection_type", inspection,
            "--corrosion_env", corrosion_env,
        ]

    def _parse_engine_tuning(self) -> tuple[float, int, int, str, str]:
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
        return gsd, stride, inf_w, inspection, corrosion_env

    def _crack_weights_or_status(self, inspection: str) -> str | None:
        if inspection.lower() != "crack":
            return None
        weights = self.weights_path
        if not os.path.exists(weights):
            self._set_status(f"Quantized YOLO model missing: {weights}")
            return ""
        return weights

    def _live_ready_flag_path(self) -> str:
        return os.path.join(self._data_dir(), "live_ready.flag")

    def copy_rpicam_command(self) -> None:
        cmd = self.rpicam_command_text()
        try:
            self.clipboard_clear()
            self.clipboard_append(cmd)
            self._set_status("Copied rpicam-vid command to clipboard.")
        except Exception as e:
            self._set_status(f"Clipboard copy failed: {e}")

    def _apply_preview_image(self, lbl, path: str) -> None:
        if lbl is None or not lbl.winfo_exists():
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if not self._live_running:
                return
            lbl.configure(image="", text=self.last_run_status.get() or "Waiting for first live frame…")
            return
        if getattr(lbl, "_lipad_mtime", None) == mtime:
            return
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            img.thumbnail((720, 405))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self._live_preview_imgtk = ctk_img
            self._live_preview_mtime = mtime
            lbl._lipad_mtime = mtime
            lbl.configure(image=ctk_img, text="")
        except Exception:
            pass

    def _refresh_live_preview_chrome(self, status_text: str | None = None) -> None:
        for caption_name in ("_live_raw_preview_caption", "_live_inference_preview_caption"):
            caption = getattr(self, caption_name, None)
            if caption is not None and caption.winfo_exists():
                caption.configure(
                    text="LIVE" if self._live_running else "NO SIGNAL",
                    text_color=self.tokens.accent if self._live_running else self.tokens.inverted_muted,
                )
        previews = (
            ("live_raw_preview_lbl", self._live_raw_preview_path(), "Waiting for raw IMX519 frames…"),
            ("live_preview_lbl", self._live_preview_path(), "Waiting for inference frames…"),
        )
        for label_name, path, fallback in previews:
            lbl = getattr(self, label_name, None)
            if lbl is None or not lbl.winfo_exists() or (os.path.exists(path) and self._live_running):
                continue
            try:
                lbl.configure(image="", text=status_text or self.last_run_status.get() or fallback)
            except Exception:
                pass

    def _clear_live_preview(self) -> None:
        self._live_preview_mtime = None
        self._live_preview_imgtk = None
        for path in (self._live_preview_path(), self._live_raw_preview_path()):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        self._refresh_live_preview_chrome("Live analysis stopped. Start again to confirm the Pi link.")

    def _schedule_live_preview(self) -> None:
        if self._live_preview_job is not None:
            try:
                self.after_cancel(self._live_preview_job)
            except Exception:
                pass
            self._live_preview_job = None
        self._tick_live_preview()

    def _tick_live_preview(self) -> None:
        self._live_preview_job = None
        if not self._live_running:
            return
        self._apply_preview_image(getattr(self, "live_raw_preview_lbl", None), self._live_raw_preview_path())
        self._apply_preview_image(getattr(self, "live_preview_lbl", None), self._live_preview_path())
        self._apply_preview_image(getattr(self, "analysis_live_preview_lbl", None), self._live_preview_path())
        self._live_preview_job = self.after(40, self._tick_live_preview)

    def _parse_pi_ssh_target(self) -> tuple[str, str, str]:
        user = (self.pi_ssh_user.get() or "lipad").strip()
        host = (self.pi_ssh_host.get() or "lipad.local").strip()
        password = self.pi_ssh_password.get() or "109791"
        raw = host
        if raw.lower().startswith("ssh "):
            raw = raw[4:].strip()
        if "@" in raw and " " not in raw:
            user, host = raw.rsplit("@", 1)
        return user, host, password

    def _ssh_exec_paramiko(self, user: str, host: str, password: str, remote: str, keep_alive: bool) -> None:
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            username=user,
            password=password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=False,
            allow_agent=False,
        )
        if keep_alive:
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(15)
            _stdin, stdout, _stderr = client.exec_command(remote, get_pty=True)
            self._pi_ssh_client = client
            self._pi_ssh_channel = stdout.channel
            return
        try:
            _stdin, stdout, stderr = client.exec_command(remote, timeout=8)
            stdout.channel.recv_exit_status()
            _ = stderr.read()
        finally:
            client.close()

    def _start_pi_rpicam(self) -> None:
        user, host, password = self._parse_pi_ssh_target()
        if not host:
            self.after(0, lambda: self._set_status(
                "Listener is up, but SSH host is empty. Set lipad@lipad.local to start the camera."
            ))
            return
        remote = self.rpicam_command_text()
        try:
            import paramiko  # noqa: F401
        except ImportError:
            self.after(0, lambda: self._set_status(
                "paramiko is missing. Run: pip install paramiko"
            ))
            return
        try:
            self._ssh_exec_paramiko(user, host, password, remote, keep_alive=True)
            self.after(0, lambda: self._set_status(
                f"Listener ready — started rpicam-vid on {user}@{host}"
            ))
        except Exception as e:
            self._pi_ssh_client = None
            self._pi_ssh_channel = None
            err = str(e)
            target = f"{user}@{host}"
            self.after(0, lambda msg=f"SSH to {target} failed ({err}). Check the Pi is on the LAN and SSH is enabled.": self._set_status(msg))

    def _stop_pi_rpicam(self) -> None:
        proc = self._pi_ssh_process
        self._pi_ssh_process = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._pi_ssh_channel = None
        client = self._pi_ssh_client
        self._pi_ssh_client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        user, host, password = self._parse_pi_ssh_target()
        if not host:
            return

        def _kill() -> None:
            try:
                self._ssh_exec_paramiko(user, host, password, "pkill -f rpicam-vid || true", keep_alive=False)
            except Exception:
                pass

        threading.Thread(target=_kill, daemon=True).start()

    def _engine_failure_hint(self) -> str:
        log_path = os.path.join(self._data_dir(), "engine_error.log")
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
        except OSError:
            return "Live engine failed. Check data/engine_error.log"
        if not lines:
            return "Live engine failed. Check data/engine_error.log"
        last = lines[-1]
        joined = "\n".join(lines)
        if "WinError 1455" in joined or "paging file is too small" in joined.lower():
            return (
                "Live engine ran out of virtual memory while starting the camera listener. "
                "Close other heavy apps and try again. Details: data/engine_error.log"
            )
        if last.startswith("OSError:") or last.startswith("RuntimeError:") or last.startswith("[CRITICAL"):
            return last
        return f"Live engine failed: {last}"

    def start_live_engine_from_ui(self) -> None:
        with self.engine_lock:
            if self.engine_process is not None:
                self._set_status("Engine already running. Stop it first.")
                return
        detected_ip = self._detect_local_ipv4()
        configured_ip = (self.live_pc_ip.get() or "").strip()
        if detected_ip and configured_ip != detected_ip:
            use_detected = messagebox.askyesno(
                "PC IP address changed",
                f"The live stream is configured for {configured_ip or 'no address'}, but Windows is "
                f"currently using {detected_ip}.\n\nUse {detected_ip} for the Raspberry Pi stream?",
                icon="warning",
                parent=self,
            )
            if use_detected:
                self.live_pc_ip.set(detected_ip)
        self._set_status("Checking Raspberry Pi connection…")
        show_pi_link_dialog(self, self.tokens, on_confirm=self._launch_live_engine)

    def _launch_live_engine(self) -> None:
        if self.current_tab != "inspection_manager":
            self.select_tab("inspection_manager")
        with self.engine_lock:
            if self.engine_process is not None:
                self._set_status("Engine already running. Stop it first.")
                return
            self.engine_stop_requested = False
            self._live_running = True

        gsd, stride, inf_w, inspection, corrosion_env = self._parse_engine_tuning()
        host, _pc_ip, port, width, height, _bitrate, protocol = self._parse_live_settings()
        os.makedirs(self._data_dir(), exist_ok=True)
        preview = self._live_preview_path()
        raw_preview = self._live_raw_preview_path()
        ready_flag = self._live_ready_flag_path()
        for stale in (preview, raw_preview, ready_flag):
            try:
                if os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass
        self._live_preview_mtime = None

        cmd = [
            sys.executable, self._engine_script_path(),
            "--live",
            "--listen_host", host,
            "--listen_port", str(port),
            "--stream_protocol", protocol,
            "--stream_width", str(width),
            "--stream_height", str(height),
            "--preview_jpeg", preview,
            "--raw_preview_jpeg", raw_preview,
            "--ready_flag", ready_flag,
            *self._engine_common_args(gsd, stride, inf_w, inspection, corrosion_env),
        ]
        weights = self._crack_weights_or_status(inspection)
        if weights == "":
            self._live_running = False
            return
        if weights:
            cmd.extend(["--weights", weights])

        def _runner() -> None:
            self.after(0, lambda: self._set_status(
                f"Listening on {host}:{port}. Waiting to auto-start rpicam-vid over SSH…"
            ))
            self.after(0, self._schedule_live_preview)
            log_file_path = os.path.join(self._data_dir(), "engine_error.log")
            try:
                with open(log_file_path, "w") as log_file:
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    with self.engine_lock:
                        self.engine_process = subprocess.Popen(
                            cmd, stdout=log_file, stderr=log_file, text=True, env=env
                        )
                        proc = self.engine_process
                    deadline = time.time() + 25
                    while time.time() < deadline:
                        if os.path.exists(ready_flag) or proc.poll() is not None:
                            break
                        time.sleep(0.1)
                    if proc.poll() is None:
                        self._start_pi_rpicam()
                    code = proc.wait()

                with self.engine_lock:
                    stopped = self.engine_stop_requested
                    self.engine_process = None
                    self.engine_stop_requested = False
                self._live_running = False
                self._stop_pi_rpicam()
                self.after(0, self._clear_live_preview)
                self.last_output_csv_path = self._morph_results_path()
                if code != 0:
                    msg = "Live analysis stopped." if stopped else self._engine_failure_hint()
                    self.after(0, lambda m=msg: self._set_status(m))
                    return
                self.after(0, lambda: self._set_status("Live analysis complete. Open Analysis Overview."))
            except Exception as ex:
                with self.engine_lock:
                    self.engine_process = None
                self._live_running = False
                self._stop_pi_rpicam()
                self.after(0, lambda: self._set_status(f"Live engine error: {ex}"))

        threading.Thread(target=_runner, daemon=True).start()

    def run_engine_from_ui(self) -> None:
        if not self.uploaded_video_path:
            self._set_status("No MP4 selected. Add/select a video first.")
            return
        with self.engine_lock:
            if self.engine_process is not None:
                self._set_status("Engine already running. Stop it first.")
                return
            self.engine_stop_requested = False
            self._live_running = False

        os.makedirs(self._data_dir(), exist_ok=True)
        annotated_dir = os.path.join(self._data_dir(), "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.uploaded_video_path))[0]
        annotated_video = os.path.join(annotated_dir, f"{base}_annotated.mp4")
        self.last_annotated_video_path = annotated_video
        gsd, stride, inf_w, inspection, corrosion_env = self._parse_engine_tuning()
        cmd = [
            sys.executable, self._engine_script_path(),
            "--video", self.uploaded_video_path,
            "--output_video", annotated_video,
            *self._engine_common_args(gsd, stride, inf_w, inspection, corrosion_env),
        ]
        weights = self._crack_weights_or_status(inspection)
        if weights == "":
            return
        if weights:
            cmd.extend(["--weights", weights])

        def _runner() -> None:
            self.after(0, lambda: self._set_status("Running quantized single-model engine..."))
            log_file_path = os.path.join(self._data_dir(), "engine_error.log")
            
            try:
                with open(log_file_path, "w") as log_file:
                    with self.engine_lock:
                        self.engine_process = subprocess.Popen(
                            cmd, stdout=log_file, stderr=log_file, text=True
                        )
                        proc = self.engine_process
                    code = proc.wait()

                with self.engine_lock:
                    stopped = self.engine_stop_requested
                    self.engine_process = None
                    self.engine_stop_requested = False

                if code != 0:
                    self.after(0, lambda: self._set_status(
                        "Analysis stopped." if stopped else "Engine failed. Check data/engine_error.log"
                    ))
                    return

                self.last_output_csv_path = self._morph_results_path()
                self.after(0, lambda: self._set_status("Quantized engine complete. Open Analysis Overview."))
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
        self._terminate_engine_process(proc)
        self._live_running = False
        self._stop_pi_rpicam()
        self._clear_live_preview()
        self._set_status("Stopping analysis...")

    def _terminate_engine_process(self, proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

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
        self._live_running = False
        self._stop_pi_rpicam()
        with self.engine_lock:
            proc = self.engine_process
        if proc is not None:
            self._terminate_engine_process(proc)
        self._telemetry.stop()
        super().destroy()


if __name__ == "__main__":
    weights_path = r"C:\Users\lenovo\Project_LIPAD_v3\Corrosion\PROJECT_LIPAD\models\best.onnx"
    app = LipadQuantizedApp(weights_path=weights_path)
    app.mainloop()
