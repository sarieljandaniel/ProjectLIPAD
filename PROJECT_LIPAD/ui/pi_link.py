"""Probe Raspberry Pi SSH reachability and confirm before live analysis."""

from __future__ import annotations

import ipaddress
import socket
import threading
from dataclasses import dataclass
from collections.abc import Callable

import customtkinter as ctk

from ui.components import body_label, meta_label, newsprint_button, sans_font
from ui.theme import ThemeTokens

ProgressFn = Callable[[str], None]


@dataclass
class PiLinkStatus:
    user: str
    host: str
    pc_ip: str
    resolved_ip: str | None
    port22_ok: bool
    ssh_ok: bool
    detail: str

    @property
    def connected(self) -> bool:
        return self.ssh_ok


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _call_with_timeout(fn, timeout: float, default=None):
    box: dict = {}

    def _worker() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return default, TimeoutError(f"timed out after {timeout:.0f}s")
    if "error" in box:
        return default, box["error"]
    return box.get("value"), None


def _resolve_host(host: str, timeout: float) -> str:
    if _is_ip(host):
        return host

    def _lookup() -> str:
        infos = socket.getaddrinfo(host, 22, type=socket.SOCK_STREAM)
        if not infos:
            raise socket.gaierror("no addresses")
        return str(infos[0][4][0])

    ip, err = _call_with_timeout(_lookup, timeout)
    if err is not None:
        raise TimeoutError(
            f"Could not resolve {host} in {timeout:.0f}s. "
            "Windows often stalls on .local names — enter the Pi IP in Pi SSH host."
        ) from err
    return str(ip)


def probe_pi_link(
    user: str,
    host: str,
    password: str,
    pc_ip: str,
    timeout: float = 6.0,
    on_progress: ProgressFn | None = None,
) -> PiLinkStatus:
    def note(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    if not host:
        return PiLinkStatus(user, host, pc_ip, None, False, False, "SSH host is empty.")

    note(f"Resolving {host}…")
    try:
        resolved = _resolve_host(host, min(4.0, timeout))
    except Exception as exc:
        return PiLinkStatus(user, host, pc_ip, None, False, False, str(exc))

    note(f"Opening TCP 22 on {resolved}…")
    try:
        with socket.create_connection((resolved, 22), timeout=min(4.0, timeout)):
            port22_ok = True
    except OSError as exc:
        return PiLinkStatus(
            user,
            host,
            pc_ip,
            resolved,
            False,
            False,
            f"TCP port 22 is closed on {resolved} ({exc}). Enable SSH on the Pi.",
        )

    try:
        import paramiko
    except ImportError:
        return PiLinkStatus(
            user,
            host,
            pc_ip,
            resolved,
            port22_ok,
            False,
            "paramiko is missing. Run: pip install paramiko",
        )

    note(f"Logging in as {user}@{resolved}…")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def _login() -> None:
        client.connect(
            hostname=resolved,
            username=user,
            password=password,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            look_for_keys=False,
            allow_agent=False,
        )

    _, login_err = _call_with_timeout(_login, timeout + 2.0)
    if login_err is not None:
        try:
            client.close()
        except Exception:
            pass
        return PiLinkStatus(
            user,
            host,
            pc_ip,
            resolved,
            True,
            False,
            f"SSH login failed: {login_err}",
        )

    try:
        transport = client.get_transport()
        ok = transport is not None and transport.is_active()
        if not ok:
            return PiLinkStatus(
                user, host, pc_ip, resolved, True, False, "SSH connected but the session is not active."
            )
        return PiLinkStatus(
            user,
            host,
            pc_ip,
            resolved,
            True,
            True,
            f"Connected as {user}@{host} ({resolved}). SSH session is active.",
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


def show_pi_link_dialog(app, tokens: ThemeTokens, on_confirm) -> None:
    previous = getattr(app, "_pi_link_win", None)
    if previous is not None:
        try:
            previous.grab_release()
        except Exception:
            pass
        try:
            previous.destroy()
        except Exception:
            pass
        app._pi_link_win = None

    user, host, password = app._parse_pi_ssh_target()
    _listen_host, pc_ip, port, width, height, _bitrate, protocol = app._parse_live_settings()

    win = ctk.CTkToplevel(app)
    app._pi_link_win = win
    win.title("Confirm Raspberry Pi link")
    win.geometry("540x460")
    win.resizable(False, False)
    win.configure(fg_color=tokens.bg)
    win.transient(app)
    generation = {"n": 0}

    inner = ctk.CTkFrame(win, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=24, pady=20)

    meta_label(inner, tokens, "Pre-flight").pack(anchor="w")
    ctk.CTkLabel(
        inner,
        text="Confirm this PC can reach the Pi",
        text_color=tokens.fg,
        font=sans_font(18, "bold"),
        anchor="w",
    ).pack(anchor="w", pady=(6, 8))
    body_label(
        inner,
        tokens,
        "Shown every time you start live analysis. If this stays on Resolving, put the Pi's "
        "numeric IP in Pi SSH host instead of lipad.local.",
        wraplength=480,
    ).pack(anchor="w")

    ctk.CTkLabel(
        inner,
        text=(
            f"This PC IP: {pc_ip}\n"
            f"Pi SSH: {user}@{host}\n"
            f"Stream: {protocol}://{pc_ip}:{port}  ({width}×{height})"
        ),
        text_color=tokens.fg,
        font=sans_font(12),
        anchor="w",
        justify="left",
    ).pack(anchor="w", pady=(14, 8))

    status_lbl = ctk.CTkLabel(
        inner,
        text="Checking SSH…",
        text_color=tokens.muted,
        font=sans_font(13),
        anchor="w",
        justify="left",
        wraplength=480,
    )
    status_lbl.pack(anchor="w", pady=(4, 16))

    btns = ctk.CTkFrame(inner, fg_color="transparent")
    btns.pack(fill="x", side="bottom")

    start_btn_holder: dict = {}

    def _alive() -> bool:
        try:
            return bool(win.winfo_exists())
        except Exception:
            return False

    def _close() -> None:
        generation["n"] += 1
        if getattr(app, "_pi_link_win", None) is win:
            app._pi_link_win = None
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass

    def _confirm() -> None:
        _close()
        on_confirm()

    def _set_progress(msg: str, ticket: int) -> None:
        if ticket != generation["n"] or not _alive():
            return
        status_lbl.configure(text=msg, text_color=tokens.muted)

    def _set_result(result: PiLinkStatus, ticket: int) -> None:
        if ticket != generation["n"] or not _alive():
            return
        btn = start_btn_holder.get("btn")
        if result.connected:
            status_lbl.configure(text=f"Connected. {result.detail}", text_color=tokens.fg)
            if btn is not None:
                btn.configure(state="normal")
            return
        status_lbl.configure(text=f"Not connected. {result.detail}", text_color=tokens.accent)
        if btn is not None:
            btn.configure(state="normal")

    def _check() -> None:
        if not _alive():
            return
        generation["n"] += 1
        ticket = generation["n"]
        status_lbl.configure(text="Checking SSH…", text_color=tokens.muted)
        btn = start_btn_holder.get("btn")
        if btn is not None:
            btn.configure(state="disabled")

        got = {"done": False}

        def _progress(msg: str) -> None:
            app.after(0, lambda m=msg, t=ticket: _set_progress(m, t))

        def _deliver(result: PiLinkStatus, t: int) -> None:
            if t != ticket:
                return
            got["done"] = True
            _set_result(result, t)

        def _work() -> None:
            try:
                result = probe_pi_link(
                    user, host, password, pc_ip, timeout=6.0, on_progress=_progress
                )
            except Exception as exc:
                result = PiLinkStatus(
                    user, host, pc_ip, None, False, False, f"SSH check crashed: {exc}"
                )
            app.after(0, lambda r=result, t=ticket: _deliver(r, t))

        threading.Thread(target=_work, daemon=True).start()

        def _watchdog() -> None:
            if got["done"] or ticket != generation["n"] or not _alive():
                return
            _deliver(
                PiLinkStatus(
                    user,
                    host,
                    pc_ip,
                    None,
                    False,
                    False,
                    "SSH check timed out. Enter the Pi IP (not lipad.local) and click Check again.",
                ),
                ticket,
            )

        win.after(14000, _watchdog)

    newsprint_button(btns, tokens, "Cancel", command=_close, variant="secondary").pack(side="left")
    newsprint_button(btns, tokens, "Check again", command=_check, variant="secondary").pack(
        side="left", padx=(8, 0)
    )
    start_btn = newsprint_button(btns, tokens, "Start live analysis", command=_confirm)
    start_btn.configure(state="disabled")
    start_btn.pack(side="right")
    start_btn_holder["btn"] = start_btn

    win.protocol("WM_DELETE_WINDOW", _close)
    _check()
