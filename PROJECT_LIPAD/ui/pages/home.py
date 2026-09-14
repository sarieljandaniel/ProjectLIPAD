"""Home tab — telemetry dashboard."""

from __future__ import annotations

import customtkinter as ctk

from ui.components import (
    badge,
    body_label,
    headline,
    meta_label,
    mono_font,
    newsprint_button,
    newsprint_card,
    sans_font,
    stat_tile,
)
from ui.theme import ThemeTokens

FIELD_STEPS = [
    "Put this PC and the Raspberry Pi on the same LAN. Confirm the PC IP (default 192.168.1.47) and that the Pi answers as lipad@lipad.local.",
    "Allow the selected live-stream port in Windows Firewall below. Open Inspection Manager, choose TCP or UDP, then start live analysis. Confirm the Pi SSH link and the app starts rpicam-vid.",
    "LiDAR distance still arrives as UDP telemetry on port 50007 and is used for GSD while the quantized YOLO model runs.",
    "When a run finishes, open Analysis Overview for detections, then export a CSV from Reports.",
]


def render_home(app, parent, tokens: ThemeTokens) -> None:
    parent.grid_columnconfigure(0, weight=2)
    parent.grid_columnconfigure(1, weight=1)

    left = ctk.CTkFrame(parent, fg_color="transparent")
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    right = ctk.CTkFrame(parent, fg_color="transparent")
    right.grid(row=0, column=1, sticky="nsew")

    # Hero
    hero = ctk.CTkFrame(left, fg_color="transparent")
    hero.pack(fill="x", pady=(0, 16))
    row = ctk.CTkFrame(hero, fg_color="transparent")
    row.pack(anchor="w", pady=(0, 8))
    badge(row, tokens, "Project LiPAD", tone="accent").pack(side="left", padx=(0, 8))
    badge(row, tokens, "LAN · SSH · IMX519").pack(side="left", padx=(0, 8))
    badge(row, tokens, "Quantized YOLO").pack(side="left")
    headline(hero, tokens, "Structural health analytics dashboard", size=26).pack(anchor="w")
    body_label(
        hero,
        tokens,
        "Project LiPAD inspects cracks and corrosion with a quantized YOLO model. "
        "Live video comes from the IMX519 over TCP or UDP: this PC listens first, then the app "
        "SSHs to lipad@lipad.local and starts rpicam-vid. LiDAR telemetry on UDP 50007 "
        "updates distance while results land in Analysis Overview.",
        wraplength=560,
    ).pack(anchor="w", pady=(12, 0))
    cta = ctk.CTkFrame(hero, fg_color="transparent")
    cta.pack(anchor="w", pady=(14, 0))
    newsprint_button(
        cta,
        tokens,
        "Open Inspection Manager",
        command=lambda: app.select_tab("inspection_manager"),
    ).pack(side="left")

    # Metrics row
    metrics = ctk.CTkFrame(left, fg_color="transparent")
    metrics.pack(fill="x", pady=(0, 16))
    metrics.grid_columnconfigure((0, 1, 2, 3), weight=1)

    link_card, app._link_status_value = stat_tile(metrics, tokens, "Telemetry link", "—")
    link_card.grid(row=0, column=0, sticky="nsew", padx=(0, 1))

    dist_card, app._distance_value = stat_tile(metrics, tokens, "LiDAR distance", "— mm")
    dist_card.grid(row=0, column=1, sticky="nsew", padx=1)

    live_card, app._live_status_value = stat_tile(metrics, tokens, "Live engine", "Idle")
    live_card.grid(row=0, column=2, sticky="nsew", padx=1)

    pkt_card, app._last_packet_value = stat_tile(metrics, tokens, "Last packet", "Awaiting Pi…")
    pkt_card.grid(row=0, column=3, sticky="nsew", padx=(1, 0))

    # Telemetry console
    console_card = newsprint_card(left, tokens)
    console_card.pack(fill="both", expand=True)
    header = ctk.CTkFrame(console_card, fg_color=tokens.console_bg, corner_radius=0, height=44)
    header.pack(fill="x")
    header.pack_propagate(False)
    meta_label(header, tokens, "Ground station telemetry", inverted=True).pack(
        side="left", padx=16, pady=12
    )
    ctk.CTkLabel(
        header,
        text="STANDBY",
        text_color=tokens.inverted_muted,
        font=mono_font(9, "bold"),
    ).pack(side="right", padx=16)
    app._telemetry_header_status = header.winfo_children()[-1]

    app.telemetry_log = ctk.CTkTextbox(
        console_card,
        height=260,
        corner_radius=0,
        fg_color=tokens.console_bg,
        text_color=tokens.console_fg,
        font=mono_font(11),
        border_width=0,
        wrap="word",
    )
    app.telemetry_log.pack(fill="both", expand=True)
    app.telemetry_log.insert(
        "0.0",
        "Waiting for UDP LiDAR packets on port 50007. "
        "Start live analysis from Inspection Manager to SSH rpicam-vid on lipad@lipad.local "
        "and stream MPEG-TS to this PC on the selected TCP or UDP port.\n",
    )
    app.telemetry_log.configure(state="disabled")

    # Firewall permission
    firewall = newsprint_card(right, tokens)
    firewall.pack(fill="x", pady=(0, 16))
    firewall_inner = ctk.CTkFrame(firewall, fg_color="transparent")
    firewall_inner.pack(fill="x", padx=16, pady=14)
    meta_label(firewall_inner, tokens, "Live-stream permission").pack(anchor="w")
    ctk.CTkLabel(
        firewall_inner,
        text="Allow the Raspberry Pi through Windows Firewall",
        text_color=tokens.fg,
        font=sans_font(16, "bold"),
        anchor="w",
    ).pack(anchor="w", pady=(6, 0))
    body_label(
        firewall_inner,
        tokens,
        "Uses the port and TCP/UDP protocol currently selected in Inspection Manager. "
        "You will be asked to approve a Windows administrator prompt. The rule permits "
        "inbound traffic on the selected port across Domain, Private, and Public networks. "
        "Only approve this on a network you trust.",
        wraplength=280,
    ).pack(anchor="w", pady=(8, 10))
    newsprint_button(
        firewall_inner,
        tokens,
        "Allow selected port",
        command=app.request_live_firewall_access,
        width=220,
    ).pack(anchor="w")
    app._live_firewall_status_value = ctk.CTkLabel(
        firewall_inner,
        textvariable=app.live_firewall_status,
        text_color=tokens.muted,
        font=mono_font(10),
        anchor="w",
        justify="left",
        wraplength=280,
    )
    app._live_firewall_status_value.pack(anchor="w", pady=(10, 0))

    # Field guide (inverted)
    guide = newsprint_card(right, tokens, inverted=True)
    guide.pack(fill="both", expand=True)
    gh = ctk.CTkFrame(guide, fg_color="transparent")
    gh.pack(fill="x", padx=16, pady=14)
    meta_label(gh, tokens, "Field guide", inverted=True).pack(anchor="w")
    ctk.CTkLabel(
        gh,
        text="LAN live inspection",
        text_color=tokens.inverted_fg,
        font=sans_font(16, "bold"),
        anchor="w",
    ).pack(anchor="w", pady=(6, 0))
    ctk.CTkFrame(guide, height=1, fg_color=tokens.inverted_muted, corner_radius=0).pack(fill="x")

    for i, step in enumerate(FIELD_STEPS, start=1):
        item = ctk.CTkFrame(guide, fg_color="transparent")
        item.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(item, text=f"0{i}", text_color=tokens.accent, font=mono_font(12, "bold")).pack(
            anchor="w"
        )
        ctk.CTkLabel(
            item,
            text=step,
            text_color=tokens.inverted_fg,
            font=sans_font(12),
            anchor="w",
            justify="left",
            wraplength=240,
        ).pack(anchor="w", pady=(4, 0))
        if i < len(FIELD_STEPS):
            ctk.CTkFrame(guide, height=1, fg_color="#333333", corner_radius=0).pack(fill="x", padx=16)

    app._refresh_home_metrics()
