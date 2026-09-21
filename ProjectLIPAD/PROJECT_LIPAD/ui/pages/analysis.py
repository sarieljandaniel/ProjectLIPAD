"""Analysis Overview tab."""

from __future__ import annotations

import os

import customtkinter as ctk
import pandas as pd
from tkinter import ttk

from ui.components import (
    badge,
    body_label,
    headline,
    meta_label,
    mono_font,
    newsprint_button,
    newsprint_card,
    page_header,
    sans_font,
    stat_tile,
    configure_tree_style,
)
from ui.theme import ThemeTokens


def _load_results(app):
    for path in (app._morph_results_path(), app._ui_results_path()):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return None


def _repair_guidance(df: pd.DataFrame) -> dict:
    has_crack = (df["Type"].astype(str).str.lower() == "crack").any() if "Type" in df.columns else False
    has_corrosion = (df["Type"].astype(str).str.lower() == "corrosion").any() if "Type" in df.columns else False
    if df.empty:
        return {"headline": "Awaiting data", "body": "Run an inspection first.", "notes": []}
    latest = df.iloc[-1]
    defect_type = str(latest.get("Type", "Unknown"))
    severity = str(latest.get("Severity", "Unknown"))
    suggestion = "No action required."
    if defect_type == "Crack" and severity == "Structural":
        suggestion = "Immediate epoxy injection and structural bracing required."
    elif defect_type == "Corrosion" and severity == "Severe":
        suggestion = "Immediate abrasive blasting and cathodic protection required."
    notes = []
    if not has_crack:
        notes.append("No cracks detected in the latest run.")
    if not has_corrosion:
        notes.append("No corrosion detected in the latest run.")
    return {
        "headline": f"Last detected: {defect_type} ({severity})",
        "body": suggestion,
        "notes": notes,
        "latest_type": defect_type,
    }


def render_analysis(app, parent, tokens: ThemeTokens) -> None:
    configure_tree_style(tokens)
    df = _load_results(app)

    header_row = ctk.CTkFrame(parent, fg_color="transparent")
    header_row.pack(fill="x")
    hdr = ctk.CTkFrame(header_row, fg_color="transparent")
    hdr.pack(side="left", fill="x", expand=True)
    page_header(hdr, tokens, "Analytics", "Analysis overview")

    actions = ctk.CTkFrame(header_row, fg_color="transparent")
    actions.pack(side="right", anchor="s", pady=(0, 20))
    newsprint_button(actions, tokens, "Refresh", command=lambda: app.select_tab("analysis_overview"), variant="secondary").pack(
        side="left", padx=(0, 8)
    )
    newsprint_button(actions, tokens, "Clear", command=app.clear_results, variant="secondary").pack(side="left", padx=(0, 8))
    newsprint_button(actions, tokens, "Watch video", command=app.watch_last_video, variant="secondary").pack(side="left")

    if getattr(app, "_live_running", False):
        live_card = newsprint_card(parent, tokens)
        live_card.pack(fill="x", pady=(0, 16))
        lv = ctk.CTkFrame(live_card, fg_color="transparent")
        lv.pack(fill="x", padx=16, pady=14)
        meta_label(lv, tokens, "Live engine output").pack(anchor="w")
        body_label(
            lv,
            tokens,
            "Annotated IMX519 frames from the runtime engine. Refresh the table to pull the latest CSV snapshot.",
            wraplength=720,
        ).pack(anchor="w", pady=(4, 8))
        app.analysis_live_preview_lbl = ctk.CTkLabel(
            lv,
            text="Waiting for annotated live frames…",
            text_color=tokens.muted,
            font=mono_font(12),
            width=640,
            height=280,
        )
        app.analysis_live_preview_lbl.pack(anchor="w")

    summary = {"total": 0, "cracks": 0, "corrosion": 0, "latest_severity": "—"}
    if df is not None and not df.empty:
        summary["total"] = len(df)
        if "Type" in df.columns:
            summary["cracks"] = int((df["Type"].astype(str).str.lower() == "crack").sum())
            summary["corrosion"] = int((df["Type"].astype(str).str.lower() == "corrosion").sum())
        if "Severity" in df.columns:
            summary["latest_severity"] = str(df.iloc[-1].get("Severity", "—"))

    stats = ctk.CTkFrame(parent, fg_color="transparent")
    stats.pack(fill="x", pady=(0, 16))
    stats.grid_columnconfigure((0, 1, 2, 3), weight=1)
    for i, (label, key) in enumerate(
        [("Total defects", "total"), ("Cracks", "cracks"), ("Corrosion", "corrosion"), ("Latest severity", "latest_severity")]
    ):
        tile, _ = stat_tile(stats, tokens, label, str(summary[key]))
        tile.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 1, 0))

    body = ctk.CTkFrame(parent, fg_color="transparent")
    body.pack(fill="both", expand=True)
    body.grid_columnconfigure(0, weight=2)
    body.grid_columnconfigure(1, weight=1)

    table_card = newsprint_card(body, tokens)
    table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    th = ctk.CTkFrame(table_card, fg_color="transparent")
    th.pack(fill="x", padx=16, pady=12)
    headline(th, tokens, "Defect analytics", size=18).pack(side="left")
    if df is not None and not df.empty and "Type" in df.columns:
        badge(th, tokens, str(df.iloc[-1]["Type"]), tone="accent").pack(side="right")

    if df is None or df.empty:
        body_label(table_card, tokens, "No defects detected yet. Run a corrosion or crack analysis in Inspection Manager.").pack(
            padx=16, pady=24
        )
    else:
        tf = ctk.CTkFrame(table_card, fg_color="transparent")
        tf.pack(fill="both", expand=True, padx=8, pady=(0, 12))
        columns = list(df.columns)
        tree = ttk.Treeview(tf, columns=columns, show="headings", style="Newsprint.Treeview")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tf, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for col in columns:
            tree.heading(col, text=col.upper())
            tree.column(col, width=120, anchor="w")
        for _, row in df.iterrows():
            tree.insert("", "end", values=["" if pd.isna(v) else str(v) for v in row.tolist()])
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

    repair = _repair_guidance(df if df is not None else pd.DataFrame())
    guide = newsprint_card(body, tokens, inverted=True)
    guide.grid(row=0, column=1, sticky="nsew")
    gh = ctk.CTkFrame(guide, fg_color="transparent")
    gh.pack(fill="x", padx=16, pady=14)
    meta_label(gh, tokens, "Repair guidance", inverted=True).pack(anchor="w")
    ctk.CTkLabel(
        gh,
        text=repair["headline"],
        text_color=tokens.inverted_fg,
        font=sans_font(16, "bold"),
        anchor="w",
        wraplength=260,
        justify="left",
    ).pack(anchor="w", pady=(8, 0))
    ctk.CTkFrame(guide, height=1, fg_color=tokens.inverted_muted, corner_radius=0).pack(fill="x")
    ctk.CTkLabel(
        guide,
        text=repair["body"],
        text_color=tokens.inverted_fg,
        font=sans_font(12),
        anchor="w",
        justify="left",
        wraplength=260,
    ).pack(anchor="w", padx=16, pady=14)
    for note in repair.get("notes", []):
        ctk.CTkLabel(
            guide,
            text=note,
            text_color=tokens.inverted_muted,
            font=sans_font(11),
            anchor="w",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))
