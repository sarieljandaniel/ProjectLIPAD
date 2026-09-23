"""Reports tab."""

from __future__ import annotations

import customtkinter as ctk

from ui.components import body_label, newsprint_button, newsprint_card, page_header
from ui.theme import ThemeTokens


def render_reports(app, parent, tokens: ThemeTokens) -> None:
    page_header(parent, tokens, "Export", "Reports")
    body_label(
        parent,
        tokens,
        "Download the latest morphological results as a CSV for archival or third-party review.",
        wraplength=520,
    ).pack(anchor="w", pady=(0, 16))

    card = newsprint_card(parent, tokens)
    card.pack(fill="x")
    inner = ctk.CTkFrame(card, fg_color="transparent")
    inner.pack(fill="x", padx=20, pady=20)
    newsprint_button(inner, tokens, "Export CSV", command=app.export_report_csv).pack(anchor="w")
    app.reports_status_lbl = body_label(inner, tokens, "", mono=True)
    app.reports_status_lbl.pack(anchor="w", pady=(12, 0))
