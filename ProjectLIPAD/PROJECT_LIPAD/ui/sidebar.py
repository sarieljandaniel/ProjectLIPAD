"""Sidebar navigation for the LiPAD desktop shell."""

from __future__ import annotations

import customtkinter as ctk

from ui.components import meta_label, newsprint_button, sans_font, serif_font
from ui.theme import ThemeTokens

NAV_ITEMS = [
    ("home", "Home"),
    ("inspection_manager", "Inspection Manager"),
    ("analysis_overview", "Analysis Overview"),
    ("reports", "Reports"),
]


def build_sidebar(app, parent, tokens: ThemeTokens) -> ctk.CTkFrame:
    sidebar = ctk.CTkFrame(
        parent,
        width=220,
        corner_radius=0,
        fg_color=tokens.bg,
        border_width=0,
    )
    sidebar.grid_propagate(False)

    header = ctk.CTkFrame(sidebar, fg_color="transparent")
    header.pack(fill="x", padx=20, pady=(24, 16))
    meta_label(header, tokens, "Structural Health").pack(anchor="w")
    ctk.CTkLabel(
        header,
        text="LiPAD AI",
        text_color=tokens.fg,
        font=serif_font(24, "bold"),
        anchor="w",
    ).pack(anchor="w", pady=(4, 0))
    ctk.CTkFrame(sidebar, height=1, fg_color=tokens.border, corner_radius=0).pack(fill="x")

    app._nav_buttons = {}
    nav = ctk.CTkFrame(sidebar, fg_color="transparent")
    nav.pack(fill="both", expand=True)

    for tab_id, label in NAV_ITEMS:
        active = app.current_tab == tab_id
        btn = ctk.CTkButton(
            nav,
            text=label.upper(),
            anchor="w",
            height=44,
            corner_radius=0,
            fg_color=tokens.button_primary_bg if active else "transparent",
            text_color=tokens.button_primary_fg if active else tokens.fg,
            hover_color=tokens.rule,
            border_width=0,
            font=sans_font(10, "bold"),
            command=lambda n=tab_id: app.select_tab(n),
        )
        btn.pack(fill="x")
        ctk.CTkFrame(nav, height=1, fg_color=tokens.rule, corner_radius=0).pack(fill="x")
        app._nav_buttons[tab_id] = btn

    ctk.CTkFrame(sidebar, height=1, fg_color=tokens.border, corner_radius=0).pack(fill="x", side="bottom")
    footer = ctk.CTkFrame(sidebar, fg_color="transparent")
    footer.pack(fill="x", side="bottom", padx=20, pady=16)

    mode_label = "Dark mode" if app.theme_mode.value == "light" else "Light mode"
    newsprint_button(footer, tokens, mode_label, command=app.toggle_theme, variant="secondary").pack(
        fill="x", pady=(0, 10)
    )
    meta_label(footer, tokens, "Edition").pack(anchor="w")
    ctk.CTkLabel(footer, text="v1.0 · Newsprint UI", text_color=tokens.fg, font=sans_font(11)).pack(anchor="w")

    sidebar.pack(fill="both", expand=True)
    return sidebar
