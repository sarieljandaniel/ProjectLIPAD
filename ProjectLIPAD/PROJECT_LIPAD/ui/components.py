"""Reusable newsprint-styled CustomTkinter widgets."""

from __future__ import annotations

import customtkinter as ctk
from tkinter import ttk

from ui.theme import FONT_MONO, FONT_SANS, FONT_SERIF, ThemeTokens


def serif_font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_SERIF[0], size=size, weight=weight)


def sans_font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_SANS[0], size=size, weight=weight)


def mono_font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_MONO[0], size=size, weight=weight)


def newsprint_button(
    parent,
    tokens: ThemeTokens,
    text: str,
    command=None,
    variant: str = "primary",
    width: int | None = None,
    **kwargs,
) -> ctk.CTkButton:
    common = dict(
        master=parent,
        text=text.upper(),
        command=command,
        height=44,
        corner_radius=0,
        font=sans_font(11, "bold"),
        **kwargs,
    )
    if width is not None:
        common["width"] = width

    if variant == "primary":
        return ctk.CTkButton(
            **common,
            fg_color=tokens.button_primary_bg,
            text_color=tokens.button_primary_fg,
            hover_color=tokens.hover,
            border_width=0,
        )
    return ctk.CTkButton(
        **common,
        fg_color="transparent",
        text_color=tokens.button_secondary_fg,
        hover_color=tokens.rule,
        border_width=1,
        border_color=tokens.border,
    )


def newsprint_card(parent, tokens: ThemeTokens, inverted: bool = False, **kwargs) -> ctk.CTkFrame:
    bg = tokens.inverted_bg if inverted else tokens.bg
    border = tokens.inverted_fg if inverted else tokens.border
    return ctk.CTkFrame(
        parent,
        fg_color=bg,
        corner_radius=0,
        border_width=1,
        border_color=border,
        **kwargs,
    )


def meta_label(parent, tokens: ThemeTokens, text: str, inverted: bool = False, **kwargs) -> ctk.CTkLabel:
    color = tokens.inverted_muted if inverted else tokens.muted
    return ctk.CTkLabel(
        parent,
        text=text.upper(),
        text_color=color,
        font=sans_font(10, "bold"),
        anchor="w",
        **kwargs,
    )


def headline(parent, tokens: ThemeTokens, text: str, size: int = 28, **kwargs) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent,
        text=text,
        text_color=tokens.fg,
        font=serif_font(size, "bold"),
        anchor="w",
        **kwargs,
    )


def body_label(parent, tokens: ThemeTokens, text: str, mono: bool = False, **kwargs) -> ctk.CTkLabel:
    font = mono_font(12) if mono else sans_font(13)
    return ctk.CTkLabel(
        parent,
        text=text,
        text_color=tokens.fg,
        font=font,
        anchor="w",
        justify="left",
        **kwargs,
    )


def badge(parent, tokens: ThemeTokens, text: str, tone: str = "default", inverted: bool = False) -> ctk.CTkLabel:
    if tone == "accent":
        fg, bg, tc = tokens.accent, tokens.bg, tokens.accent
    elif tone == "inverted" or inverted:
        fg, bg, tc = tokens.inverted_fg, tokens.inverted_bg, tokens.inverted_fg
    else:
        fg, bg, tc = tokens.border, tokens.rule, tokens.fg
    frame = ctk.CTkFrame(parent, fg_color=bg, corner_radius=0, border_width=1, border_color=fg)
    lbl = ctk.CTkLabel(frame, text=text.upper(), text_color=tc, font=sans_font(9, "bold"))
    lbl.pack(padx=8, pady=4)
    return frame


def labeled_entry(
    parent,
    tokens: ThemeTokens,
    label: str,
    textvariable=None,
    width: int = 160,
    show: str | None = None,
) -> tuple[ctk.CTkFrame, ctk.CTkEntry]:
    wrap = ctk.CTkFrame(parent, fg_color="transparent")
    meta_label(wrap, tokens, label).pack(anchor="w", pady=(0, 4))
    entry_kwargs = {}
    if show is not None:
        entry_kwargs["show"] = show
    entry = ctk.CTkEntry(
        wrap,
        textvariable=textvariable,
        width=width,
        height=36,
        corner_radius=0,
        border_width=0,
        fg_color="transparent",
        text_color=tokens.fg,
        font=mono_font(12),
        **entry_kwargs,
    )
    entry.pack(fill="x")
    ctk.CTkFrame(wrap, height=1, fg_color=tokens.border, corner_radius=0).pack(fill="x", pady=(2, 0))
    return wrap, entry


def page_header(parent, tokens: ThemeTokens, meta: str, title: str) -> ctk.CTkFrame:
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    meta_label(frame, tokens, meta).pack(anchor="w")
    headline(frame, tokens, title, size=30).pack(anchor="w", pady=(4, 12))
    ctk.CTkFrame(frame, height=4, fg_color=tokens.border, corner_radius=0).pack(fill="x", pady=(0, 16))
    return frame


def stat_tile(parent, tokens: ThemeTokens, label: str, value: str) -> tuple[ctk.CTkFrame, ctk.CTkLabel]:
    card = newsprint_card(parent, tokens)
    inner = ctk.CTkFrame(card, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=16, pady=14)
    meta_label(inner, tokens, label).pack(anchor="w")
    value_lbl = ctk.CTkLabel(inner, text=value, text_color=tokens.fg, font=mono_font(24, "bold"), anchor="w")
    value_lbl.pack(anchor="w", pady=(8, 0))
    return card, value_lbl


def configure_tree_style(tokens: ThemeTokens) -> ttk.Style:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Newsprint.Treeview",
        background=tokens.bg,
        foreground=tokens.fg,
        fieldbackground=tokens.bg,
        bordercolor=tokens.border,
        rowheight=28,
        font=(FONT_MONO[0], 10),
    )
    style.configure(
        "Newsprint.Treeview.Heading",
        background=tokens.rule,
        foreground=tokens.muted,
        relief="flat",
        font=(FONT_SANS[0], 9, "bold"),
    )
    style.map(
        "Newsprint.Treeview",
        background=[("selected", tokens.button_primary_bg)],
        foreground=[("selected", tokens.button_primary_fg)],
    )
    return style
