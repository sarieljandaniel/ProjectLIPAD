"""Newsprint design tokens for the LiPAD desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThemeMode(str, Enum):
    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True)
class ThemeTokens:
    bg: str
    fg: str
    border: str
    rule: str
    muted: str
    accent: str
    inverted_bg: str
    inverted_fg: str
    inverted_muted: str
    button_primary_bg: str
    button_primary_fg: str
    button_secondary_fg: str
    console_bg: str
    console_fg: str
    hover: str


LIGHT = ThemeTokens(
    bg="#F9F9F7",
    fg="#111111",
    border="#111111",
    rule="#E5E5E0",
    muted="#737373",
    accent="#CC0000",
    inverted_bg="#111111",
    inverted_fg="#F9F9F7",
    inverted_muted="#E5E5E0",
    button_primary_bg="#111111",
    button_primary_fg="#F9F9F7",
    button_secondary_fg="#111111",
    console_bg="#111111",
    console_fg="#F9F9F7",
    hover="#737373",
)

DARK = ThemeTokens(
    bg="#111111",
    fg="#F9F9F7",
    border="#F9F9F7",
    rule="#2A2A2A",
    muted="#A3A3A3",
    accent="#CC0000",
    inverted_bg="#1C1C1C",
    inverted_fg="#F9F9F7",
    inverted_muted="#E5E5E0",
    button_primary_bg="#F9F9F7",
    button_primary_fg="#111111",
    button_secondary_fg="#F9F9F7",
    console_bg="#0A0A0A",
    console_fg="#F9F9F7",
    hover="#737373",
)


def get_tokens(mode: ThemeMode) -> ThemeTokens:
    return LIGHT if mode == ThemeMode.LIGHT else DARK


# Typography
FONT_SERIF = ("Times New Roman", "Georgia")
FONT_SANS = ("Segoe UI", "Helvetica Neue", "Arial")
FONT_MONO = ("Consolas", "Courier New")
