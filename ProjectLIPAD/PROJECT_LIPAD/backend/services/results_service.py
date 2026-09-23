from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import MORPH_CSV, UI_CSV


def _resolve_csv() -> Path | None:
    if MORPH_CSV.exists():
        return MORPH_CSV
    if UI_CSV.exists():
        return UI_CSV
    return None


def load_results() -> dict:
    path = _resolve_csv()
    if path is None:
        return {"rows": [], "columns": [], "path": None, "summary": {}}

    df = pd.read_csv(path)
    if df.empty:
        return {"rows": [], "columns": list(df.columns), "path": str(path), "summary": {}}

    rows = df.fillna("").astype(str).to_dict(orient="records")
    summary = {
        "total": len(df),
        "cracks": int((df["Type"].astype(str).str.lower() == "crack").sum()) if "Type" in df.columns else 0,
        "corrosion": int((df["Type"].astype(str).str.lower() == "corrosion").sum()) if "Type" in df.columns else 0,
    }
    if "Severity" in df.columns and len(df) > 0:
        latest = df.iloc[-1]
        summary["latest_type"] = str(latest.get("Type", ""))
        summary["latest_severity"] = str(latest.get("Severity", ""))

    return {
        "rows": rows,
        "columns": list(df.columns),
        "path": str(path),
        "summary": summary,
    }


def repair_guidance() -> dict:
    data = load_results()
    rows = data["rows"]
    if not rows:
        return {"headline": "Awaiting analysis data", "body": "Run an inspection in Inspection Manager first.", "severity": "none"}

    df = pd.DataFrame(rows)
    has_crack = (df["Type"].str.lower() == "crack").any() if "Type" in df.columns else False
    has_corrosion = (df["Type"].str.lower() == "corrosion").any() if "Type" in df.columns else False

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
        "severity": severity.lower(),
        "has_crack": bool(has_crack),
        "has_corrosion": bool(has_corrosion),
    }


def clear_results() -> dict:
    cleared = 0
    for path in (MORPH_CSV, UI_CSV):
        try:
            if path.exists():
                path.unlink()
                cleared += 1
        except OSError:
            pass
    return {"cleared": cleared}
