from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MORPH_CSV = DATA_DIR / "MorphologicalResults.csv"
UI_CSV = DATA_DIR / "results.csv"
ENGINE_SCRIPT = REPO_ROOT / "Project_LIPAD_AI" / "lipad_runtime_engine.py"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "best.pt"
ANNOTATED_DIR = DATA_DIR / "annotated"
TELEMETRY_PORT = 50007
