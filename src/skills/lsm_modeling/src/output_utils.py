"""Independent run folders and consistent success/failure diagnostics."""
from datetime import datetime
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd


def create_run(root: Path, seed: int) -> Path:
    """Create an exclusive run directory; collisions never overwrite prior work."""
    run = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}")
    run.mkdir(parents=True, exist_ok=False)
    for name in ("config", "data", "models", "metrics", "figures", "spatial", "qc", "logs"):
        (run / name).mkdir()
    return run


def setup_logging(run: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(run / "logs/run.log", encoding="utf-8")])


def write_json(path: Path, obj: dict) -> None:
    """Serialize numpy scalars and non-finite metadata to standard JSON."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        return value
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_qc(run: Path, report: dict) -> None:
    """Write a readable audit trail, including partial progress for failed runs."""
    sections = [("1. Samples", "samples"), ("2. Factor rasters", "rasters"),
        ("3. Sample extraction", "extraction"), ("4. Train / Test split", "split"),
        ("5. Random Forest", "model"), ("6. Model performance", "metrics"),
        ("7. Full-area prediction", "prediction"), ("8. Output validation", "output_validation")]
    lines = ["# LSM RF Modeling QC Report", ""]
    for title, key in sections:
        lines.extend([f"## {title}", ""])
        for name, value in report.get(key, {}).items():
            lines.append(f"- {name}: {value}")
        lines.append("")
    lines += ["Categorical rasters are retained using their existing integer class codes.",
              "Continuous features use raw values, without normalization. No one-hot encoding is used.",
              "Scores are RF-predicted landslide susceptibility scores, not calibrated future-event probabilities.",
              "Evaluation uses a stratified random holdout when configured; it is not independent spatial validation.",
              "Feature importance is impurity-based and is reported for reference, without feature selection.", "",
              f"QC Status: {report.get('qc_status', 'FAIL')}"]
    if report.get("error"):
        lines.extend(["", f"Error: {report['error']}"])
    (run / "qc/qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
