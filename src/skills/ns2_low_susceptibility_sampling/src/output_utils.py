"""Independent runs and transparent partial/success reports."""
from datetime import datetime
from pathlib import Path
from skills.lsm_modeling.src.output_utils import setup_logging, write_json, write_csv


def create_run(root: Path, seed: int) -> Path:
    run = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}")
    run.mkdir(parents=True, exist_ok=False)
    for folder in ("config", "samples", "spatial/iv_factors", "qc", "logs"):
        (run / folder).mkdir(parents=True)
    return run


def write_report(run: Path, report: dict):
    """Report definitions and diagnostics without altering classifications."""
    titles = [("Configuration", "configuration"), ("Positive samples", "positive"), ("Raster alignment", "rasters"),
        ("Reclassification and Information Value", "iv_factors"), ("IV Total", "iv_total"),
        ("Low-susceptibility candidate pool", "low_susceptibility"), ("NS2 samples", "sampling"),
        ("Spatial exclusion checks", "conflicts"), ("Geometry", "geometry"), ("Output validation", "output_validation")]
    lines = ["# NS2 Low-Susceptibility Sampling QC Report", ""]
    for i, (title, key) in enumerate(titles, 1):
        lines += [f"## {i}. {title}", ""] + [f"- {k}: {v}" for k, v in report.get(key, {}).items()] + [""]
    lines += ["IV background = factor_valid AND boundary covers pixel center AND NOT water.",
        "Base candidate = IV background AND NOT existing buffer. Buffer never changes IV area counts.",
        "IV = ln(((L_j+alpha)/(L+alpha*K))/(A_j/A)); K includes only classes with A_j>0.",
        "Empty classes have no IV; diagnostics never change breaks or thresholds.",
        "Quantile uses float32 IV_total inside base candidates, linear interpolation; <= includes all ties.",
        "IV_total is a negative-sample screening surface, not final LSM prediction.", "",
        f"QC Status: {report.get('qc_status', 'FAIL')}"]
    if report.get("error"):
        lines += ["", f"Error: {report['error']}"]
    (run / "qc/qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
