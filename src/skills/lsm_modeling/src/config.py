"""Configuration, explicit feature order, and portable paths."""
from pathlib import Path
import re
import yaml


def load_config(path: Path) -> tuple[dict, dict, dict, Path]:
    """Validate modeling choices without changing the research design."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    root = Path(cfg["project_root"]).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    cfg["project_root"] = str(root)
    name = cfg["experiment_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("experiment_name must contain only letters, numbers, underscores or hyphens")
    factors = cfg["factors"]
    order = factors.get("feature_order")
    if not isinstance(order, list) or not order or any(not isinstance(name, str) or not name for name in order):
        raise ValueError("factors.feature_order is required and must be a nonempty list of feature names; no fallback to files or variable types")
    continuous, categorical = factors["continuous"], factors["categorical"]
    if (not order or len(set(order)) != len(order) or set(order) != set(factors["files"])
            or len(continuous + categorical) != len(set(continuous + categorical))
            or set(continuous + categorical) != set(order)):
        raise ValueError("Factor files, feature_order and disjoint variable types must describe the same unique features")
    if cfg["reference_raster"] not in order:
        raise ValueError("reference_raster must name a configured factor")
    if set(cfg["samples"]) != {"positive_samples", "negative_samples"}:
        raise ValueError("samples must contain only positive_samples and negative_samples")
    if cfg["model"]["type"] != "RandomForest":
        raise ValueError("Only RandomForest is supported")
    if "random_state" in cfg["model"]["params"]:
        raise ValueError("Set random_state in model.random_state, not model.params")
    for seed in (cfg["split"]["random_seed"], cfg["model"]["random_state"]):
        if type(seed) is not int or not 0 <= seed <= 2**32-1:
            raise ValueError("Seeds must be integers in [0, 2**32-1]")
    if not 0 < cfg["split"]["test_size"] < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    batch = cfg["prediction"]["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("prediction.batch_size must be a positive integer")
    cfg.setdefault("allow_imbalanced_samples", False)
    cfg.setdefault("data_quality", {}).setdefault("stop_on_invalid_sample", True)
    for value in (cfg["allow_imbalanced_samples"], cfg["data_quality"]["stop_on_invalid_sample"], cfg["split"]["stratify"]):
        if type(value) is not bool:
            raise ValueError("allow_imbalanced_samples, stop_on_invalid_sample and stratify must be booleans")
    paths = {key: (root / factors["directory"] / factors["files"][key]).resolve() for key in order}
    samples = {key: (root / value).resolve() for key, value in cfg["samples"].items()}
    output = (root / cfg["output"]["root"]).resolve()
    if any(output == p.parent or output.is_relative_to(p.parent) for p in [*paths.values(), *samples.values()]):
        raise ValueError("Output root must be outside source-data directories")
    return cfg, paths, samples, output
