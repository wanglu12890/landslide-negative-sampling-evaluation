"""Configuration validation and path resolution."""
from pathlib import Path
import yaml

INPUT_KEYS = ("boundary", "water", "historical_landslides",
              "historical_landslide_buffer", "reference_raster")


def load_config(path: Path) -> tuple[dict, dict[str, Path], Path]:
    """Resolve relative project_root against the YAML directory, inputs against root."""
    with path.open(encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    root = Path(cfg["project_root"]).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    cfg["project_root"] = str(root)
    if set(cfg["input"]) != set(INPUT_KEYS):
        raise ValueError(f"input must contain exactly {INPUT_KEYS}; check {path}")
    paths = {key: (root / cfg["input"][key]).resolve() for key in INPUT_KEYS}
    sampling = cfg["sampling"]
    if sampling["sample_count_mode"] not in ("auto", "manual"):
        raise ValueError("sample_count_mode must be auto or manual")
    if sampling["sample_count_mode"] == "manual":
        count = sampling.get("manual_sample_count")
        if type(count) is not int or count <= 0:
            raise ValueError("manual_sample_count must be a positive integer")
    seed = sampling["random_seed"]
    if type(seed) is not int or seed < 0 or seed > 2**63 - 1:
        raise ValueError("random_seed must be an integer in [0, 2**63-1]")
    if sampling.get("one_negative_per_raster_cell") is not True:
        raise ValueError("one_negative_per_raster_cell must be true; replacement is prohibited")
    if cfg["positive_samples"]["representative_point_method"] != "centroid_with_inside_fallback":
        raise ValueError("Only centroid_with_inside_fallback is supported")
    processing = cfg.setdefault("processing", {})
    rows = processing.setdefault("chunk_rows", 128)
    processing.setdefault("vector_encoding", None)
    if type(rows) is not int or rows <= 0:
        raise ValueError("processing.chunk_rows must be a positive integer")
    output = (root / cfg["output"]["root"]).resolve()
    if any(output == p.parent or output.is_relative_to(p.parent) for p in paths.values()):
        raise ValueError("Output root must not be inside any input data directory")
    return cfg, paths, output
