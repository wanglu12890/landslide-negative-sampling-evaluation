"""Left-closed fixed-break classes and unchanged integer category codes."""
import numpy as np


def classify(values, factor: str, cfg: dict) -> np.ndarray:
    """Classify prefiltered valid values; values on a break enter the upper class."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{factor}: exclude non-finite values before classification")
    if factor in cfg["factors"]["continuous"]:
        return np.searchsorted(np.asarray(cfg["reclassification"][factor]["breaks"], dtype="float64"),
                               values.astype("float64"), side="right") + 1
    if not np.equal(values, np.floor(values)).all() or (np.abs(values) >= 2**63).any():
        raise ValueError(f"{factor}: categorical codes must be int64-compatible integers")
    return values.astype("int64")


def class_description(factor: str, code: int, cfg: dict) -> dict:
    if factor in cfg["factors"]["categorical"]:
        return dict(class_label=str(code), lower_bound=None, upper_bound=None, category_code=code)
    breaks = cfg["reclassification"][factor]["breaks"]
    lower = breaks[code-2] if code > 1 else -np.inf
    upper = breaks[code-1] if code <= len(breaks) else np.inf
    return dict(class_label=f"[{lower}, {upper})", lower_bound=lower, upper_bound=upper, category_code=None)
