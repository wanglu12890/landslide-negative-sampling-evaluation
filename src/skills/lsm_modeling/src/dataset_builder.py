"""Variable audits, explicit invalid-sample handling and holdout construction."""
from collections import Counter
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from .utils import windows, valid_values, class_counts
from .output_utils import write_csv


def build_dataset(extracted, bad, cfg: dict, destination, report: dict):
    """Default to stopping on invalid samples; explicit opt-out records deletions."""
    if bad.any() and cfg["data_quality"]["stop_on_invalid_sample"]:
        raise ValueError(f"{int(bad.sum())} samples have invalid factor values; inspect data/invalid_samples.csv. No model trained")
    data = extracted.loc[~bad].copy().reset_index(drop=True)
    counts = class_counts(data)
    report["extraction"].update(retained_class_distribution=counts, dropped_sample_count=int(bad.sum()))
    if not counts["positive"] or not counts["negative"]:
        raise ValueError("Valid modeling dataset must contain both classes")
    if counts["positive"] != counts["negative"] and not cfg["allow_imbalanced_samples"]:
        raise ValueError("Valid dataset is imbalanced after explicit exclusions; inspect invalid_samples.csv and allow_imbalanced_samples")
    columns = ["sample_id", "label", "x", "y", "row", "col", *cfg["factors"]["feature_order"], "sample_type"]
    data = data[columns]
    write_csv(data, destination)
    return data


def summarize_factors(data, paths: dict, cfg: dict, run, report: dict) -> None:
    """Audit integer codes across each full raster and raw sample statistics."""
    categorical_rows, summaries = [], {}
    for factor in cfg["factors"]["categorical"]:
        sample_values = data[factor].to_numpy()
        if not np.equal(sample_values, np.floor(sample_values)).all():
            raise ValueError(f"{factor}: categorical samples have noninteger values; check source raster, do not recode silently")
        full = Counter()
        with rasterio.open(paths[factor]) as raster:
            for window in windows(raster.width, raster.height, cfg["prediction"]["batch_size"]):
                array = raster.read(1, window=window, masked=True)
                values = array.data[valid_values(array)]
                if not np.equal(values, np.floor(values)).all():
                    raise ValueError(f"{factor}: {paths[factor]} has noninteger categorical raster codes")
                unique, counts = np.unique(values, return_counts=True)
                full.update({int(value): int(count) for value, count in zip(unique, counts)})
        sample_counts = data[factor].value_counts()
        positive_counts = data.loc[data.label == 1, factor].value_counts()
        negative_counts = data.loc[data.label == 0, factor].value_counts()
        for code in sorted(full):
            categorical_rows.append(dict(factor=factor, category=code,
                sample_count=int(sample_counts.get(code, 0)), positive_count=int(positive_counts.get(code, 0)),
                negative_count=int(negative_counts.get(code, 0)), full_raster_cell_count=full[code],
                unseen_in_samples=code not in sample_counts.index))
        summaries[factor] = dict(sample_category_count=len(sample_counts), full_raster_category_count=len(full),
                                 unseen_in_samples=sorted(set(full)-set(sample_counts.index)))
    write_csv(pd.DataFrame(categorical_rows, columns=["factor", "category", "sample_count", "positive_count",
        "negative_count", "full_raster_cell_count", "unseen_in_samples"]), run / "qc/categorical_summary.csv")
    continuous_rows = []
    for group, frame in (("All", data), ("Positive", data[data.label == 1]), ("Negative", data[data.label == 0])):
        for factor in cfg["factors"]["continuous"]:
            values = frame[factor]
            continuous_rows.append(dict(factor=factor, group=group, count=len(values), min=values.min(),
                max=values.max(), mean=values.mean(), std=values.std(ddof=1), median=values.median()))
    write_csv(pd.DataFrame(continuous_rows, columns=["factor", "group", "count", "min", "max", "mean", "std", "median"]),
              run / "qc/continuous_factor_summary.csv")
    report["extraction"]["categorical_summary"] = summaries


def split_dataset(data, cfg: dict, run, report: dict):
    """Split entire records so saved IDs and raw feature values remain traceable."""
    split = cfg["split"]
    train, test = train_test_split(data, test_size=split["test_size"], random_state=split["random_seed"],
                                  stratify=data.label if split["stratify"] else None)
    if train.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError("Both train and test must contain both classes; inspect split size and sample counts")
    write_csv(train, run / "data/train_samples.csv")
    write_csv(test, run / "data/test_samples.csv")
    train_cells = set(zip(train.row, train.col))
    test_cells = set(zip(test.row, test.col))
    report["split"] = dict(train_count=len(train), test_count=len(test), train_class_distribution=class_counts(train),
        test_class_distribution=class_counts(test), **split, shared_train_test_cell_count=len(train_cells & test_cells))
    report["split"]["test_only_categories"] = {factor: sorted(set(test[factor])-set(train[factor]))
                                               for factor in cfg["factors"]["categorical"]}
    return train, test
