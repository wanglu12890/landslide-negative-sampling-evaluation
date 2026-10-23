"""Random Forest only; all hyperparameters come from configuration."""
from datetime import datetime
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .output_utils import write_json
from .utils import class_counts


def train_rf(train, test, cfg: dict, run, report: dict):
    """Fit the training subset once; never tune using the holdout set."""
    features = cfg["factors"]["feature_order"]
    model = RandomForestClassifier(random_state=cfg["model"]["random_state"], **cfg["model"]["params"])
    logging.info("RF parameters: %s", model.get_params())
    logging.info("Training samples=%d, features=%d, classes=%s", len(train), len(features), class_counts(train))
    model.fit(train[features], train.label)
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Unexpected RF class order: {model.classes_}")
    joblib.dump(model, run / "models/rf_model.joblib")
    metadata = dict(model_type="RandomForest", random_state=cfg["model"]["random_state"], feature_names=features,
        continuous_features=cfg["factors"]["continuous"], categorical_features=cfg["factors"]["categorical"],
        training_sample_count=len(train), test_sample_count=len(test), class_distribution=class_counts(train),
        parameters=model.get_params(), training_timestamp=datetime.now().astimezone().isoformat(),
        preprocessing="Raw continuous values and existing integer category codes; no scaling or encoding",
        input_paths=report["input_paths"], versions=report["versions"])
    write_json(run / "models/model_metadata.json", metadata)
    report["model"] = metadata
    return model


def susceptibility_scores(model, features) -> np.ndarray:
    """Return the class-1 score, respecting the serialized feature order."""
    if list(features.columns) != list(model.feature_names_in_):
        raise ValueError("Prediction feature order differs from training feature order")
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError("Model must contain exactly one landslide class (1)")
    scores = model.predict_proba(features)[:, positive[0]]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("RF produced non-finite or out-of-range susceptibility scores")
    return scores
