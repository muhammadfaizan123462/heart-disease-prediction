"""Model training, prediction and validation for the Heart Disease Prediction app.

Reproduces the user's own Colab notebook exactly:
- LogisticRegression() with default settings (max_iter=100)
- train_test_split(test_size=0.2, stratify=Y, random_state=2)
- Labels are the strings 'Presence' / 'Absence' (capitalised).

Measured on the UCI StatLog heart dataset (270 rows):
- train accuracy 0.875, test accuracy 0.8333, 5-fold CV 0.8481

BUG FIX (notebook cell 28): the notebook compared
``prediction[0] == 'absence'`` (lowercase) while the real label is
'Absence', so the example input printed the WRONG message
("the person have heart disease" for a patient the model called
'Absence'). :func:`result_message` compares the label correctly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

DATA_PATH = Path(__file__).parent / "data" / "Heart_Disease_Prediction.csv"
METRICS_PATH = Path(__file__).parent / "metrics.json"

FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
TARGET = "Heart Disease"

# Sensible input ranges: (min, max) per feature, in FEATURES order.
RANGES = {
    "Age": (1, 120),
    "Sex": (0, 1),
    "Chest pain type": (1, 4),
    "BP": (50, 250),
    "Cholesterol": (50, 700),
    "FBS over 120": (0, 1),
    "EKG results": (0, 2),
    "Max HR": (40, 250),
    "Exercise angina": (0, 1),
    "ST depression": (0.0, 7.0),
    "Slope of ST": (1, 3),
    "Number of vessels fluro": (0, 3),
    "Thallium": (3, 7),
}

# Discrete allowed values (in addition to the ranges above).
ALLOWED = {
    "Thallium": {3, 6, 7},
}
EXAMPLE_INPUT = (57, 1, 2, 124, 261, 0, 0, 141, 0, 0.3, 1, 0, 7)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame | None = None):
    """Train exactly like the notebook. Returns (model, X_train, X_test,
    Y_train, Y_test)."""
    df = df if df is not None else load_data()
    X = df.drop(columns=TARGET)
    Y = df[TARGET]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    with warnings.catch_warnings():
        # The notebook's default max_iter=100 does not fully converge;
        # that warning is a known, documented limitation (see README).
        warnings.simplefilter("ignore")
        model = LogisticRegression()
        model.fit(X_train, Y_train)
    return model, X_train, X_test, Y_train, Y_test


def evaluate(model, X_train, X_test, Y_train, Y_test, X, Y) -> dict:
    """Train/test accuracy + 5-fold CV (all measured, never invented)."""
    return {
        "train_accuracy": round(
            accuracy_score(Y_train, model.predict(X_train)), 4
        ),
        "test_accuracy": round(
            accuracy_score(Y_test, model.predict(X_test)), 4
        ),
        "cv5_accuracy": round(
            float(cross_val_score(model, X, Y, cv=5).mean()), 4
        ),
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def validate_inputs(values) -> list[float]:
    """Validate 13 feature values. Raises ValueError on anything invalid."""
    if len(values) != 13:
        raise ValueError(f"Expected 13 features, got {len(values)}")
    out = []
    for name, v in zip(FEATURES, values):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: not a number ({v!r})")
        lo, hi = RANGES[name]
        if not (lo <= v <= hi):
            raise ValueError(f"{name}: {v} out of range [{lo}, {hi}]")
        if name in ALLOWED and v not in ALLOWED[name]:
            raise ValueError(
                f"{name}: {v} not allowed (allowed: {sorted(ALLOWED[name])})"
            )
        out.append(v)
    return out


def predict(model, values) -> tuple[str, float]:
    """Return (label, probability of Presence).

    Label is exactly 'Presence' or 'Absence' — the comparison bug in
    notebook cell 28 (lowercase 'absence') is fixed here.
    """
    validated = validate_inputs(values)
    arr = np.asarray(validated, dtype=float).reshape(1, -1)
    label = str(model.predict(arr)[0])
    # predict_proba columns follow model.classes_ order: ['Absence','Presence']
    proba = model.predict_proba(arr)[0]
    idx = list(model.classes_).index("Presence")
    return label, float(proba[idx])


def result_message(label: str) -> str:
    """Correct human message for a label (the notebook got this wrong)."""
    if label == "Absence":
        return "the person does not have a heart disease"
    if label == "Presence":
        return "the person has heart disease"
    raise ValueError(f"Unknown label: {label!r}")


if __name__ == "__main__":
    df = load_data()
    model, X_train, X_test, Y_train, Y_test = train_model(df)
    metrics = evaluate(
        model, X_train, X_test, Y_train, Y_test,
        df.drop(columns=TARGET), df[TARGET],
    )
    print(json.dumps(metrics, indent=2))
    label, prob = predict(model, EXAMPLE_INPUT)
    print(f"Example input -> {label} (P(Presence)={prob:.3f}): "
          f"{result_message(label)}")
