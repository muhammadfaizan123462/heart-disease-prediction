"""Export the trained LogisticRegression to model.json for Vercel.

Trains exactly like the user's notebook (see ../model.py):
- LogisticRegression() with default settings (max_iter=100)
- train_test_split(test_size=0.2, stratify=Y, random_state=2)

Writes pure numbers — coefficients, intercept, feature order, classes,
metrics — so the Vercel function can predict with plain stdlib sigmoid
math. No sklearn is needed at runtime.

Run from this directory with the project venv:
    ../.venv/bin/python export_model.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "Heart_Disease_Prediction.csv"
OUT_PATH = ROOT / "model.json"

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


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    Y = df[TARGET]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    with warnings.catch_warnings():
        # Default max_iter=100 does not fully converge (notebook warning);
        # that is a known, documented limitation — see README.
        warnings.simplefilter("ignore")
        model = LogisticRegression()
        model.fit(X_train, Y_train)

    payload = {
        "features": FEATURES,
        "classes": [str(c) for c in model.classes_],
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "metrics": {
            "train_accuracy": round(
                accuracy_score(Y_train, model.predict(X_train)), 4
            ),
            "test_accuracy": round(
                accuracy_score(Y_test, model.predict(X_test)), 4
            ),
            "cv5_accuracy": round(
                float(cross_val_score(model, X, Y, cv=5).mean()), 4
            ),
            "n_rows": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(payload["metrics"], indent=2))
    return payload


if __name__ == "__main__":
    main()
