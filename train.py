"""Train the heart-disease model exactly like the user's Colab notebook
and write metrics.json. Run:  .venv/bin/python train.py
"""

import json

from model import DATA_PATH, METRICS_PATH, evaluate, load_data, train_model

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, X_train, X_test, Y_train, Y_test = train_model(df)
    X = df.drop(columns="Heart Disease")
    Y = df["Heart Disease"]
    metrics = evaluate(model, X_train, X_test, Y_train, Y_test, X, Y)
    metrics["dataset"] = "UCI StatLog Heart Disease (270 rows, 14 columns, CC0)"
    metrics["algorithm"] = "LogisticRegression (default settings, max_iter=100)"
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {METRICS_PATH}")
