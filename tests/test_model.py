"""Tests for model training, prediction and the notebook bug fix."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (  # noqa: E402
    EXAMPLE_INPUT,
    evaluate,
    load_data,
    predict,
    result_message,
    train_model,
    validate_inputs,
)


@pytest.fixture(scope="module")
def trained():
    df = load_data()
    model, X_train, X_test, Y_train, Y_test = train_model(df)
    return model, df, X_train, X_test, Y_train, Y_test


def test_training_reproduces_notebook_metrics(trained):
    model, df, X_train, X_test, Y_train, Y_test = trained
    metrics = evaluate(model, X_train, X_test, Y_train, Y_test,
                       df.drop(columns="Heart Disease"), df["Heart Disease"])
    assert metrics["train_accuracy"] == pytest.approx(0.875, abs=0.001)
    assert metrics["test_accuracy"] == pytest.approx(0.8333, abs=0.001)
    assert metrics["n_rows"] == 270


def test_example_input_predicts_absence_with_correct_message(trained):
    model = trained[0]
    label, prob = predict(model, EXAMPLE_INPUT)
    assert label == "Absence"
    # The notebook's bug printed "the person have heart disease" here.
    assert result_message(label) == "the person does not have a heart disease"
    assert 0.0 <= prob <= 1.0


def test_known_presence_row(trained):
    model = trained[0]
    # First row of the CSV, labelled Presence.
    label, prob = predict(model, (70, 1, 4, 130, 322, 0, 2, 109, 0, 2.4, 2, 3, 3))
    assert label == "Presence"
    assert result_message(label) == "the person has heart disease"
    assert prob > 0.5


def test_result_message_both_labels():
    assert result_message("Absence") == "the person does not have a heart disease"
    assert result_message("Presence") == "the person has heart disease"
    with pytest.raises(ValueError):
        result_message("absence")  # lowercase is NOT a valid label


def test_out_of_range_rejected(trained):
    model = trained[0]
    bad = list(EXAMPLE_INPUT)
    bad[0] = 200  # Age out of range
    with pytest.raises(ValueError, match="Age"):
        predict(model, bad)
    bad = list(EXAMPLE_INPUT)
    bad[2] = 5  # Chest pain type only 1-4
    with pytest.raises(ValueError, match="Chest pain type"):
        predict(model, bad)
    bad = list(EXAMPLE_INPUT)
    bad[12] = 5  # Thallium only 3, 6, 7 in practice; range guard 3-7
    with pytest.raises(ValueError, match="Thallium"):
        predict(model, bad)


def test_wrong_feature_count_rejected(trained):
    model = trained[0]
    with pytest.raises(ValueError, match="13 features"):
        predict(model, (57, 1, 2))
    with pytest.raises(ValueError):
        validate_inputs(["not", "numbers"] * 7)


def test_probability_is_sane(trained):
    model = trained[0]
    _, prob = predict(model, EXAMPLE_INPUT)
    assert 0.0 <= prob <= 1.0
