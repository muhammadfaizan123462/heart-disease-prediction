"""Tests for the Vercel build (api/index.py + web/index.html).

Covers:
- model.json shape and metrics
- pure-stdlib prediction EQUALS sklearn on all 270 rows + example input
- input validation (unit level)
- live HTTP endpoints via a local server (health, root HTML, predict
  array/object/validation, speak MP3 in Urdu + English)
"""

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent  # vercel/
sys.path.insert(0, str(ROOT.parent))  # heart-disease/ (for model.py)

from model import EXAMPLE_INPUT, load_data, train_model  # noqa: E402


def _load_handler():
    spec = importlib.util.spec_from_file_location(
        "vercel_api_index", ROOT / "api" / "index.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


api = _load_handler()


# ---------------------------------------------------------------- helpers
def _http_get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=60) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _http_post(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _is_mp3(data: bytes) -> bool:
    # gTTS output starts with an ID3 tag; raw MPEG frames start 0xFF.
    return data[:3] == b"ID3" or (len(data) > 0 and data[0] == 0xFF)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api.handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ---------------------------------------------------------------- model.json
def test_model_json_shape():
    payload = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    assert len(payload["features"]) == 13
    assert payload["classes"] == ["Absence", "Presence"]
    assert len(payload["coefficients"]) == 13
    assert isinstance(payload["intercept"], float)
    assert payload["metrics"]["train_accuracy"] == 0.875
    assert payload["metrics"]["test_accuracy"] == 0.8333
    assert payload["metrics"]["n_rows"] == 270


# ---------------------------------------------------------------- equivalence
def test_pure_python_matches_sklearn_all_rows():
    """The stdlib sigmoid prediction must equal sklearn on every row."""
    df = load_data()
    model, _xt, _xte, _yt, _yte = train_model(df)
    presence_idx = list(model.classes_).index("Presence")
    X = df.drop(columns="Heart Disease")
    max_prob_diff = 0.0
    for _, row in X.iterrows():
        values = [float(v) for v in row.tolist()]
        label, prob = api.predict(values)
        sk_label = str(model.predict([values])[0])
        sk_prob = float(model.predict_proba([values])[0][presence_idx])
        assert label == sk_label, f"label mismatch on row {values}"
        max_prob_diff = max(max_prob_diff, abs(prob - sk_prob))
    assert max_prob_diff < 1e-6, f"max prob diff {max_prob_diff}"


def test_example_input():
    label, prob = api.predict(list(EXAMPLE_INPUT))
    assert label == "Absence"
    assert 0.23 < prob < 0.25  # measured sklearn value is ~0.238


def test_result_sentences_both_labels_and_languages():
    ur_abs = api.result_sentence("Absence", "ur")
    ur_pre = api.result_sentence("Presence", "ur")
    en_abs = api.result_sentence("Absence", "en")
    en_pre = api.result_sentence("Presence", "en")
    assert "خطرہ نہیں" in ur_abs and "خطرہ" in ur_pre
    assert "no risk" in en_abs and "risk" in en_pre
    assert len({ur_abs, ur_pre, en_abs, en_pre}) == 4


# ---------------------------------------------------------------- validation
def test_validation_wrong_count():
    with pytest.raises(ValueError, match="13 features"):
        api.validate_inputs([1] * 12)


def test_validation_out_of_range():
    bad = list(EXAMPLE_INPUT)
    bad[3] = 999  # BP out of [50, 250]
    with pytest.raises(ValueError, match="BP"):
        api.validate_inputs(bad)


def test_validation_bad_thallium():
    bad = list(EXAMPLE_INPUT)
    bad[12] = 5  # Thallium must be 3, 6 or 7
    with pytest.raises(ValueError, match="Thallium"):
        api.validate_inputs(bad)


def test_validation_non_numeric():
    bad = list(EXAMPLE_INPUT)
    bad[0] = "abc"
    with pytest.raises(ValueError, match="Age"):
        api.validate_inputs(bad)


# ---------------------------------------------------------------- endpoints
def test_health_endpoint(server):
    status, ctype, body = _http_get(server, "/api/health")
    assert status == 200
    assert "application/json" in ctype
    data = json.loads(body)
    assert data == {"ok": True, "model_loaded": True}


def test_root_serves_urdu_html(server):
    status, ctype, body = _http_get(server, "/")
    assert status == 200
    assert "text/html" in ctype
    text = body.decode("utf-8")
    assert "دل کی بیماری کی پیش گوئی" in text
    assert "پیش گوئی کریں" in text
    assert "Heart Disease Prediction" in text


def test_predict_endpoint_array(server):
    status, body = _http_post(server, "/api/predict", list(EXAMPLE_INPUT))
    assert status == 200
    data = json.loads(body)
    assert data["label"] == "Absence"
    assert 0.23 < data["probability"] < 0.25


def test_predict_endpoint_object(server):
    payload = dict(zip(api.FEATURES, EXAMPLE_INPUT))
    status, body = _http_post(server, "/api/predict", payload)
    assert status == 200
    data = json.loads(body)
    assert data["label"] == "Absence"
    assert 0.23 < data["probability"] < 0.25


def test_predict_endpoint_validation_error(server):
    status, body = _http_post(server, "/api/predict", [1] * 12)
    assert status == 400
    data = json.loads(body)
    assert "error" in data and "13 features" in data["error"]


def test_predict_endpoint_bad_json(server):
    req = urllib.request.Request(
        server + "/api/predict",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=60)
        raise AssertionError("expected 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "invalid JSON" in e.read().decode("utf-8")


def test_speak_urdu_mp3(server):
    status, ctype, body = _http_get(
        server, "/api/speak?label=Absence&lang=ur"
    )
    assert status == 200, f"speak ur failed: {body[:200]!r}"
    assert "audio/mpeg" in ctype
    assert len(body) > 1000, f"suspiciously small audio: {len(body)} bytes"
    assert _is_mp3(body)


def test_speak_english_mp3(server):
    status, ctype, body = _http_get(
        server, "/api/speak?label=Absence&lang=en"
    )
    assert status == 200, f"speak en failed: {body[:200]!r}"
    assert "audio/mpeg" in ctype
    assert len(body) > 1000, f"suspiciously small audio: {len(body)} bytes"
    assert _is_mp3(body)


def test_speak_invalid_label(server):
    status, _ctype, body = _http_get(server, "/api/speak?label=Maybe&lang=ur")
    assert status == 400
    assert "error" in json.loads(body)
