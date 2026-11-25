"""Vercel serverless handler for the Heart Disease Prediction app.

Pure stdlib + gTTS/edge-tts for speech. NO sklearn, pandas, streamlit
or fastapi imports — the function stays light.

Prediction is sigmoid math on the coefficients in model.json (exported
from the user's notebook model by export_model.py):
    P(Presence) = sigmoid(dot(coefficients, x) + intercept)
    label = 'Presence' if P >= 0.5 else 'Absence'

Routes:
  GET  /                        -> web/index.html
  GET  /api/health              -> {"ok": true, "model_loaded": true}
  POST /api/predict             -> {"label": ..., "probability": ...}
  GET  /api/speak?label&lang    -> MP3 audio of the result sentence

Vercel's Python runtime serves this file as a serverless function when
it defines a `handler` class subclassing BaseHTTPRequestHandler.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ------------------------------------------------------------------ model
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

# Sensible input ranges: (min, max) per feature — same as ../model.py.
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

ALLOWED = {
    "Thallium": {3, 6, 7},
}


def _bundle_path(relative: str) -> Path:
    """Find a file inside the deployment bundle.

    Vercel runs the function with the project root in the bundle, so
    paths are resolved relative to this file first, then the cwd.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / relative,
        here / relative,
        Path.cwd() / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_model() -> dict:
    path = _bundle_path("model.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


MODEL = _load_model()
COEF = MODEL["coefficients"]
INTERCEPT = MODEL["intercept"]
CLASSES = MODEL["classes"]  # ['Absence', 'Presence']
PRESENCE_INDEX = CLASSES.index("Presence")


def _sigmoid(z: float) -> float:
    # Overflow-safe sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def validate_inputs(values) -> list:
    """Validate 13 feature values. Raises ValueError on anything invalid."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("Expected a JSON array of 13 numbers")
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


def predict_probability(values) -> float:
    """P(Presence) by pure-stdlib sigmoid math (no sklearn)."""
    xs = validate_inputs(values)
    z = INTERCEPT + sum(c * x for c, x in zip(COEF, xs))
    return _sigmoid(z)


def predict(values) -> tuple:
    """Return (label, P(Presence)). Label is 'Presence' or 'Absence'."""
    prob = predict_probability(values)
    return ("Presence" if prob >= 0.5 else "Absence"), prob


def _extract_values(payload) -> list:
    """Accept a JSON array of 13 numbers, or an object keyed by feature
    name (in FEATURES order)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        missing = [name for name in FEATURES if name not in payload]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        return [payload[name] for name in FEATURES]
    raise ValueError("Body must be a JSON array of 13 numbers or an "
                     "object keyed by feature name")


# ------------------------------------------------------------------ speech
# Result sentences — copied exactly from ../tts.py (do not import it:
# it lives outside the Vercel root).
def result_sentence(label: str, lang: str) -> str:
    if lang == "ur":
        if label == "Presence":
            return (
                "ماڈل کے مطابق دل کی بیماری کا خطرہ ہو سکتا ہے۔ "
                "گھبرائیں نہیں، یہ صرف ایک تعلیمی اندازہ ہے۔ "
                "جلد از جلد دل کے ڈاکٹر سے مل کر مکمل معائنہ کرائیں۔"
            )
        return (
            "ماڈل کے مطابق دل کی بیماری کا خطرہ نہیں ہے۔ "
            "پھر بھی صحت مند خوراک، ورزش اور باقاعدہ چیک اپ جاری رکھیں۔ "
            "اگر کوئی علامت ہو تو فوراً ڈاکٹر سے رابطہ کریں۔"
        )
    if label == "Presence":
        return (
            "According to the model, there may be a risk of heart disease. "
            "Do not panic — this is only an educational estimate. "
            "Please see a heart doctor soon for a complete check-up."
        )
    return (
        "According to the model, there is no risk of heart disease. "
        "Still, keep eating healthy, exercising, and getting regular check-ups. "
        "If you notice any symptom, contact a doctor immediately."
    )


EDGE_VOICES = {"ur": "ur-PK-AsadNeural", "en": "en-US-AriaNeural"}

# In-memory cache: (label, lang) -> (audio bytes, engine name).
_AUDIO_CACHE: dict = {}


def _edge_tts_bytes(text: str, voice: str) -> bytes:
    import edge_tts

    async def _run() -> str:
        path = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)
        return path

    path = asyncio.run(asyncio.wait_for(_run(), timeout=45))
    try:
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if len(data) < 1000:
        raise RuntimeError("edge-tts returned suspiciously small audio")
    return data


def _gtts_bytes(text: str, lang: str) -> bytes:
    from gtts import gTTS

    path = tempfile.mktemp(suffix=".mp3")
    try:
        gTTS(text=text, lang=lang).save(path)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if len(data) < 1000:
        raise RuntimeError("gTTS returned suspiciously small audio")
    return data


def speak_result(label: str, lang: str) -> tuple:
    """Generate the spoken result for (label, lang).

    Tries edge-tts first, then gTTS. Returns (audio_bytes, engine).
    Raises ValueError for a bad label/lang and RuntimeError if both
    engines fail — audio is never faked.
    """
    if label not in ("Presence", "Absence"):
        raise ValueError("label must be Presence or Absence")
    if lang not in ("ur", "en"):
        raise ValueError("lang must be ur or en")
    key = (label, lang)
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    text = result_sentence(label, lang)
    errors = []
    for name, fn in (
        ("edge-tts", lambda t: _edge_tts_bytes(t, EDGE_VOICES[lang])),
        ("gtts", lambda t: _gtts_bytes(t, lang)),
    ):
        try:
            audio = fn(text)
            _AUDIO_CACHE[key] = (audio, name)
            return audio, name
        except Exception as exc:  # noqa: BLE001 - fall through to next engine
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


# ---------------------------------------------------------------- handler
class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep server logs quiet
        pass

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            html_path = _bundle_path("web/index.html")
            try:
                html = html_path.read_bytes()
            except OSError:
                self._send_json(500, {"error": "index.html not found in bundle"})
                return
            self._send_bytes(200, html, "text/html; charset=utf-8")
        elif path == "/api/health":
            self._send_json(200, {"ok": True, "model_loaded": True})
        elif path == "/api/speak":
            qs = parse_qs(parsed.query)
            label = (qs.get("label") or [None])[0]
            lang = (qs.get("lang") or ["ur"])[0]
            try:
                audio, _engine = speak_result(label, lang)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - speech failure -> 503
                self._send_json(
                    503, {"error": f"speech generation failed: {exc}"}
                )
                return
            self._send_bytes(200, audio, "audio/mpeg")
        else:
            self._send_json(404, {"error": f"not found: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._send_json(404, {"error": f"not found: {parsed.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        try:
            values = _extract_values(payload)
            label, prob = predict(values)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, {"label": label, "probability": prob})
