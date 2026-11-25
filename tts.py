"""Urdu + English text-to-speech for the Heart Disease Prediction app.

The result is spoken in BOTH languages (user request): Urdu first, then
English — so users who cannot read in either language can still hear it.

Tries, in order, per language:
1. edge-tts (Microsoft neural voices: ur-PK-AsadNeural, en-US-AriaNeural)
   — best quality, needs network access to speech.platform.bing.com.
2. gTTS (Google TTS, lang='ur' / 'en') — works where edge-tts is blocked.

If both fail for a language, returns (None, None, error) for that language
so the UI can fall back to showing the text only. Audio is NEVER faked:
bytes are returned only when a real TTS engine produced them.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

EDGE_VOICE_UR = "ur-PK-AsadNeural"
EDGE_VOICE_EN = "en-US-AriaNeural"


def result_sentence_urdu(label: str) -> str:
    """Short Urdu sentence for the spoken result (kept short for speed)."""
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


def result_sentence_english(label: str) -> str:
    """Short English sentence for the spoken result (kept short for speed)."""
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


def speak(text: str, lang: str = "ur") -> tuple[bytes | None, str | None, str | None]:
    """Generate audio for *text* in *lang* ('ur' or 'en').

    Returns (audio_bytes, engine_name, error). engine_name is 'edge-tts'
    or 'gtts' on success; on failure audio_bytes is None and error
    explains why, so callers can show text only.
    """
    voice = EDGE_VOICE_EN if lang == "en" else EDGE_VOICE_UR
    errors = []
    for name, fn in (("edge-tts", lambda t: _edge_tts_bytes(t, voice)),
                     ("gtts", lambda t: _gtts_bytes(t, lang))):
        try:
            return fn(text), name, None
        except Exception as exc:  # noqa: BLE001 - fall through to next engine
            errors.append(f"{name}: {exc}")
    return None, None, "; ".join(errors)


def speak_urdu(text: str) -> tuple[bytes | None, str | None, str | None]:
    """Generate Urdu audio for *text* (kept for compatibility)."""
    return speak(text, "ur")


def speak_english(text: str) -> tuple[bytes | None, str | None, str | None]:
    """Generate English audio for *text*."""
    return speak(text, "en")


def speak_result_both(label: str) -> dict:
    """Generate the spoken result in BOTH Urdu and English.

    Returns {'ur': (audio, engine, error), 'en': (audio, engine, error)}.
    Each language is independent: if one fails, the other still plays.
    """
    return {
        "ur": speak(result_sentence_urdu(label), "ur"),
        "en": speak(result_sentence_english(label), "en"),
    }


if __name__ == "__main__":
    for lang, sentence in (("ur", result_sentence_urdu("Absence")),
                           ("en", result_sentence_english("Absence"))):
        audio, engine, error = speak(sentence, lang)
        if audio:
            print(f"OK [{lang}] via {engine}: {len(audio)} bytes")
        else:
            print(f"FAILED [{lang}]: {error}")
