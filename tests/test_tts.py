"""Tests for Urdu TTS. Audio must be real bytes from a real engine —
never faked. If no engine can reach the network, the function must
return a clear failure (None, None, error) instead of fake audio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tts import (result_sentence_english, result_sentence_urdu, speak_english,
                   speak_result_both, speak_urdu)  # noqa: E402


def test_result_sentence_both_labels():
    absence = result_sentence_urdu("Absence")
    presence = result_sentence_urdu("Presence")
    assert len(absence) > 20
    assert len(presence) > 20
    assert absence != presence
    assert "خطرہ نہیں" in absence
    assert "خطرہ" in presence


def test_speak_urdu_returns_real_audio_or_clear_failure():
    audio, engine, error = speak_urdu(result_sentence_urdu("Absence"))
    if audio is not None:
        # Real MP3 bytes from a real engine.
        assert isinstance(audio, bytes)
        assert len(audio) > 1000, f"suspiciously small audio: {len(audio)} bytes"
        assert engine in ("edge-tts", "gtts")
        assert error is None
    else:
        # Honest failure, no fake audio.
        assert engine is None
        assert isinstance(error, str) and len(error) > 0


def test_result_sentence_english_both_labels():
    absence = result_sentence_english("Absence")
    presence = result_sentence_english("Presence")
    assert len(absence) > 20
    assert len(presence) > 20
    assert absence != presence
    assert "no risk" in absence
    assert "risk" in presence


def test_speak_english_returns_real_audio_or_clear_failure():
    audio, engine, error = speak_english(result_sentence_english("Absence"))
    if audio is not None:
        assert isinstance(audio, bytes)
        assert len(audio) > 1000, f"suspiciously small audio: {len(audio)} bytes"
        assert engine in ("edge-tts", "gtts")
        assert error is None
    else:
        assert engine is None
        assert isinstance(error, str) and len(error) > 0


def test_speak_result_both_returns_both_languages():
    both = speak_result_both("Absence")
    assert set(both) == {"ur", "en"}
    for lang in ("ur", "en"):
        audio, engine, error = both[lang]
        if audio is not None:
            assert isinstance(audio, bytes)
            assert len(audio) > 1000, f"[{lang}] suspiciously small audio"
            assert engine in ("edge-tts", "gtts")
        else:
            assert engine is None
            assert isinstance(error, str) and len(error) > 0
