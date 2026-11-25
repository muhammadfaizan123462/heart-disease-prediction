# Heart Disease Prediction — دل کی بیماری کی پیش گوئی

A Streamlit web app built from the user's own Colab notebook
(`notebooks/Heart_Disease_Prediction.ipynb`). Urdu-first interface with
RTL layout, an English toggle, and — the key feature — the result is
**spoken aloud in Urdu and English** for users who cannot read.

## Run it

```bash
cd ~/workspace/heart-disease
.venv/bin/streamlit run app.py
# open http://localhost:8501
```

Train + write metrics (also runs automatically at app startup):

```bash
.venv/bin/python train.py
.venv/bin/python -m pytest tests/ -q
```

## Model (the user's own approach, reproduced exactly)

- **Algorithm:** `LogisticRegression()` with default settings (`max_iter=100`)
- **Split:** `train_test_split(test_size=0.2, stratify=Y, random_state=2)`
- **Dataset:** UCI StatLog Heart Disease — 270 patients, 14 columns
  (13 features + `Heart Disease` label `Presence`/`Absence`), CC0 public domain.
  Source file: `data/Heart_Disease_Prediction.csv` (recovered from the
  notebook's Colab environment; the original `heart.dat` is also kept).

### Measured metrics (reproduced, not claimed)

| Model | Train | Test | 5-fold CV |
|---|---|---|---|
| LogisticRegression, notebook settings | 0.875 | **0.8333** | 0.8481 |
| Scaled + `max_iter=1000` (alternative) | 0.8889 | 0.7963 | 0.8370 |

The notebook's settings were kept because the scaled variant is **not
better** on test accuracy (0.7963 vs 0.8333).

### Known limitation — convergence warning

With default `max_iter=100`, LBFGS does not fully converge and scikit-learn
emits a `ConvergenceWarning` (the same warning the notebook shows). This is
documented, not hidden. The scaled + `max_iter=1000` variant removes the
warning but scores worse on the test set, so it was not adopted.

## Bug fix — the notebook's wrong message (cell 28)

The notebook compared the prediction against lowercase `'absence'`:

```python
if prediction[0] == 'absence':   # BUG: real label is 'Absence'
    print('the person does not have a heart disease')
else:
    print('the person have heart disease')
```

For the notebook's own example patient
`(57, 1, 2, 124, 261, 0, 0, 141, 0, 0.3, 1, 0, 7)` the model predicts
`Absence`, but the notebook printed *"the person have heart disease"* —
the wrong message. `model.py:result_message()` compares the label
correctly (`'Absence'` / `'Presence'`), and the test suite covers it.

## Urdu + English voice

`tts.py` speaks the result in BOTH Urdu and English (short sentences,
for speed) — Urdu first, then English:

1. **edge-tts** first (`ur-PK-AsadNeural` / `en-US-AriaNeural` neural voices).
2. Falls back to **gTTS** (`lang='ur'` / `lang='en'`) if edge-tts fails.
3. If both fail for a language, the UI shows the written result with a
   note — audio is never faked. Each language is independent: if one
   fails, the other still plays.

Verified locally: gTTS produced real MP3 audio for both languages
(Urdu 113,856 bytes, English 116,160 bytes).
Note: in this sandbox edge-tts cannot reach `speech.platform.bing.com`
(SSL interception), so gTTS is the working engine here; on Streamlit
Cloud / a normal network, edge-tts is tried first.

In the app: a prominent **سنیں (Sunain / Listen)** button plays the
result in both languages (Urdu auto-plays, English has its own player
below it so the two voices never talk over each other), and a checkbox
auto-plays the result right after prediction.

## Files

| File | What it is |
|---|---|
| `app.py` | Streamlit app — Urdu-first RTL UI, English toggle, prediction + voice |
| `model.py` | Training, prediction, validation, the label-bug fix |
| `train.py` | CLI trainer, writes `metrics.json` |
| `tts.py` | Urdu + English TTS (edge-tts → gTTS → honest failure) |
| `data/Heart_Disease_Prediction.csv` | UCI StatLog dataset (270 rows, CC0) |
| `notebooks/Heart_Disease_Prediction.ipynb` | The user's original Colab notebook |
| `tests/` | pytest suite (12 tests) |
| `requirements.txt` | streamlit, pandas, scikit-learn, edge-tts, gtts |

## Limitations (honest)

- Trained on only 270 patients; test accuracy 83.3% means ~1 in 6
  predictions is wrong. Treat the output as an educational estimate only.
- The model uses unscaled features with default settings, exactly as the
  notebook did (see convergence note above).
- Voice (Urdu + English) depends on a network service (edge-tts or
  gTTS); offline it falls back to text only.
- Pashto voice is not included — Urdu and English only, as requested.

## Disclaimer

⚠️ **This app is for education only. It is NOT a medical diagnosis.**
Whatever the result, always consult a doctor about your health. In case
of chest pain, breathing difficulty or fainting, seek emergency help
immediately.

⚠️ **یہ ایپ صرف تعلیم کے لیے ہے۔ یہ طبی تشخیص نہیں ہے۔**
