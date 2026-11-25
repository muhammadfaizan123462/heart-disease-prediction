# Heart Disease Prediction — Vercel build

A lightweight serverless version of the Heart Disease Prediction app,
for Vercel (Streamlit cannot run on Vercel, and Streamlit Community
Cloud is blocked for this account). The Streamlit app in `../app.py`
is untouched — this is a second, Vercel-specific build.

## How it works

- `export_model.py` trains the user's notebook model (LogisticRegression,
  default settings, same split) and writes `model.json` with the
  coefficients, intercept, feature order, classes and metrics.
- `api/index.py` is the Vercel Python serverless handler (stdlib only,
  plus gTTS/edge-tts for speech). Prediction is pure sigmoid math on
  `model.json` — no sklearn at runtime:
  `P(Presence) = sigmoid(dot(coefficients, x) + intercept)`.
- `web/index.html` is a single self-contained page: Urdu-first RTL UI
  with an English toggle, the same 13 inputs, result card, and Urdu +
  English voice playback via `/api/speak`.
- `vercel.json` is exactly `{}` — no rewrites. (A previous project broke
  because rewrites replaced the request path with the function path;
  native routing serves `/` and `/api/*` correctly.)

## Run it locally

```bash
cd vercel
../.venv/bin/python export_model.py   # regenerate model.json
../.venv/bin/python -m pytest tests/ -q
../.venv/bin/python serve_local.py    # http://127.0.0.1:8000
```

## Deploy

```bash
vercel --prod
# or: import the GitHub repo in the Vercel dashboard, root directory vercel/
```

## Pure-Python prediction equivalence (measured)

`tests/test_api.py` compares the stdlib prediction against sklearn's
`predict` / `predict_proba` on **all 270 rows** of the dataset plus the
notebook's example input: labels match on every row, probabilities
agree within 1e-6. Example input
`(57,1,2,124,261,0,0,141,0,0.3,1,0,7)` → `Absence`, P(Presence) ≈ 0.238.

Model metrics (measured, same as the notebook): train accuracy 0.875,
test accuracy 0.8333, 5-fold CV 0.8481.

## Limitations (honest)

- Trained on only 270 patients; test accuracy 83.3% means ~1 in 6
  predictions is wrong. Educational estimate only — never a diagnosis.
- The model uses unscaled features with default `max_iter=100`, exactly
  as the notebook did, so it does not fully converge (the notebook's
  warning). The scaled + `max_iter=1000` alternative was tried and was
  not better (test 79.6%).
- Speech needs a network service (edge-tts or gTTS); if both fail the
  API returns 503 and the UI falls back to the written result. Audio is
  never faked. In this sandbox edge-tts is blocked (SSL), so gTTS is
  the working engine here; on Vercel, edge-tts is tried first.
- Urdu + English voice only — no Pashto voice in this app.

## Disclaimer

⚠️ **This app is for education only. It is NOT a medical diagnosis.**
Whatever the result, always consult a doctor about your health. In case
of chest pain, breathing difficulty or fainting, seek emergency help
immediately.
