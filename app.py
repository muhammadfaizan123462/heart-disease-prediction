"""Heart Disease Prediction — Urdu-first Streamlit app.

Built from the user's own Colab notebook (notebooks/Heart_Disease_Prediction.ipynb):
LogisticRegression, same split (test_size=0.2, stratify, random_state=2).
Train accuracy 0.875, test accuracy 0.8333.

Key features:
- Urdu-first UI (RTL), English toggle.
- Result SPOKEN in Urdu AND English (Urdu first, then English) for users
  who cannot read — "Sunain" button + optional auto-play.
- Prominent medical disclaimer: educational demo, NOT a diagnosis.
- The notebook's label bug (cell 28 compared 'absence' lowercase) is
  fixed: 'Presence'/'Absence' are handled correctly here.
"""

from __future__ import annotations

import streamlit as st

from model import EXAMPLE_INPUT, evaluate, load_data, predict, train_model
from tts import speak_result_both

# ---------------------------------------------------------------- strings
STR = {
    "ur": {
        "title": "دل کی بیماری کی پیش گوئی",
        "subtitle": "اپنی صحت کی معلومات درج کریں — ماڈل بتائے گا کہ دل کی بیماری کا خطرہ ہے یا نہیں",
        "lang_label": "زبان / Language",
        "predict_btn": "پیش گوئی کریں",
        "result_presence": "دل کی بیماری کا خطرہ ہے",
        "result_absence": "دل کی بیماری کا خطرہ نہیں ہے",
        "prob_label": "خطرے کا امکان",
        "listen_btn": "سنیں",
        "listen_urdu": "اردو میں سنیں",
        "listen_en": "انگریزی میں سنیں",
        "autoplay": "نتیجہ اردو اور انگریزی دونوں میں خود بخود سنائیں",
        "audio_note": "آواز تیار ہو رہی ہے…",
        "audio_fail": "آواز تیار نہیں ہو سکی — نیچے لکھا نتیجہ پڑھ لیں۔",
        "accuracy_note": "ماڈل کی درستگی (ٹیسٹ ڈیٹا): 83.3٪ — یہ صرف ایک اندازہ ہے",
        "disclaimer_title": "اہم انتباہ",
        "disclaimer": (
            "یہ ایپ صرف تعلیم کے لیے ہے۔ یہ طبی تشخیص نہیں ہے۔ "
            "نتیجہ چاہے کچھ بھی ہو، اپنی صحت کے بارے میں ہمیشہ ڈاکٹر سے مشورہ کریں۔ "
            "سینے میں درد، سانس لینے میں دشواری یا بے ہوشی کی صورت میں فوراً ایمرجنسی مدد حاصل کریں۔"
        ),
        "inputs_title": "آپ کی معلومات",
        "model_title": "ماڈل کی معلومات",
        "example_btn": "مثال والا مریض بھریں",
        "example_note": "مثال: 57 سالہ مرد — نوٹ بک والا مریض",
    },
    "en": {
        "title": "Heart Disease Prediction",
        "subtitle": "Enter your health information — the model will tell you whether there is a risk of heart disease",
        "lang_label": "Language / زبان",
        "predict_btn": "Predict",
        "result_presence": "Risk of heart disease",
        "result_absence": "No risk of heart disease",
        "prob_label": "Risk probability",
        "listen_btn": "Listen",
        "listen_urdu": "Listen in Urdu",
        "listen_en": "Listen in English",
        "autoplay": "Play the result automatically in Urdu and English",
        "audio_note": "Generating audio…",
        "audio_fail": "Could not generate audio — please read the written result below.",
        "accuracy_note": "Model accuracy (test data): 83.3% — this is only an estimate",
        "disclaimer_title": "Important warning",
        "disclaimer": (
            "This app is for education only. It is NOT a medical diagnosis. "
            "Whatever the result, always consult a doctor about your health. "
            "In case of chest pain, breathing difficulty or fainting, seek emergency help immediately."
        ),
        "inputs_title": "Your information",
        "model_title": "Model information",
        "example_btn": "Fill in the example patient",
        "example_note": "Example: 57-year-old man — the patient from the notebook",
    },
}

# (key, urdu label, english label, widget type, options/default)
FIELDS = [
    ("age", "عمر", "Age", ("int", 1, 120, 57)),
    ("sex", "جنس", "Sex", ("select", ["مرد (Mard)", "عورت (Aurat)"])),
    ("cp", "سینے کے درد کی قسم", "Chest pain type", ("select", [
        "1 — عام اینجائنا (Typical angina)",
        "2 — غیر معمولی اینجائنا (Atypical angina)",
        "3 — غیر اینجائنا درد (Non-anginal pain)",
        "4 — بغیر علامت (Asymptomatic)",
    ])),
    ("bp", "آرام کے وقت بلڈ پریشر", "Resting blood pressure (BP)", ("int", 50, 250, 124)),
    ("chol", "کولیسٹرول", "Cholesterol", ("int", 50, 700, 261)),
    ("fbs", "خالی پیٹ شوگر 120 سے زیادہ؟", "Fasting blood sugar > 120?", ("select", ["نہیں (No)", "ہاں (Yes)"])),
    ("ekg", "ای سی جی نتیجہ", "EKG results", ("select", [
        "0 — نارمل (Normal)",
        "1 — ایس ٹی ٹی خرابی (ST-T wave abnormality)",
        "2 — بائیں وینٹریکل کی موٹائی (LV hypertrophy)",
    ])),
    ("hr", "ورزش کے دوران زیادہ سے زیادہ دل کی دھڑکن", "Maximum heart rate", ("int", 40, 250, 141)),
    ("angina", "ورزش سے سینے میں درد؟", "Exercise-induced angina?", ("select", ["نہیں (No)", "ہاں (Yes)"])),
    ("st", "ایس ٹی ڈپریشن", "ST depression", ("float", 0.0, 7.0, 0.3)),
    ("slope", "ایس ٹی کی ڈھلان", "Slope of ST segment", ("select", [
        "1 — اوپر کی طرف (Upsloping)",
        "2 — برابر (Flat)",
        "3 — نیچے کی طرف (Downsloping)",
    ])),
    ("vessels", "رنگ سے دیکھی گئی بند نالیاں", "Major vessels (fluoroscopy)", ("int", 0, 3, 0)),
    ("thal", "تھیلیم ٹیسٹ", "Thallium test", ("select", [
        "3 — نارمل (Normal)",
        "6 — مستقل خرابی (Fixed defect)",
        "7 — قابل واپسی خرابی (Reversible defect)",
    ])),
]

YES_NO = {"نہیں (No)": 0, "ہاں (Yes)": 1}


def _first_int(option: str) -> int:
    return int(option.split("—")[0].strip())


# ---------------------------------------------------------------- model
@st.cache_resource
def get_model():
    df = load_data()
    model, X_train, X_test, Y_train, Y_test = train_model(df)
    metrics = evaluate(
        model, X_train, X_test, Y_train, Y_test,
        df.drop(columns="Heart Disease"), df["Heart Disease"],
    )
    return model, metrics


def collect_values() -> list:
    vals = []
    for key, ur, en, spec in FIELDS:
        label = f"{ur} ({en})"
        if spec[0] == "int":
            vals.append(st.number_input(label, min_value=spec[1], max_value=spec[2],
                                        value=spec[3], step=1, key=key))
        elif spec[0] == "float":
            vals.append(st.number_input(label, min_value=spec[1], max_value=spec[2],
                                        value=spec[3], step=0.1, key=key))
        else:
            choice = st.selectbox(label, spec[1], key=key)
            if key == "sex":
                vals.append(1 if choice.startswith("مرد") else 0)
            elif key in ("fbs", "angina"):
                vals.append(YES_NO[choice])
            else:
                vals.append(_first_int(choice))
    return vals


def fill_example():
    vals = list(EXAMPLE_INPUT)
    st.session_state["age"] = vals[0]
    st.session_state["sex"] = "مرد (Mard)" if vals[1] == 1 else "عورت (Aurat)"
    st.session_state["cp"] = FIELDS[2][3][1][vals[2] - 1]
    st.session_state["bp"] = vals[3]
    st.session_state["chol"] = vals[4]
    st.session_state["fbs"] = "ہاں (Yes)" if vals[5] == 1 else "نہیں (No)"
    st.session_state["ekg"] = FIELDS[6][3][1][vals[6]]
    st.session_state["hr"] = vals[7]
    st.session_state["angina"] = "ہاں (Yes)" if vals[8] == 1 else "نہیں (No)"
    st.session_state["st"] = vals[9]
    st.session_state["slope"] = FIELDS[10][3][1][vals[10] - 1]
    st.session_state["vessels"] = vals[11]
    st.session_state["thal"] = {3: FIELDS[12][3][1][0],
                                6: FIELDS[12][3][1][1],
                                7: FIELDS[12][3][1][2]}[vals[12]]


def play_result_audio(label: str):
    """Play the result in BOTH Urdu and English (Urdu first).

    Urdu auto-plays; English is a separate player below it so the two
    voices never talk over each other.
    """
    with st.spinner(STR[st.session_state.lang]["audio_note"]):
        both = speak_result_both(label)
    audio_ur, _engine_ur, error_ur = both["ur"]
    audio_en, _engine_en, error_en = both["en"]
    if audio_ur:
        st.write(f"🔊 {STR[st.session_state.lang]['listen_urdu']}")
        st.audio(audio_ur, format="audio/mp3", autoplay=True)
    else:
        st.warning(STR[st.session_state.lang]["audio_fail"])
        if error_ur:
            st.caption(f"Audio error (Urdu): {error_ur}")
    if audio_en:
        st.write(f"🔊 {STR[st.session_state.lang]['listen_en']}")
        st.audio(audio_en, format="audio/mp3")
    else:
        st.warning(STR[st.session_state.lang]["audio_fail"])
        if error_en:
            st.caption(f"Audio error (English): {error_en}")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="دل کی بیماری کی پیش گوئی | Heart Disease Prediction",
                   page_icon="❤️", layout="centered")

if "lang" not in st.session_state:
    st.session_state.lang = "ur"

lang = st.radio("زبان / Language",
                options=["ur", "en"],
                format_func=lambda x: "اردو" if x == "ur" else "English",
                horizontal=True, key="lang_choice",
                index=0 if st.session_state.lang == "ur" else 1)
st.session_state.lang = lang
S = STR[lang]
rtl = lang == "ur"

st.markdown(
    f"""<style>
    .stApp {{ direction: {'rtl' if rtl else 'ltr'}; text-align: {'right' if rtl else 'left'};
             font-family: 'Noto Nastaliq Urdu', 'Noto Naskh Arabic', Tahoma, sans-serif; }}
    .result-box {{ padding: 1.2rem; border-radius: 12px; text-align: center;
                   font-size: 1.6rem; font-weight: bold; margin: 1rem 0; }}
    .result-risk {{ background: #fde8e8; color: #9b1c1c; border: 2px solid #e02424; }}
    .result-ok {{ background: #e6f7ec; color: #14532d; border: 2px solid #16a34a; }}
    .disclaimer {{ background: #fff7e6; border: 2px solid #d97706; border-radius: 12px;
                   padding: 1rem; margin: 1rem 0; }}
    </style>""",
    unsafe_allow_html=True,
)

st.title(f"❤️ {S['title']}")
st.write(S["subtitle"])

model, metrics = get_model()

# Disclaimer first and prominent
st.markdown(f"<div class='disclaimer'><b>⚠️ {S['disclaimer_title']}</b><br>{S['disclaimer']}</div>",
            unsafe_allow_html=True)

st.header(S["inputs_title"])
if st.button(S["example_btn"]):
    fill_example()
    st.caption(S["example_note"])

values = collect_values()
autoplay = st.checkbox(S["autoplay"], value=True)

if st.button(S["predict_btn"], type="primary", use_container_width=True):
    try:
        label, prob = predict(model, values)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    cls = "result-risk" if label == "Presence" else "result-ok"
    st.markdown(f"<div class='result-box {cls}'>{S['result_' + label.lower()]}</div>",
                unsafe_allow_html=True)
    st.progress(prob if label == "Presence" else 1 - prob)
    st.write(f"**{S['prob_label']}: {prob * 100:.1f}%**")
    st.caption(S["accuracy_note"])

    if autoplay:
        play_result_audio(label)
    elif st.button(f"🔊 {S['listen_btn']}", use_container_width=True):
        play_result_audio(label)

# ---------------------------------------------------------------- model info
with st.expander(S["model_title"]):
    if lang == "ur":
        st.markdown(f"""
- **ڈیٹا سیٹ:** UCI StatLog Heart Disease — 270 مریض، 14 کالم (CC0، پبلک ڈومین)
- **الگورتھم:** Logistic Regression (نوٹ بک والی ترتیبات)
- **درستگی:** ٹریننگ {metrics['train_accuracy']*100:.1f}٪، ٹیسٹ {metrics['test_accuracy']*100:.1f}٪، 5-fold CV {metrics['cv5_accuracy']*100:.1f}٪
- **نوٹ بک کی غلطی درست کی گئی:** سیل 28 میں `'absence'` (چھوٹے حروف) سے موازنہ کیا گیا تھا جبکہ اصل لیبل `'Absence'` ہے — اس لیے مثال والے مریض کو غلط پیغام ملا تھا۔ اس ایپ میں لیبل درست طریقے سے جانچا جاتا ہے۔
- **حد:** ڈیفالٹ max_iter=100 پر ماڈل مکمل converge نہیں ہوتا (نوٹ بک والی وارننگ) — scaled + max_iter=1000 والا متبادل آزمایا گیا، بہتر نہیں تھا (ٹیسٹ 79.6٪)۔
- **آواز:** نتیجہ اردو اور انگریزی دونوں میں بولا جاتا ہے — edge-tts (ur-PK-AsadNeural / en-US-AriaNeural) سے، ناکامی پر gTTS سے۔
""")
    else:
        st.markdown(f"""
- **Dataset:** UCI StatLog Heart Disease — 270 patients, 14 columns (CC0, public domain)
- **Algorithm:** Logistic Regression (notebook settings)
- **Accuracy:** train {metrics['train_accuracy']*100:.1f}%, test {metrics['test_accuracy']*100:.1f}%, 5-fold CV {metrics['cv5_accuracy']*100:.1f}%
- **Notebook bug fixed:** cell 28 compared against `'absence'` (lowercase) while the real label is `'Absence'` — so the example patient got the wrong message. Labels are compared correctly here.
- **Limitation:** with default max_iter=100 the model does not fully converge (the notebook's warning) — the scaled + max_iter=1000 alternative was tried and was not better (test 79.6%).
- **Voice:** the result is spoken in both Urdu and English — via edge-tts (ur-PK-AsadNeural / en-US-AriaNeural), falling back to gTTS.
""")
