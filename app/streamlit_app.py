"""Streamlit prototype: News Check.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import explain_baseline, load_model, predict_proba, verdict  # noqa: E402

st.set_page_config(page_title="News Check", page_icon="📰", layout="centered")

EXAMPLES = {
    "Example: neutral news style":
        "The central bank kept its key interest rate unchanged on Thursday, citing slowing "
        "inflation and steady employment, officials said after a two-day meeting.",
    "Example: sensational style":
        "SHOCKING! You won't believe what this politician just did. The mainstream media "
        "will never tell you the truth. Share before it gets deleted!!!",
}

VERDICTS = {
    "fake": ("The text shows signs of fake news", "error"),
    "real": ("The style resembles reliable news", "success"),
    "uncertain": ("The model is uncertain: manual review is needed", "warning"),
}


@st.cache_resource(show_spinner="Loading the model...")
def get_model(name: str):
    return load_model(name)


def available_models() -> list[str]:
    names = []
    if config.DNN_PATH.exists():
        names.append("dnn")
    if config.BILSTM_PATH.exists():
        names.append("bilstm")
    if config.BASELINE_PATH.exists():
        names.append("baseline")
    return names


def load_metrics() -> dict:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return {}


# ---------------------------------------------------------------- sidebar
st.sidebar.title("About")
st.sidebar.write(
    "A deep neural network estimates how closely the style of an English-language "
    "article resembles the style of fake news sources."
)
st.sidebar.info(
    "The model **does not verify facts**. It analyses the language of the text. "
    "The result is a signal for a fact-checker, not a verdict."
)
models = available_models()
if not models:
    st.error("No trained model found. Run `python -m src.train` and `python -m src.evaluate`.")
    st.stop()
labels = {"dnn": "Deep neural network (main)", "bilstm": "BiLSTM (extension)", "baseline": "TF-IDF + logistic regression (baseline)"}
model_name = st.sidebar.radio("Model", models, format_func=labels.get)

metrics = load_metrics().get(model_name)
if metrics:
    st.sidebar.subheader("Test-set quality")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("F1 (fake)", f"{metrics['f1_fake']:.3f}")
    c1.metric("Precision", f"{metrics['precision_fake']:.1%}")
    c2.metric("Recall", f"{metrics['recall_fake']:.1%}")
    st.sidebar.caption(f"Evaluated on {metrics['n_test']} articles unseen during training.")

# ---------------------------------------------------------------- main
st.title("📰 News Check")
st.write("Paste the headline and text of an English-language article. The service estimates the probability that it is fake.")

choice = st.selectbox("Or choose an example", ["Own text", *EXAMPLES])
default = EXAMPLES.get(choice, "")
text = st.text_area("Article text", value=default, height=220,
                    placeholder="Paste the headline and the article here...")

if st.button("Check", type="primary", use_container_width=True):
    if len(text.split()) < 5:
        st.warning("The text is too short. Add at least one full sentence.")
        st.stop()

    model = get_model(model_name)
    p_fake = float(predict_proba(model, model_name, [text])[0])
    label, kind = VERDICTS[verdict(p_fake)]
    getattr(st, kind)(f"**{label}**")

    st.progress(p_fake, text=f"Probability of fake: {p_fake:.0%}")
    st.caption(
        f"Below {config.UNCERTAIN_LOW:.0%}: resembles reliable news · "
        f"{config.UNCERTAIN_LOW:.0%} to {config.UNCERTAIN_HIGH:.0%}: uncertain · "
        f"above {config.UNCERTAIN_HIGH:.0%}: signs of fake news"
    )

    if "baseline" in models:
        with st.expander("Words that influenced the score (baseline model)"):
            contributions = explain_baseline(get_model("baseline"), text)
            if not contributions:
                st.write("The model found no known words in the text.")
            for word, weight in contributions:
                side = "towards fake" if weight > 0 else "towards reliable"
                st.write(f"`{word}` · {side} ({weight:+.3f})")

st.divider()
st.caption(
    "Trained on the Fake and Real News corpus (Kaggle, ISOT): English-language news on US politics, "
    "2015-2018. Accuracy may be lower on other topics, languages and periods."
)
