"""Streamlit prototype: «Перевірка новини».

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

st.set_page_config(page_title="Перевірка новини", page_icon="📰", layout="centered")

EXAMPLES = {
    "Приклад: стриманий новинний стиль":
        "The central bank kept its key interest rate unchanged on Thursday, citing slowing "
        "inflation and steady employment, officials said after a two-day meeting.",
    "Приклад: сенсаційний стиль":
        "SHOCKING! You won't believe what this politician just did. The mainstream media "
        "will never tell you the truth. Share before it gets deleted!!!",
}

VERDICTS = {
    "fake": ("Текст має ознаки фейкової новини", "error"),
    "real": ("Стиль тексту схожий на надійні новини", "success"),
    "uncertain": ("Модель не впевнена: потрібна ручна перевірка", "warning"),
}


@st.cache_resource(show_spinner="Завантаження моделі...")
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
st.sidebar.title("Про сервіс")
st.sidebar.write(
    "Глибока нейронна мережа оцінює, наскільки стиль англомовної "
    "новини схожий на стиль джерел фейкових новин."
)
st.sidebar.info(
    "Модель **не перевіряє факти**. Вона аналізує мову тексту. Результат є сигналом "
    "для фактчекера, а не вироком."
)
models = available_models()
if not models:
    st.error("Моделі не знайдено. Виконайте `python -m src.train` і `python -m src.evaluate`.")
    st.stop()
labels = {"dnn": "Глибока нейронна мережа (основна)", "bilstm": "BiLSTM (розширення)", "baseline": "TF-IDF + логістична регресія (базова)"}
model_name = st.sidebar.radio("Модель", models, format_func=labels.get)

metrics = load_metrics().get(model_name)
if metrics:
    st.sidebar.subheader("Якість на тестовій вибірці")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Точність", f"{metrics['accuracy']:.1%}")
    c2.metric("F1 (фейк)", f"{metrics['f1_fake']:.3f}")
    c1.metric("Precision", f"{metrics['precision_fake']:.1%}")
    c2.metric("Recall", f"{metrics['recall_fake']:.1%}")
    st.sidebar.caption(f"Перевірено на {metrics['n_test']} статтях, яких модель не бачила.")

# ---------------------------------------------------------------- main
st.title("📰 Перевірка новини")
st.write("Вставте заголовок і текст англомовної новини. Сервіс оцінить імовірність того, що це фейк.")

choice = st.selectbox("Або оберіть приклад", ["Власний текст", *EXAMPLES])
default = EXAMPLES.get(choice, "")
text = st.text_area("Текст новини", value=default, height=220,
                    placeholder="Paste the headline and the article here...")

if st.button("Перевірити", type="primary", use_container_width=True):
    if len(text.split()) < 5:
        st.warning("Текст закороткий. Додайте хоча б одне повне речення.")
        st.stop()

    model = get_model(model_name)
    p_fake = float(predict_proba(model, model_name, [text])[0])
    label, kind = VERDICTS[verdict(p_fake)]
    getattr(st, kind)(f"**{label}**")

    st.progress(p_fake, text=f"Імовірність фейку: {p_fake:.0%}")
    st.caption(
        f"Нижче {config.UNCERTAIN_LOW:.0%}: схоже на надійну новину · "
        f"від {config.UNCERTAIN_LOW:.0%} до {config.UNCERTAIN_HIGH:.0%}: модель не впевнена · "
        f"вище {config.UNCERTAIN_HIGH:.0%}: ознаки фейку"
    )

    if "baseline" in models:
        with st.expander("Які слова вплинули на оцінку (за базовою моделлю)"):
            contributions = explain_baseline(get_model("baseline"), text)
            if not contributions:
                st.write("Модель не знайшла знайомих слів у тексті.")
            for word, weight in contributions:
                side = "до «фейк»" if weight > 0 else "до «надійна»"
                st.write(f"`{word}` · {side} ({weight:+.3f})")

st.divider()
st.caption(
    "Навчено на корпусі Fake and Real News (Kaggle, ISOT): англомовні новини про політику США "
    "2015-2018 років. На інших темах, мовах і періодах точність може бути нижчою."
)
