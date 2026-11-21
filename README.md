# Fake News Detection with Deep Learning

Course project, *Deep Learning*, Vasyl Stefanyk Carpathian National University, 2026.
Team 10: Diana Hasiuk, Yuliia Dolishnia (group IPZ-41).

## Overview

The project estimates whether an English-language news article comes from an unreliable source, using only the language of the text. The main model is a deep neural network on TF-IDF features. A logistic regression on the same features serves as the baseline. A Streamlit prototype exposes the model to a non-technical user.

The model does not verify facts. Its output is intended to prioritise articles for manual fact-checking.

## Key methodological finding

In the source corpus the class label is almost fully determined by publisher artefacts: the Reuters dateline, the `subject` field and image credits. A classifier that sees only the `subject` field reaches F1 = 1.000. These artefacts and duplicate texts are removed before the train/test split, so the model learns from the language of the article rather than from source markers. The audit is documented in section 2.3 of the [research notebook](notebooks/01_research.ipynb).

## Results

Held-out test set: 5,824 articles not used for training or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.9878 | 0.9891 | 0.9844 | 0.9867 | 0.9992 |
| **Deep neural network** | **0.9902** | 0.9888 | **0.9899** | **0.9894** | **0.9996** |

The network has significantly higher recall on the fake class (exact McNemar test, p = 0.008). The difference in overall accuracy is not significant (p = 0.076).

## Repository structure

```
app/            Streamlit prototype
data/           raw and processed data (not stored in the repository)
docs/           project management
models/         trained models (created by training)
notebooks/      research notebook
reports/        metrics and figures
src/            data preparation, models, training, evaluation, inference
tests/          preprocessing tests
```

## Quick start

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` in `data/raw/`.

```bash
python -m src.data                   # cleaning and stratified split
python -m src.train                  # model training
python -m src.evaluate               # test-set evaluation
streamlit run app/streamlit_app.py   # prototype
pytest -q                            # tests
```

## Limitations

Labels are assigned by publisher, not by verification of individual claims. The corpus covers English-language news on US politics from 2015 to 2018. Part of the difference between classes is topical, and topic cannot be fully separated from style within this corpus.
