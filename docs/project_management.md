# Project management

Process: a Scrum-like cycle of four one-week sprints. Hierarchy: **epic → user story → task → subtask**.
Each epic is a GitHub milestone and an epic issue. Each user story is an issue; its tasks are a checklist.
Commit messages reference the related issue number, for example `(#4)`.

## Definition of Done

* the change is committed with a reference to its issue;
* tests pass and the notebook runs top to bottom;
* results and decisions are recorded in the notebook or the README.

## Sprint plan

| Sprint | Focus | Stories |
|---|---|---|
| 1 | Data and feature leakage | US1, US2, US8 |
| 2 | Models | US3, US4 |
| 3 | Evaluation and prototype | US5, US6, US7 |
| 4 | Documentation and defence | US9, US10 |

## E1. Data and exploration

*Goal:* Obtain a clean, leakage-audited dataset and understand its structure.

### US1. Load and describe the dataset

As a researcher, I want to load and describe the dataset, so that I know its size, class balance and quality.

**Tasks:**

- [ ] Download Fake.csv and True.csv from Kaggle into data/raw
- [ ] Check missing values, empty texts and duplicates
- [ ] Plot the article length histogram and boxplot
- [ ] Test length normality (D'Agostino-Pearson test, Q-Q plot)

**Acceptance criteria:** Section 2 of the notebook runs without errors; key figures are reported in markdown.

### US2. Detect feature leakage

As a researcher, I want to detect feature leakage, so that the model learns from the language of the text rather than from source artefacts.

**Tasks:**

- [ ] Cross-tabulate subject against label
- [ ] Measure the share of (Reuters), Getty Images and pic.twitter.com markers by class
- [ ] Experiment: F1 of a model trained on subject only and on the Reuters marker only
- [ ] Implement strip_leakage in src/data.py
- [ ] Deduplicate before splitting; stratified 70/15/15 split

**Acceptance criteria:** After cleaning, fewer than 0.01% of texts mention 'reuters'; tests/test_data.py passes.

## E2. Modelling and evaluation

*Goal:* Build the baseline and the deep model and compare them fairly on the held-out test set.

### US3. Build a TF-IDF + logistic regression baseline

As an ML engineer, I want a TF-IDF + logistic regression baseline, so that I know whether a deep model is justified.

**Tasks:**

- [ ] build_baseline in src/models.py
- [ ] Training via src/train.py --model baseline
- [ ] Validation metrics report

**Acceptance criteria:** The model is saved to models/; validation metrics are reported in the notebook.

### US4. Train a deep neural network

As an ML engineer, I want to train a deep neural network, so that it outperforms the baseline.

**Tasks:**

- [ ] build_dnn_parts: TF-IDF + MLP (256, 64)
- [ ] Epoch-wise training with early stopping on validation log-loss
- [ ] Save the best epoch to models/
- [ ] Loss and F1 learning curves
- [ ] BiLSTM as an optional extension

**Acceptance criteria:** The model can be saved and loaded; learning curves are saved to reports/figures.

### US5. Evaluate both models on the test set

As a researcher, I want to evaluate both models on the test set, so that I can compare precision, recall, F1 and ROC-AUC.

**Tasks:**

- [ ] src/evaluate.py: metrics.json, confusion matrices, ROC curves
- [ ] McNemar test and bootstrap confidence intervals
- [ ] Uncertainty zone analysis
- [ ] Analysis of the 10 most confident errors

**Acceptance criteria:** reports/metrics.json exists; sections 3 and 5 of the notebook report the figures.

## E3. Service prototype

*Goal:* Give a non-technical user a simple tool for checking a text.

### US6. Score a news article for fake probability

As a journalist, I want to paste an article and get a fake probability, so that I can decide whether manual review is needed.

**Tasks:**

- [ ] Streamlit page with a text field and examples
- [ ] Three-zone verdict: reliable, uncertain, fake
- [ ] Explanation: words that influenced the score

**Acceptance criteria:** streamlit run app/streamlit_app.py works locally; response time is under 2 seconds.

### US7. Show model quality and limitations

As a user, I want to see the model's quality and limitations, so that I do not trust it blindly.

**Tasks:**

- [ ] Test-set metrics from metrics.json in the sidebar
- [ ] Warning: the model does not verify facts
- [ ] Description of the corpus and period

**Acceptance criteria:** Limitations are visible without scrolling.

## E4. Quality, documentation and defence

*Goal:* Make the project reproducible and prepare the presentation.

### US8. Add preprocessing tests

As a developer, I want automated preprocessing tests, so that changes do not break the cleaning.

**Tasks:**

- [ ] Tests for removal of the Reuters dateline and image credits
- [ ] Tests for deduplication and stratification

**Acceptance criteria:** pytest passes.

### US9. Reproduce the project from the README

As a lecturer, I want to reproduce the project from the README, so that I can verify the results.

**Tasks:**

- [ ] README: structure, installation, quick start
- [ ] requirements.txt
- [ ] Fixed random seed

**Acceptance criteria:** The project runs from scratch by following the instructions.

### US10. Prepare a presentation for a non-technical audience

As a presenter, I want a presentation for a non-technical audience, so that I can explain the value and limits of the service.

**Tasks:**

- [ ] Jargon-free slides
- [ ] Live Streamlit demo
- [ ] 7-10 minute rehearsal

**Acceptance criteria:** The presentation uses real metrics from metrics.json.
