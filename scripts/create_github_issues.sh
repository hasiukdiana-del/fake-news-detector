#!/usr/bin/env bash
# Creates labels, milestones (epics) and issues (user stories with task lists).
# Requirements: GitHub CLI, `gh auth login`, run inside the cloned repository.
set -euo pipefail

gh label create epic --color 123A2F --description 'Epic' --force
gh label create user-story --color 2E6B52 --description 'User story' --force
for l in data modelling app docs; do gh label create "$l" --color 9FB8A9 --force; done

gh api repos/{owner}/{repo}/milestones -f title='E1: Data and exploration' -f description='Obtain a clean, leakage-audited dataset and understand its structure.' >/dev/null || true
gh issue create --title 'US1: Load and describe the dataset' --body 'As a researcher, I want to load and describe the dataset, so that I know its size, class balance and quality.

**Tasks**
- [ ] Download Fake.csv and True.csv from Kaggle into data/raw
- [ ] Check missing values, empty texts and duplicates
- [ ] Plot the article length histogram and boxplot
- [ ] Test length normality (D'"'"'Agostino-Pearson test, Q-Q plot)

**Acceptance criteria:** Section 2 of the notebook runs without errors; key figures are reported in markdown.

Epic: E1: Data and exploration' --label user-story --label data --milestone 'E1: Data and exploration'
gh issue create --title 'US2: Detect feature leakage' --body 'As a researcher, I want to detect feature leakage, so that the model learns from the language of the text rather than from source artefacts.

**Tasks**
- [ ] Cross-tabulate subject against label
- [ ] Measure the share of (Reuters), Getty Images and pic.twitter.com markers by class
- [ ] Experiment: F1 of a model trained on subject only and on the Reuters marker only
- [ ] Implement strip_leakage in src/data.py
- [ ] Deduplicate before splitting; stratified 70/15/15 split

**Acceptance criteria:** After cleaning, fewer than 0.01% of texts mention '"'"'reuters'"'"'; tests/test_data.py passes.

Epic: E1: Data and exploration' --label user-story --label data --milestone 'E1: Data and exploration'

gh api repos/{owner}/{repo}/milestones -f title='E2: Modelling and evaluation' -f description='Build the baseline and the deep model and compare them fairly on the held-out test set.' >/dev/null || true
gh issue create --title 'US3: Build a TF-IDF + logistic regression baseline' --body 'As an ML engineer, I want a TF-IDF + logistic regression baseline, so that I know whether a deep model is justified.

**Tasks**
- [ ] build_baseline in src/models.py
- [ ] Training via src/train.py --model baseline
- [ ] Validation metrics report

**Acceptance criteria:** The model is saved to models/; validation metrics are reported in the notebook.

Epic: E2: Modelling and evaluation' --label user-story --label modelling --milestone 'E2: Modelling and evaluation'
gh issue create --title 'US4: Train a deep neural network' --body 'As an ML engineer, I want to train a deep neural network, so that it outperforms the baseline.

**Tasks**
- [ ] build_dnn_parts: TF-IDF + MLP (256, 64)
- [ ] Epoch-wise training with early stopping on validation log-loss
- [ ] Save the best epoch to models/
- [ ] Loss and F1 learning curves
- [ ] BiLSTM as an optional extension

**Acceptance criteria:** The model can be saved and loaded; learning curves are saved to reports/figures.

Epic: E2: Modelling and evaluation' --label user-story --label modelling --milestone 'E2: Modelling and evaluation'
gh issue create --title 'US5: Evaluate both models on the test set' --body 'As a researcher, I want to evaluate both models on the test set, so that I can compare precision, recall, F1 and ROC-AUC.

**Tasks**
- [ ] src/evaluate.py: metrics.json, confusion matrices, ROC curves
- [ ] McNemar test and bootstrap confidence intervals
- [ ] Uncertainty zone analysis
- [ ] Analysis of the 10 most confident errors

**Acceptance criteria:** reports/metrics.json exists; sections 3 and 5 of the notebook report the figures.

Epic: E2: Modelling and evaluation' --label user-story --label modelling --milestone 'E2: Modelling and evaluation'

gh api repos/{owner}/{repo}/milestones -f title='E3: Service prototype' -f description='Give a non-technical user a simple tool for checking a text.' >/dev/null || true
gh issue create --title 'US6: Score a news article for fake probability' --body 'As a journalist, I want to paste an article and get a fake probability, so that I can decide whether manual review is needed.

**Tasks**
- [ ] Streamlit page with a text field and examples
- [ ] Three-zone verdict: reliable, uncertain, fake
- [ ] Explanation: words that influenced the score

**Acceptance criteria:** streamlit run app/streamlit_app.py works locally; response time is under 2 seconds.

Epic: E3: Service prototype' --label user-story --label app --milestone 'E3: Service prototype'
gh issue create --title 'US7: Show model quality and limitations' --body 'As a user, I want to see the model'"'"'s quality and limitations, so that I do not trust it blindly.

**Tasks**
- [ ] Test-set metrics from metrics.json in the sidebar
- [ ] Warning: the model does not verify facts
- [ ] Description of the corpus and period

**Acceptance criteria:** Limitations are visible without scrolling.

Epic: E3: Service prototype' --label user-story --label app --milestone 'E3: Service prototype'

gh api repos/{owner}/{repo}/milestones -f title='E4: Quality, documentation and defence' -f description='Make the project reproducible and prepare the presentation.' >/dev/null || true
gh issue create --title 'US8: Add preprocessing tests' --body 'As a developer, I want automated preprocessing tests, so that changes do not break the cleaning.

**Tasks**
- [ ] Tests for removal of the Reuters dateline and image credits
- [ ] Tests for deduplication and stratification

**Acceptance criteria:** pytest passes.

Epic: E4: Quality, documentation and defence' --label user-story --label docs --milestone 'E4: Quality, documentation and defence'
gh issue create --title 'US9: Reproduce the project from the README' --body 'As a lecturer, I want to reproduce the project from the README, so that I can verify the results.

**Tasks**
- [ ] README: structure, installation, quick start
- [ ] requirements.txt
- [ ] Fixed random seed

**Acceptance criteria:** The project runs from scratch by following the instructions.

Epic: E4: Quality, documentation and defence' --label user-story --label docs --milestone 'E4: Quality, documentation and defence'
gh issue create --title 'US10: Prepare a presentation for a non-technical audience' --body 'As a presenter, I want a presentation for a non-technical audience, so that I can explain the value and limits of the service.

**Tasks**
- [ ] Jargon-free slides
- [ ] Live Streamlit demo
- [ ] 7-10 minute rehearsal

**Acceptance criteria:** The presentation uses real metrics from metrics.json.

Epic: E4: Quality, documentation and defence' --label user-story --label docs --milestone 'E4: Quality, documentation and defence'

