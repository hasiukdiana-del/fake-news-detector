#!/usr/bin/env bash
# Creates labels, milestones (epics) and issues (user stories with task lists).
# Requirements: GitHub CLI, `gh auth login`, run inside the cloned repository.
set -euo pipefail

gh label create user-story --color 2E6B52 --description 'User story' --force
for l in data modelling app docs; do gh label create "$l" --color 9FB8A9 --force; done

gh api repos/{owner}/{repo}/milestones -f title='E1: Дані та дослідження' -f description='Отримати чистий, перевірений на витік ознак датасет і зрозуміти його структуру.' >/dev/null || true
gh issue create --title 'US1: Завантажити й описати датасет' --body 'Як дослідниця, я хочу завантажити й описати датасет, щоб знати його обсяг, баланс класів і якість.

**Tasks**
- [ ] Завантажити Fake.csv і True.csv з Kaggle у data/raw
- [ ] Перевірити пропуски, порожні тексти, дублікати
- [ ] Побудувати гістограму й boxplot довжини статей
- [ ] Перевірити нормальність довжини (тест Д'"'"'Агостіно, QQ-plot)

**Acceptance criteria:** Розділ 2 записника виконується без помилок; ключові числа зафіксовано в markdown.

Epic: E1: Дані та дослідження' --label user-story --label data --milestone 'E1: Дані та дослідження'
gh issue create --title 'US2: Виявити витік ознак' --body 'Як дослідниця, я хочу виявити витік ознак, щоб модель вчилася на мові тексту, а не на слідах джерела.

**Tasks**
- [ ] Таблиця subject x label
- [ ] Частка маркерів (Reuters), Getty Images, pic.twitter.com за класами
- [ ] Експеримент: F1 моделі лише на subject і лише на маркері Reuters
- [ ] Реалізувати strip_leakage у src/data.py
- [ ] Дедуплікація до розбиття, стратифікований split 70/15/15

**Acceptance criteria:** Після очищення частка '"'"'reuters'"'"' у текстах менша за 0,01%; тести tests/test_data.py проходять.

Epic: E1: Дані та дослідження' --label user-story --label data --milestone 'E1: Дані та дослідження'

gh api repos/{owner}/{repo}/milestones -f title='E2: Моделювання й оцінювання' -f description='Побудувати базову і глибоку моделі та чесно порівняти їх на тестовій вибірці.' >/dev/null || true
gh issue create --title 'US3: Мати базову модель TF-IDF + LogReg' --body 'Як ML-інженерка, я хочу мати базову модель TF-IDF + LogReg, щоб знати, чи виправдана глибока модель.

**Tasks**
- [ ] build_baseline у src/models.py
- [ ] Навчання в src/train.py --model baseline
- [ ] Classification report на validation

**Acceptance criteria:** Модель збережено в models/; метрики validation записано в ноутбуці.

Epic: E2: Моделювання й оцінювання' --label user-story --label modelling --milestone 'E2: Моделювання й оцінювання'
gh issue create --title 'US4: Навчити глибоку нейронну мережу' --body 'Як ML-інженерка, я хочу навчити глибоку нейронну мережу, щоб перевершити базову модель.

**Tasks**
- [ ] build_dnn_parts: TF-IDF + MLP (256, 64)
- [ ] Навчання по епохах з ранньою зупинкою за validation log-loss
- [ ] Збереження найкращої епохи в models/
- [ ] Криві навчання loss і F1
- [ ] BiLSTM як необов'"'"'язкове розширення

**Acceptance criteria:** Модель зберігається й завантажується; криві навчання збережено в reports/figures.

Epic: E2: Моделювання й оцінювання' --label user-story --label modelling --milestone 'E2: Моделювання й оцінювання'
gh issue create --title 'US5: Оцінити обидві моделі на тестовій вибірці' --body 'Як дослідниця, я хочу оцінити обидві моделі на тестовій вибірці, щоб порівняти їх за precision, recall, F1 і ROC-AUC.

**Tasks**
- [ ] src/evaluate.py: metrics.json, confusion matrices, ROC
- [ ] Аналіз зони невизначеності
- [ ] Аналіз 10 найвпевненіших помилок

**Acceptance criteria:** reports/metrics.json існує; висновки в розділах 3 і 5 записника заповнено цифрами.

Epic: E2: Моделювання й оцінювання' --label user-story --label modelling --milestone 'E2: Моделювання й оцінювання'

gh api repos/{owner}/{repo}/milestones -f title='E3: Прототип сервісу' -f description='Дати нетехнічному користувачеві простий інструмент перевірки тексту.' >/dev/null || true
gh issue create --title 'US6: Вставити текст новини й отримати оцінку ймовірності фейку' --body 'Як журналістка, я хочу вставити текст новини й отримати оцінку ймовірності фейку, щоб вирішити, чи потрібна ручна перевірка.

**Tasks**
- [ ] Сторінка Streamlit із полем тексту й прикладами
- [ ] Вердикт у трьох зонах: надійна, невпевнено, фейк
- [ ] Пояснення: слова, що вплинули на оцінку

**Acceptance criteria:** streamlit run app/streamlit_app.py працює локально; відповідь за менше ніж 2 секунди.

Epic: E3: Прототип сервісу' --label user-story --label app --milestone 'E3: Прототип сервісу'
gh issue create --title 'US7: Бачити якість моделі та її обмеження' --body 'Як користувач, я хочу бачити якість моделі та її обмеження, щоб не довіряти їй сліпо.

**Tasks**
- [ ] Метрики з metrics.json у боковій панелі
- [ ] Попередження: модель не перевіряє факти
- [ ] Опис корпусу й періоду

**Acceptance criteria:** Обмеження видно без прокручування сторінки.

Epic: E3: Прототип сервісу' --label user-story --label app --milestone 'E3: Прототип сервісу'

gh api repos/{owner}/{repo}/milestones -f title='E4: Якість, документація, захист' -f description='Зробити проєкт відтворюваним і підготувати презентацію.' >/dev/null || true
gh issue create --title 'US8: Мати автотести препроцесингу' --body 'Як розробниця, я хочу мати автотести препроцесингу, щоб зміни не зламали очищення.

**Tasks**
- [ ] Тести на вилучення датлайну Reuters і підписів до фото
- [ ] Тести на дедуплікацію і стратифікацію

**Acceptance criteria:** pytest проходить без помилок.

Epic: E4: Якість, документація, захист' --label user-story --label docs --milestone 'E4: Якість, документація, захист'
gh issue create --title 'US9: Відтворити проєкт за README' --body 'Як викладач, я хочу відтворити проєкт за README, щоб перевірити результати.

**Tasks**
- [ ] README: структура, встановлення, запуск, Colab
- [ ] requirements.txt
- [ ] Фіксований seed

**Acceptance criteria:** Проєкт запускається з нуля за інструкцією.

Epic: E4: Якість, документація, захист' --label user-story --label docs --milestone 'E4: Якість, документація, захист'
gh issue create --title 'US10: Презентацію для нетехнічної аудиторії' --body 'Як доповідачка, я хочу презентацію для нетехнічної аудиторії, щоб пояснити цінність і межі сервісу.

**Tasks**
- [ ] Слайди без жаргону
- [ ] Живе демо Streamlit
- [ ] Репетиція на 7-10 хвилин

**Acceptance criteria:** Презентація заповнена реальними метриками з metrics.json.

Epic: E4: Якість, документація, захист' --label user-story --label docs --milestone 'E4: Якість, документація, захист'

echo "Done. Add the issues to a GitHub Project board (Todo / In progress / Done)."
