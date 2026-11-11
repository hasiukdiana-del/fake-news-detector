# Виявлення фейкових новин за допомогою глибокої нейронної мережі

Проєкт з курсу «Глибинне навчання», ІПЗ-41. Авторка: Діана Гасюк.

Сервіс оцінює, наскільки стиль англомовної новини схожий на стиль джерел фейкових новин. В основі сервісу глибока нейронна мережа (багатошаровий перцептрон на ознаках TF-IDF). Для порівняння є базова модель TF-IDF + логістична регресія. Двонаправлена LSTM реалізована як необов'язкове розширення. Прототип зроблено на Streamlit.

> Модель не перевіряє факти. Вона аналізує мову тексту, тож її результат слід сприймати як сигнал для фактчекера.

## Головна методологічна ідея

У датасеті майже всі справжні статті починаються з «МІСТО (Reuters) -», а поле `subject` має різні значення для двох файлів. Модель, навчена на сирих даних, досягає майже ідеальної точності, тому що вчиться впізнавати джерело, а не фейк. У проєкті цей витік ознак виявлено й усунуто (див. розділ 2.3 дослідницького записника і `src/data.py`), а дублікати вилучено до розбиття на вибірки.

## Структура

```
fake-news-detector/
├── app/streamlit_app.py        # Streamlit prototype
├── data/raw/                   # Fake.csv, True.csv (not in git)
├── data/processed/             # train/val/test splits (generated)
├── docs/project_management.md  # epics, user stories, tasks, DoD
├── models/                     # trained models (generated)
├── notebooks/01_research.ipynb # дослідницький записник: аналіз даних, аудит витоку, моделі, оцінювання
├── reports/                    # metrics.json, figures (generated)
├── scripts/create_github_issues.sh
├── src/
│   ├── config.py               # paths, seed, hyperparameters
│   ├── data.py                 # loading, leakage-aware cleaning, split
│   ├── models.py               # baseline, DNN and optional BiLSTM
│   ├── train.py                # training CLI
│   ├── evaluate.py             # test-set metrics and plots
│   └── predict.py              # inference and explanations
└── tests/test_data.py
```

## Встановлення

Потрібен Python 3.10 або новіший. TensorFlow потрібен лише для необов'язкової BiLSTM.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Дані

Завантажте [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) і покладіть `Fake.csv` та `True.csv` у `data/raw/`.

Можна завантажити й через Kaggle API (потрібен `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
```

## Запуск

```bash
python -m src.data                 # clean, deduplicate and split -> data/processed
python -m src.train                # train baseline and DNN -> models/ (about 3 min on CPU)
python -m src.evaluate             # test metrics and figures -> reports/
pytest -q                          # unit tests
streamlit run app/streamlit_app.py # prototype at http://localhost:8501
```

Окремий текст можна перевірити з командного рядка:

```bash
python -m src.predict --model dnn "Lawmakers approved the budget bill after a long debate."
```

Дослідницький записник: `jupyter notebook notebooks/01_research.ipynb`. Записник виконує ті самі кроки з поясненнями й графіками та зберігає моделі в `models/`.

## Необов'язкове розширення: BiLSTM

```bash
pip install tensorflow
python -m src.train --model bilstm   # GPU recommended, e.g. Google Colab
python -m src.evaluate
```

## Модель

| Компонент | Значення |
|---|---|
| Вхід | заголовок + текст, очищені від слідів джерела |
| Ознаки | TF-IDF, 30 000 уні- і біграм |
| Архітектура | Dense 256 ReLU → Dense 64 ReLU → sigmoid, 7,7 млн параметрів |
| Навчання | Adam, L2 `alpha=1e-4`, батч 256, рання зупинка за validation log-loss |
| Розбиття | стратифіковане 70 / 15 / 15, seed 42, дедуплікація до розбиття |
| Метрики | accuracy, precision, recall, F1 для класу «фейк», ROC-AUC |

## Результати

Тестова вибірка: 5 824 статті, яких модель не бачила (`reports/metrics.json`).

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Пропущені фейки | Хибні тривоги |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9878 | 0.9891 | 0.9844 | 0.9867 | 0.9992 | 42 | 29 |
| **DNN** | **0.9902** | 0.9888 | **0.9899** | **0.9894** | **0.9996** | **27** | 30 |

**Статистична значущість** (точний тест Мак-Немара, Dietterich, 1998): загальна різниця в точності не значуща (p = 0.076, 95% бутстреп-інтервал різниці від -0.02 до 0.50 п.п.), а перевага DNN у повноті значуща (p = 0.008: 22 фейки знайшла лише DNN, 7 лише базова модель).

Без очищення від слідів джерела модель лише на полі `subject` досягає F1 = 1.000, а лише на маркері «(Reuters)» F1 = 0.996 (записник, розділ 2.3).

## Обмеження

* Мітка присвоєна за джерелом статті, а не за фактчекінгом кожного тексту.
* Корпус охоплює англомовні новини про політику США 2015-2018 років.
* Модель вчиться стилю джерел. Стримано написаний фейк вона може пропустити.
* Частина відмінностей тематична: Reuters пише про міжнародні події, фейкові сайти про внутрішню політику США.

## Управління проєктом

Епіки, юзер сторі, таски і критерії приймання описано в `docs/project_management.md`. Створити їх як issues і milestones на GitHub можна однією командою: `bash scripts/create_github_issues.sh`.

## Джерела

* Ahmad, H., Traore, I., Saad, S. (2017). Detection of online fake news using n-gram analysis and machine learning techniques. ISDDC 2017, LNCS 10618, 127-138.
* Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.
* Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2, 665-673.
