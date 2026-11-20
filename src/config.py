"""Central configuration: paths, random seed and model hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # Fake.csv and True.csv from Kaggle go here
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42

# Split proportions: train / validation / test = 70 / 15 / 15
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Label convention used everywhere in the project
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_FAKE: "fake"}

# Baseline: TF-IDF + logistic regression
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)

# Main model: deep neural network (multilayer perceptron) on TF-IDF features
DNN_MAX_FEATURES = 30_000
DNN_NGRAM_RANGE = (1, 2)
DNN_HIDDEN = (256, 64)     # two hidden ReLU layers
DNN_ALPHA = 1e-4           # L2 regularisation
DNN_BATCH = 256
DNN_LR = 1e-3
DNN_MAX_EPOCHS = 15
DNN_PATIENCE = 2

# Optional extension: BiLSTM (needs TensorFlow, see README)
VOCAB_SIZE = 20_000
MAX_LEN = 300          # tokens kept per article (the median article is shorter)
EMBED_DIM = 128
LSTM_UNITS = 64
DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 8             # upper bound; early stopping usually ends training earlier
PATIENCE = 2

# Decision rule shown to users of the service
UNCERTAIN_LOW = 0.35   # below: "looks like reliable news"
UNCERTAIN_HIGH = 0.65  # above: "shows signs of fake news"

BASELINE_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
DNN_PATH = MODELS_DIR / "dnn_tfidf_mlp.joblib"
BILSTM_PATH = MODELS_DIR / "bilstm.keras"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
