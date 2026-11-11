"""Model definitions: a classical baseline, the main deep neural network
(multilayer perceptron) and an optional bidirectional LSTM.

TensorFlow is imported lazily so that the baseline, the tests and the data
pipeline work on machines without TensorFlow.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


def build_baseline() -> Pipeline:
    """TF-IDF over word uni- and bigrams + L2 logistic regression.

    A strong, transparent reference point: a deep model is only worth its
    cost if it beats this.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                max_features=config.TFIDF_MAX_FEATURES,
                ngram_range=config.TFIDF_NGRAM_RANGE,
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=config.SEED)),
        ]
    )


def build_dnn_parts():
    """Vectoriser and multilayer perceptron for the main deep model.

    Architecture: TF-IDF (30 000 uni- and bigrams) -> Dense 256 ReLU ->
    Dense 64 ReLU -> sigmoid output, trained with Adam and L2 regularisation.
    The two parts are returned separately because training runs epoch by
    epoch with early stopping on the validation split (see train.fit_dnn).
    """
    from sklearn.neural_network import MLPClassifier

    vectorizer = TfidfVectorizer(
        max_features=config.DNN_MAX_FEATURES,
        ngram_range=config.DNN_NGRAM_RANGE,
        min_df=3,
        sublinear_tf=True,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=config.DNN_HIDDEN,
        activation="relu",
        solver="adam",
        alpha=config.DNN_ALPHA,
        batch_size=config.DNN_BATCH,
        learning_rate_init=config.DNN_LR,
        random_state=config.SEED,
    )
    return vectorizer, mlp


def build_bilstm(train_texts):
    """Embedding -> BiLSTM -> Dense classifier with text vectorisation inside.

    Putting TextVectorization inside the model means the saved .keras file
    takes raw (cleaned) strings, so training and serving cannot drift apart.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(config.SEED)

    vectorizer = layers.TextVectorization(
        max_tokens=config.VOCAB_SIZE,
        output_sequence_length=config.MAX_LEN,
        standardize=None,              # text is already cleaned by src.data.clean_text
        name="vectorizer",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(list(train_texts)).batch(1024))

    inputs = keras.Input(shape=(1,), dtype="string", name="text")
    x = vectorizer(inputs)
    x = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, mask_zero=True, name="embedding")(x)
    x = layers.Bidirectional(layers.LSTM(config.LSTM_UNITS), name="bilstm")(x)
    x = layers.Dropout(config.DROPOUT)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_fake")(x)

    model = keras.Model(inputs, outputs, name="bilstm_fake_news")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


def to_tensor(texts):
    """Shape a list of strings as the (n, 1) string tensor the BiLSTM expects."""
    import tensorflow as tf
    return tf.constant([[str(t)] for t in texts])
