import logging
from typing import cast

import numpy as np
import pytest

from mlfeel.activations.activation import LeakyReLU
from mlfeel.losses.ce import CrossEntropy
from mlfeel.model import Linear, Sequential, train_model
from mlfeel.optimizers.adam import Adam
from mlfeel.vectorizer import TF_IDF_Vectorizer

logger = logging.getLogger(__name__)


def _one_hot(labels: list[int], num_classes: int) -> np.ndarray:
    """Encode des labels entiers (N,) au format one-hot (N, C)."""
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    for row, label in enumerate(labels):
        encoded[row, label] = 1.0
    return encoded


@pytest.mark.model
def test_sentiment_classification_training() -> None:
    """
    Entraîne un MLP à classer des critiques en positif/négatif à partir
    de features TF-IDF, en utilisant la perte CrossEntropy.

    Valide que :
      - la perte décroît bien au fil des époques (l'optimiseur apprend),
      - le modèle sépare correctement les deux classes (accuracy parfaite
        sur un corpus jouet linéairement séparable).
    """
    np.random.seed(42)  # Déterminisme

    # 1. Corpus jouet : critiques positives (label 1) et négatives (label 0)
    positives = [
        "Un film excellent et captivant",
        "Une oeuvre magnifique et brillante",
        "Acteurs geniaux, scenario captivant",
        "Magnifique, excellent, je recommande",
        "Brillant et captivant du debut a la fin",
    ]
    negatives = [
        "Un film horrible et ennuyeux",
        "Une oeuvre ratee et mediocre",
        "Acteurs mauvais, scenario ennuyeux",
        "Horrible, mediocre, je deconseille",
        "Rate et ennuyeux du debut a la fin",
    ]

    corpus = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    # 2. Vectorisation TF-IDF
    vectorizer = TF_IDF_Vectorizer()
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus).astype(np.float32)
    y_train = _one_hot(labels, num_classes=2)

    assert X_train.shape == (len(corpus), vectorizer.dim)
    assert y_train.shape == (len(corpus), 2)

    # 3. Graphe de calcul : MLP -> 2 logits (une par classe)
    model = Sequential(
        [
            Linear(input_features=vectorizer.dim, output_features=16),
            LeakyReLU(alpha=0.01),
            Linear(input_features=16, output_features=2),
        ]
    )

    loss_fn = CrossEntropy()
    optimizer = Adam(learning_rate=0.05, beta_momentum=0.9, beta_rms=0.999)

    # 4. Entraînement
    epochs = 200
    losses = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=4,
    )

    initial_loss = losses[0]
    final_loss = losses[-1]
    logger.info(
        f"Sentiment - Perte Initiale: {initial_loss:.6f} -> Finale: {final_loss:.6f}"
    )

    # 5. Assertions sur l'apprentissage
    assert final_loss < initial_loss, "La perte CrossEntropy n'a pas décru."
    assert final_loss < 0.1, f"Convergence insuffisante, loss finale: {final_loss:.6f}"

    # 6. Accuracy sur le corpus d'entraînement
    logits = model.forward(X_train)
    predictions = cast(np.ndarray, np.argmax(logits, axis=1))
    expected = cast(np.ndarray, np.argmax(y_train, axis=1))
    matches = cast(np.ndarray, predictions == expected)
    accuracy = cast(np.floating, matches.sum()).item() / matches.size
    logger.info(f"Sentiment - Accuracy entraînement: {accuracy:.2%}")

    assert accuracy == 1.0, f"Le classifieur n'a pas tout appris (acc={accuracy:.2%})."
