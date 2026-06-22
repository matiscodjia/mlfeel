import logging

import numpy as np
import pytest

from mlfeel.model import MSE, Adam, LeakyReLU, Linear, Sequential, train_model

logger = logging.getLogger(__name__)


# Enregistrement d'un marqueur personnalisé auprès de pytest
@pytest.mark.model
def test_mlp_convergence() -> None:
    """
    Vérifie programmatiquement et visuellement que le réseau
    apprend une fonction logique simple (ici un XOR ou une séparation linéaire).
    """
    np.random.seed(42)  # Déterminisme

    # 1. Génération de données jouets (Batch_size=4, In_features=2)
    # Problème simple : prédire la somme des entrées
    X_train = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
    )
    y_train = np.array([[0.3], [0.7], [1.1], [1.5]], dtype=np.float32)

    # 2. Instanciation du graphe de calcul
    model = Sequential(
        [
            Linear(input_features=2, output_features=4),
            LeakyReLU(alpha=0.01),
            Linear(input_features=4, output_features=1),
        ]
    )

    loss_fn = MSE()
    optimizer = Adam(learning_rate=0.1, beta_momentum=0.9, beta_rms=0.999)

    # 3. Exécution de l'entraînement
    epochs = 50
    losses = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=2,
    )

    # 4. Assertions programmatiques (La machine valide le comportement)
    initial_loss = losses[0]
    final_loss = losses[-1]

    logger.info(
        f"Test Fin - Perte Initiale: {initial_loss:.6f} -> Perte Finale: {final_loss:.6f}"
    )

    assert (
        final_loss < initial_loss
    ), "L'optimiseur n'a pas fait décroître la fonction de coût."
    assert (
        final_loss < 1e-2
    ), f"La convergence est insuffisante, loss finale: {final_loss:.6f}"
