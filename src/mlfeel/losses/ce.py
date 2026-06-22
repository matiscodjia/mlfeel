from typing import cast

import numpy as np
from typing_extensions import override

from mlfeel.losses.loss import Loss


class CrossEntropy(Loss):
    # On ajoute une annotation pour le typage strict de la matrice de probabilités
    softmax_preds: np.ndarray

    @override
    def _func(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        pred: Logits bruts sortant du modèle (N, C)
        target: Cibles au format One-Hot Encoding (N, C)
        """
        N = cast(float, pred.shape[0])

        # 1. Softmax stable : soustraction du max par ligne via le broadcasting
        max_logits = cast(np.ndarray, np.max(pred, axis=1, keepdims=True))
        exp_logits: np.ndarray = np.exp(pred - max_logits)
        self.softmax_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 2. Calcul de l'entropie croisée (on ajoute un EPS pour parer à log(0))
        EPS = 1e-15
        clipped_preds = np.clip(self.softmax_preds, EPS, 1.0 - EPS)
        total_loss = cast(np.floating, np.sum(target * np.log(clipped_preds)))
        loss_val = -total_loss.item() / N

        return loss_val

    @override
    def _deriv(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Renvoie la dérivée analytique directe par rapport aux logits bruts.
        """
        N = cast(float, pred.shape[0])
        # dL/dZ = (P - Y) / N
        return (self.softmax_preds - target) / N
