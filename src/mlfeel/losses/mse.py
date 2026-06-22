from typing import cast

import numpy as np
from typing_extensions import override

from mlfeel.losses.loss import Loss


class MSE(Loss):
    @override
    def _func(self, pred: np.ndarray, target: np.ndarray) -> float:
        N = cast(float, pred.shape[0])
        error_vector = target - pred
        return ((np.linalg.norm(error_vector) ** 2) * 0.5).item() / N

    @override
    def _deriv(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        N = cast(float, pred.shape[0])
        return (pred - target) / N
