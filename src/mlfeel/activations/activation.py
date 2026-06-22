from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override

from mlfeel.model import Module


class Activation(Module, ABC):
    @abstractmethod
    def _func(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _deriv(self, X: np.ndarray) -> np.ndarray:
        pass

    @override
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._deriv(self.entry)

    @override
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.entry: np.ndarray = X
        return self._func(X)

    @override
    def parameters(self) -> list[dict[str, np.ndarray]]:
        return []


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha: float = alpha

    @override
    def _func(self, X: np.ndarray) -> np.ndarray:
        # np.where(condition, valeur_si_vrai, valeur_si_faux)
        return np.where(X > 0, X, self.alpha * X)

    @override
    def _deriv(self, X: np.ndarray) -> np.ndarray:
        # Renvoie 1.0 là où X > 0, et alpha partout ailleurs
        return np.where(X > 0, 1.0, self.alpha).astype(X.dtype)
