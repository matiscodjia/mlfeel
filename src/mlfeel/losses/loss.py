from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def _func(self, pred: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def _deriv(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        # Grad_out = ∂L/∂L = 1
        return self._deriv(self.pred, self.target)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred: np.ndarray = pred
        self.target: np.ndarray = target
        return self._func(pred, target)
