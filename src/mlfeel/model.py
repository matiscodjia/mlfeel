from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override


class Module(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pass


class Linear(Module):
    weights: np.ndarray
    bias: np.ndarray
    entry: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    dx: np.ndarray

    def __init__(self, input_features: int, output_features: int):
        self.weights = np.random.randn(output_features, input_features)
        self.bias = np.random.randn(output_features, 1)

    @override
    def forward(self, X: np.ndarray):
        self.entry = X
        return (
            self.weights @ X + self.bias
        )  # w(out in) @ (in, N) + (out, 1) -> (out, N)

    @override
    def backward(self, grad_out: np.ndarray):
        self.dW = grad_out @ self.entry.T
        # grad_out(out, N) @ entry.T(N, in) -> (out, in)
        self.db = grad_out.sum(axis=1, keepdims=True)  # -> (out, 1)
        self.dx = (
            grad_out.T @ self.weights
        )  # grad_out.T(N, out) @ w(out, in) -> (N, in)
        return self.dx

    @override
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.weights, self.bias)]


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
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class ReLU(Activation):
    @override
    def _func(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @override
    def _deriv(self, X: np.ndarray) -> np.ndarray:
        return (X > 0).astype(X.dtype)


class Sequential(Module):
    layers: list[Module]

    def __init__(self, modules: list[Module]):
        self.layers = modules

    @override
    def forward(self, X: np.ndarray) -> np.ndarray:
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    @override
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in self.layers:
            grad = layer.backward(grad)
        return grad

    @override
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            params.append(
                layer.parameters()[0]
            )  # chaque couche renvoie une liste de tuple
        return params
