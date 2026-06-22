import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from typing_extensions import override

from mlfeel.losses.loss import Loss
from mlfeel.optimizers.adam import Adam


class Module(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def parameters(
        self,
    ) -> list[dict[str, np.ndarray]]:
        pass


class Linear(Module):
    weights: np.ndarray
    bias: np.ndarray
    entry: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    dx: np.ndarray

    def __init__(self, input_features: int, output_features: int):
        self.weights = np.random.randn(input_features, output_features)
        self.bias = np.random.randn(1, output_features)
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    @override
    def forward(self, X: np.ndarray):
        self.entry = X
        return X @ self.weights + self.bias
        # (N, in) @ w(in, out) = (N, out) + (1, out) -> (N, out)

    @override
    def backward(self, grad_out: np.ndarray):
        self.dW[:] = self.entry.T @ grad_out  # (in, N) @ (N, out) = (in, out)

        self.db[:] = grad_out.sum(axis=0, keepdims=True)  # -> (1, out)
        self.dx = (
            grad_out @ self.weights.T
        )  # grad_out (N, out) @ w.T (out, in) -> (N, in)
        return self.dx

    @override
    def parameters(self) -> list[dict[str, np.ndarray]]:
        return [
            {"param": self.weights, "grad": self.dW},
            {
                "param": self.bias,
                "grad": self.db,
            },
        ]


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
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    @override
    def parameters(self) -> list[dict[str, np.ndarray]]:
        params: list[dict[str, np.ndarray]] = []
        for layer in self.layers:
            params += (
                layer.parameters()
            )  # chaque couche renvoie une liste de dictionnaires
        return params


logger = logging.getLogger(__name__)


def train_model(
    model: Sequential,
    loss_fn: Loss,
    optimizer: Adam,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
) -> list[float]:
    loss_history: list[float] = []
    n_samples: int = cast(int, X_train.shape[0])

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            current_Xbatch = X_shuffled[start_idx:end_idx]
            current_ybatch = y_shuffled[start_idx:end_idx]

            # Forward
            predictions = model.forward(current_Xbatch)

            # Compute loss
            loss = loss_fn.forward(predictions, current_ybatch)
            epoch_loss += loss
            num_batches += 1
            logger.debug(f"Batch {num_batches} - Loss locale: {loss:.6f}")
            # Backward
            grad_out = loss_fn.backward()
            _ = model.backward(grad_out)

            optimizer.step(model.parameters())
        average_epoch_loss = epoch_loss / num_batches
        loss_history.append(average_epoch_loss)
        logger.info(
            f"Époque {epoch + 1}/{epochs} - Perte Moyenne : {average_epoch_loss:.6f}"
        )

    return loss_history
