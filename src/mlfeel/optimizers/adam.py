import numpy as np
from typing_extensions import override

from mlfeel.optimizers.optim import Optimzer


class Adam(Optimzer):
    def __init__(self, learning_rate: float, beta_momentum: float, beta_rms: float):
        self.t: int = 0
        self.beta_1: float = beta_momentum
        self.beta_2: float = beta_rms
        self.lr: float = learning_rate
        self.state: dict[int, dict[str, np.ndarray]] = {}

    @override
    def step(self, params: list[dict[str, np.ndarray]]):
        EPS = 10e-8
        self.t += 1
        for param_dict in params:
            parameter = param_dict["param"]
            param_grad = param_dict["grad"]

            pid = id(parameter)
            if pid not in self.state:
                self.state[pid] = {
                    "m": np.zeros_like(parameter),
                    "v": np.zeros_like(parameter),
                }
            m = self.state[pid]["m"]
            v = self.state[pid]["v"]

            m_current = self.beta_1 * m + (1 - self.beta_1) * param_grad
            v_current = self.beta_2 * v + (1 - self.beta_2) * (param_grad**2)
            m_current_corrected = m_current / (1 - self.beta_1**self.t)
            v_current_corrected = v_current / (1 - self.beta_2**self.t)
            self.state[pid]["m"] = m_current
            self.state[pid]["v"] = v_current

            parameter -= (
                self.lr * m_current_corrected / (np.sqrt(v_current_corrected) + EPS)
            )
