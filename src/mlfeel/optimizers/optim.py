from abc import ABC, abstractmethod

import numpy as np


class Optimzer(ABC):

    @abstractmethod
    def step(self, params: list[dict[str, np.ndarray]]):
        pass
