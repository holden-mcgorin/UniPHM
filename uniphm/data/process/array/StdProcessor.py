import numpy as np
from numpy import ndarray

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class StdProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Std"

    def _compute(self, window: ndarray) -> float:
        return np.std(window)
