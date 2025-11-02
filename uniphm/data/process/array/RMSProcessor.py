import numpy as np
from numpy import ndarray

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class RMSProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "RMS"

    def _compute(self, window: ndarray) -> float:
        return np.sqrt(np.mean(np.square(window)))
