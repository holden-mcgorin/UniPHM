import numpy as np
from numpy import ndarray

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class PTPProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "PTP"

    def _compute(self, window: ndarray) -> float:
        return np.ptp(window)
