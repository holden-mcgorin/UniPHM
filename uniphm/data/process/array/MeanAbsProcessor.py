import numpy as np
from numpy import ndarray

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class MeanAbsProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Mean_abs"

    def _compute(self, window: ndarray) -> float:
        return np.mean(np.abs(window))
