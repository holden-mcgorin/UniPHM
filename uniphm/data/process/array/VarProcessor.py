import numpy as np
from numpy import ndarray

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class VarProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Var"

    def _compute(self, window: ndarray) -> float:
        return np.var(window)
