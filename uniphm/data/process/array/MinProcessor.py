import numpy as np
from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class MinProcessor(WindowedProcessor):
    @property
    def name(self):
        return "Min"

    def _compute(self, window):
        return np.min(window)
