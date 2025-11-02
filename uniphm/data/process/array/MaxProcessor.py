import numpy as np
from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class MaxProcessor(WindowedProcessor):
    @property
    def name(self):
        return "Max"

    def _compute(self, window):
        return np.max(window)
