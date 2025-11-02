import numpy as np
from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class MeanProcessor(WindowedProcessor):
    @property
    def name(self):
        return "Mean"

    def _compute(self, window):
        return np.mean(window)
