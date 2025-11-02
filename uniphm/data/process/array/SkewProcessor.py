from numpy import ndarray

from scipy.stats import skew

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class SkewProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Skew"

    def _compute(self, window: ndarray) -> float:
        return skew(window)
