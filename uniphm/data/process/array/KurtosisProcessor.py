from numpy import ndarray
from scipy.stats import kurtosis

from uniphm.data.process.array.WindowedProcessor import WindowedProcessor


class KurtosisProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Kurtosis"

    def _compute(self, window: ndarray) -> float:
        return kurtosis(window)
