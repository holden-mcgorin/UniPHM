from abc import abstractmethod

import numpy as np
from numpy import ndarray

from uniphm.data.process.array.ABCBaseProcessor import ABCBaseProcessor


class WindowedProcessor(ABCBaseProcessor):
    """基于滑动窗口的统计量计算模板"""

    def __init__(self, window_size: int, window_step: int = -1):
        self.window_size = window_size
        self.window_step = window_step if window_step != -1 else window_size

    def _sliding_windows(self, source: ndarray) -> ndarray:
        num_windows = (len(source) - self.window_size) // self.window_step + 1
        result = np.zeros(num_windows)
        for i in range(num_windows):
            start_idx = i * self.window_step
            end_idx = start_idx + self.window_size
            window = source[start_idx:end_idx]
            result[i] = self._compute(window)
        return result

    @abstractmethod
    def _compute(self, window: ndarray) -> float:
        """计算单个窗口的特征"""
        raise NotImplementedError

    def run(self, source: ndarray) -> ndarray:
        return self._sliding_windows(source)
