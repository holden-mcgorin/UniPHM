import numpy as np
from numpy import ndarray

from uniphm.data.process.array.ABCBaseProcessor import ABCBaseProcessor


class AverageProcessor(ABCBaseProcessor):
    def __init__(self, window_size):
        self.window_size = window_size

    @property
    def name(self) -> str:
        return 'Average'

    def run(self, source: ndarray) -> ndarray:
        if self.window_size <= 0:
            raise ValueError("Window size must be greater than 0")
        if self.window_size > len(source):
            raise ValueError("Window size must be less than or equal to the length of the array")

        # 使用 np.cumsum 计算累积和，再用滑动窗口大小的差分来计算移动平均
        cumsum = np.cumsum(np.insert(source, 0, 0))
        moving_avg = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / self.window_size
        return moving_avg
