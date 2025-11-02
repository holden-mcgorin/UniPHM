import numpy as np
from numpy import ndarray
from scipy import interpolate

from uniphm.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from uniphm.util.Logger import Logger


class SlideWindowProcessor(ABCBaseProcessor):
    """
    使用滑动窗口处理后会多一个维度在最外部（轴0）
    """

    def __init__(self, window_size: int, window_step: int = 1, time_axis: int = 0):
        """
        :param window_size: 用于滑动窗口的区间大小
        :param window_step: 滑动窗口的步长，默认为1
        :param time_axis: 指定时间轴，沿哪个轴进行滑动窗口操作，默认为第0轴
        """
        self.window_size = window_size
        self.window_step = window_step
        self.time_axis = time_axis

    @property
    def name(self) -> str:
        return 'SlideWindow'

    def run(self, source: ndarray) -> ndarray:
        # 如果窗口大小小于等于0则跳过
        if self.window_size <= 0:
            source = np.expand_dims(source, axis=0)
            return source

        # 将时间轴移到最前面
        source = np.moveaxis(source, self.time_axis, 0)

        # 当数据长度小于窗口长度时，采用B样条插值
        if self.window_size > source.shape[0]:
            output = self.bspline_expand(source, self.window_size)
            Logger.warning(f'The data length [{source.shape[0]}] is less than the window length [{self.window_size}],'
                           ' and B-spline interpolation method has been used to expand the data length')
            return np.moveaxis(output, 0, self.time_axis)

        # 计算滑动窗口的数量num_windows,初始化存储窗口数据的数组 output
        num_windows = (source.shape[0] - self.window_size) // self.window_step + 1
        new_shape = (num_windows, self.window_size) + source.shape[1:]
        output = np.zeros(new_shape)

        # 遍历所有窗口以输入数据
        for i in range(num_windows):
            start_index = i * self.window_step
            end_index = start_index + self.window_size
            output[i] = source[start_index:end_index]

        # 将轴移回原来的位置
        return np.moveaxis(output, 0, self.time_axis)

    @staticmethod
    def bspline_expand(data, window_size) -> ndarray:
        """
        使用B样条插值将输入数据扩展为指定长度。
        """
        sensors = []
        for s in range(data.shape[1]):  # 遍历传感器
            x = np.linspace(0, window_size - 1, data.shape[0])
            x_new = np.linspace(0, window_size - 1, window_size)
            tck = interpolate.splrep(x, data[:, s])
            y_new = interpolate.splev(x_new, tck)
            sensors.append(y_new.tolist())
        output = np.array(sensors).T
        return output[np.newaxis, :]
