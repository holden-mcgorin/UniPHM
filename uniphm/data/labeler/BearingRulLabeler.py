from typing import Union, List

import numpy as np

from uniphm.data.Entity import Entity
from uniphm.data.labeler.ABCLabeler import ABCLabeler
from uniphm.data.Dataset import Dataset
from uniphm.data.process.array.SlideWindowProcessor import SlideWindowProcessor


class BearingRulLabeler(ABCLabeler):
    """
    本框架数据集的x默认形状： (batch, time, features)
    数据的维度建议，维度转换在模型处设置
    RNN / LSTM / GRU / Transformer	(batch, time, features) ✅
    1D CNN	(batch, features, time) ✅
    """

    def __init__(self, window_size: int, window_step=None,
                 is_from_fpt=False, is_rectified=False, is_squeeze=False):
        """
        :param window_size:滑动窗口长度
        :param window_step:滑动窗口步长
        :param is_from_fpt:是否fpt后才开始生成数据
        :param is_rectified:fpt之前rul是否固定为1
        :param is_squeeze:压缩x中长度为1的轴（只有一个特征时去掉特征轴）,该操作决定在卷积网络中是否需要扩充通道轴
        """
        self.window_size = window_size
        self.window_step = window_step if window_step is not None else window_size
        self.is_from_fpt = is_from_fpt
        self.is_rectified = is_rectified
        self.is_squeeze = is_squeeze
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, bearing: Entity, key: Union[str, List[str]]) -> Dataset:
        # --- 原始 x, y, z 生成逻辑不变 ---
        # 处理 key
        if isinstance(key, str):
            data = np.array(bearing[key])
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        elif isinstance(key, list):
            arrays = []
            for k in key:
                arr = np.array(bearing[k])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            data = np.hstack(arrays)
        else:
            raise TypeError("key must be a str or a list of str")

        data_size = data.shape[0]
        life = bearing['life']

        if self.is_from_fpt:
            fpt = bearing['fpt']
            fpt_index_x = int(data_size * (fpt / life))
            data_segment = data[fpt_index_x:, :]
            x = self.sliding_window(data_segment)
            sample_size = (data_size - fpt_index_x) // self.window_size
            z = np.linspace(fpt, life, sample_size).reshape(-1, 1)
        else:
            x = self.sliding_window(data)
            sample_size = data_size // self.window_size
            z = np.linspace(0, life, sample_size).reshape(-1, 1)

        if self.is_squeeze and x.shape[-1] == 1:
            x = x.squeeze(axis=-1)

        if self.is_rectified and not self.is_from_fpt:
            fpt = bearing['fpt']
            fpt_index_x = int(data_size * (fpt / life))
            fpt_index_y = fpt_index_x // self.window_size
            y1 = np.ones((fpt_index_y, 1))
            y2 = np.linspace(1, 0, x.shape[0] - fpt_index_y).reshape(-1, 1)
            y = np.vstack((y1, y2))
        else:
            y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)

        return Dataset(x=x, y=y, z=z, name=bearing.name)
