import numpy as np
from typing import Union, List
from uniphm.data.Entity import Entity
from uniphm.data.labeler.ABCLabeler import ABCLabeler
from uniphm.data.Dataset import Dataset
from uniphm.data.process.array.SlideWindowProcessor import SlideWindowProcessor


class TurbofanRulLabeler(ABCLabeler):

    def __init__(self, window_size, window_step=1, max_rul=-1, last_sample: bool = False):
        """
        涡扇发动机数据打标器
        对于涡扇发动机数据集来说，评价指标通常只评价整个发动机的RUL，而不是滑动窗口生成的所有切片，
        对于常使用的score指标来说，测试集越多分数会越大，因此在某些情况下需要仅取每个发动机最后一个样本作为测试集评价预测结果
        :param window_size: 窗口的大小
        :param window_step: 窗口的滑动步长
        :param max_rul: 最大rul值，在C-MAPSS数据集中常取130
        :param last_sample: 是否只取每个发动机最后一个样本（该选择对score评价指标至关重要）
        """
        self.window_size = window_size
        self.window_step = window_step
        self.max_rul = max_rul
        self.last_sample = last_sample
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, entity: Entity, key: Union[str, List[str]]) -> Dataset:
        # 将单列 key 转为列表
        keys = [key] if isinstance(key, str) else key
        # 合并多列数据，形成 (n, m) ndarray
        data_list = [entity[k] for k in keys]
        data = np.hstack(data_list) if len(data_list) > 1 else data_list[0]

        # 生成滑动窗口样本
        x = self.sliding_window(data)

        # 生成标签 y
        y = np.arange(x.shape[0], 0, -1).reshape(-1, 1) + entity['rul'] - 1
        y[y > self.max_rul] = self.max_rul

        # 生成 z
        z = np.arange(x.shape[0]) + self.window_size
        z = z.reshape(-1, 1)

        # 是否仅取最后一个样本
        if self.last_sample:
            x = x[np.newaxis, -1]
            y = y[np.newaxis, -1]
            z = z[np.newaxis, -1]

        return Dataset(x=x, y=y, z=z, name=entity.name)
