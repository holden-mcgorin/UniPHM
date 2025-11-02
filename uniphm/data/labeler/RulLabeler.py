import numpy as np

from uniphm.data import Dataset
from uniphm.data.Entity import Entity
from uniphm.data.labeler.ABCLabeler import ABCLabeler
from uniphm.data.process.array.SlideWindowProcessor import SlideWindowProcessor


class RulLabeler(ABCLabeler):
    """
    通用RUL打标器
    既适用于涡扇发动机（C-MAPSS），也适用于轴承数据集。
    通过参数选择不同的RUL生成策略。
    """

    def __init__(self, window_size: int, window_step: int = None,
                 is_normalized: bool = False,  # False: linear (涡扇发动机风格) / True: normalized (轴承风格)
                 max_rul: int = -1,
                 is_from_fpt: bool = False,
                 is_rectified: bool = True,
                 is_squeeze: bool = False,
                 last_sample: bool = False):
        """
        :param window_size: 滑动窗口长度
        :param window_step: 滑动窗口步长
        :param is_normalized: 是否使用 normalized 风格 (轴承风格)
        :param max_rul: RUL最大值，linear模式下生效
        :param is_from_fpt: 是否仅从FPT后生成数据（轴承常用）
        :param is_rectified: 是否对FPT之前RUL固定为1（轴承常用）
        :param is_squeeze: 是否去掉特征维度长度为1的轴
        :param last_sample: 是否只取最后一个样本（涡扇发动机常用）
        """
        self.window_size = window_size
        self.window_step = window_step if window_step is not None else window_size
        self.is_normalized = is_normalized
        self.max_rul = max_rul
        self.last_sample = last_sample
        self.is_from_fpt = is_from_fpt
        self.is_rectified = is_rectified
        self.is_squeeze = is_squeeze
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, entity: Entity, key: str) -> Dataset:
        data = entity[key].values
        data_size = data.shape[0]

        # =====================
        # 1. 滑窗数据
        # =====================
        if self.is_from_fpt and "fpt" in entity and "life" in entity:
            # 针对轴承: 从FPT后取
            life = entity["life"]
            fpt = entity["fpt"]
            fpt_index = int(data_size * (fpt / life))
            data = data[fpt_index:, :]
            x = self.sliding_window(data)
            # 时间轴：从 fpt 到 life 均匀分布
            z = np.linspace(fpt, life, x.shape[0]).reshape(-1, 1)
        else:
            # 全数据
            x = self.sliding_window(data)
            z = np.linspace(0, entity["life"], x.shape[0]).reshape(-1, 1)

        # 发动机 cycle 数据从 1 开始计数
        if entity.meta.get('time_unit') == 'cycle':
            z = np.arange(x.shape[0]).reshape(-1, 1) + self.window_size

        if self.is_squeeze:
            x = x.squeeze()

        # =====================
        # 2. 标签生成
        # =====================
        if not self.is_normalized:
            y = np.arange(x.shape[0], 0, -1).reshape(-1, 1) + entity['rul'] - 1
            if self.max_rul > 0:
                y[y > self.max_rul] = self.max_rul
        else:
            # 轴承风格：归一化到 [1→0]
            life = entity.meta.get("life", data_size)
            fpt = entity.meta.get("fpt", 0)
            if self.is_rectified and fpt > 0:
                fpt_index = int(fpt / life * x.shape[0])
                y1 = np.ones((fpt_index, 1))
                y2 = np.linspace(1, 0, x.shape[0] - fpt_index).reshape(-1, 1)
                y = np.vstack((y1, y2))
            else:
                y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)

        # =====================
        # 3. 是否只取最后样本
        # =====================
        if self.last_sample:
            x, y, z = x[np.newaxis, -1], y[np.newaxis, -1], z[np.newaxis, -1]

        return Dataset(x=x, y=y, z=z, name=entity.name)
