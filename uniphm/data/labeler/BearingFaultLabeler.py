import numpy as np
from typing import Union, List
from uniphm.data.Entity import Entity
from uniphm.data.labeler.ABCLabeler import ABCLabeler
from uniphm.data.Dataset import Dataset
from uniphm.data.loader.XJTULoader import Fault
from uniphm.data.process.array.SlideWindowProcessor import SlideWindowProcessor
from uniphm.util.Logger import Logger


class BearingFaultLabeler(ABCLabeler):
    @property
    def name(self):
        return 'fault'

    def __init__(self, window_size: int, fault_types: list, window_step=None,
                 is_multi_hot=True, is_from_fpt=False, is_squeeze=False):

        self.window_size = window_size
        self.window_step = window_step if window_step is not None else window_size
        self.fault_types = fault_types
        self.is_multi_hot = is_multi_hot
        self.is_from_fpt = is_from_fpt
        self.is_squeeze = is_squeeze
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    def _label(self, bearing: Entity, key: Union[str, List[str]]) -> Dataset:
        cls_name = self.__class__.__name__
        Logger.info(f"[{cls_name}] -> Labeling entity '{bearing.name}', key '{key}'")

        # ---------------- 获取数据 ----------------
        if isinstance(key, str):
            data = bearing[key]
        else:  # key 是列表
            data_list = [bearing[k] for k in key]
            # 拼接成 (n, m)
            data = np.hstack(data_list)

        # 保证 (n,1) 形式
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        data_size = data.shape[0]
        life = bearing['life']
        fpt = bearing['fpt']
        fpt_index_x = int(data_size * (fpt / life))
        Logger.debug(f"[{cls_name}] Data shape: {data.shape}, life: {life}, fpt_index_x: {fpt_index_x}")

        # ---------------- 生成 x 和 z ----------------
        if self.is_from_fpt:
            data_segment = data[fpt_index_x:, :]
            x = self.sliding_window(data_segment)
            sample_size = (data_size - fpt_index_x) // self.window_size
            z = np.linspace(fpt, life, sample_size).reshape(-1, 1)
            Logger.debug(f"[{cls_name}] Using degradation data, x shape: {x.shape}, z shape: {z.shape}")
        else:
            x = self.sliding_window(data)
            sample_size = data_size // self.window_size
            z = np.linspace(0, life, sample_size).reshape(-1, 1)
            Logger.debug(f"[{cls_name}] Using full-life data, x shape: {x.shape}, z shape: {z.shape}")

        if self.is_squeeze and x.shape[-1] == 1:
            x = x.squeeze(axis=-1)
            Logger.debug(f"[{cls_name}] Squeezed x to shape: {x.shape}")

        # ---------------- 生成 y ----------------
        try:
            normal_index = self.fault_types.index(Fault.NC)
        except ValueError:
            normal_index = None

        fault_indices = [i for i, e in enumerate(self.fault_types) if e in bearing['fault_type']]
        num_type = len(self.fault_types)
        num_normal = fpt_index_x // self.window_size
        num_fault = sample_size - num_normal

        if self.is_multi_hot:
            y_normal = np.zeros((num_normal, num_type))
            if normal_index is not None:
                y_normal[:, normal_index] = 1
            y_fault = np.zeros((num_fault, num_type))
            y_fault[:, fault_indices] = 1
        else:
            y_normal = np.zeros((num_normal, 1))
            if normal_index is not None:
                y_normal[:, 0] = normal_index
            y_fault = np.zeros((num_fault, 1))
            y_fault[:, 0] = fault_indices[0]

        y = y_fault if y_normal is None or self.is_from_fpt else np.vstack((y_normal, y_fault))
        Logger.info(f"[{cls_name}] Finished labeling, final y shape: {y.shape}")

        return Dataset(x=x, y=y, z=z, name=bearing.name)
