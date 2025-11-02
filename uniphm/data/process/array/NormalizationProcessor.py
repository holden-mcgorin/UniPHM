import numpy as np
from numpy import ndarray

from uniphm.data.Dataset import Dataset
from uniphm.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from uniphm.engine.Result import Result
from uniphm.util.Logger import Logger


class NormalizationProcessor(ABCBaseProcessor):
    def __init__(self, mode='[0,1]',
                 arr_min=None, arr_max=None,
                 arr_mean=None, arr_std=None):
        self.arr_min = arr_min
        self.arr_max = arr_max
        self.arr_mean = arr_mean
        self.arr_std = arr_std
        self.mode = mode  # 归一化模式，可选项：'[0,1]'、'[-1,1]'、'z-score','None'

    @property
    def name(self) -> str:
        return 'Normalization'

    def run(self, source: ndarray) -> ndarray:
        # 获取最小值和最大值
        return self.normalize(source)

    def auto_get_min_max(self, source):
        if self.arr_min is None:
            Logger.warning('[Normalization]  The minimum value of NormalizationProcessor is not specified and'
                           ' will be automatically obtained from the input')
            self.arr_min = np.min(source, axis=0)  # 每列的最小值

        if self.arr_max is None:
            Logger.warning('[Normalization]  The maximum value of NormalizationProcessor is not specified and'
                           ' will be automatically obtained from the input')
            self.arr_max = np.max(source, axis=0)  # 每列的最大值

    def auto_get_mean_std(self, source):
        if self.arr_mean is None:
            Logger.warning('[Normalization]  The mean value of NormalizationProcessor is not specified and'
                           ' will be automatically obtained from the input')
            self.arr_mean = np.mean(source, axis=0)  # 每列的最大值

        if self.arr_std is None:
            Logger.warning('[Normalization]  The std value of NormalizationProcessor is not specified and'
                           ' will be automatically obtained from the input')
            self.arr_std = np.std(source, axis=0)  # 每列的最大值

    def norm(self, obj):
        if isinstance(obj, ndarray):
            return self.normalize(obj)
        elif isinstance(obj, Dataset):
            return self.norm_label(obj)
        elif isinstance(obj, Result):
            return self.norm_result(obj)
        else:
            raise Exception(f'class {type(obj).__name__} does not support normalization or denormalization')

    def denorm(self, obj):
        if isinstance(obj, ndarray):
            return self.denormalize(obj)
        elif isinstance(obj, Dataset):
            return self.denorm_label(obj)
        elif isinstance(obj, Result):
            return self.denorm_result(obj)
        else:
            raise Exception(f'class {type(obj).__name__} does not support normalization or denormalization')

    def normalize(self, x: ndarray):
        if self.mode == '[0,1]':
            self.auto_get_min_max(x)
            x_norm = (x - self.arr_min) / (self.arr_max - self.arr_min)
        elif self.mode == '[-1,1]':
            self.auto_get_min_max(x)
            x_norm = 2 * (x - self.arr_min) / (self.arr_max - self.arr_min) - 1
        elif self.mode == 'z-score':
            self.auto_get_mean_std(x)
            x_norm = (x - self.arr_mean) / self.arr_std
        else:  # None
            Logger.warning('[Normalization]  the normalization mode of NormalizationProcessor is not specified,'
                           ' the original data will be automatically returned')
            return x
        return x_norm

    def denormalize(self, x_norm: ndarray):
        if self.mode == '[0,1]':
            x = x_norm * (self.arr_max - self.arr_min) + self.arr_min
        elif self.mode == '[-1,1]':
            x = ((x_norm + 1) / 2) * (self.arr_max - self.arr_min) + self.arr_min
        elif self.mode == 'z-score':
            x = x_norm * self.arr_mean + self.arr_std
        else:  # None
            Logger.warning('[Normalization]  the normalization mode of NormalizationProcessor is not specified,'
                           ' the original data will be automatically returned')
            return x_norm
        return x

    def norm_label(self, dataset: Dataset) -> Dataset:
        new_dataset = Dataset(x=dataset.x, y=self.normalize(dataset.y), z=dataset.y, label_map=dataset.label_map,
                              name=dataset.name, entity_map=dataset.entity_map)
        return new_dataset

    def norm_result(self, result: Result) -> Result:
        new_result = result.__copy__()
        new_result.y_hat = self.normalize(new_result.y_hat)
        return new_result

    def denorm_label(self, dataset: Dataset) -> Dataset:
        new_dataset = Dataset(x=dataset.x, y=self.denormalize(dataset.y), z=dataset.y, label_map=dataset.label_map,
                              name=dataset.name, entity_map=dataset.entity_map)
        return new_dataset

    def denorm_result(self, result: Result) -> Result:
        new_result = result.__copy__()
        new_result.y_hat = self.denormalize(new_result.y_hat)
        return new_result

    def __str__(self):
        return f'NormalizationProcessor(min={self.arr_min}, max={self.arr_max}, mean={self.arr_mean}, std={self.arr_std}, mode={self.mode})'
