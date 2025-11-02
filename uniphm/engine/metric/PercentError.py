import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.metric.ABCMetric import ABCMetric
from uniphm.engine.Result import Result


class PercentError(ABCMetric):
    @property
    def name(self) -> str:
        return 'PercentError'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y

        r = np.sum(r)
        r_hat = np.sum(r_hat)

        # 计算百分比误差
        percent_error = (r - r_hat) / r

        return float(np.mean(percent_error) * 100)

    def value2(self, test_set: Dataset, result: Result) -> float:
        """
        老版计算百分误差，计算所有样本的百分比误差，再取平均
        :param test_set:
        :param result:
        :return:
        """
        r_hat = result.y_hat
        r = test_set.y

        # 去掉真实值为0的项
        zero_indices = np.where(r == 0)
        r = np.delete(r, zero_indices)
        r_hat = np.delete(r_hat, zero_indices)

        # 计算百分比误差
        percent_error = (r - r_hat) / r

        return float(np.mean(percent_error) * 100)

    def format(self, value: float) -> str:
        return f"{value:.2f}%"
