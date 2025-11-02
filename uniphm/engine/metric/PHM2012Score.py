import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.metric.ABCMetric import ABCMetric
from uniphm.engine.Result import Result


class PHM2012Score(ABCMetric):

    @property
    def name(self) -> str:
        return 'PHM2012Score'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y

        # 去掉真实值为0的项
        zero_indices = np.where(r == 0)
        r = np.delete(r, zero_indices)
        r_hat = np.delete(r_hat, zero_indices)

        # 计算百分比误差
        percent_error = (r - r_hat) * 100 / r

        # 计算分数
        scores = np.where(percent_error <= 0,
                          np.exp(-np.log(0.5) * (percent_error / 5)),
                          np.exp(np.log(0.5) * (percent_error / 20)))

        return float(np.mean(scores))
