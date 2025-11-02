import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.Result import Result
from uniphm.engine.metric.ABCMetric import ABCMetric


class MAE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y
        return float(np.mean(np.abs(r - r_hat)))
