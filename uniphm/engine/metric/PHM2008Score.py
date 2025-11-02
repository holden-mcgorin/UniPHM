import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.metric.ABCMetric import ABCMetric
from uniphm.engine.Result import Result


class PHM2008Score(ABCMetric):
    """
    别名：PHM2008 Score、NASA Score
    """
    @property
    def name(self) -> str:
        return 'PHM2008Score'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat.reshape(-1)
        r = test_set.y.reshape(-1)

        # 初始化 Score
        score = 0.0

        # 计算每个 S_i 并累加
        for i in range(len(r_hat)):
            if r_hat[i] >= r[i]:
                s = np.exp((r_hat[i] - r[i]) / 10) - 1
            else:
                s = np.exp((r[i] - r_hat[i]) / 13) - 1

            score += s

        return float(score)
