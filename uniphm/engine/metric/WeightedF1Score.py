import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.Result import Result
from uniphm.engine.metric.ABCMetric import ABCMetric


class WeightedF1Score(ABCMetric):
    @property
    def name(self) -> str:
        return 'WeightedF1Score'

    @property
    def is_higher_better(self) -> bool:
        return True

    def value(self, test_set: Dataset, result: Result) -> float:
        from sklearn.metrics import f1_score

        # 若为分类标签
        if test_set.y.shape[1] == 1:
            y_pred = np.argmax(result.y_hat, axis=1).reshape(-1)  # 找出每行最大值的下标
            y_true = test_set.y.reshape(-1)

        # 若为独热标签
        else:
            y_true = test_set.y.astype(int)
            y_pred = (result.y_hat > 0.5).astype(int)

        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        return float(weighted_f1)
