import numpy as np

from uniphm.data.Dataset import Dataset
from uniphm.engine.Result import Result
from uniphm.engine.metric.ABCMetric import ABCMetric


class Accuracy(ABCMetric):
    @property
    def name(self) -> str:
        return 'Accuracy'

    @property
    def is_higher_better(self) -> bool:
        return True

    def value(self, test_set: Dataset, result: Result) -> float:

        # 若为分类标签
        if test_set.y.shape[1] == 1:
            y_true = test_set.y
            y_pred = np.argmax(result.y_hat, axis=1).reshape(-1, 1)  # 找出每行最大值的下标

        # 若为独热标签
        else:
            y_true = test_set.y
            y_pred = (result.y_hat > 0.5).astype(int)

        equal_rows = np.all(y_pred == y_true, axis=1)
        correct_predictions = np.sum(equal_rows)
        return correct_predictions / test_set.y.shape[0]
