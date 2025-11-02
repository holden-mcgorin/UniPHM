from abc import abstractmethod, ABC

from uniphm.data.Dataset import Dataset
from uniphm.engine.Result import Result


class ABCMetric(ABC):
    """
    所有评价指标的抽象父类
    所有子类必须完成下列功能
    1. 返回评价指标的名称
    2. 完成评价指标的具体计算方法，返回评价结果
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        定义此评价指标的名称
        :return: 此评价指标的名称
        """
        pass

    @property
    @abstractmethod
    def is_higher_better(self) -> bool:
        pass

    @abstractmethod
    def value(self, test_set: Dataset, result: Result) -> float:
        """
        获取评估得到的原始数值
        :param test_set:
        :param result:
        :return:
        """
        pass

    def format(self, value: float) -> str:
        """
        对原始数值格式化
        :param value:
        :return:
        """
        return str(f'{value:.4f}')

    def __call__(self, test_set: Dataset, result: Result) -> str:
        """
        此评价指标的计算方法
        :return: 评价指标字符串（数字、区间、百分比...）
        """
        return self.format(self.value(test_set, result))
