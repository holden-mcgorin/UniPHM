from abc import ABC, abstractmethod

from numpy import ndarray


class ABCBaseProcessor(ABC):
    """
    所有数据处理器的抽象基类
    作用：将向量转为另一个向量
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        定义此评价指标的名称
        :return: 此评价指标的名称
        """
        pass

    @abstractmethod
    def run(self, source: ndarray) -> ndarray:
        """
        对 ndarray 进行数据处理
        :param source:
        :return:
        """
        raise NotImplementedError

    def __call__(self, source: ndarray) -> ndarray:
        return self.run(source)
