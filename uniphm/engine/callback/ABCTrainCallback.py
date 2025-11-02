from abc import ABC, abstractmethod
from typing import Union, List



class ABCTrainCallback(ABC):
    """
    该类及实现类不建议在实验过程中重复使用，有可能产生意想不到的冲突
    """

    def on_train_begin(self, model) -> bool:
        return True

    def on_epoch_begin(self, model, epoch: int) -> bool:
        return True

    def on_epoch_end(self, model, epoch: int, avg_loss: Union[float, List[float]]) -> bool:
        return True

    def on_train_end(self, model) -> bool:
        return True

    def __str__(self):
        class_name = self.__class__.__name__
        # fields = [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
        # return f"{class_name}({', '.join(fields)})"
        return class_name
