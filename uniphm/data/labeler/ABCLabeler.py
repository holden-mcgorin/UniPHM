from abc import ABC, abstractmethod
from typing import Union, List

from uniphm.data.Dataset import Dataset
from uniphm.data.Entity import Entity
from uniphm.util.Logger import Logger


class ABCLabeler(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _label(self, entity: Entity, key: Union[str, List[str]]) -> Dataset:
        raise NotImplementedError

    def label(self, entity: Entity, key: Union[str, List[str]]) -> Dataset:

        Logger.info(f"[{self.__class__.__name__}]  -> Generating dataset for entity: '{entity.name}', key: '{key}'")

        # 待实现的抽象方法
        dataset = self._label(entity, key)

        # 给数据集添加信息：名称、标签映射表、实体映射表
        if self.name is not None:
            dataset.label_map[self.name] = (0, dataset.y.shape[1])
        if entity.name is not None:
            dataset.entity_map[entity.name] = (0, dataset.x.shape[0])

        Logger.info(f"[{self.__class__.__name__}]  ✓ Finished labeling for entity: '{entity.name}'")
        return dataset

    def __call__(self, entity: Entity, key: Union[str, List[str]]) -> Dataset:
        return self.label(entity, key)
