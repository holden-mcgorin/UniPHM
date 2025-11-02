from typing import Union, List
from uniphm.data.Entity import Entity
from uniphm.data.labeler.ABCLabeler import ABCLabeler
from uniphm.data.Dataset import Dataset


class MultiLabeler(ABCLabeler):
    def __init__(self, generators: List[ABCLabeler]):
        self.generators = generators

    @property
    def name(self):
        """
        :return: None，防止自动添加至生成子标签字典
        """
        return None

    def _label(self, bearing: Entity, key: Union[str, List[str]]) -> Dataset:
        # 先用第一个生成器生成 dataset
        dataset = self.generators[0](bearing, key)
        # 依次调用其他生成器并追加标签
        for generator in self.generators[1:]:
            another_dataset = generator(bearing, key)
            dataset.append_label(another_dataset)
        return dataset
