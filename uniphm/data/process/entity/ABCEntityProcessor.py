from abc import ABC, abstractmethod

from uniphm.data.Entity import Entity


class ABCEntityProcessor(ABC):
    """
    所有对实体的处理器的抽象基类
    """

    @abstractmethod
    def run(self, entity: Entity, key: str) -> Entity:
        raise NotImplementedError

    def __call__(self, entity: Entity, key: str) -> Entity:
        return self.run(entity, key)
