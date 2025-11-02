from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from uniphm.data.Entity import Entity
from uniphm.util.Logger import Logger


class ABCLoader(ABC):
    """
    所有数据读取器的抽象基类、
    采用懒加载的方式加载数据
    所有子类必须重写下列方法，对应加载数据的3个步骤
        1. _register（注册：建立数据到文件的映射表）
        2. _load（装载：从文件中读取数据）
        3. _assemble（组装；返回实体对象、设置实体对象的属性，如RUL、故障模式、工况）
    """

    def __init__(self, root: str):
        """
        获取数据集根目录，确定各个数据项的位置
        :param root: 数据集的根目录
        """
        # 此数据集根目录
        self._root = root
        # {数据名称-文件地址}字典、{实体名称-数据}字典
        self._file_dict, self._entity_dict = self._register(root)

        Logger.debug(str(self))

    @property
    @abstractmethod
    def meta(self) -> dict:
        pass

    def __call__(self, entity_name: str,
                 include: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None,
                 exclude: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None
                 ) -> Entity:
        return self.load_entity(entity_name, include, exclude)

    def load_entity(self, entity_name: str,
                    include: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None,
                    exclude: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None
                    ) -> Entity:
        """
        获取实体
        :param entity_name: 实体名称
        :param include: 全局或局部的列选择
                        - List[str] 或 List[int]：所有 DataFrame 都应用
                        - Dict[str, List]：只对指定 DataFrame 应用
        :param exclude: 全局或局部的列排除
        :return:
        """
        Logger.info(f'[DataLoader]  -> Loading data entity: {entity_name}')
        raw_data = self._load(entity_name)
        raw_data = self._filter(raw_data, include, exclude)
        entity = self._assemble(entity_name, raw_data)
        Logger.info(f'[DataLoader]  ✓ Successfully loaded: {entity_name}')
        return entity

    @abstractmethod
    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        """
        file_dict：数据项名称 -> 数据项文件地址
        entity_dict：数据项名称 -> 数据项对象
        键：数据项名称
        值：数据项文件目录
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _load(self, entity_name: str) -> DataFrame:
        """
        根据数据项名称从数据集中获取数据
        :param entity_name:数据项名称
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _filter(
            df: DataFrame,
            include: Optional[Union[int, str, List[Union[int, str]]]] = None,
            exclude: Optional[Union[int, str, List[Union[int, str]]]] = None
    ) -> DataFrame:
        """
        对单个 DataFrame 按列进行过滤。
        - include / exclude: 支持 int (列索引)、str (列名)、或它们的列表。
        - 优先应用 include（若有），然后在结果上应用 exclude（若有）。
        - 无效的索引/列名会被忽略（不会抛异常）。
        返回一个新的 DataFrame（浅拷贝）。
        """

        def _to_list(spec):
            if spec is None:
                return None
            if isinstance(spec, (int, str)):
                return [spec]
            return list(spec)

        def _resolve(cols, df_cols):
            """把 int 索引转为列名，并只保留在 df_cols 中存在的列名"""
            if cols is None:
                return None
            resolved = []
            for c in cols:
                if isinstance(c, int):
                    if 0 <= c < len(df_cols):
                        resolved.append(df_cols[c])
                elif isinstance(c, str):
                    if c in df_cols:
                        resolved.append(c)
            return resolved

        include_list = _to_list(include)
        exclude_list = _to_list(exclude)

        df_cols = list(df.columns)

        inc_cols = _resolve(include_list, df_cols)
        exc_cols = _resolve(exclude_list, df_cols)

        # 初始为所有列
        cols = df_cols

        # 先 apply include（若指定）
        if inc_cols is not None:
            # 保持原列顺序，但只保留 include 指定的列（按 df 原顺序）
            cols = [c for c in cols if c in inc_cols]

        # 再 apply exclude（若指定）
        if exc_cols is not None:
            cols = [c for c in cols if c not in exc_cols]

        return df[cols].copy()

    @abstractmethod
    def _assemble(self, entity_name: str, df: DataFrame) -> Entity:
        """
        组装成实体对象
        :param entity_name:
        :return:
        """
        raise NotImplementedError

    def __str__(self) -> str:
        items = '\n'.join([f"\t✓ {key}, location: {value}" for key, value in self._file_dict.items()])
        return f'\n[DataLoader]  Root directory: {self._root}\n{items}'

    def __iter__(self):
        return NameIterator(list(self._entity_dict.keys()))

    # -------------------------------- 字典接口 --------------------------------
    def __getitem__(self, key: str):
        """返回元数据"""
        if key in self.meta:
            return self.meta[key]

    def __contains__(self, entity_name: str) -> bool:
        return entity_name in self._entity_dict

    def keys(self):
        return self._entity_dict.keys()

    def values(self):
        return self._entity_dict.values()

    def items(self):
        return self._entity_dict.items()


class NameIterator:
    # 用来遍历所有已加载的数据
    def __init__(self, name_list: list):
        self.name_list = name_list
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.name_list):
            result = self.name_list[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration
