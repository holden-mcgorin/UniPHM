import os
from enum import Enum
from typing import Dict, Union, Optional, List

import pandas as pd
from pandas import DataFrame

from uniphm.data.Entity import Entity
from uniphm.data.loader.ABCLoader import ABCLoader
from uniphm.util.Logger import Logger


class Fault(Enum):
    """
    轴承故障类型枚举
    """
    HPC = 'Normal Condition'
    FAN = 'Fan Degradation'

    def __str__(self):
        return self.name


class CMAPSSLoader(ABCLoader):
    """
    归一化参数设置
    """
    # 按照什么分组归一化（归一化的最大值和最小值来源）
    # 可选项："self"、"train"、"condition" 通常来说，self预测误差较大，其余差异不大
    normalization_from = 'condition'

    # 归一化模式
    # 可选项：'[0,1]'、'[-1,1]'、'None' 通常来说，该设置对预测的影响不大
    normalization_mode = '[0,1]'

    # setting1离散化参数（根据该值划分6种工况）
    # （low, high）-> val
    ranges = [
        (0.0, 0.003, 0.0),
        (9.998, 10.008, 10.0),
        (19.998, 20.008, 20.0),
        (24.998, 25.008, 25.0),
        (34.998, 35.008, 35.0),
        (41.998, 42.008, 42.0)
    ]

    # 存储每个数据集每个工况的最大最小值，e.g. group_min['FD002']['0.0']
    group_min = {'FD001': {},
                 'FD002': {},
                 'FD003': {},
                 'FD004': {}}
    group_max = {'FD001': {},
                 'FD002': {},
                 'FD003': {},
                 'FD004': {}}

    """
    硬数据
    """

    @property
    def meta(self) -> dict:
        return {'category': 'Turbofan',
                'time_unit': 'cycle'}

    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30',
              'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF',
              'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

    num_trajectories = {
        'FD001_train': 100,
        'FD002_train': 260,
        'FD003_train': 100,
        'FD004_train': 249,
        'FD001_test': 100,
        'FD002_test': 259,
        'FD003_test': 100,
        'FD004_test': 248,
        'FD001_RUL': 100,
        'FD002_RUL': 259,
        'FD003_RUL': 100,
        'FD004_RUL': 248
    }

    fault_type_dict = {
        'FD001': [Fault.HPC],
        'FD002': [Fault.HPC],
        'FD003': [Fault.HPC, Fault.FAN],
        'FD004': [Fault.HPC, Fault.FAN]
    }

    conditions = {
        'FD001': 'ONE (Sea Level)',
        'FD002': 'SIX',
        'FD003': 'ONE (Sea Level)',
        'FD004': 'SIX'
    }

    min_cycle = {
        'FD001_train': 128,
        'FD002_train': 128,
        'FD003_train': 145,
        'FD004_train': 128,
        'FD001_test': 31,
        'FD002_test': 21,
        'FD003_test': 38,
        'FD004_test': 19,
    }

    max_cycle = {
        'FD001_train': 362,
        'FD002_train': 378,
        'FD003_train': 525,
        'FD004_train': 543,
        'FD001_test': 303,
        'FD002_test': 267,
        'FD003_test': 475,
        'FD004_test': 486,
    }

    def batch_load(self, trajectory: str,
                   include: Optional[Union[List[str], List[int], Dict[str, List[Union[str, int]]]]] = None,
                   exclude: Optional[Union[List[str], List[int], Dict[str, List[Union[str, int]]]]] = None
                   ) -> [Entity]:
        """
        批量加载数据：例如批量加载 FD001_train
        :param exclude:
        :param include:
        :param trajectory: 子数据集名称
        :return:
        """
        entities = []
        num_trajectory = CMAPSSLoader.num_trajectories[trajectory]
        for i in range(1, num_trajectory + 1):
            entities.append(self.load_entity(trajectory + '_' + str(i), include, exclude))

        return entities

    def load_entity(self, entity_name: str,
                    include: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None,
                    exclude: Optional[Union[int, str, List[str], List[int], Dict[str, List[Union[str, int]]]]] = None
                    ) -> Entity:
        """
        加载单个实体数据，并进行归一化和拆分处理。
        归一化仅针对未被过滤的列。
        """
        Logger.info(f'[DataLoader]  -> Loading data entity: {entity_name}')
        raw_data = self._load(entity_name)
        raw_data = self._filter(raw_data, include, exclude)

        entity = self._assemble(entity_name, raw_data)

        # 归一化处理
        norm_df = self._norm(entity_name)
        norm_df = self._filter(norm_df, include, exclude)

        # 对归一化列也应用相同的过滤
        if include is not None or exclude is not None:
            # 仅保留原始数据最终留下的列
            filtered_cols = list(entity.data.keys())
            norm_df = norm_df[filtered_cols]

        # 拆分归一化数据，每列 (n,1) ndarray，key = 'norm_' + 原列名
        for col in norm_df.columns:
            entity.data[f'norm_{col}'] = norm_df[[col]].to_numpy()

        Logger.info(f'[DataLoader]  ✓ Successfully loaded: {entity_name}')
        return entity

    def _register(self, root) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        """
        记录数据文件位置
        """
        file_dict = {}
        entity_dict = {}
        for filename in os.listdir(root):
            raw_name = filename[:-4]
            split = raw_name.split('_')
            if len(split) < 2:  # 非数据文件则跳过
                continue
            post_name = split[1] + '_' + split[0]
            if post_name in self.num_trajectories:
                file_dict[post_name] = os.path.join(root, filename)

                # 注册数据
                entity_dict[post_name] = None  # 注册完整数据
                entity_dict[post_name + '_norm'] = None  # 注册完整归一化数据
                for i in range(1, self.num_trajectories[post_name] + 1):  # 注册单个发动机数据
                    entity_dict[post_name + '_' + str(i)] = None
                    entity_dict[post_name + '_' + str(i) + '_norm'] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> DataFrame:
        """
        加载完整数据文件进入内存
        """
        # 否则第一次查找则加载
        if self._entity_dict[entity_name] is None:
            split = entity_name.split('_')
            raw_name = split[1] + '_' + split[0]  # e.g. test_FD001
            post_name = split[0] + '_' + split[1]  # e.g. FD001_test

            # 读取数据文件并设置表头
            df = pd.read_csv(os.path.join(self._root, raw_name + '.txt'), header=None, sep='\\s+')
            df.columns = ['num'] + self.header

            # 保存完整数据
            self._entity_dict[post_name] = df
            # 保存单个实体的数据
            grouped = df.groupby(df.columns[0])
            dfs = {group: pd.DataFrame(data) for group, data in grouped}
            for i in range(1, self.num_trajectories[post_name] + 1):
                self._entity_dict[post_name + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

            # 若读取的发动机属于测试集需要额外读取RUL文件
            if split[1] == 'test':
                df = pd.read_csv(os.path.join(self._root, 'RUL_' + split[0] + '.txt'), header=None, sep='\\s+')
                for i in range(1, self.num_trajectories[split[0] + '_RUL'] + 1):
                    self._entity_dict[split[0] + '_RUL_' + str(i)] = int(df.iloc[i - 1, 0])

        # 因为C-MAPSS数据集每个实体仅有一个数据表格，该数据表格命名为 sensor_signals
        return self._entity_dict[entity_name]

    def _assemble(self, entity_name: str, df: DataFrame) -> Entity:
        """
        生成实体（C-MAPSS涡扇发动机）对象
        """
        split = entity_name.split('_')

        # 设置实体类别 & 工况
        meta = {
            'fault_type': self.fault_type_dict[split[0]],  # 故障类型
            'condition': self.conditions[split[0]]  # 工况
        }
        meta = {**self.meta, **meta}

        # 设置 rul & life（测试集发动机 RUL 从文件中读取；训练集发动机 RUL 为 0）
        if len(split) == 3:  # 单个发动机
            if split[1] == 'train':
                meta['rul'] = 0
                meta['life'] = len(df)
            elif split[1] == 'test':
                meta['rul'] = self._entity_dict[split[0] + '_RUL_' + split[2]]
                meta['life'] = len(df) + meta['rul']

        # 拆分 DataFrame，每列生成 (n,1) ndarray
        array_dict = {}
        for col in df.columns:
            array_dict[col] = df[[col]].to_numpy()  # (n, 1)

        # 返回 Entity 对象
        return Entity(name=entity_name, data=array_dict, meta=meta)

    def _norm(self, entity_name: str) -> DataFrame:
        """
        生成归一化的子数据集整体数据
        :param entity_name:
        :return:
        """
        # 如果字典已被加载过则直接返回
        if self._entity_dict[entity_name + '_norm'] is not None:
            return self._entity_dict[entity_name + '_norm']

        split = entity_name.split('_')
        raw_df = self._load(split[0] + '_' + split[1])
        df = raw_df.copy()

        if self.normalization_from == "condition":

            # setting1离散化
            def discretize(val):
                for low, high, target in self.ranges:
                    if low <= val <= high:
                        return target
                Logger.warning(f'During the discretization, '
                               f'the value {val} exceeded all intervals and has been returned as None')
                return None

            # 添加两个辅助列
            if split[0] in ['FD001', 'FD003']:
                df['condition'] = 1
            elif split[0] in ['FD002', 'FD004']:
                df['condition'] = df['setting1'].apply(discretize)
            else:
                raise KeyError(f"{split[0]} does not exist")
            df['original_index'] = df.index
            grouped = []
            for name, group in df.groupby('condition'):
                features = group.iloc[:, 2:-2]  # 去掉前两列和最后两个辅助列
                if split[1] == 'test':
                    self._norm(split[0] + '_train')
                if split[1] == 'train':
                    self.group_min[split[0]][name] = features.min()
                    self.group_max[split[0]][name] = features.max()
                normalized = self._cal(features, self.group_min[split[0]][name], self.group_max[split[0]][name])
                group.update(normalized)  # 更新归一化后的特征列
                grouped.append(group)
            result = pd.concat(grouped).sort_values('original_index').drop(columns='original_index').drop(
                columns='condition')
        elif self.normalization_from == "self":
            result = self._cal(df, df.min(), df.max())
        elif self.normalization_from == "train":
            train_df = self._load(split[0] + '_train')
            train_df = train_df['sensor_signals']
            result = self._cal(df, train_df.min(), train_df.max())
        else:
            raise KeyError(f"parameter 'normalization_from' now is {self.normalization_from}, which is invalid!")

        # 保存完整数据
        self._entity_dict[split[0] + '_' + split[1] + '_norm'] = result
        # 保存单个实体的数据
        grouped = result.groupby(result.columns[0])
        dfs = {group: pd.DataFrame(data) for group, data in grouped}
        for i in range(1, self.num_trajectories[split[0] + '_' + split[1]] + 1):
            self._entity_dict[split[0] + '_' + split[1] + '_' + str(i) + '_norm'] = dfs[i].drop(dfs[i].columns[0],
                                                                                                axis=1)

        return self._entity_dict[entity_name + '_norm']

    def _cal(self, features, min_val, max_val):
        """
        归一化计算算法
        :param features:
        :param min_val:
        :param max_val:
        :return:
        """
        if self.normalization_mode == '[0,1]':
            normalized = (features - min_val) / (max_val - min_val)
        elif self.normalization_mode == '[-1,1]':
            normalized = 2 * (features - min_val) / (max_val - min_val) - 1
        elif self.normalization_mode == 'None':
            normalized = features
        else:
            raise KeyError(f"parameter 'normalization_mode' now is {self.normalization_mode}, which is invalid!")

        return normalized
