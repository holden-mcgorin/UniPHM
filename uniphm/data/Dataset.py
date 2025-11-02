from __future__ import annotations

import copy
from typing import Tuple, List, Union

import numpy as np
from numpy import ndarray

from uniphm.util.Logger import Logger


# todo 1. 将x、y、z设计为[ndarray]的结构，用于处理不规则的输入和输出，列表维度代表样本维度
# todo 2. 将单个样本的x、y设计为{str:ndarray}的结构，用来处理多输入多输出问题，此时可以去掉 label_map
# todo 3. 如果同时实现1和2，则x和y的结构为[{str:ndarray}]

class Dataset:
    def __init__(self, name: str = None,
                 x: ndarray = None, y: ndarray = None, z: ndarray = None,
                 label_map: dict = None, entity_map: dict = None,
                 extra: dict = None):
        """
        每个entity获取连续的一片空间
        :param name:
        :param x: 输入 / 特征
        :param y: 输出 / 标签
        :param z: 该样本采集结束时的物理时间，对于第i个样本，z应该满足 z[i] + rul[i] = life
        :param label_map: 标签映射表
        :param entity_map: 实体映射表
        """
        self.name = name

        self.x = x  # 输入（特征）
        self.y = y  # 输出（标签）
        self.z = z  # 该样本采集结束时的物理时间，对于第i个样本，z应该满足 z[i] + rul[i] = life
        self.extra = extra  # 兼容其他结构数据，如

        # 针对y的 e.g. {'label_1': (start, end)}
        self.label_map = label_map if label_map is not None else {}
        # 针对x的 e.g. {'entity_1': (start, end)}
        self.entity_map = entity_map if entity_map is not None else {}

        self.__validate(x, y, z)

    @property
    def num_label(self):
        """
        :return:标签类别数量（兼容单标签与多标签）
        """
        return len(self.label_map)

    @property
    def num_entity(self):
        """
        :return:标签类别数量（兼容单标签与多标签）
        """
        return len(self.entity_map)

    def add(self, another_dataset: Dataset) -> None:
        """
        常用方法设置别名（append_entity）
        :param another_dataset:
        :return:
        """
        self.append_entity(another_dataset)

    def append_entity(self, another_dataset: Dataset) -> None:
        """
        原地操作，合并另一个 Dataset。
        若 entity 名称重复，则将数据拼接到该实体末尾，并更新所有后续 entity 的起止位置。
        """
        # 合并的数据集为空，直接跳过
        if another_dataset.x is None:
            return

        # 当前数据集为空，直接替换
        if self.x is None:
            self.name = another_dataset.name
            self.x = another_dataset.x
            self.y = another_dataset.y
            self.z = another_dataset.z
            self.label_map = another_dataset.label_map
            self.entity_map = another_dataset.entity_map
            return

        # 确保标签类别一致
        if set(self.label_map.keys()) != set(another_dataset.label_map.keys()):
            raise ValueError("label_map's keys are not the same, cannot merge.")

        for entity, (start_new, end_new) in another_dataset.entity_map.items():
            data_x = another_dataset.x[start_new:end_new]
            data_y = another_dataset.y[start_new:end_new]
            data_z = another_dataset.z[start_new:end_new]
            new_len = end_new - start_new

            if entity in self.entity_map:
                start_old, end_old = self.entity_map[entity]
                # 插入到旧 entity 的末尾
                insert_pos = end_old

                # 插入数据
                self.x = np.insert(self.x, insert_pos, data_x, axis=0)
                self.y = np.insert(self.y, insert_pos, data_y, axis=0)
                self.z = np.insert(self.z, insert_pos, data_z, axis=0)

                # 更新该 entity 的映射
                self.entity_map[entity] = (start_old, end_old + new_len)

                # 更新所有在该 entity 之后的 entity 映射
                for key in self.entity_map:
                    if key != entity:
                        s, e = self.entity_map[key]
                        if s >= end_old:
                            self.entity_map[key] = (s + new_len, e + new_len)
            else:
                # 不冲突则直接追加
                start_new_idx = self.x.shape[0]
                self.x = np.concatenate([self.x, data_x], axis=0)
                self.y = np.concatenate([self.y, data_y], axis=0)
                self.z = np.concatenate([self.z, data_z], axis=0)
                self.entity_map[entity] = (start_new_idx, start_new_idx + new_len)

        self.name = f"{self.name}; {another_dataset.name}"

    def append_label(self, another_dataset: Dataset) -> None:
        """
        添加额外的标签
        """
        # 合并的数据集为空，直接跳过
        if another_dataset.x is None:
            return

        # 当前数据集为空，直接替换
        if self.x is None:
            self.name = another_dataset.name
            self.x = another_dataset.x
            self.y = another_dataset.y
            self.z = another_dataset.z
            self.label_map = another_dataset.label_map
            self.entity_map = another_dataset.entity_map
            return

        if another_dataset.y.shape[0] != self.y.shape[0]:
            raise Exception(f'标签行数不匹配，源标签行数：{self.y.shape[0]}，目标标签行数：{another_dataset.y.shape[0]}')

        another_datasets = another_dataset.split_by_label()
        for another_dataset in another_datasets:
            self.label_map[next(iter(another_dataset.label_map))] = (
                self.y.shape[1], self.y.shape[1] + another_dataset.y.shape[1])
            self.y = np.hstack((self.y, another_dataset.y))

    def split_by_ratio(self, ratio, seed=None) -> Tuple[Dataset, Dataset]:
        """
        将数据集按比例拆分为两个子集
        注：按照实体对象分组拆分
        :param ratio: 0~1之间的比例，表示第一个子集所占比例
        :param seed: 随机种子，示例：42
        :return: 两个新的 Dataset 实例
        """

        # 设置随机种子
        if seed is None:
            seed = np.random.SeedSequence().entropy  # 自动生成高熵种子
        Logger.info(f"[Dataset]  Splitting data by ratio {ratio} using random seed: {seed}")

        # 先分为多个只包含单个entity的Dataset，再逐个按比例分裂
        uppers = []
        lowers = []
        for entity_dataset in self.split_by_entity():
            upper, lower = entity_dataset.__split(ratio, seed)
            uppers.append(upper)
            lowers.append(lower)

        # 合并
        upper_result = Dataset()
        for upper in uppers:
            upper_result.append_entity(upper)
        lower_result = Dataset()
        for lower in lowers:
            lower_result.append_entity(lower)

        return upper_result, lower_result

    def split_by_label(self) -> Tuple[Dataset, ...]:
        """
        按子标签分裂数据集，即保持 x、z 不变，按 y 分裂
        :return: 每个子标签一个 Dataset
        """

        results = []
        for key, (start, end) in self.label_map.items():
            sub_label_map = {key: (0, end - start)}
            results.append(
                Dataset(
                    x=self.x,
                    y=self.y[:, start:end],
                    z=self.z,
                    label_map=sub_label_map,
                    entity_map=self.entity_map,  # 保留原始实体信息
                    name=f"{self.name}"
                )
            )
        return tuple(results)

    def split_by_entity(self) -> Tuple[Dataset, ...]:
        """
        按 entity_map 中的每个实体拆分数据集
        :return: 每个实体一个 Dataset
        """
        results = []
        for key, (start, end) in self.entity_map.items():
            x_part = self.x[start:end]
            y_part = self.y[start:end]
            z_part = self.z[start:end]

            # entity_map 只保留当前实体
            sub_entity_map = {key: (0, end - start)}

            results.append(
                Dataset(
                    x=x_part,
                    y=y_part,
                    z=z_part,
                    label_map=self.label_map,  # 所有实体共享同一个 label_map
                    entity_map=sub_entity_map,
                    name=f"{key}"
                )
            )
        return tuple(results)

    def select_by_features(self, indices: Union[List[int], int], squeeze=True) -> Dataset:
        """
        选择x中的指定列，生成新的Dataset
        :param squeeze:
        :param indices:
        :return:
        """
        if self.x.ndim != 3:
            raise Exception(
                'The feature selection process only supports input with three dimensions:(batch_size, seq_length, feature_size')
        # 当输入是单个索引时统一为索引列表
        if isinstance(indices, int):
            indices = [indices]

        new_x = self.x[:, :, indices]
        if squeeze:
            new_x = new_x.squeeze()

        return Dataset(
            x=new_x,
            y=self.y,
            z=self.z,
            label_map=copy.deepcopy(self.label_map),
            entity_map=copy.deepcopy(self.entity_map),
            name=self.name
        )

    def get(self, entity_name) -> Dataset:
        """
        非原地操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        return Dataset(
            x=self.x[start:end],
            y=self.y[start:end],
            z=self.z[start:end],
            label_map=self.label_map,
            entity_map={entity_name: (0, end - start)},
            name=entity_name
        )

    def remove(self, entity_name: str) -> None:
        """
        原地删除操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        shift = end - start
        # 生成新的entity_map
        if end == self.x.shape[0]:  # x中最后的entity
            del self.entity_map[entity_name]
        else:
            last = end
            while True:
                flag = False
                for k, (s, e) in self.entity_map.items():
                    if s == last:  # 找到下一个entity
                        self.entity_map[k] = (s - shift, e - shift)
                        last = e
                        flag = True
                        break
                if not flag:
                    break
            del self.entity_map[entity_name]

        self.x = np.concatenate([self.x[:start], self.x[end:]], axis=0),
        self.y = np.concatenate([self.y[:start], self.y[end:]], axis=0),
        self.z = np.concatenate([self.z[:start], self.z[end:]], axis=0),
        self.name.replace(entity_name, '')

    def include(self, entity_names: Union[str, List[str]]) -> Dataset:
        """
        非原地批量选择
        :param entity_names:
        :return:
        """
        entity_names = [entity_names] if isinstance(entity_names, str) else entity_names
        datasets = []
        for entity_name in entity_names:
            datasets.append(self.get(entity_name))

        new_dataset = datasets[0]
        for i in range(1, len(datasets)):
            new_dataset.append_entity(datasets[i])
        return new_dataset

    def exclude(self, entity_names: Union[str, List[str]]) -> Dataset:
        entity_names = [entity_names] if isinstance(entity_names, str) else entity_names
        new_dataset = self
        for entity_name in entity_names:
            new_dataset = new_dataset.__exclude(entity_name)

        if not new_dataset.entity_map:
            Logger.warning(f"[Dataset]  After exclude operation for {entity_names}, the entity_map is empty. ")

        return new_dataset

    def __exclude(self, entity_name: str) -> Dataset:
        """
        非原地删除操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        shift = end - start
        new_entity_map = copy.deepcopy(self.entity_map)
        # 生成新的entity_map
        if end == self.x.shape[0]:  # x中最后的entity
            del new_entity_map[entity_name]
        else:
            last = end
            while True:
                flag = False
                for k, (s, e) in new_entity_map.items():
                    if s == last:  # 找到下一个entity
                        new_entity_map[k] = (s - shift, e - shift)
                        last = e
                        flag = True
                        break
                if not flag:
                    break
            del new_entity_map[entity_name]
        new_name = copy.copy(self.name).replace(entity_name, '')

        return Dataset(
            x=np.concatenate([self.x[:start], self.x[end:]], axis=0),
            y=np.concatenate([self.y[:start], self.y[end:]], axis=0),
            z=np.concatenate([self.z[:start], self.z[end:]], axis=0),
            label_map=copy.deepcopy(self.label_map),
            entity_map=new_entity_map,
            name=new_name
        )

    def clear(self) -> None:
        """
        原地操作
        :return:
        """
        self.name = None
        self.x = None
        self.y = None
        self.z = None
        self.label_map = {}
        self.entity_map = {}

    def __split(self, ratio, seed=None) -> Tuple[Dataset, Dataset]:
        """
        当数据集中只有一个entity时由概率分裂
        :param ratio:
        :return:
        """
        num_samples = self.x.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_samples)

        # # 指定随机种子
        # if seed is not None:
        #     rng = np.random.default_rng(seed=seed)  # 创建一个带种子的随机生成器
        #     indices = rng.permutation(num_samples)
        # # 不指定随机种子
        # else:
        #     indices = np.random.permutation(num_samples)

        train_size = int(ratio * num_samples)
        upper_indices = indices[:train_size]
        lower_indices = indices[train_size:]

        upper_entity_map = {self.name: (0, train_size)}
        lower_entity_map = {self.name: (0, num_samples - train_size)}

        upper = Dataset(x=self.x[upper_indices], y=self.y[upper_indices], z=self.z[upper_indices],
                        label_map=self.label_map, entity_map=upper_entity_map, name=self.name)
        lower = Dataset(x=self.x[lower_indices], y=self.y[lower_indices], z=self.z[lower_indices],
                        label_map=self.label_map, entity_map=lower_entity_map, name=self.name)

        return upper, lower

    def __validate(self, x, y, z) -> None:
        """
        验证数据集是否合法
        :param x:
        :param y:
        :param z:
        :return:
        """
        if x is None and y is None and z is None:
            return
        # 验证x、y、z的样本数是否一致
        if x is not None and y is not None and z is not None:
            def get_n(array: np.ndarray):
                return array.shape[0] if array.ndim >= 1 else 1

            x_n, y_n, z_n = get_n(x), get_n(y), get_n(z)
            assert x_n == y_n == z_n, f"x样本数:{x_n}, y样本数:{y_n}, z样本数:{z_n} 不一致"
        else:
            raise Exception(f'x、y、z需要同时初始化或同时为None：x={x},y={y},z={z}')

        if self.label_map is None:
            self.label_map = {}
        if self.entity_map is None:
            self.entity_map = {}

        # 检查数值异常（NaN / Inf / 极端大值）
        for arr, name in [(x, "x"), (y, "y"), (z, "z")]:
            if arr is not None and arr.size > 0:
                if np.isnan(arr).any():
                    raise ValueError(f"检测到 {name} 中存在 NaN 值")
                if np.isinf(arr).any():
                    raise ValueError(f"检测到 {name} 中存在 Inf 值")
                if np.abs(arr).max() > 1e6:
                    raise ValueError(f"检测到 {name} 中存在异常大值 (>|1e6|)，最大值: {np.abs(arr).max()}")

        # 分裂时产生空的数据集
        if self.x.shape[0] == 0:
            self.clear()

    # ---------- 字典风格接口 ----------
    def __getitem__(self, entity_name: str) -> Dataset:
        """
        支持字典风格访问单个 entity
        示例：
            ds = Dataset(...)
            entity_ds = ds["entity_1"]
        """
        if entity_name not in self.entity_map:
            raise KeyError(f"Entity '{entity_name}' does not exist. Available entities: {list(self.entity_map.keys())}")
        return self.get(entity_name)

    def __delitem__(self, entity_name: str) -> None:
        """ 支持 del ds[entity_name] 删除指定 entity """
        if entity_name not in self.entity_map:
            raise KeyError(f"Entity '{entity_name}' does not exist")
        self.remove(entity_name)

    def __contains__(self, entity_name: str) -> bool:
        """ 支持 'entity_name' in ds """
        return entity_name in self.entity_map

    def __len__(self):
        return self.x.shape[0]

    def __copy__(self):
        return Dataset(name=self.name if self.name is not None else None,
                       x=self.x.copy() if self.x is not None else None,
                       y=self.y.copy() if self.y is not None else None,
                       z=self.z.copy() if self.z is not None else None,
                       label_map=self.label_map.copy() if self.label_map is not None else None,
                       entity_map=self.entity_map.copy() if self.label_map is not None else None
                       )
