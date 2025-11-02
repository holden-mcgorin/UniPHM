from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
from numpy import ndarray


class Result:
    """
    预测结果
    结构类似 Dataset，但更简单
    """

    def __init__(self, name: str = None, y_hat: ndarray = None,
                 entity_map: dict = None) -> None:

        self.name = name
        self.y_hat = y_hat

        self.entity_map = entity_map

    def add(self, another_result: Result) -> None:
        self.append_entity(another_result)

    def get(self, entity_name) -> Result:
        """
        非原地操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        return Result(
            y_hat=self.y_hat[start:end],
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
        if end == self.y_hat.shape[0]:  # x中最后的entity
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

        self.y_hat = np.concatenate([self.y_hat[:start], self.y_hat[end:]], axis=0),
        self.name.replace(entity_name, '')

    def exclude(self, entity_name: str) -> Result:
        """
        非原地删除操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        shift = end - start
        new_entity_map = copy.deepcopy(self.entity_map)
        # 生成新的entity_map
        if end == self.y_hat.shape[0]:  # x中最后的entity
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

        return Result(
            y_hat=np.concatenate([self.y_hat[:start], self.y_hat[end:]], axis=0),
            entity_map=new_entity_map,
            name=new_name
        )

    def append_entity(self, another_result: Result) -> None:
        """
        原地操作。将另一个 Result 合并到当前对象中。
        如果 entity 名称冲突，则将数据插入到该 entity 的末尾位置，并更新后续实体的映射。
        """
        if another_result.y_hat is None:
            return

        if self.y_hat is None:
            self.name = another_result.name
            self.y_hat = another_result.y_hat
            self.entity_map = another_result.entity_map.copy()
            return

        for entity, (start_new, end_new) in another_result.entity_map.items():
            y_hat_part = another_result.y_hat[start_new:end_new]
            new_len = end_new - start_new

            if entity in self.entity_map:
                start_old, end_old = self.entity_map[entity]
                insert_pos = end_old

                # 插入数据
                self.y_hat = np.insert(self.y_hat, insert_pos, y_hat_part, axis=0)

                # 更新该 entity 的映射
                self.entity_map[entity] = (start_old, end_old + new_len)

                # 更新后续 entity 的映射
                for key, (s, e) in self.entity_map.items():
                    if key != entity and s >= end_old:
                        self.entity_map[key] = (s + new_len, e + new_len)
            else:
                # 不冲突直接追加
                start_new_idx = self.y_hat.shape[0]
                self.y_hat = np.concatenate([self.y_hat, y_hat_part], axis=0)
                self.entity_map[entity] = (start_new_idx, start_new_idx + new_len)

        self.name = f"{self.name}; {another_result.name}"

    def split_by_entity(self) -> Tuple[Result, ...]:
        """
        按 entity_map 中的每个实体拆分数据集
        :return: 每个实体一个 Dataset
        """
        results = []
        for key, (start, end) in self.entity_map.items():
            y_hat_part = self.y_hat[start:end]

            # entity_map 只保留当前实体
            sub_entity_map = {key: (0, end - start)}
            results.append(
                Result(
                    name=f"{key}",
                    y_hat=y_hat_part,
                    entity_map=sub_entity_map
                )
            )

        return tuple(results)

    def __copy__(self):
        return Result(
            name=self.name if self.name is not None else None,
            y_hat=self.y_hat.copy() if self.y_hat is not None else None,
            entity_map=self.entity_map.copy() if self.entity_map is not None else None
        )

    def __len__(self):
        return self.y_hat.shape[0]

# 测试用例
# # 构造 Result A：entity1(0,2), entity2(2,4)
# y_hat_a = np.array([[1], [2], [3], [4]])
# entity_map_a = {'entity1': (0, 2), 'entity2': (2, 4)}
# A = Result(name="A", y_hat=y_hat_a, entity_map=entity_map_a)
#
# # 构造 Result B：entity1(0,2), entity3(2,4)
# y_hat_b = np.array([[5], [6], [7], [8]])
# entity_map_b = {'entity1': (0, 2), 'entity3': (2, 4)}
# B = Result(name="B", y_hat=y_hat_b, entity_map=entity_map_b)
#
# # 合并
# A.append(B)
#
# print("Merged y_hat:")
# print(A.y_hat)
#
# print("Merged entity_map:")
# print(A.entity_map)
#
# # 期望 entity_map:
# # entity1: (0, 4) -> 1,2 (原) + 5,6 (插入)
# # entity2: (4, 6) -> 3,4 原来后移2位
# # entity3: (6, 8) -> 7,8 新增在末尾
#
# expected_y_hat = np.array([[1], [2], [5], [6], [3], [4], [7], [8]])
# expected_entity_map = {
#     'entity1': (0, 4),
#     'entity2': (4, 6),
#     'entity3': (6, 8)
# }
#
# assert np.array_equal(A.y_hat, expected_y_hat), "y_hat mismatch"
# assert A.entity_map == expected_entity_map, "entity_map mismatch"
#
# print("✅ Result.append 测试通过")
