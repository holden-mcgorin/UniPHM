import numpy as np
from torch import nn
import os
import random
import torch

from uniphm.data import Dataset
from uniphm.engine.Result import Result
from uniphm.util.Logger import Logger

"""
用于验证、排查bug的工具集合
"""


class Utils:
    @staticmethod
    def only_failure(test_set: Dataset, result: Result, enable=True):
        """
        仅取失效后数据，即过滤掉失效前的数据
        """
        if not enable:
            return test_set, result
        index = np.where(test_set.y != 1)[0]
        sub_test_set = Dataset(x=test_set.x[index, :], y=test_set.y[index, :], z=test_set.z[index, :],
                               label_map=test_set.label_map, name=test_set.name, entity_map=test_set.entity_map)

        sub_result = result.__copy__()
        sub_result.outputs = sub_result.outputs[index, :]
        return sub_test_set, sub_result

    @staticmethod
    def set_seed(seed: int = 42, deterministic: bool = True):
        """
        设置所有可控随机种子，确保实验可复现

        :param seed: 随机种子值
        :param deterministic: 是否使用确定性操作（会降低性能）
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"[Seed]  Random seed set to {seed} (deterministic={deterministic})")

    @staticmethod
    def random_seed() -> int:
        return np.random.SeedSequence().entropy

    @staticmethod
    def seed_worker(worker_id):
        """
        DataLoader 中 worker 进程的随机种子初始化函数

        :param worker_id: worker进程编号
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def count_paras(model: nn.Module) -> int:
        count = sum(p.numel() for p in model.parameters())
        Logger.info(f'The parameter count of {model.__class__.__name__} is {count}')
        return count

    @staticmethod
    def compare_dataset(ds1, ds2, name1="Dataset1", name2="Dataset2"):
        """
        对比两个 Dataset 对象
        :param ds1: Dataset 对象
        :param ds2: Dataset 对象
        :param name1: 第一个 Dataset 名称
        :param name2: 第二个 Dataset 名称
        """
        differences = []

        # 检查样本数
        x_shape1 = ds1.x.shape if ds1.x is not None else None
        x_shape2 = ds2.x.shape if ds2.x is not None else None
        if x_shape1 != x_shape2:
            differences.append(f"x shape 不一致: {name1}: {x_shape1}, {name2}: {x_shape2}")

        y_shape1 = ds1.y.shape if ds1.y is not None else None
        y_shape2 = ds2.y.shape if ds2.y is not None else None
        if y_shape1 != y_shape2:
            differences.append(f"y shape 不一致: {name1}: {y_shape1}, {name2}: {y_shape2}")

        z_shape1 = ds1.z.shape if ds1.z is not None else None
        z_shape2 = ds2.z.shape if ds2.z is not None else None
        if z_shape1 != z_shape2:
            differences.append(f"z shape 不一致: {name1}: {z_shape1}, {name2}: {z_shape2}")

        # 检查数值是否相同
        for arr_name, arr1, arr2 in [("x", ds1.x, ds2.x), ("y", ds1.y, ds2.y), ("z", ds1.z, ds2.z)]:
            if arr1 is not None and arr2 is not None:
                if arr1.shape == arr2.shape and not np.allclose(arr1, arr2, atol=1e-8, equal_nan=True):
                    diff_count = np.sum(~np.isclose(arr1, arr2, atol=1e-8, equal_nan=True))
                    differences.append(f"{arr_name} 值不一致，共 {diff_count} 个不同元素")

        # 检查 label_map
        if ds1.label_map != ds2.label_map:
            differences.append(f"label_map 不一致: {name1}: {ds1.label_map}, {name2}: {ds2.label_map}")

        # 检查 entity_map
        if ds1.entity_map != ds2.entity_map:
            differences.append(f"entity_map 不一致: {name1}: {ds1.entity_map}, {name2}: {ds2.entity_map}")

        if not differences:
            print(f"{name1} 与 {name2} 一致 ✅")
        else:
            print(f"{name1} 与 {name2} 存在差异：")
            for diff in differences:
                print("  -", diff)
