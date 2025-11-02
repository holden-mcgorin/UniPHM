import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from uniphm.data.Entity import Entity
from uniphm.data.loader.ABCLoader import ABCLoader


class PHM2008Loader(ABCLoader):
    """
        实体名词示例：
            train_3
            test_7
            final_test_8
    """

    @property
    def meta(self) -> dict:
        pass

    arr_min = {}
    arr_max = {}

    trajectories = {
        'train': 218,
        'test': 218,
        'final_test': 435
    }

    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
              'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF', 'NRc', 'BPR', 'farB',
              'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for filename in os.listdir(root):
            name = filename[:-4]
            if name in self.trajectories:
                file_dict[name] = os.path.join(root, filename)
                # 注册未分裂的完整数据
                entity_dict[name] = None
                for i in range(1, self.trajectories[name] + 1):
                    entity_name = name + '_' + str(i)
                    entity_dict[entity_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> Dict[str, Union[DataFrame, None]]:
        # 如果字典为None（第一次查找）则加载
        if self._entity_dict[entity_name] is None:
            split = entity_name.split('_')
            prefix_raw = split[0]  # e.g. test
            if prefix_raw == 'final':
                prefix_raw = prefix_raw + '_test'

            df = pd.read_csv(os.path.join(self._root, prefix_raw + '.txt'), header=None, sep='\\s+')
            df.columns = ['num'] + self.header

            # 保存数据文件每列的最大值和最小值，用于归一化操作
            arr_min = np.min(df.values[:, 1:], axis=0)  # 每列的最小值
            arr_max = np.max(df.values[:, 1:], axis=0)  # 每列的最大值
            self.arr_min[prefix_raw] = arr_min
            self.arr_max[prefix_raw] = arr_max

            # 保存未分裂前的完整数据
            self._entity_dict[prefix_raw] = df.drop(df.columns[0], axis=1)

            # 按第一列分裂
            grouped = df.groupby(df.columns[0])
            dfs = {group: pd.DataFrame(data) for group, data in grouped}

            for i in range(1, self.trajectories[prefix_raw] + 1):
                self._entity_dict[prefix_raw + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

        return {'sensor_signals': self._entity_dict[entity_name]}

    def _assemble(self, entity_name: str, raw_data: Dict[str, Union[DataFrame, None]]) -> Entity:

        # 组装实体对象
        aux_data = {
            'category': 'Turbofan'
        }

        return Entity(name=entity_name, raw_data=raw_data, aux_data=aux_data)
