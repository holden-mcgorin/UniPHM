import os
import re
from enum import Enum
from typing import Dict, Union

import pandas as pd
from pandas import DataFrame

from uniphm.data.Entity import Entity
from uniphm.data.loader.ABCLoader import ABCLoader


class Fault(Enum):
    """
    轴承故障类型枚举
    """
    NC = 'Normal Condition'
    OF = 'Outer Race Fault'
    IF = 'Inner Race Fault'
    CF = 'Cage Fault'
    BF = 'Ball Fault'

    def __str__(self):
        return self.name


class XJTULoader(ABCLoader):
    @property
    def meta(self) -> dict:
        return {
            'frequency': 25600,  # 采样频率
            'continuum': 32768,  # 连续采样区间
            'time_unit': 'minute',  # 时间相关的数据单位为分钟
            'span': 1,  # 连续采样区间代表时长
            'rul': 0  # 因为该数据集是全寿命数据，轴承当前已失效
        }

    fault_type_dict = {
        'Bearing1_1': [Fault.OF],
        'Bearing1_2': [Fault.OF],
        'Bearing1_3': [Fault.OF],
        'Bearing1_4': [Fault.CF],
        'Bearing1_5': [Fault.IF, Fault.OF],
        'Bearing2_1': [Fault.IF],
        'Bearing2_2': [Fault.OF],
        'Bearing2_3': [Fault.CF],
        'Bearing2_4': [Fault.OF],
        'Bearing2_5': [Fault.OF],
        'Bearing3_1': [Fault.OF],
        'Bearing3_2': [Fault.IF, Fault.OF, Fault.CF, Fault.BF],
        'Bearing3_3': [Fault.IF],
        'Bearing3_4': [Fault.IF],
        'Bearing3_5': [Fault.OF],
    }

    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root, condition)
            for bearing_name in os.listdir(condition_dir):
                file_dict[bearing_name] = os.path.join(root, condition, bearing_name)
                entity_dict[bearing_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name: str) -> DataFrame:
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param entity_name:
        :return: Bearing对象（包含raw_data)
        """
        bearing_dir = self._file_dict[entity_name]

        # 读取csv数据并合并
        dataframes = []
        files = sorted(os.listdir(bearing_dir), key=self.__extract_number)
        for file_name in files:
            df = pd.read_csv(os.path.join(bearing_dir, file_name))
            dataframes.append(df)
        dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

        # 规范列名
        dataframe.rename(columns={'Horizontal_vibration_signals': 'Horizontal Vibration',
                                  'Vertical_vibration_signals': 'Vertical Vibration'},
                         inplace=True)

        return dataframe

    def _assemble(self, entity_name: str, df: DataFrame) -> Entity:
        meta = {
            'fault_type': self.fault_type_dict[entity_name],  # 故障类型
            'life': len(df) / self.meta['continuum'] * self.meta['span']
        }
        meta = {**self.meta, **meta}

        # 拆分 DataFrame，每列生成 (n,1) ndarray
        array_dict = {}
        for col in df.columns:
            array_dict[col] = df[[col]].to_numpy()  # (n, 1)

        return Entity(name=entity_name, data=array_dict, meta=meta)

    # 自定义排序函数，从文件名中提取数字
    @staticmethod
    def __extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
