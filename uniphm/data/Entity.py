import pandas as pd
import numpy as np
from typing import Dict, Any, Union


class Entity:
    """
    数据实体类，支持像字典一样访问数据。
    - data: 存储ndarray 形状一般是(n,1)的二维ndarray
    - meta: 存储标量或其他辅助信息

        数据实体以下数据：
        实体名:如：
            Bearing1_2
            FD001_test_89

        数据
        ---------------   原始数据：传感器收集到的时序数据    ---------------
            sensor_signals     默认命名 or 未命名，如 C-MAPSS 多传感器数据
            vibration_signal
                horizontal_vibration    水平振动信号
                vertical_vibration      垂直振动信号
            temperature_signal      温度信号
        ---------------   特征数据：原始数据经过处理后得到的时序特征数据    ---------------
            RMS     滑动窗口得到的均方根
            max     滑动窗口得到的最大值
            min     滑动窗口得到的最小值
            avg     滑动窗口得到的平均值
            norm    归一化后的原始数据

        元数据：
            ---------------   数据集标注的标量或由原始数据计算得出的标量数据    ---------------
            category    实体类别，如：Bearing、Turbofan、Battery
            fault_type  故障类型，如：Normal Condition、Outer Race Fault、HPC Degradation
            condition   工况，如：ONE (Sea Level)、SIX
            frequency   采样频率，如：
            continuum   此轴承的连续采样区间长度
            threshold   失效阈值
            time_unit   时间单位，如：second、minute、hour、day、month、year
            ---------------   下述时间标量的时间单位均为 time_unit    ---------------
            span        此轴承连续采样代表的时间
            life        全寿命时长 or 生命周期时长
            rul         剩余使用寿命
            fpt         first predict time 首次预测点 or 早期故障点
            eol         end of life 完全失效时间点 or 失效阈值时间
    """

    def __init__(self, name: str,
                 data: Dict[str, np.ndarray] = None,
                 meta: Dict[str, Any] = None):
        """
        :param name:
        :param data: 一般是(n,1)的二维ndarray
        :param meta: 其他辅助性的元数据
        """
        self.name = name
        self.data = data if data is not None else {}
        self.meta = meta if meta is not None else {}
        self._validate_all()

    # ===== 字典风格访问 =====
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif key in self.meta:
            return self.meta[key]
        else:
            raise KeyError(f"No data named '{key}' found.")

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            self.data[key] = value
        else:
            self.meta[key] = value
        self._validate_single(key, value)

    def __delitem__(self, key):
        if key in self.data:
            del self.data[key]
        elif key in self.meta:
            del self.meta[key]
        else:
            raise KeyError(f"No data named '{key}' found.")

    def __contains__(self, key):
        return key in self.data or key in self.meta

    def keys(self):
        return list(self.data.keys()) + list(self.meta.keys())

    # ===== 数据验证 =====
    @staticmethod
    def _validate_single(key, value):
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError(f"ndarray '{key}' is empty.")
            if not np.isfinite(value).all():
                raise ValueError(f"ndarray '{key}' contains non-finite values (NaN/Inf).")

    def _validate_all(self):
        for k, v in self.data.items():
            self._validate_single(k, v)

    # ===== 获取数据方法 =====
    def get(self, key):
        return self[key]

    # ===== 打印信息 =====
    def __repr__(self):
        return f"<Entity {self.name}: {len(self.data)} ndarray, {len(self.meta)} meta items>"

# if __name__ == '__main__':
#     entity = Entity("Bearing1")
#     entity["sensor"] = pd.DataFrame(np.random.rand(100, 3), columns=["x", "y", "z"])
#     entity["features"] = np.random.rand(100, 10, 5)  # 高维数组
#     entity["life"] = 150
#     entity["fpt"] = 50
#
#     print(entity["sensor"].shape)
#     print(entity["features"].shape)
#     print(entity["life"])
#
#     # 遍历所有数据
#     for key in entity.keys():
#         print(key, type(entity[key]))
