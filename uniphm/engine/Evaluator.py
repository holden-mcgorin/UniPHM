from typing import List

import pandas as pd

from uniphm.data.Dataset import Dataset
from uniphm.engine.Result import Result
from uniphm.engine.metric import ABCMetric
from uniphm.util.Logger import Logger


class Evaluator:
    """
    指标评价器
    先使用add_metric添加需要的指标
    再调用evaluate计算所有的指标

    两种方法传入评价指标：
        1. 构造传入
        2. 方法传入
    """

    def __init__(self, *metrics: ABCMetric) -> None:
        # 用于保存评价指标
        self.metrics: List[ABCMetric] = list(metrics)

    def __call__(self, test_set: Dataset, result: Result, title: str = None, only_avg=None) -> {}:
        return self.evaluate(test_set, result, title, only_avg)

    def evaluate(self, test_set: Dataset, result: Result, title: str = None, only_avg=None) -> {}:
        """
        根据已经添加的评价指标开始计算
        :param test_set:
        :param result:
        :param title:
        :param only_avg: Ture:只输出平均结果 False:所有entity都会单独评估 None:智能选择
        :return:
        """
        # 验证输入的合法性
        sample_num = test_set.x.shape[0]
        if sample_num != result.y_hat.shape[0]:
            raise Exception(f'测试样本量：{sample_num}与测试结果数量：{result.y_hat.shape[0]} 不匹配')

        test_sets = test_set.split_by_entity()
        results = result.split_by_entity()
        n_row = len(test_sets)
        n_col = len(self.metrics)

        # 所有评估结果用dataframe保存并返回,加一行平均值
        data = [[None for col in range(n_col)] for row in range(n_row + 2 if n_row > 1 else 1)]
        for col in range(n_col):
            total_value = 0.0
            for row in range(n_row):
                value = self.metrics[col].value(test_sets[row], results[row])
                total_value += value * len(test_sets[row])
                data[row][col] = self.metrics[col].format(value)
            if n_row > 1:  # 当实体多于1个时计算平均值
                data[-2][col] = self.metrics[col](test_set, result)  # 全局平均值
                data[-1][col] = self.metrics[col].format(total_value / len(test_set))  # 分组平均值

        row_names = [test_set.name for test_set in test_sets]
        if n_row > 1:  # 当实体多于1个时显示平均值
            row_names += ['mean(global)', 'mean(group)']
        column_names = [metric.name for metric in self.metrics]
        df = pd.DataFrame(data, index=row_names, columns=column_names)

        if only_avg:
            df = df.loc[['mean(global)']]

        title = f'Performance Evaluation' if title is None else title
        Logger.info(f'\n[Evaluator]  {title}:\n{df.to_string(max_rows=None, max_cols=None)}\n')

        return df

    def add(self, *args: ABCMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)

    def clear(self):
        self.metrics = []
