import copy
from typing import Union, List

from uniphm.data import Dataset
from uniphm.engine.metric.ABCMetric import ABCMetric
from uniphm.engine.tester.ABCTester import ABCTester
from uniphm.engine.callback.ABCTrainCallback import ABCTrainCallback
from uniphm.util.Logger import Logger
from uniphm.util.Cache import Cache
import uuid


class EarlyStoppingCallback(ABCTrainCallback):
    """
    两种模式
    1. 在训练过程中选择 loss 最小的模型
    2. 选择在验证集上指标最佳的模型
    """

    def __init__(self, patience=5, delta=0.0,
                 val_set: Dataset = None, metric: ABCMetric = None, tester: ABCTester = None,
                 cache_in_memory=True,
                 task_id=0):
        self.patience = patience  # 容忍验证损失不下降的轮数
        self.delta = delta  # 最小改善值
        self.counter = 0  # 当前累计不提升轮数
        self.min_loss = None  # 当前最佳分数（一般为 -val_loss）
        self.cache_in_memory = cache_in_memory  # 最佳模型缓存方法 1.内存缓存 2.硬盘缓存
        self.best_model = None  # 1. 内存缓存
        self.cache_id = None  # 2. 硬盘缓存

        # 模式二（在验证集测试获取最佳模型）
        self.val_set = val_set
        self.metric = metric
        self.tester = tester

        # 兼容多任务模型
        self.task_id = task_id

    def on_train_begin(self, model) -> bool:

        if not self.cache_in_memory:
            self.cache_id = str(uuid.uuid4())

        loss = float('inf')

        # 模式二
        if self.val_set and self.metric:
            result = self.tester.test(model, self.val_set)
            val_set = self.val_set

            #  兼容多任务模型
            if isinstance(result, tuple):
                val_set = self.val_set.split_by_label()[self.task_id]
                result = result[self.task_id]

            score = self.metric.value(val_set, result)
            if self.metric.is_higher_better:
                loss = -1 * score
            else:
                loss = score
            Logger.debug(
                f'[EarlyStopping]  On the validation set {val_set.name},'
                f' the {self.metric.name} is {self.metric.format(score)}')

        self.min_loss = loss
        self.__save_best_model(model)

        return True

    def on_epoch_end(self, model, epoch: int, losses: Union[float, List[float]]) -> bool:

        loss = None

        # 模式二
        if self.val_set and self.metric:
            result = self.tester.test(model, self.val_set)
            val_set = self.val_set

            #  兼容多任务模型
            if isinstance(result, tuple):
                val_set = self.val_set.split_by_label()[self.task_id]
                result = result[self.task_id]

            score = self.metric.value(val_set, result)
            if self.metric.is_higher_better:
                loss = -1 * score
            else:
                loss = score
            Logger.debug(
                f'[EarlyStopping]  On the validation set {val_set.name},'
                f' the {self.metric.name} is {self.metric.format(score)}')

        if loss is None:
            loss = sum(losses) if isinstance(losses, list) else losses

        # 当性能没有提升
        elif loss > self.min_loss - self.delta:
            self.counter += 1
            Logger.debug(f"[EarlyStopping]  No improvement for [{self.counter}/{self.patience}] epochs")
            if self.counter >= self.patience:
                self.__load_best_model(model)
                return False
        # 当性能提升
        else:
            self.min_loss = loss
            self.__save_best_model(model)
            self.counter = 0

        return True

    def on_train_end(self, model) -> bool:
        self.__delete_best_model()
        return True

    def __load_best_model(self, model):
        if self.cache_in_memory:
            model.load_state_dict(self.best_model)
        else:
            model.load_state_dict(Cache.load(f"{self.cache_id}.pth"))

    def __save_best_model(self, model):
        if self.cache_in_memory:
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            Cache.save(model.state_dict(), f"{self.cache_id}.pth")

    def __delete_best_model(self):
        if self.cache_in_memory:
            pass
        else:
            Cache.delete(f"{self.cache_id}.pth")
