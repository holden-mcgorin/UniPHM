import textwrap
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Dict, List, Union

from uniphm.data import Dataset
from uniphm.util.Logger import Logger


class ABCTrainer(ABC):
    """
    训练器接口
    回调函数在这里处理
    钩子函数（Hook Function）
    在框架代码中预留的可被用户“挂接”执行自定义逻辑的函数。
    框架在特定的时机自动调用这些函数，允许用户在不修改主流程的情况下，自定义行为。
    """

    def __init__(self, config: dict = None):
        # 初始化训练配置
        self.config = config if config else {}

        # 初始化回调算法
        callbacks = self.config['callbacks'] if self.config.get('callbacks') else []  # 当无回调时转换为空列表
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]  # 当单个回调时转换为单元素列表
        self.config['callbacks'] = callbacks

    def __call__(self, model, train_set: Dataset) -> Union[List, Dict]:
        return self.train(model, train_set)

    def train(self, model, train_set: Dataset) -> Union[List, Dict]:
        """
        训练接口
        :param model:
        :param train_set:
        :return: 损失（单个类型损失返回list，多个类型损失返回dict{list}）
        """
        # ---------------------------------------- 1. 准备训练阶段 -----------------------------------------
        self._on_train_begin(model, train_set)
        self.__log_train_config(train_set)  # 日志记录训练配置信息

        # 训练前置回调
        keep_training = True

        # 初始化损失字典，用来保存不同损失的历史
        loss_history_dicts = {}
        for key in self.losses_keys:
            loss_history_dicts[key] = []

        for callback in self.config['callbacks']:
            keep_training = callback.on_train_begin(model)
            if not keep_training:
                break
        if not keep_training:
            Logger.warning(f'[{self.__class__.__name__}]  Not in the training phase, exiting early')
            return loss_history_dicts

        for epoch in range(1, self.config['epochs'] + 1):
            # ---------------------------------------- 2. epoch 前 -----------------------------------------
            # epoch前置回调
            for callback in self.config['callbacks']:
                keep_training = callback.on_epoch_begin(model, epoch)
                if not keep_training:
                    break
            if not keep_training:
                Logger.warning(f'[{self.__class__.__name__}]  End early before epoch')
                break

            # ---------------------------------------- 3. epoch 中 -----------------------------------------
            losses = self._on_epoch(model, train_set)
            # 打印所有损失
            losses = losses if isinstance(losses, tuple) else tuple([losses])
            losses_str = ' || '.join([f'{self.losses_keys[i]}:{losses[i]:.4f}' for i in range(len(self.losses_keys))])
            Logger.info(f"[{self.__class__.__name__}]  Epoch [{epoch}/{self.config['epochs']}], {losses_str}")
            # 将所有loss记录到字典
            for i in range(len(self.losses_keys)):
                loss_history_dicts[self.losses_keys[i]].append(losses[i] if isinstance(losses, tuple) else losses)

            # ---------------------------------------- 4. epoch 后 -----------------------------------------
            # epoch后置回调
            for callback in self.config['callbacks']:
                keep_training = callback.on_epoch_end(model, epoch, losses)
                if not keep_training:
                    break
            if not keep_training:
                Logger.warning(f'[{self.__class__.__name__}]  End early after epoch')
                break

        # ---------------------------------------- 5. 训练结束阶段 -----------------------------------------
        # 训练结束回调
        for callback in self.config['callbacks']:
            keep_training = callback.on_train_end(model)
            if not keep_training:
                break
        if not keep_training:
            Logger.warning(f'[{self.__class__.__name__}]  At the end of the training phase, exit early')
            return loss_history_dicts

        return loss_history_dicts

    @property
    def losses_keys(self) -> tuple:
        """
        作为 loss_history_dicts 的 key
        :return:
        """
        # 当多个损失时示例
        # return tuple('loss_1', 'loss_2')
        return tuple([self.config['criterion'].__class__.__name__])

    @abstractmethod
    def _on_train_begin(self, model, train_set: Dataset):
        """
        训练前准备
        需要完成以下职责：
        1. 初始化DataLoader、优化器并传入object_dict
        2. 统一数值类型（如float32）
        3. 统一设备（如cuda或cpu）
        :return:
        """
        pass

    @abstractmethod
    def _on_epoch(self, model, train_set: Dataset) -> Union[float, List[float]]:
        """
        单次迭代代码
        需要完成以下职责：
        1. 单次训练逻辑
        2. Logger打印单次epoch训练平均损失
        :return: 单次迭代的平均损失（当多个损失时返回list）
        """
        pass

    def __log_train_config(self, train_set):
        config_str = f'\n[Trainer]  Start training by {self.__class__.__name__}:'
        if train_set.name is None:
            config_str += f'\n\ttraining set: unnamed'
        else:
            config_str += f'\n\ttraining set: {textwrap.shorten(train_set.name, width=40, placeholder="…")}'

        for k, v in self.config.items():
            if k == 'optimizer':
                format_v = v.__class__.__name__
            elif k == 'data_loader':
                continue
            elif k == 'tasks':
                format_v = ' + '.join(f"{x['weight']} * {x['criterion'].__class__.__name__}" for x in v)
            elif isinstance(v, Iterable) and all(hasattr(i, '__class__') for i in v):
                format_v = "[" + ", ".join(type(i).__name__ for i in v) + "]"
            else:
                format_v = str(v)
            config_str += f'\n\t{k}: {format_v}'

        Logger.info(config_str)
