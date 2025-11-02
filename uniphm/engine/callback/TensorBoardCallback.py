from typing import Union, List

import torch
from torch.utils.tensorboard import SummaryWriter

from uniphm.engine.callback.ABCTrainCallback import ABCTrainCallback
import time

from uniphm.util.Logger import Logger


class TensorBoardCallback(ABCTrainCallback):
    def __init__(self, log_dir=None):

        if log_dir is None:
            log_dir = f"tb_logs/exp_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, model, epoch: int, losses: Union[float, List[float]]) -> bool:
        # # 记录损失
        # self.writer.add_scalar('Loss/train', losses, epoch)

        for name, param in model.named_parameters():
            # 检查参数
            if torch.isnan(param).any() or torch.isinf(param).any():
                Logger.error(f"[TensorBoardCallback] NaN/Inf detected in parameter: {name} at epoch {epoch}")
                self.writer.add_text("Error", f"NaN/Inf detected in parameter {name} (epoch {epoch})", epoch)
                return False  # 你也可以 raise RuntimeError 来强制停训

            # 记录参数
            self.writer.add_histogram(f'Params/{name}', param, epoch)
            self.writer.add_scalar(f'ParamMean/{name}', param.data.mean(), epoch)
            self.writer.add_scalar(f'ParamStd/{name}', param.data.std(), epoch)

            # 检查梯度
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    Logger.error(f"[TensorBoardCallback] NaN/Inf detected in gradient: {name} at epoch {epoch}")
                    self.writer.add_text("Error", f"NaN/Inf detected in gradient {name} (epoch {epoch})", epoch)
                    return False

                # 记录梯度
                self.writer.add_histogram(f'Grads/{name}', param.grad, epoch)
                self.writer.add_scalar(f'GradMean/{name}', param.grad.mean(), epoch)
                self.writer.add_scalar(f'GradStd/{name}', param.grad.std(), epoch)

        return True
