from typing import Union, List

import torch

from uniphm.engine.callback.ABCTrainCallback import ABCTrainCallback

from uniphm.util.Logger import Logger


class CheckGradientsCallback(ABCTrainCallback):
    """
    梯度检查回调器
    当梯度异常（过小/过大/NaN/Inf）时日志输出警告
    """

    def __init__(self, min_threshold=1e-5, max_threshold=1e+3):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def on_epoch_end(self, model, epoch: int, avg_loss: Union[float, List[float]]) -> bool:
        for name, param in model.named_parameters():
            # 检查参数本身
            if torch.isnan(param).any():
                Logger.error(f"[CheckGradients] {name} contains NaN values in parameters! Stop training.")
                return False
            if torch.isinf(param).any():
                Logger.error(f"[CheckGradients] {name} contains Inf values in parameters! Stop training.")
                return False

            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()

                # 检查梯度是否 NaN 或 Inf
                if torch.isnan(param.grad).any():
                    Logger.error(f"[CheckGradients] {name} contains NaN values in gradients! Stop training.")
                    return False
                if torch.isinf(param.grad).any():
                    Logger.error(f"[CheckGradients] {name} contains Inf values in gradients! Stop training.")
                    return False

                # 检查梯度幅度
                if grad_mean < self.min_threshold:
                    Logger.warning(f"[CheckGradients] {name} gradient is very small: {grad_mean:.2e}")
                if grad_mean > self.max_threshold:
                    Logger.warning(f"[CheckGradients] {name} gradient is very large: {grad_mean:.2e}")
            else:
                Logger.warning(f"[CheckGradients] {name} has no gradient")

        return True

