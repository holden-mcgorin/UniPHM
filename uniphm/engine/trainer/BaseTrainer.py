from uniphm.data import Dataset
from uniphm.data.process.array.NormalizationProcessor import NormalizationProcessor
from uniphm.engine.trainer.ABCTrainer import ABCTrainer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer(ABCTrainer):
    """
    基础训练器
    """

    def _on_train_begin(self, model, train_set: Dataset):
        # 使用默认配置补充缺少项
        default_config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'dtype': torch.float32,
            'epochs': 100,
            'batch_size': 256,
            'criterion': nn.MSELoss(),
            'lr': 0.01,
            'weight_decay': 0.0,
        }
        for k, v in default_config.items():
            self.config.setdefault(k, v)

        config = self.config

        # 初始化模型
        model.to(device=config['device'], dtype=config['dtype'])

        # 若配置了归一化器则对训练数据标签归一化
        if config.get('norm') is not None:
            train_set = config['norm'].norm_label(train_set)

        # 初始化数据加载器
        if config.get('data_loader') is None:
            x = torch.tensor(train_set.x, dtype=config['dtype'], device=config['device'])
            # 当损失函数是交叉熵函数时需要设置 config['y_dtype'] 为 torch.long
            if isinstance(config['criterion'], nn.CrossEntropyLoss):
                y = torch.tensor(train_set.y, dtype=torch.long, device=config['device']).reshape(-1)
            else:
                y = torch.tensor(train_set.y, dtype=config['dtype'], device=config['device'])
            config['data_loader'] = DataLoader(TensorDataset(x, y), batch_size=config['batch_size'],
                                               shuffle=True)

        # 初始化优化器
        if config.get('optimizer') is None:
            config['optimizer'] = optim.Adam(model.parameters(),
                                             lr=config['lr'],
                                             weight_decay=config['weight_decay'])

    def _on_epoch(self, model, train_set: Dataset):
        config = self.config
        model.train()

        total_loss = 0.0
        for inputs, labels in config['data_loader']:
            config['optimizer'].zero_grad()  # 梯度清零
            outputs = model(inputs)

            # 标签反归一化计算损失
            if isinstance(config.get('norm'), NormalizationProcessor):
                outputs = config['norm'].denormalize(outputs)
                labels = config['norm'].denormalize(labels)

            loss = config['criterion'](outputs, labels)
            loss.backward()  # 反向传播
            config['optimizer'].step()  # 更新权重

            total_loss += loss.item()
        avg_loss = total_loss / len(config['data_loader'])

        return avg_loss
