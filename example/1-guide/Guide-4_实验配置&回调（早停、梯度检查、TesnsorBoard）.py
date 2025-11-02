import torch
from torch import nn

from uniphm.data import Dataset
from uniphm.data.labeler.BearingFaultLabeler import BearingFaultLabeler
from uniphm.data.loader.XJTULoader import XJTULoader, Fault
from uniphm.data.process.EntityPipeline import EntityPipeline
from uniphm.data.process.array.RMSProcessor import RMSProcessor
from uniphm.data.process.entity.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from uniphm.engine.Evaluator import Evaluator
from uniphm.engine.callback.CheckGradientsCallback import CheckGradientsCallback
from uniphm.engine.callback.EarlyStoppingCallback import EarlyStoppingCallback
from uniphm.engine.callback.TensorBoardCallback import TensorBoardCallback
from uniphm.engine.metric.Accuracy import Accuracy
from uniphm.engine.metric.WeightedF1Score import WeightedF1Score
from uniphm.engine.tester.BaseTester import BaseTester
from uniphm.engine.trainer.BaseTrainer import BaseTrainer
from uniphm.model.basic.CNN import CNN
from uniphm.util.Plotter import Plotter

# 配置数据加载器
Plotter.DPI = 80
data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')

# 配置特征提取流水线
pipeline = EntityPipeline()
pipeline.register(
    processor=RMSProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_RMS'
)
pipeline.register(
    processor=ThreeSigmaFPTCalculator(),
    input_key='H_RMS',
)

# 构造数据集
fault_types = [Fault.NC, Fault.OF, Fault.IF, Fault.CF]
labeler = BearingFaultLabeler(2048, fault_types, is_multi_hot=False)
dataset = Dataset()
for bearing_name in ['Bearing1_1', 'Bearing1_2', 'Bearing1_4', 'Bearing2_1', 'Bearing2_3']:
    bearing = data_loader(bearing_name, 'Horizontal Vibration')
    pipeline.execute(bearing)
    dataset.add(labeler(bearing, 'Horizontal Vibration'))

# 划分数据集：训练集（70%）、验证集（15%）、测试集（15%）
# 训练集：用于模型参数学习
# 验证集：用于模型调参、早停（early stopping）、选择最佳模型
# 测试集：只用于最终性能评估，不参与任何训练和调参过程
train_set, test_set = dataset.split_by_ratio(0.7)
val_set, test_set = test_set.split_by_ratio(0.5)

# 配置测试算法
# 因为没有什么可设置的参数，其实可以使用默认配置。这里使用自定义配置仅为示例
test_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
}
tester = BaseTester(test_config)


# 自定义配置训练算法&早停、tensorboard、梯度检测回调
# EarlyStoppingCallback，早停回调，耐心10，使用准确率指标选取最佳模型
# TensorBoardCallback，自动将模型参数保存至tensorboard，文件保存至runs文件夹下
# CheckGradientsCallback，检查模型梯度，当过大（梯度爆炸），过小（梯度消失）时日志记录警告信息
train_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.01,
    'weight_decay': 0.0,
    'criterion': nn.CrossEntropyLoss(),
    'callbacks': [
        EarlyStoppingCallback(patience=10,
                              val_set=val_set,
                              metric=Accuracy(),
                              tester=tester),
        TensorBoardCallback(),
        CheckGradientsCallback()
    ]
}
trainer = BaseTrainer(train_config)

# 定义模型并训练
model = CNN(2048, len(fault_types))
losses = trainer.train(model, train_set)

Plotter.loss(losses)

result = tester.test(model, test_set)

# 预测结果评估
Plotter.confusion_matrix(test_set, result, types=fault_types)

evaluator = Evaluator()
evaluator.add(Accuracy(), WeightedF1Score())
evaluator(test_set, result)
