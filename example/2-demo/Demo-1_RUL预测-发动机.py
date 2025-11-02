import torch
from torch import nn
from uniphm.data import Dataset, TurbofanRulLabeler
from uniphm.data.loader.CMAPSSLoader import CMAPSSLoader
from uniphm.data.process.array.NormalizationProcessor import NormalizationProcessor
from uniphm.engine.Evaluator import Evaluator
from uniphm.engine.metric.MSE import MSE
from uniphm.engine.metric.MAE import MAE
from uniphm.engine.metric.PHM2008Score import PHM2008Score
from uniphm.engine.metric.PHM2012Score import PHM2012Score
from uniphm.engine.metric.PercentError import PercentError
from uniphm.engine.metric.RMSE import RMSE
from uniphm.engine.tester.BaseTester import BaseTester
from uniphm.engine.trainer.BaseTrainer import BaseTrainer
from uniphm.model.classic.Transfromer import TransformerEncoderModel
from uniphm.engine.callback.CheckGradientsCallback import CheckGradientsCallback
from uniphm.engine.callback.EarlyStoppingCallback import EarlyStoppingCallback
from uniphm.engine.callback.TensorBoardCallback import TensorBoardCallback
from uniphm.util.Cache import Cache
from uniphm.util.Plotter import Plotter

cache_model = False
Plotter.DPI = 70
Plotter.SIZE = (10, 7.5)

# 基本信息
data_loader = CMAPSSLoader('D:\\data\\dataset\\CMAPSSData')
sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRF', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
sensor_norm = ['norm_' + s for s in sensor]

# 发动机数据可视化
turbofan = data_loader('FD001_train_1', include=sensor)
Plotter.entity(turbofan)
Plotter.entity(turbofan, sensor_norm)

# 配置标签器
labeler_all_sample = TurbofanRulLabeler(window_size=30, window_step=1, max_rul=130)
labeler_last_sample = TurbofanRulLabeler(window_size=30, window_step=1, max_rul=130, last_sample=True)

# 加载训练集
turbofans_train = data_loader.batch_load('FD001_train', include=sensor)
train_set = Dataset()
for turbofan in turbofans_train:
    train_set.append_entity(labeler_all_sample(turbofan, sensor_norm))
train_set.name = 'FD001_train'

# 加载测试集
turbofan_test = data_loader.batch_load('FD001_test', include=sensor)
test_set_last_sample = Dataset()
test_set_all_sample = Dataset()
for turbofan in turbofan_test:
    test_set_last_sample.append_entity(labeler_last_sample(turbofan, sensor_norm))
    test_set_all_sample.append_entity(labeler_all_sample(turbofan, sensor_norm))
test_set_last_sample.name = 'FD001_test'

# 标签归一化训练能显著降低梯度消失的概率
norm = NormalizationProcessor(arr_min=0, arr_max=130, mode='[0,1]')

# 配置测试算法
test_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
    'norm': norm
}
tester = BaseTester(config=test_config)

# 配置训练算法
train_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
    'epochs': 120,
    'batch_size': 512,
    'lr': 0.01,
    'weight_decay': 0.01,
    'criterion': nn.MSELoss(),
    'norm': norm,
    'callbacks': [
        EarlyStoppingCallback(patience=20,
                              val_set=train_set,
                              metric=RMSE(),
                              tester=tester),
        CheckGradientsCallback(),
        TensorBoardCallback()]
}
trainer = BaseTrainer(config=train_config)

# 加载模型
model = Cache.load('CMAPSS_model', cache_model)
if model is None:
    model = TransformerEncoderModel(14, 1)

    # 开始训练
    loss_history_dicts = trainer(model=model, train_set=train_set)
    Plotter.loss(loss_history_dicts)
    Cache.save(model, 'CMAPSS_model')

# 配置评价指标与评价器
evaluator = Evaluator(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())

# 仅测试最后一个时间窗口
result_last_sample = tester(model, test_set_last_sample)
evaluator(test_set_last_sample, result_last_sample, title='Evaluation of last sample')
Plotter.rul_ascending(test_set_last_sample, result_last_sample,
                      is_scatter=False, label_x='Test Engine ID', label_y='RUL (cycle)')

# 测试所有时间窗口
result_all_sample = tester(model, test_set_all_sample)
evaluator(test_set_all_sample, result_all_sample, title='Evaluation of all sample')
Plotter.rul_end2end(test_set_all_sample, result_all_sample,
                    is_scatter=False, label_x='Time (cycle)', label_y='RUL (cycle)')

# 仅绘图某个发动机预测结果
Plotter.rul_end2end(test_set_all_sample.get('FD001_test_76'), result_all_sample.get('FD001_test_76'),
                    is_scatter=False)
