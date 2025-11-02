import torch
from torch.nn import MSELoss

from uniphm.data import Dataset
from uniphm.data.loader.XJTULoader import XJTULoader
from uniphm.data.labeler.BearingRulLabeler import BearingRulLabeler
from uniphm.data.process.EntityPipeline import EntityPipeline
from uniphm.data.process.array.RMSProcessor import RMSProcessor
from uniphm.engine.metric.PHM2008Score import PHM2008Score
from uniphm.engine.metric.PHM2012Score import PHM2012Score
from uniphm.engine.metric.PercentError import PercentError
from uniphm.engine.metric.MAE import MAE
from uniphm.engine.metric.MSE import MSE
from uniphm.engine.metric.RMSE import RMSE
from uniphm.data.process.entity.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from uniphm.engine.tester.BaseTester import BaseTester
from uniphm.engine.trainer.BaseTrainer import BaseTrainer
from uniphm.model.basic.CNN import CNN
from uniphm.engine.Evaluator import Evaluator
from uniphm.engine.callback.CheckGradientsCallback import CheckGradientsCallback
from uniphm.engine.callback.EarlyStoppingCallback import EarlyStoppingCallback
from uniphm.engine.callback.TensorBoardCallback import TensorBoardCallback
from uniphm.util.Cache import Cache
from uniphm.util.Plotter import Plotter

cache_dataset = True
# cache_dataset = False
# cache_model = True
cache_model = False

# 获取数据集
dataset = Cache.load('prognosis_bearing_dataset', is_able=cache_dataset)
if dataset is None:
    data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    labeler = BearingRulLabeler(2048, is_from_fpt=False, is_rectified=True)

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

    dataset = Dataset()
    for bearing_name in ['Bearing1_1', 'Bearing1_2', 'Bearing1_3',
                         'Bearing2_1', 'Bearing2_2',
                         'Bearing3_1', 'Bearing3_2']:
        bearing = data_loader(bearing_name, 'Horizontal Vibration')
        pipeline.execute(bearing)
        dataset.append_entity(labeler(bearing, key='Horizontal Vibration'))
    Cache.save(dataset, 'prognosis_bearing_dataset')

# 划分测试集、训练集、验证集
test_set = dataset.include(['Bearing1_1', 'Bearing1_3'])
train_set = dataset.exclude(['Bearing1_1', 'Bearing1_3'])
val_set, train_set = train_set.split_by_ratio(0.3)

# 配置测试算法
test_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
}
tester = BaseTester(config=test_config)

# 配置训练算法
train_config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
    'epochs': 20,
    'batch_size': 256,
    'lr': 0.01,
    'weight_decay': 0.0,
    'criterion': MSELoss(),
    'callbacks': [
        EarlyStoppingCallback(patience=5,
                              val_set=val_set,
                              metric=RMSE(),
                              tester=tester),
        TensorBoardCallback(),
        CheckGradientsCallback()
    ]
}
trainer = BaseTrainer(config=train_config)

# 定义模型并训练
model = Cache.load('prognosis_bearing_model', cache_model)
if model is None:
    model = CNN(2048, 1, end_with_sigmoid=False)
    # 开始训练
    losses = trainer.train(model=model, train_set=train_set)
    Plotter.loss(losses)
    Cache.save(model, 'prognosis_bearing_model')

result = tester.test(model, test_set)
Plotter.rul_end2end(test_set, result, is_scatter=True, label_x='Time (min)', label_y='Relative RUL')

# 预测结果评价
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
evaluator(test_set, result)
