from uniphm.data.labeler.BearingRulLabeler import BearingRulLabeler
from uniphm.data.loader.XJTULoader import XJTULoader
from uniphm.data.process.EntityPipeline import EntityPipeline
from uniphm.data.process.array.RMSProcessor import RMSProcessor
from uniphm.data.process.entity.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from uniphm.engine.trainer.BaseTrainer import BaseTrainer
from uniphm.model.basic.MLP import MLP
from uniphm.util.Cache import Cache

# 读取原始数据
data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
bearing = data_loader("Bearing1_3", 'Horizontal Vibration')

# 提取特征
pipeline = EntityPipeline()
pipeline.step(
    entity=bearing,
    processor=RMSProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_RMS'
)
pipeline.step(
    entity=bearing,
    processor=ThreeSigmaFPTCalculator(),
    input_key='H_RMS',
)

# 构造数据集
labeler = BearingRulLabeler(2048, is_rectified=True, is_squeeze=True)
dataset = labeler(bearing, 'Horizontal Vibration')

# 缓存数据集
Cache.save(dataset, 'dataset')

# 加载数据集
dataset = Cache.load('dataset')

# 训练模型
trainer = BaseTrainer()
model = MLP(2048, 16, 1)
trainer.train(model, dataset)

# 缓存模型
Cache.save(model, 'model')

# 读取模型
model = Cache.load('model')
