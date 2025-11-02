from uniphm.data.loader.CMAPSSLoader import CMAPSSLoader
from uniphm.util.Plotter import Plotter

"""
为方便归一化计算，CMAPSSLoader自动计算归一化特征，无需像轴承数据那样再用EntityPipeline处理
"""


data_loader = CMAPSSLoader('D:\\data\\dataset\\CMAPSSData')
sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRF', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
sensor_norm = ['norm_' + s for s in sensor]

entity = data_loader('FD001_train_1', include=sensor)
Plotter.entity(entity)
Plotter.entity(entity, 'T50')
Plotter.entity(entity, sensor, title='raw sensor')
Plotter.entity(entity, sensor_norm, title='norm sensor')
