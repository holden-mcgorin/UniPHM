from uniphm.data.loader.PHM2012Loader import PHM2012Loader
from uniphm.data.loader.XJTULoader import XJTULoader
from uniphm.data.process.EntityPipeline import EntityPipeline
from uniphm.data.process.array.KurtosisProcessor import KurtosisProcessor
from uniphm.data.process.array.MaxProcessor import MaxProcessor
from uniphm.data.process.array.MeanAbsProcessor import MeanAbsProcessor
from uniphm.data.process.array.MeanProcessor import MeanProcessor
from uniphm.data.process.array.MinProcessor import MinProcessor
from uniphm.data.process.array.PTPProcessor import PTPProcessor
from uniphm.data.process.array.RMSProcessor import RMSProcessor
from uniphm.data.process.array.SkewProcessor import SkewProcessor
from uniphm.data.process.array.StdProcessor import StdProcessor
from uniphm.data.process.array.VarProcessor import VarProcessor
from uniphm.data.process.entity.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from uniphm.util.Plotter import Plotter

# 定义数据加载器
# data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
data_loader = PHM2012Loader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')

# 选择性读取数据(include模式)
bearing = data_loader("Bearing1_1", include='Horizontal Vibration')
Plotter.entity(bearing)

# 选择性读取数据(exclude模式)
bearing = data_loader("Bearing1_1", exclude='Horizontal Vibration')
Plotter.entity(bearing)

# 读取完整数据
bearing = data_loader("Bearing1_1")
Plotter.entity(bearing)

# 特征提取流水线（即时模式）
pipeline = EntityPipeline()
pipeline.step(
    entity=bearing,
    processor=RMSProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='H_RMS'
)
pipeline.step(
    entity=bearing,
    processor=KurtosisProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_RMS'
)
Plotter.entity(bearing)

# 特征提取流水线（注册模式）
pipeline = EntityPipeline()
pipeline.register(
    processor=RMSProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_RMS'
)
pipeline.register(
    processor=KurtosisProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Kurtosis'
)
pipeline.register(
    processor=MinProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Min'
)
pipeline.register(
    processor=MeanProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Mean'
)
pipeline.register(
    processor=MaxProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Max'
)
pipeline.register(
    processor=MeanAbsProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_MeanAbs'
)
pipeline.register(
    processor=PTPProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_PTP'
)
pipeline.register(
    processor=SkewProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Skew'
)
pipeline.register(
    processor=VarProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Var'
)
pipeline.register(
    processor=StdProcessor(data_loader['continuum']),
    input_key='Vertical Vibration',
    output_key='V_Std'
)
pipeline.register(
    processor=KurtosisProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Kurtosis'
)
pipeline.register(
    processor=MinProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Min'
)
pipeline.register(
    processor=MeanProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Mean'
)
pipeline.register(
    processor=MaxProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Max'
)
pipeline.register(
    processor=MeanAbsProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_MeanAbs'
)
pipeline.register(
    processor=PTPProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_PTP'
)
pipeline.register(
    processor=SkewProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Skew'
)
pipeline.register(
    processor=VarProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Var'
)
pipeline.register(
    processor=StdProcessor(data_loader['continuum']),
    input_key='Horizontal Vibration',
    output_key='H_Std'
)

pipeline.execute(bearing)
Plotter.entity(bearing)

# ThreeSigmaFPTCalculator作为特殊的数据处理器，因为不产生新的特征，因此可以省略output_key
pipeline.step(
    entity=bearing,
    processor=ThreeSigmaFPTCalculator(),
    input_key='H_RMS'
)
Plotter.entity(bearing)

# 选择性可视化
Plotter.entity(bearing, ['H_RMS', 'H_Kurtosis', 'H_Min', 'H_Mean', 'H_Max', 'H_MeanAbs'])
