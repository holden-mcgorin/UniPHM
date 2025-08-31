<div align="center">
    <h1>⚡ FastPHM ⚡</h1>
</div>

<div align="center"><h3>✨ 
快速上手、快速运行的 PHM 实验框架！✨</h3></div>

<div align="center">

[![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Gitee star](https://gitee.com/holdenmcgorin/FastPHM/badge/star.svg?theme=dark)](https://gitee.com/holdenmcgorin/FastPHM/stargazers)
[![GitHub stars](https://img.shields.io/github/stars/holden-mcgorin/FastPHM.svg?style=social)](https://github.com/holden-mcgorin/FastPHM/stargazers)

</div>

<div align="center">

[简体中文](README.md) | [English](readme-en.md)

</div>

<div align="center">
    <a href="https://gitee.com/holdenmcgorin/FastPHM" target="_blank">Gitee</a> •
    <a href="https://github.com/holden-mcgorin/FastPHM" target="_blank">GitHub</a>
</div>

###  
> 本框架面向故障预测与健康管理（PHM）领域，专为基于深度学习方法的 PHM 实验（如**剩余使用寿命（RUL）预测**、**故障诊断**、**异常检测**等任务）设计，旨在提供一个高效、易用、低资源消耗的实验平台，帮助用户快速上手并搭建 PHM 相关实验流程，大幅简化代码开发工作，提高研究与开发效率。  
> 本项目将持续更新，逐步集成基于该框架实现的论文复现案例，欢迎大家 ⭐star 项目并多多交流！

## 🚀    功能概览
- ✅ **兼容多种深度学习框架**：支持 PyTorch、TensorFlow、Pyro 等主流框架灵活构建模型

- 📦 **数据集自动导入**：内置支持 XJTU-SY、PHM2012、C-MAPSS、PHM2008 等常用数据集

- 📝 **自动记录实验配置与结果**：包括模型结构、正则化系数、迭代次数、采样策略等参数

- 🔁 **每个 Epoch 支持自定义回调**：内置 EarlyStopping、TensorBoard，均通过回调实现

- 🛠 **模型训练过程可监控**：支持 TensorBoard 训练可视化与梯度异常（如消失/爆炸）记录与报警

- 🔍 **多种预处理与特征提取方法**：滑动窗口、归一化、均方根、峭度等信号处理手段

- 🧠 **多种退化阶段划分策略**：支持 3σ 原则、FPT（First Predictable Time）等算法

- 🔮 **多种预测方式支持**：端到端预测、单/多步滚动预测、不确定性建模等

- 📊 **实验结果可视化**：支持混淆矩阵、退化阶段图、预测结果曲线、注意力热图等

- 📁 **多种文件格式支持**：模型、数据、缓存与结果支持 CSV、PKL 等多种格式导入与导出

- 📈 **内置多种评价指标**：MAE、MSE、RMSE、MAPE、PHM2012 Score、NASA Score 等

- 🔧 **灵活组件化设计**：支持用户快速扩展和接入自定义算法模块


## 💻    实验示例

以下是完成一次 PHM 实验（RUL预测）的**极简流程示例**，仅包含**数据加载、模型训练与评估**的最基本步骤，便于快速上手。

> 本示例专注于最小可运行流程，框架还支持更强大的功能，详见项目根目录下的 `Notebook 示例`。

只需十几行代码，即可完成端到端实验流程：

```python
# Step 1: Initialize the data loader and labeler
data_loader = CMAPSSLoader('D:\\data\\dataset\\CMAPSSData')
labeler = TurbofanRulLabeler(window_size=30, max_rul=130)

# Step 2.1: Load and label the training dataset
turbofans_train = data_loader.batch_load('FD001_train', columns_to_drop=[0, 1, 2, 3, 4, 8, 9, 13, 19, 21, 22])
train_set = Dataset()
for turbofan in turbofans_train:
    train_set.add(labeler(turbofan))

# Step 2.2: Load and label the test dataset
turbofans_test = data_loader.batch_load('FD001_test', columns_to_drop=[0, 1, 2, 3, 4, 8, 9, 13, 19, 21, 22])
test_set = Dataset()
for turbofan in turbofans_test:
    test_set.add(labeler(turbofan))

# Step 3: Initialize the model and trainer, then begin training
model = MyLSTM()
trainer = BaseTrainer()
trainer.train(model, train_set)

# Step 4: Evaluate the trained model on the test dataset
tester = BaseTester()
result = tester.test(model, test_set)

# Step 5: Configure evaluation metrics and compute performance scores
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
evaluator(test_set, result)
```

在添加可视化代码和其他功能组件后，程序在 CMD 环境中的运行效果如下所示。  
（ 该示例展示程序在 CMD 环境下的运行过程。实际上，在本地开发时，推荐使用如 PyCharm、VSCode、Jupyter Notebook 等集成开发环境（IDE））

![demo](show.gif)


## 📚 论文复现
> 本项目支持快速搭建 PHM 相关实验流程，并已尝试复现若干学术论文中的方法与实验结果。   
> 本项目对原作者的研究成果保持充分尊重。若复现结果与原论文存在一定偏差，可能是实现方式或实验条件不同，也可能是复现过程存在疏漏。欢迎读者在 issue 区指出问题或提出建议。  


### ✅ 已复现论文示例

整理中

[//]: # (| 论文标题 | 出处 | 方法关键词 | 数据集 | 复现文件路径 |)

[//]: # (|----------|------|------------|--------|------------------|)

[//]: # (| A BiGRU method for RUL prediction | Measurement, 2020 | BiGRU | C-MAPSS | `reproduction/Bigru_RUL.ipynb` |)

[//]: # (| Prognostics uncertainty using Bayesian deep learning | IEEE TIE, 2019 | Bayesian DL | C-MAPSS | `reproduction/Bayesian_Uncertainty.py` |)


## 📂    文件结构说明
- fastphm —— 框架代码
- doc —— 框架详细说明文档（编写自定义组件时建议查看）
- example —— 试验代码示例（原生python）

### 📦 数据集来源

| 名称             | 描述                                  | 链接                                                                 |
|------------------|-------------------------------------|----------------------------------------------------------------------|
| XJTU-SY 数据集   | 西安交通大学发布的滚动轴承寿命退化数据                 | [点击访问](https://biaowang.tech/xjtu-sy-bearing-datasets/)         |
| PHM2012 数据集   | IEEE PHM 2012 大赛提供的轴承故障数据，包含多个运行工况  | [点击访问](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) |
| C-MAPSS 数据集   | NASA 提供的模拟涡扇发动机退化数据，广泛用于 RUL 预测任务   | [点击访问](https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) |
| PHM2008 数据集   | NASA 提供的早期涡轮设备预测数据集，来源于 PHM08 数据挑战  | [点击访问](https://data.nasa.gov/download/nk8v-ckry/application%2Fzip) |
| NASA 数据集仓库  | NASA 智能系统部汇总的多个设备健康数据集，覆盖多领域 PHM 任务 | [点击访问](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |


## ⚠    注意事项
> - 该框架使用Python 3.8.10编写，使用其他版本python运行可能会出现兼容性问题，若出现问题欢迎在issue提问
> - 读取数据集时，不要改变原始数据集内部文件的相对位置（可以只保留部分数据），不同的位置可能导致无法读取数据



觉得项目写的还行的大佬们点个star呗，觉得哪里写得不行的地方也欢迎issue一下，您的关注是我最大的更新动力！😀


##### @键哥工作室 @AndrewStudio
##### 📧 个人邮箱：andrewstudio@foxmail.com
##### 🌐 个人网站：http://8.138.46.66/#/home

