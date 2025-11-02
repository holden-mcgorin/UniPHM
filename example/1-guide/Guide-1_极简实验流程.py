from uniphm.data.labeler.BearingRulLabeler import BearingRulLabeler
from uniphm.data.loader.XJTULoader import XJTULoader
from uniphm.engine.Evaluator import Evaluator
from uniphm.engine.metric.MSE import MSE
from uniphm.engine.metric.MAE import MAE
from uniphm.engine.metric.PHM2008Score import PHM2008Score
from uniphm.engine.metric.PHM2012Score import PHM2012Score
from uniphm.engine.metric.PercentError import PercentError
from uniphm.engine.metric.RMSE import RMSE
from uniphm.engine.tester.BaseTester import BaseTester
from uniphm.engine.trainer.BaseTrainer import BaseTrainer
from uniphm.model.basic.CNN import CNN

# Step 1: Load raw data
data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
bearing = data_loader.load_entity('Bearing1_1')

# Step 2: Construct dataset
labeler = BearingRulLabeler(2048)
dataset = labeler.label(bearing, 'Horizontal Vibration')
train_set, test_set = dataset.split_by_ratio(0.7)

# Step 3: Train model
model = CNN(input_size=2048, output_size=1)
trainer = BaseTrainer()
trainer.train(model, train_set)

# Step 4: Test model
tester = BaseTester()
result = tester.test(model, test_set)

# Step 5: Evaluate results
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
evaluator(test_set, result)
