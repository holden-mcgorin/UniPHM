import copy

import torch

from uniphm.data import Dataset
from uniphm.engine.Result import Result
from uniphm.engine.tester.ABCTester import ABCTester


class BaseTester(ABCTester):
    """
    基础测试器
    """

    def test(self, model, test_set: Dataset) -> Result:
        # 使用默认配置补充缺少项
        default_config = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'dtype': torch.float32
        }
        for k, v in default_config.items():
            self.config.setdefault(k, v)

        # 输入数据类型转换
        input_data = torch.from_numpy(test_set.x).to(dtype=self.config['dtype'], device=self.config['device'])

        # 输入模型
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        model.train()

        # 结果打包
        result = Result(y_hat=output.cpu().numpy(),
                        name=test_set.name,
                        entity_map=copy.deepcopy(test_set.entity_map))

        # 若配置了归一化器则反归一化结果
        if self.config.get('norm') is not None:
            result = self.config['norm'].denorm_result(result)

        return result
