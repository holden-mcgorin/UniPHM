import numpy as np

from uniphm.data.Entity import Entity
from uniphm.data.process.entity.ABCEntityProcessor import ABCEntityProcessor
from uniphm.util.Logger import Logger


class ThreeSigmaFPTCalculator(ABCEntityProcessor):
    """
    改进版的3-sigma法则
    """

    def __init__(self, ratio=3, max_consecution=5, consecution_ratio=0.3, healthy_ratio=0.3, min_bound=0.1, max_rms=2):
        """
        :param ratio: 几倍sigma
        :param max_consecution: 最大的连续区间
        :param consecution_ratio: 几次连续出现异常值才判定FPT
        :param healthy_ratio: 默认健康比例
        :param min_bound: 最小阈值增量
        :param max_rms: RMS阈值，超出即为fpt
        """

        # 默认3倍标准差，即3σ
        self.ratio = ratio
        self.healthy_ratio = healthy_ratio
        self.consecution_ratio = consecution_ratio
        self.min_bound = min_bound
        self.max_consecution = max_consecution
        self.max_rms = max_rms

    def run(self, entity: Entity, key: str) -> Entity:
        feat = entity[key].reshape(-1)

        # 如果取全局计算均值和标准差
        sliced_data = feat[:int(len(feat) * self.healthy_ratio)]
        mu = np.mean(sliced_data)
        sigma = np.std(sliced_data)
        mu_plus_sigma = mu + self.ratio * sigma + self.min_bound
        # mu_minus_sigma = mu - self.ratio * sigma

        consecution_max = int(self.consecution_ratio * len(feat))
        if consecution_max > self.max_consecution:
            consecution_max = self.max_consecution

        # print(f'mu_plus_sigma={mu_plus_sigma}')
        # print(f'consecution_max={consecution_max} ')

        fpt_feature = 0
        consecution_count = 0
        success = False
        for i in range(len(feat)):
            x = feat[i]
            if x > mu_plus_sigma or x > self.max_rms:
                consecution_count += 1

                # 两个确认异常标志
                if consecution_count == consecution_max:
                    fpt_feature = i - consecution_max
                    success = True
                    break
                if i == len(feat) - 1:
                    fpt_feature = i - consecution_count
                    success = True
                    break
            else:
                consecution_count = 0

        fpt = entity['life'] * (fpt_feature / len(feat))
        entity['fpt'] = fpt

        if not success:
            Logger.warning('fail to identify FPT, used default value 0')
        return entity
