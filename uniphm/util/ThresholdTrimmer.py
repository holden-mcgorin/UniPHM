from uniphm.engine.Result import Result


class ThresholdTrimmer:
    """
    去掉超过阈值的部分
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def trim(self, result: Result) -> Result:
        trimmed_result = result.__copy__()

        # 裁剪均值线
        mean = trimmed_result.mean
        if mean is not None:
            reach_index_mean = -1
            for i in range(mean.shape[0]):
                if mean[i] > self.threshold:
                    mean[i] = self.threshold
                    reach_index_mean = i
                    break
            if reach_index_mean != -1:
                trimmed_result.mean = mean[:reach_index_mean + 1]

        # 裁剪不确定性区间
        if trimmed_result.lower is not None and trimmed_result.upper is not None:
            lower = trimmed_result.lower
            upper = trimmed_result.upper

            # 检查超过开始阈值的下标
            reach_index_low, reach_index_up = -1, -1
            length = len(upper)
            for i in range(length):
                if upper[i] > self.threshold:
                    reach_index_up = i
                    break
            for i in range(length):
                if lower[i] > self.threshold:
                    reach_index_low = i
                    break

            # 贴合阈值
            if reach_index_up != -1:
                for i in range(reach_index_up, length):
                    upper[i] = self.threshold
            if reach_index_low != -1:
                for i in range(reach_index_low, length):
                    upper[i] = self.threshold

            # 去除多余数据
            upper = upper[:reach_index_low]
            lower = lower[:reach_index_low]

            trimmed_result.lower = lower
            trimmed_result.upper = upper

        return trimmed_result
