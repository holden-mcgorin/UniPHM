"""
co-occurrence
共现处理
"""
import numpy as np
from numpy import ndarray

from uniphm.data import Dataset


def co_oc_matrix(label_matrix: ndarray):
    if not isinstance(label_matrix, np.ndarray):
        raise TypeError("输入必须是NumPy数组。")

    if label_matrix.ndim != 2:
        raise ValueError("输入必须是二维数组，形状为(num_samples, num_labels)。")

    co_matrix = label_matrix.T @ label_matrix
    return co_matrix


def norm_co_oc_matrix(co_matrix: ndarray):
    # 共现矩阵归一化
    diag = np.diag(co_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = co_matrix / diag[:, None]
        norm_matrix[np.isnan(norm_matrix)] = 0

    # # 生成概率矩阵归一化版本
    # diag = np.diag(co_matrix)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     norm_matrix = co_matrix / diag[:, None]
    #     norm_matrix[np.isnan(norm_matrix)] = 0
    #     norm_matrix = norm_matrix / norm_matrix.sum(axis=1, keepdims=True)
    return norm_matrix


def softmax(x, axis=1):
    # 数值稳定：先减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x


def pseudo_label(label_matrix, co_matrix_norm) -> Dataset:
    soft_label = label_matrix @ co_matrix_norm

    # 后归一化
    # soft_label = softmax(soft_label)
    # 计算每一行的和
    row_sum = soft_label.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    soft_label = soft_label / row_sum

    # mask = (row_sum != 0)
    # result = soft_label.copy()
    # result[mask[:, 0]] = result[mask[:, 0]] / row_sum[mask][:, None]
    return soft_label


def co_oc(dataset: Dataset, multi_label_index=0) -> Dataset:
    datasets = dataset.split_by_label()
    multi_label_dataset: Dataset = datasets[multi_label_index]
    label_matrix = multi_label_dataset.y
    pseudo = pseudo_label(label_matrix, norm_co_oc_matrix(co_oc_matrix(label_matrix)))
    multi_label_dataset.y = pseudo

    result = Dataset()
    for i in range(len(dataset.label_map)):
        if i == multi_label_dataset:
            result.append_label(multi_label_dataset)
        else:
            result.append_label(datasets[i])

    return result

# # 示例标签矩阵
# matrix = np.array([
#     [1, 1, 0],  # 样本1
#     [1, 0, 1],  # 样本2
#     [0, 1, 1],  # 样本3
#     [1, 0, 0]  # 样本4
# ])
#
# # 计算共现矩阵
# co_matrix = co_oc_matrix(matrix)
#
# print("标签矩阵:")
# print(matrix)
# print("\n共现矩阵:")
# print(co_matrix)

# import numpy as np
#
# # 示例 soft_label
# soft_label = np.array([
#     [0.2, 0.3, 0.5],
#     [0.0, 0.0, 0.0],
#     [0.1, 0.9, 0.0]
# ])
#
# # 计算每一行的和
# row_sum = soft_label.sum(axis=1, keepdims=True)
#
# # 创建掩码：行和不为0的位置
# mask = (row_sum != 0)
#
# # 初始化 result 为 soft_label 的副本
# result = soft_label.copy()
#
# # 只对和不为0的行做归一化
# result[mask[:, 0]] = result[mask[:, 0]] / row_sum[mask][:, None]
#
# print(result)
