from numpy import ndarray


def tsne(x: ndarray, n_components=2, perplexity=30, random_state=0):
    """
    :param x:输入数据(2维数据，形状为【样本数，特征数】）
    :param n_components:降维后的特征数
    :param perplexity:必须要小于样本数，表示每个点在高维空间中“考虑多少个邻居”。
    :param random_state:
    :return:降维结果【样本数，降维后的特征数】
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(x)
