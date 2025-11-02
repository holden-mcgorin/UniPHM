if __name__ == '__main__':
    # 测试用例
    import numpy as np

    # 假设你的 Dataset 类已定义，包含 split_by_label 和 split_by_entity 方法

    # 构造一个虚拟数据集
    from uniphm.data import Dataset

    x = np.random.rand(10, 5)  # 10 条样本，每条5维特征
    y = np.random.rand(10, 6)  # 每条样本有6个标签值
    z = np.arange(10)  # 运行时间可以简单设为 0~9

    label_map = {
        'A': (0, 2),  # y 的第0列到第2列（不含2）
        'B': (2, 4),  # 第2列到第4列
        'C': (4, 6),  # 第4列到第6列
    }

    entity_map = {
        'E1': (0, 4),  # 第0~3行
        'E2': (4, 7),  # 第4~6行
        'E3': (7, 10),  # 第7~9行
    }

    dataset = Dataset(
        name="TestData",
        x=x,
        y=y,
        z=z,
        label_map=label_map,
        entity_map=entity_map
    )

    # ==== 测试 split_by_label ====
    print("=== Split by Label ===")
    label_splits = dataset.split_by_label()
    for i, ds in enumerate(label_splits):
        print(f"Sub-dataset {i}:")
        print("  Name:", ds.name)
        print("  y shape:", ds.y.shape)  # 访问私有变量
        print("  label_map:", ds.label_map)
        print()

    # ==== 测试 split_by_entity ====
    print("=== Split by Entity ===")
    entity_splits = dataset.split_by_entity()
    for i, ds in enumerate(entity_splits):
        print(f"Sub-dataset {i}:")
        print("  Name:", ds.name)
        print("  x shape:", ds.x.shape)
        print("  entity_map:", ds.entity_map)
        print()

    a, b = dataset.split_by_ratio(0.3)
    c = dataset.get('E1')
    d = dataset.remove('E2')

    # 构造原始数据集 A（有 entity1 和 entity2）
    x1 = np.array([[1], [2], [3], [4]])
    y1 = np.array([[10], [20], [30], [40]])
    z1 = np.array([[0], [0], [1], [1]])
    label_map = {'RUL': [0, 1]}
    entity_map1 = {'entity1': (0, 2), 'entity2': (2, 4)}
    A = Dataset(x=x1, y=y1, z=z1, label_map=label_map, entity_map=entity_map1, name="A")

    # 构造另一个数据集 B（含 entity1，和新的 entity3）
    x2 = np.array([[5], [6], [7], [8]])
    y2 = np.array([[50], [60], [70], [80]])
    z2 = np.array([[0], [0], [1], [1]])
    entity_map2 = {'entity1': (0, 2), 'entity3': (2, 4)}
    B = Dataset(x=x2, y=y2, z=z2, label_map=label_map, entity_map=entity_map2, name="B")

    # 合并 B 到 A
    A.append_entity(B)

    # 检查合并后数据
    print("x:")
    print(A.x)
    print("entity_map:")
    print(A.entity_map)

    # 期望结果：
    # entity1: 原 [0,2] + 新 [4,6] → [0,4]
    # entity2: 被插入后推迟 → [4,6]
    # entity3: 接在末尾 → [6,8]
    expected_x = np.array([[1], [2], [5], [6], [3], [4], [7], [8]])
    expected_entity_map = {
        'entity1': (0, 4),
        'entity2': (4, 6),
        'entity3': (6, 8)
    }

    assert np.array_equal(A.x, expected_x)
    assert A.entity_map == expected_entity_map

    print("✅ 测试通过 append_entity (冲突合并)")
