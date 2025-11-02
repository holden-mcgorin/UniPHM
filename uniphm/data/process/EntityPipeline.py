from typing import List, Callable, Union, Dict, Any

import numpy as np
import pandas as pd

from uniphm.data.Entity import Entity
from uniphm.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from uniphm.data.process.entity.ABCEntityProcessor import ABCEntityProcessor
from uniphm.util.Logger import Logger


class EntityPipeline:
    """
    一个流式处理器管道，存储一系列处理器（惰性执行），在调用 process(entity) 时执行。
    """

    # 存储处理器步骤的类型定义
    PipelineStep = Dict[str, Any]

    def __init__(self, name: str = "DefaultPipeline"):
        """
        :param name: 管道名称
        """
        self.name = name
        self._steps = []

    def register(self, processor: Union[ABCBaseProcessor, ABCEntityProcessor],
                 input_key: str,
                 output_key: str = None) -> 'EntityPipeline':
        """
        注册一个处理器步骤到管道中，不立即执行。

        :param processor: 处理器实例 (ABCBaseProcessor 或 ABCEntityProcessor)。
        :param input_key: 处理器操作的数据键。
        :param output_key: (仅对 ABCBaseProcessor 有效) 提取出的特征存储的新键。
        :return: self，用于链式调用。
        """
        step = {
            'processor': processor,
            'input_key': input_key,
            'output_key': output_key
        }

        if output_key is None and isinstance(processor, ABCBaseProcessor):
            output_key = processor.name

        self._steps.append(step)
        Logger.info(f"[EntityPipeline]  Registered step: {type(processor).__name__} on '{input_key}' -> '{output_key}'")

        return self

    def execute(self, entity: Entity) -> Entity:
        """
        对传入的 Entity 实例按顺序执行所有已注册的处理器步骤。

        :param entity: 要处理的 Entity 实例。
        :return: 处理完成后的 Entity 实例。
        """
        Logger.debug(f"[EntityPipeline]  STARTING EXECUTION of Pipeline '{self.name}' on Entity '{entity.name}'")

        current_entity = entity

        for i, step in enumerate(self._steps):
            processor = step['processor']
            input_key = step['input_key']
            output_key = step['output_key']

            step_name = f"[{i + 1}/{len(self._steps)}]"

            # === 1. 实体级处理器 ===
            if isinstance(processor, ABCEntityProcessor):
                Logger.debug(
                    f"[EntityPipeline]  {step_name} Executing ABCEntityProcessor: {type(processor).__name__} on key '{input_key}'..."
                )
                current_entity = processor.run(current_entity, input_key)

            # === 2. 特征级处理器 ===
            elif isinstance(processor, ABCBaseProcessor):
                if output_key is None:
                    output_key = processor.name
                if input_key not in current_entity:
                    raise KeyError(
                        f"Step {i + 1}: Input key '{input_key}' not found in Entity '{current_entity.name}'."
                    )

                Logger.debug(
                    f"[EntityPipeline]  {step_name} Executing ABCBaseProcessor: {type(processor).__name__} "
                    f"on '{input_key}' -> '{output_key}'..."
                )

                # 取出源数据（现在是 ndarray）
                source_array = current_entity[input_key]
                if not isinstance(source_array, np.ndarray):
                    raise TypeError(
                        f"Step {i + 1}: Input data for '{input_key}' is not np.ndarray."
                    )

                # 运行处理器
                extracted_array = processor.run(source_array)

                # 验证返回值
                if not isinstance(extracted_array, np.ndarray):
                    raise TypeError(
                        f"Step {i + 1}: Processor '{type(processor).__name__}' did not return np.ndarray."
                    )

                # 保存结果到 Entity（直接以 ndarray 形式）
                current_entity[output_key] = extracted_array

            # === 3. 其他类型处理器 ===
            else:
                raise TypeError(
                    f"Step {i + 1}: Unsupported processor type: {type(processor).__name__}."
                )

        Logger.debug(f"[EntityPipeline]  FINISHED EXECUTION of Pipeline '{self.name}'")
        return current_entity

    @staticmethod
    def step(entity: Entity,
             processor: Union[ABCBaseProcessor, ABCEntityProcessor],
             input_key: str,
             output_key: str = None) -> Entity:
        """
        立即执行单个处理步骤，不保存到管道中。

        :param entity: 要处理的 Entity 实例
        :param processor: 处理器实例（ABCBaseProcessor 或 ABCEntityProcessor）
        :param input_key: 输入数据的键
        :param output_key: 输出数据的键（仅对 ABCBaseProcessor 有效）
        :return: 处理后的 Entity 实例
        """
        Logger.debug(f"[EntityPipeline]  RUN ONCE using {type(processor).__name__} on key '{input_key}'")

        # === 实体级处理器 ===
        if isinstance(processor, ABCEntityProcessor):
            Logger.debug(f"[EntityPipeline]  Executing ABCEntityProcessor '{type(processor).__name__}'...")
            return processor.run(entity, input_key)

        # === 特征级处理器 ===
        elif isinstance(processor, ABCBaseProcessor):
            if output_key is None:
                output_key = processor.name
            if input_key not in entity:
                raise KeyError(f"[EntityPipeline]  Input key '{input_key}' not found in Entity '{entity.name}'.")

            source_array = entity[input_key]
            if not isinstance(source_array, np.ndarray):
                raise TypeError(
                    f"[EntityPipeline]  Input data '{input_key}' must be np.ndarray, got {type(source_array)}")

            Logger.debug(
                f"[EntityPipeline]  Running ABCBaseProcessor '{type(processor).__name__}' on '{input_key}' -> '{output_key}'")
            processed_array = processor.run(source_array)

            if not isinstance(processed_array, np.ndarray):
                raise TypeError(f"[EntityPipeline]  Processor '{type(processor).__name__}' did not return np.ndarray.")

            entity[output_key] = processed_array
            return entity

        # === 不支持的类型 ===
        else:
            raise TypeError(f"[EntityPipeline]  Unsupported processor type: {type(processor).__name__}")
