import os
import pickle

# import dill

from uniphm.util.Logger import Logger


class Cache:
    __CACHE_DIR = '.\\cache'

    if not os.path.exists(__CACHE_DIR):
        os.makedirs(__CACHE_DIR)

    def __init__(self, cache_dir: str):
        Cache.__CACHE_DIR = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    @classmethod
    def __get_cache_file(cls, name: str) -> str:
        """
        根据信息获取缓存文件（名称及位置）
        :param name:
        :return:
        """
        # hash_input = str(kwargs).encode('utf-8')
        # hash_value = hashlib.md5(hash_input).hexdigest()
        return os.path.join(cls.__CACHE_DIR, f'{name}.pkl')

    @classmethod
    def save(cls, target, name):
        """
        保存缓存到文件
        :return:
        """
        cache_file = cls.__get_cache_file(name)
        with open(cache_file, 'wb') as f:
            Logger.debug(f"[Cache]  Generating cache file: {cache_file}")
            pickle.dump(target, f)
            # dill.dump(target, f)
        Logger.debug(f"[Cache]  Generated cache file: {cache_file}")

    @classmethod
    def load(cls, name, is_able=True):
        """
        从文件加载缓存
        :return:
        """
        if not is_able:
            return None

        cache_file = cls.__get_cache_file(name)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                Logger.debug(f"[Cache]  -> Loading cache file: {cache_file}")
                cache = pickle.load(f)
                # cache = dill.load(f)
                Logger.debug(f"[Cache]  ✓ Successfully loaded: {cache_file}")
                return cache
        else:
            Logger.warning(f'[Cache]  × Cache file {cache_file} does not exist!')
            return None

    @classmethod
    def delete(cls, name, is_able=True):
        """
        删除缓存文件
        :param name:
        :param is_able:
        :return:
        """
        if not is_able:
            return None

        cache_file = cls.__get_cache_file(name)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            Logger.debug(f"[Cache]  Deleted cache file {cache_file}")
        else:
            Logger.warning(f"[Cache]  {cache_file} does not exist and cannot be deleted!")
