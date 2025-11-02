import logging
import sys

from colorama import Fore, Style, init


class Logger(logging.Formatter):
    """ 自定义日志格式化器，支持彩色输出 + 等级对齐 """
    __instance = None

    __level = logging.DEBUG
    __log_format = '[%(levelname)s %(asctime)s]  %(message)s'

    __COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA,
        'RESET': Style.RESET_ALL
    }

    def __init__(self, log_format=None):
        init(autoreset=True)

        # 检测是否在Jupyter环境中
        if 'ipykernel' in sys.modules:
            self.enable_color = False
        else:
            self.enable_color = True
            init(autoreset=True)

        if log_format is None:
            log_format = self.__log_format
        super(Logger, self).__init__(log_format, datefmt='%H:%M:%S')

        logger = logging.getLogger(__name__)
        logger.setLevel(self.__level)

        ch = logging.StreamHandler()
        ch.setLevel(self.__level)
        ch.setFormatter(self)

        logger.addHandler(ch)
        self.__logger = logger
        Logger.__instance = self

    def format(self, record):
        # 等级对齐为 7 个字符（如 "INFO   ", "WARNING"）
        levelname_raw = record.levelname
        padded_levelname = f"{levelname_raw:<7}"  # 左对齐，宽度7

        original_fmt = self._style._fmt.replace("%(levelname)s", padded_levelname)

        if self.enable_color:
            color = self.__COLORS.get(levelname_raw, self.__COLORS['RESET'])
            colored_fmt = f"{color}{original_fmt}{self.__COLORS['RESET']}"
        else:
            colored_fmt = original_fmt  # 不加颜色

        formatter = logging.Formatter(colored_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls()
        return cls.__instance

    @classmethod
    def set_level_debug(cls):
        Logger.__level = logging.DEBUG
        instance = cls.get_instance()
        instance.__logger.setLevel(cls.__level)

    @classmethod
    def set_level_info(cls):
        Logger.__level = logging.INFO
        instance = cls.get_instance()
        instance.__logger.setLevel(cls.__level)

    @classmethod
    def set_level_warning(cls):
        Logger.__level = logging.WARNING
        instance = cls.get_instance()
        instance.__logger.setLevel(cls.__level)

    @classmethod
    def set_level_error(cls):
        Logger.__level = logging.ERROR
        instance = cls.get_instance()
        instance.__logger.setLevel(cls.__level)

    @classmethod
    def set_level_critical(cls):
        Logger.__level = logging.CRITICAL
        instance = cls.get_instance()
        instance.__logger.setLevel(cls.__level)

    @classmethod
    def debug(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.debug(string)

    @classmethod
    def info(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.info(string)

    @classmethod
    def warning(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.warning(string)

    @classmethod
    def error(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.error(string)

    @classmethod
    def critical(cls, string: str):
        instance = cls.get_instance()
        instance.__logger.critical(string)

    # def get_symbols(use_unicode=True):
    #     if use_unicode:
    #         return {
    #             'ok': '✔',✓
    #             'fail': '✘',×
    #             'info': 'ℹ',
    #             'arrow': '➜'->
    #         }
    #     else:
    #         return {
    #             'ok': '[OK]',
    #             'fail': '[FAIL]',
    #             'info': '[INFO]',
    #             'arrow': '-->'
    #         }
