import threading
import time


class Timer:
    """
    自定义计时器
    start()开始计时
    stop()结束计时
    """
    keep_result = True
    __start_time = None
    __timer_thread = None
    __running = False

    def __init__(self):
        raise NotImplementedError("不需要实例，直接Timer.start()开始计时，Timer.stop()结束计时")

    @staticmethod
    def start():
        Timer.__start_time = time.time()
        Timer.__running = True
        Timer.__timer_thread = threading.Thread(target=Timer.__count)
        Timer.__timer_thread.start()

    @staticmethod
    def stop():
        Timer.__running = False
        Timer.__timer_thread.join()

    @staticmethod
    def __count():
        while Timer.__running:
            time.sleep(0.1)
            print(f'计时中：{round(time.time() - Timer.__start_time, 2)} s', end='\r')
        if Timer.keep_result:
            print(f'计时时长：{round(time.time() - Timer.__start_time, 2)} s')
