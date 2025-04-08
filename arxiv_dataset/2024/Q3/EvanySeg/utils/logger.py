import logging
import colorlog


class Logger:
    def __init__(self, log_path):
        self.logger = logging.getLogger("tech_stu")
        self.logger.setLevel(logging.INFO)

        # 文件流
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_format = logging.Formatter(fmt='%(levelname)s    %(asctime)s    %(message)s')   # logging.Formatter
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # 控制台流
        console_hander = logging.StreamHandler()
        console_format = colorlog.ColoredFormatter(fmt='%(log_color)s %(message)s', log_colors={"INFO": "green", "WARNING": "yellow", "ERROR": "red"})  # colorformater
        console_hander.setFormatter(console_format)
        self.logger.addHandler(console_hander)

    def get_logger(self):
        return self.logger
