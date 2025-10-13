import logging

def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

logger = setup_logger()