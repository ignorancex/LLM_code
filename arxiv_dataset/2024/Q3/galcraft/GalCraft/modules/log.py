import logging


class Logger:
    def __init__(self, logfile, logger=None, level=logging.INFO, mode='a'):
        self.logger = logging.getLogger(logger)
        self.logger.propagate = False
        self.logger.setLevel(level)
        fh = logging.FileHandler(logfile, mode=mode, encoding='utf-8')
        fh.setLevel(level)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        fh.close()
        sh.close()

    def get_log(self):
        return self.logger


