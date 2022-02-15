import logging
import logging.handlers
import datetime


def logger_basic(filename='out.log'):
    LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"
    logging.basicConfig(filename=filename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    return logging


def logger_custom(filename='out.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    
    # 按天滚动
    # rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    # rf_handler.setFormatter(formatter)
    # 输出到file
    f_handler = logging.FileHandler(filename)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    # 输出到控制台
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)

    # logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    return logger


if __name__ == '__main__':
    logger = logger_custom()
    logger.info("This is a info log.")