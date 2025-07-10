import logging


def get_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

# Adaptation: Create new function similar to get_logger, but it moves logging to a txt file instead of spamming the console on the notebook
def get_logger_to_file(name='root', log_file='deepsort_log.txt'):
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Adaptation: Use FileHandler instead of StreamHandler to write logs to the specified log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Adaptation: Clear existing handlers to avoid duplicate logs
    logger.handlers = []
    logger.addHandler(file_handler)
    return logger
