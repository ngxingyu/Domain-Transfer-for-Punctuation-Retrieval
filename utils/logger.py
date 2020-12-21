import logging
import yaml

def get_logger(name: str, create_file:bool = True):
    """Logs a message
    Args:
    name(str): name of logger
    create_file(bool): if file should be created
    """
    formatter=logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('{}.log'.format(name), 'w', 'utf-8')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if create_file: logger.addHandler(fh)
    return logger