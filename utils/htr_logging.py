import logging

def get_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
