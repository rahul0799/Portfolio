import logging
import datetime
import os


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s | %(message)s")

    logfile = "./logs/" + str(datetime.date.today()) + ".log"

    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, "logs")

    if os.path.isdir(log_dir):
        pass
    else:
        os.mkdir(log_dir)

    file_handler = logging.FileHandler(logfile)

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
