import os
import logging


def get_logger_directory(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            # Extract the directory from the file handler's path
            return os.path.dirname(handler.baseFilename)
    return None
