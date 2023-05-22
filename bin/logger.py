import logging
import os


class LoggingFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[33;92m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: format,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_logger(ckpt_dir, seed):
    logger = logging.getLogger(str(seed))
    logger.propagate = False  # Prevent the logger from passing messages to its parent logger

    # remove all handlers if they exist
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    ch = logging.StreamHandler()
    ch.setFormatter(LoggingFormatter())
    logger.addHandler(ch)
    logger.addHandler(logging.FileHandler(os.path.join(ckpt_dir, "{}.log".format(seed)), mode='a+'))

    logger.setLevel(logging.DEBUG)

    addLoggingLevel("SETTINGS", levelNum=logging.DEBUG)

    return logger  # return logger instance


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present.

    :param levelName: The name of the level.
    :param levelNum: The numeric value of the level.
    :param methodName: The name of the method defined in `logging` and its logger
                        classes to call the new level. Defaults to `levelName.lower()`
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError(f'Logging level {levelName} already defined')
    if hasattr(logging, methodName):
        raise AttributeError(f'Method {methodName} already defined in logging module')

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
