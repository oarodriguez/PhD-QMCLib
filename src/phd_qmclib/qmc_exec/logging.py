import logging

import colorama
from colorlog import ColoredFormatter

__all__ = [
    'exec_logger'
]

# Disable colorama processing.
colorama.deinit()

# The name of the logger.
QMC_EXEC_LOG_NAME = f'QMC Exec'

# The basic format of the log messages.
LOG_FORMAT = "%(bg_white)s%(fg_black)s%(asctime)-15s%(reset)s | " \
             "%(log_color)s%(name)-s - " \
             "%(levelname)-5s%(reset)s : %(message)s"

# console_formatter = logging.Formatter(LOG_FORMAT)
log_colors = {
    'DEBUG': 'bold_cyan',
    'INFO': 'bold_green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bg_red,bold_white',
}
console_formatter = ColoredFormatter(LOG_FORMAT, reset=True,
                                     log_colors=log_colors)
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.DEBUG)

exec_logger = logging.getLogger(QMC_EXEC_LOG_NAME)
exec_logger.setLevel(logging.DEBUG)
exec_logger.addHandler(console_handler)
