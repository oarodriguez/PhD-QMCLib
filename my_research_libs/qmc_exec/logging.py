import logging

# The name of the logger.
QMC_EXEC_LOG_NAME = f'QMC Exec'

# The basic format of the log messages.
LOG_FORMAT = "%(asctime)-15s | %(name)-s - %(levelname)-5s: %(message)s"

console_formatter = logging.Formatter(LOG_FORMAT)
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

exec_logger = logging.getLogger(QMC_EXEC_LOG_NAME)
exec_logger.setLevel(logging.INFO)
exec_logger.addHandler(console_handler)
