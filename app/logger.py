import logging
import sys

from app.util import ModelConfig, ApiConfig


def configure_logger(model_config: ModelConfig):
    from loguru import logger

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists.
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logger_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.add(model_config.model_name + "_{time}.log", format=logger_format, rotation="50MB", level="DEBUG")
