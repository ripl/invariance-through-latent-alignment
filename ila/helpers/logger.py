#!/usr/bin/env python3

from logging import DEBUG
import colorlog

def patch_logger_info(_logger):
    original_logger_info = _logger.info
    def __info(*args, **kwargs):
        original_logger_info(' '.join([str(arg) for arg in args]), **kwargs)
    return __info

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s: %(message)s'))

logger = colorlog.getLogger('iti')
logger.addHandler(handler)

logger.info = patch_logger_info(logger)
logger.print = logger.info

logger.setLevel(DEBUG)
