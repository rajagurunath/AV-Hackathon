print(__name__)

import os
import sys
from loguru import logger as loguru_logger

# Setup LOGGER
VERSION = os.environ.get('VERSION', 'not-set')
LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>| " + \
    "<level>{level: <8}</level>" \
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
loguru_logger.remove()
loguru_logger.add(sys.stderr, format=LOGURU_FORMAT)
# loguru_logger.add("databricks-jobs-api_{time}.log", level="TRACE", rotation="100 MB")

logger = loguru_logger

def diagnose(name):
    logger.info(f'Loading file {name}')
    logger.info(f'sys.path is {sys.path}')
    try:
        pythonpath = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        pythonpath = []
    logger.info(f'PYTHONPATH is {pythonpath}')

# logger.diagnose = diagnose

