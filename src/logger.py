import logging
import sys
import os
from src.config import OUTPUTS_DIR

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        os.makedirs(os.path.join(OUTPUTS_DIR, "logs"), exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "logs", "pipeline.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
