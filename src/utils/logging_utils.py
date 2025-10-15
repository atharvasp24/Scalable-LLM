"""
logging_utils.py
----------------
Custom logger for tracking training and evaluation events.
"""

import logging

def setup_logger(name="scalable_llm"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger
