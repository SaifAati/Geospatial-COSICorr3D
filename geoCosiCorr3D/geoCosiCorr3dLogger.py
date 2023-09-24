"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from geoCosiCorr3D.geoCore.constants import SOFTWARE

# START THE LOGGER:
# logger = geoCosiCorr3DLog("app")
# logger.start_logging()

class geoCosiCorr3DLog:
    DEFAULT_LOG_DIR = SOFTWARE.WKDIR
    DEFAULT_LOG_LEVEL = logging.INFO

    def __init__(self, log_prefix: str, log_dir: Optional[str] = None):
        """
        Initializes the logger.

        Args:
            log_prefix (str): Prefix for the log file.
            log_dir (Optional[str], optional): Directory for the log files. Defaults to DEFAULT_LOG_DIR.
        """
        self.log_prefix = log_prefix
        self.log_dir = log_dir or self.DEFAULT_LOG_DIR
        self._prepare_log_dir()

    def _prepare_log_dir(self):
        """Creates the log directory if it doesn't exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except PermissionError:
            print(f"Permission denied: Cannot create directory {self.log_dir}.")
            sys.exit(1)

    def start_logging(self, level: Optional[int] = None):
        """
        Starts the logging with the given log level.

        Args:
            level (Optional[int], optional): Logging level. Defaults to DEFAULT_LOG_LEVEL.
        """
        now = datetime.now()
        generation_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        log_filename = f"{self.log_prefix}_{generation_time}.log"

        logging.basicConfig(
            level=level or self.DEFAULT_LOG_LEVEL,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, log_filename)),
                logging.StreamHandler(sys.stdout)
            ]
        )

# class geoCosiCorr3DLog:
#     def __init__(self, log_prefix: str, log_dir: Optional[str] = None):
#         if log_dir is None:
#             log_dir = SOFTWARE.WKDIR
#
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         now = datetime.now()
#         GENERATION_TIME = now.strftime("%m-%d-%Y-%H-%M-%S")
#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s [%(levelname)s] %(message)s",
#             handlers=[
#                 logging.FileHandler(os.path.join(log_dir, log_prefix + "_" + GENERATION_TIME + '.log')),
#                 logging.StreamHandler(sys.stdout)
#             ]
#         )
