"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os, sys
from datetime import datetime
from typing import Optional

from geoCosiCorr3D.geoCore.constants import SOFTWARE


class GeoCosiCorr3DLog:
    def __init__(self, log_prefix: str, log_dir: Optional[str] = None):
        # Set default log directory based on some application-wide setting
        self.log_dir = log_dir or getattr(SOFTWARE, 'WKDIR', 'default_log_dir')

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a unique timestamp for the log file
        generation_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        log_filename = f"{log_prefix}_{generation_time}.log"
        full_log_path = os.path.join(self.log_dir, log_filename)

        # Set up the basic configuration for logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] - %(message)s",
            handlers=[
                logging.FileHandler(full_log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
