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


class geoCosiCorr3DLog:
    def __init__(self, log_prefix: str, log_dir: Optional[str] = None):
        if log_dir is None:
            log_dir = SOFTWARE.WKDIR

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        now = datetime.now()
        GENERATION_TIME = now.strftime("%m-%d-%Y-%H-%M-%S")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_prefix + "_" + GENERATION_TIME + '.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
