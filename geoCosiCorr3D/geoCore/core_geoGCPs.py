"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import geoCosiCorr3D.georoutines.geo_utils as geoRT
import numpy as np
from geoCosiCorr3D.geoCore.base.base_geoGCPs import BaseTP2GCP


class RawTP2GCP(BaseTP2GCP):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        logging.info(f'{self.__class__.__name__}: GCP generation')

    def ingest(self) -> None:
        if self.debug:
            logging.info(f'{self.__class__.__name__}: Matching file:{self.tp_file}')
            logging.info(f'{self.__class__.__name__}: ref_img:{self.ref_img_path}')
            logging.info(f'{self.__class__.__name__}: dem_path:{self.dem_path}')
        self.fetch_tps()
        if self.dem_path is not None:
            self.dem_info = geoRT.cRasterInfo(self.dem_path)
        else:
            self.dem_info = None

        if self.output_gcp_path is None:
            self.output_gcp_path = os.path.join(os.path.dirname(self.tp_file), Path(self.tp_file).stem + "_GCP.csv")
        elif os.path.isdir(self.output_gcp_path):
            if not os.path.exists(self.output_gcp_path):
                os.makedirs(self.output_gcp_path)
            self.output_gcp_path = os.path.join(self.output_gcp_path, Path(self.tp_file).stem + "_GCP.csv")
        logging.info(f'{self.__class__.__name__}: GCP file: {self.output_gcp_path}')
        self.ref_img_info = geoRT.cRasterInfo(self.ref_img_path)
        return

    def fetch_tps(self):
        if os.path.exists(self.tp_file):
            self.tp_array = self.conv_tp_file_2_array(file_path=self.tp_file)
            self.ref_img_tps, self.base_img_tps = self.tp_array[:, 0:2], self.tp_array[:, 2:4]
            self.nb_Tps = np.shape(self.tp_array)[0]
            logging.info(f'{self.__class__.__name__}: # of matching pts:{self.nb_Tps}')
        else:
            er_msg = f'Matching file:{self.tp_file} does not exist !!'
            logging.error(er_msg)
            sys.exit(er_msg)
        return

    @staticmethod
    def conv_tp_file_2_array(file_path: str, comment: Optional[str] = ";") -> np.ndarray:
        try:
            arrayData = np.genfromtxt(file_path, comments=comment, dtype=np.dtype("f8"))
            return (arrayData)
        except:
            er_msg = f'Enable to convert :{file_path} to numpy array (comment:{comment})!!'
            logging.error(er_msg)
            sys.exit(er_msg)

    def write_gcps(self) -> None:

        pass

    def set_gcp_alt(self):
        pass

    def run_tp_to_gcp(self):
        pass
