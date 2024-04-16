"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseTP2GCP(ABC):
    def __init__(self, in_tp_file: str,
                 ref_img_path: str,
                 base_img_path: str,
                 dem_path: Optional[str] = None,
                 output_gcp_path: Optional[str] = None,
                 debug: bool = False):

        self.tp_file = in_tp_file
        self.ref_img_path = ref_img_path
        self.base_img_path = base_img_path
        self.dem_path = dem_path
        self.output_gcp_path = output_gcp_path
        self.debug = debug


    @abstractmethod
    def write_gcps(self) -> None:

        pass

    @abstractmethod
    def set_gcp_alt(self):
        pass

    @abstractmethod
    def run_tp_to_gcp(self):
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass
