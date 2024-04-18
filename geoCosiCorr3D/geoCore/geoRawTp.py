"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.georoutines.file_cmd_routines import FilesInDirectory
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo
from geoCosiCorr3D.georoutines.utils.plt_utils import plot_matches


@dataclasses.dataclass
class KpsKeys:
    REF_IMG_FN = 'ref_img_gn'
    TARGET_IMG_FN = 'target_img_fn'
    REF_COL = 'ref_col'
    REF_LIN = 'ref_lin'
    TARGET_COL = 'target_col'
    TARGET_LIN = 'target_lin'


class RawGeoTP(ABC):

    @staticmethod
    def plot_matches_v1(img_i: np.ndarray, img_j: np.ndarray, matches: np.ndarray, o_plot_fn: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 10))

        plt.gray()
        keypoints1 = matches[:, 0:2]
        keypoints1[:, [0, 1]] = keypoints1[:, [1, 0]]
        keypoints2 = matches[:, 2:4]
        keypoints2[:, [0, 1]] = keypoints2[:, [1, 0]]
        list = np.arange(0, keypoints1.shape[0], 1)
        matches12 = np.array([list, list]).T
        # from geoCosiCorr3D.georoutines.geoplt_misc import GenerateColors
        # colors = GenerateColors(matches.shape[0])
        plot_matches(ax, img_i, img_j, keypoints1, keypoints2, matches12, matches_color='b', only_keypoints=True)
        ax.axis('off')
        ax.set_title(f" #matches:{matches12.shape[0]}")
        if o_plot_fn:
            plt.savefig(o_plot_fn, dpi=100)
        plt.close(fig)
        return

    @staticmethod
    def plot_matches_v2(img_i, img_j, matches_file):
        fig, ax = plt.subplots(figsize=(10, 10))
        tp_array = np.loadtxt(matches_file, comments=";")  # TODO
        plt.gray()
        img1 = cRasterInfo(img_i).raster_array[0]
        img2 = cRasterInfo(img_j).raster_array[0]
        keypoints1 = tp_array[:, 0:2]
        keypoints1[:, [0, 1]] = keypoints1[:, [1, 0]]
        keypoints2 = tp_array[:, 2:4]
        keypoints2[:, [0, 1]] = keypoints2[:, [1, 0]]
        list = np.arange(0, keypoints1.shape[0], 1)
        matches12 = np.array([list, list]).T

        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12, keypoints_color='r', only_keypoints=True)
        ax.axis('off')
        ax.set_title("{} \n #matches:{}".format(Path(matches_file).stem, str(matches12.shape[0])))
        plt.savefig(os.path.join(os.path.dirname(matches_file), Path(matches_file).stem + ".png"), dpi=100)
        plt.close(fig)
        return

    @staticmethod
    def write_kp_fn(ref_img: str, target_img: str, method: str, matches: np.ndarray, o_kp_fn: str):
        header = (f"; COSI-Corr tie points file (method:{method})\n"
                  f"; reference file:{ref_img}\n"
                  f"; target file:{target_img}\n"
                  f"; Base Image (col,lin), Target Image (col,lin)\n;\n")

        data_dic: Dict = {KpsKeys.REF_COL: matches[:, 0],
                          KpsKeys.REF_LIN: matches[:, 1],
                          KpsKeys.TARGET_COL: matches[:, 2],
                          KpsKeys.TARGET_LIN: matches[:, 3]}
        data_df = pandas.DataFrame(data_dic)
        with open(o_kp_fn, 'w') as file:
            file.write(header)
            data_df.to_csv(file, header=False, index=False, sep='\t', float_format='%.6f')

        return o_kp_fn


class RawMMTP(RawGeoTP):
    MM_LIB_PATH = C.ASIFT_TP_PARAMS.MM_LIB

    def __init__(self, config: Dict):
        self.in_config = config
        self._ingest()

    def _ingest(self):
        self.scale_factor = float(C.ASIFT_TP_PARAMS.SCALE_FACTOR \
                                      if self.in_config.get("scale_factor", C.ASIFT_TP_PARAMS.SCALE_FACTOR) is None \
                                      else self.in_config.get("scale_factor", C.ASIFT_TP_PARAMS.SCALE_FACTOR))

        self.mode = self.in_config.get("mode", C.ASIFT_TP_PARAMS.MODE)  ## mode : MulScale,All, Line,Georef
        self.tp_format = self.in_config.get("tp_format", "COSI-Corr")
        self.tmp_dir = self.in_config.get("mm_temp_folder", None)
        self.max_pts = 60 if self.in_config.get('max_pts', None) is None else self.in_config.get('max_pts')

    @staticmethod
    def run_mm_tapioca(mm_lib_path, mode, in_imgs_folder, img_size=C.ASIFT_TP_PARAMS.IMG_SIZE):

        cmd = [mm_lib_path + " Tapioca"]
        cmd.extend([mode])
        cmd.extend([os.path.join(in_imgs_folder, ".*tif")])
        if mode == C.ASIFT_TP_PARAMS.MODE:
            cmd.extend([str(img_size)])
        cmd.extend(["ExpTxt=true"])
        logging.info(cmd)
        call = ""
        for cmd_ in cmd:
            call += cmd_ + " "
        logging.info(call)
        os.system(call)
        return

    @staticmethod
    def mm_tps(img_i: str, img_j: str, homol_dir: str, format_cosi_corr: Optional[bool] = False,
               max_tps: Optional[int] = None) -> Optional[Dict]:
        ref_img_name = os.path.basename(img_i)
        raw_img_name = os.path.basename(img_j)
        file_list = FilesInDirectory(path=homol_dir, displayFile=True)
        if os.path.join(homol_dir, "Pastis" + ref_img_name) in file_list:
            tempPath__ = os.path.join(homol_dir, "Pastis" + ref_img_name)
            if not os.path.exists(tempPath__):
                logging.error("Error: " + tempPath__ + "  does not exit !!! ")
                return None
            if os.path.exists(os.path.join(tempPath__, raw_img_name + ".txt")):
                data = np.loadtxt(os.path.join(tempPath__, raw_img_name + ".txt"))
                nb_tps = data.shape[0]
                if max_tps is not None:
                    step = int(nb_tps / max_tps)
                    if step > 1:
                        data = data[0::int(step)]
                        logging.info(f'Reducing matching(target max pts {max_tps}) :{nb_tps}-->{data.shape[0]}')
                data_dic: Dict = {KpsKeys.REF_COL: data[:, 0],
                                  KpsKeys.REF_LIN: data[:, 1],
                                  KpsKeys.TARGET_COL: data[:, 2],
                                  KpsKeys.TARGET_LIN: data[:, 3]}
                data_df = pandas.DataFrame(data_dic)
                filePath_ = os.path.join(tempPath__,
                                         f'{Path(img_i).stem}_VS_{Path(img_j).stem}_{C.TP_DETECTION_METHODS.ASIFT}.pts')

                if format_cosi_corr:
                    RawGeoTP.write_kp_fn(ref_img_name, raw_img_name, C.TP_DETECTION_METHODS.ASIFT, data, filePath_)
                    return {KpsKeys.REF_IMG_FN: img_i, KpsKeys.TARGET_IMG_FN: img_j,
                            "DataFrame": data_df, "TpFile": filePath_}
                else:
                    filePath_ = os.path.join(tempPath__, Path(img_i).stem + "_VS_" + Path(img_j).stem + "_matches.pts")
                    np.savetxt(filePath_, data[:, 0:4], delimiter='    ')
                    return {KpsKeys.REF_IMG_FN: img_i, KpsKeys.TARGET_IMG_FN: img_j,
                            "DataFrame": data_df, "TpFile": filePath_}
            else:
                logging.error("Error: " + os.path.join(tempPath__, raw_img_name + ".txt") + "  does not exit !!! ")
                return

        else:
            logging.warning("No tie points file !! ")
            return None

    @abstractmethod
    def run_mm_tp(self):
        pass

    def set_img_size(self, img_width, img_height):
        if self.scale_factor != 1:
            return int(min(img_width, img_height) * self.scale_factor)
        else:
            return -1
