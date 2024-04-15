"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import geoCosiCorr3D.geoCore.constants as C
import numpy as np
import pandas
from geoCosiCorr3D.georoutines.file_cmd_routines import FilesInDirectory


class RawGeoTP(ABC):
    def __init__(self):
        pass

    @staticmethod
    def plot_matches(img_i, img_j, matches, tpPath, plot_matches: Optional[bool] = True):
        import matplotlib.pyplot as plt
        import rasterio
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        fig.suptitle("# Matches:" + str(matches.shape[0]))
        if img_i is not None:
            src1 = rasterio.open(img_i)
            axs[0].imshow(src1.read(1), cmap='gray')
        if img_j is not None:
            src2 = rasterio.open(img_j)
            axs[1].imshow(src2.read(1), cmap='gray')

        from geoCosiCorr3D.georoutines.geoplt_misc import GenerateColors
        colors = GenerateColors(matches.shape[0])

        axs[0].scatter(matches["ref_xPix"], matches["ref_yPix"], marker="+", s=80)  # , color=colors)
        axs[1].scatter(matches["target_xPix"], matches["target_yPix"], marker="+", s=80)  # , color=colors)
        axs[0].axes.xaxis.set_visible(False)
        axs[0].axes.yaxis.set_visible(False)
        axs[1].axes.xaxis.set_visible(False)
        axs[1].axes.yaxis.set_visible(False)
        axs[0].set_title("Ref image")
        axs[1].set_title("Target image")

        plt.savefig(os.path.join(os.path.dirname(tpPath), Path(tpPath).stem + ".svg"), dpi=100)
        # plt.close(fig)
        return

    @staticmethod
    def plot_matches_v2(img_i, img_j, matches_file):  # , plot_matches: Optional[bool] = True):
        import matplotlib.pyplot as plt
        import numpy as np
        from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo
        from geoCosiCorr3D.georoutines.utils.plt_utils import plot_matches
        fig, ax = plt.subplots(figsize=(10, 10))

        tp_array = np.loadtxt(matches_file, comments=";")
        plt.gray()
        img1 = cRasterInfo(img_i).raster_array[0]
        img2 = cRasterInfo(img_j).raster_array[0]
        keypoints1 = tp_array[:, 0:2]
        keypoints1[:, [0, 1]] = keypoints1[:, [1, 0]]
        keypoints2 = tp_array[:, 2:4]
        keypoints2[:, [0, 1]] = keypoints2[:, [1, 0]]
        list = np.arange(0, keypoints1.shape[0], 1)
        matches12 = np.array([list, list]).T
        # plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12, keypoints_color='r', only_keypoints=True)
        ax.axis('off')
        ax.set_title("{} \n #matches:{}".format(Path(matches_file).stem, str(matches12.shape[0])))
        plt.savefig(os.path.join(os.path.dirname(matches_file), Path(matches_file).stem + ".png"), dpi=100)
        return


class RawMMTP(RawGeoTP):
    MM_LIB_PATH =  C.ASIFT_TP_PARAMS.MM_LIB

    def __init__(self, config: Dict):
        super().__init__()
        self.in_config = config
        self._ingest()

    def _ingest(self):
        self.scale_factor = float( C.ASIFT_TP_PARAMS.SCALE_FACTOR \
                                      if self.in_config.get("scale_factor",  C.ASIFT_TP_PARAMS.SCALE_FACTOR) is None \
                                      else self.in_config.get("scale_factor",  C.ASIFT_TP_PARAMS.SCALE_FACTOR))

        self.mode = self.in_config.get("mode",  C.ASIFT_TP_PARAMS.MODE)  ## mode : MulScale,All, Line,Georef
        self.tp_format = self.in_config.get("tp_format", "COSI-Corr")
        self.tmp_dir = self.in_config.get("mm_temp_folder", None)
        self.max_pts = 60 if self.in_config.get('max_pts', None) is None else self.in_config.get('max_pts')

    @staticmethod
    def run_mm_tapioca(mm_lib_path, mode, in_imgs_folder, img_size= C.ASIFT_TP_PARAMS.IMG_SIZE):

        cmd = [mm_lib_path + " Tapioca"]
        cmd.extend([mode])
        cmd.extend([os.path.join(in_imgs_folder, ".*tif")])
        if mode ==  C.ASIFT_TP_PARAMS.MODE:
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
                data_dic: Dict = {"ref_xPix": data[:, 0], "ref_yPix": data[:, 1], "target_xPix": data[:, 2],
                                  "target_yPix": data[:, 3]}
                data_df = pandas.DataFrame(data_dic)
                filePath_ = os.path.join(tempPath__, Path(img_i).stem + "_VS_" + Path(img_j).stem + ".pts")

                if format_cosi_corr:
                    header = "; COSI-Corr tie points file (from Micmac)\n; base file:" + ref_img_name + "\n; warp file:" + \
                              raw_img_name+ "\n; Base Image (x,y), Warp Image (x,y)\n;\n"
                    with open(filePath_, 'w') as file:
                        file.write(header)
                        data_df.to_csv(file, header=False, index=False, sep='\t', float_format='%.6f')
                    return {"RefImgPath": img_i, "targetImgPath": img_j, "DataFrame": data_df, "TpFile": filePath_}
                else:
                    filePath_ = os.path.join(tempPath__, Path(img_i).stem + "_VS_" + Path(img_j).stem + "_matches.pts")
                    np.savetxt(filePath_, data[:, 0:4], delimiter='    ')
                    return {"RefImgPath": img_i, "targetImgPath": img_j, "DataFrame": data_df, "TpFile": filePath_}
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
