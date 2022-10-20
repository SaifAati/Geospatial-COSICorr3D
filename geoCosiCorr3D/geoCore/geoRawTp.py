"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

from abc import abstractmethod, ABC
from typing import Optional, Dict

import os
import logging
import pandas
import numpy as np
from pathlib import Path

from geoCosiCorr3D.georoutines.file_cmd_routines import FilesInDirectory
from geoCosiCorr3D.geoConfig import cgeoCfg

geoCfg = cgeoCfg()


class RawGeoTP(ABC):
    def __init__(self):
        pass

    @staticmethod
    def plot_matches(img_i, img_j, matches, tpPath, plot_matches: Optional[bool] = True):
        import rasterio
        import matplotlib.pyplot as plt
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
        from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo
        import numpy as np
        from geoCosiCorr3D.georoutines.utils.plt_utils import plot_matches
        fig, ax = plt.subplots( figsize=(10,10))

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

        plt.savefig(os.path.join(os.path.dirname(matches_file), Path(matches_file).stem + ".png"), dpi=200)
        return


class RawMMTP(RawGeoTP):
    MM_LIB_PATH = geoCfg.mmLib

    def __init__(self, config: Dict):
        super().__init__()
        self.in_config = config
        self._ingest()

    def _ingest(self):
        self.scale_factor = self.in_config.get("scale_factor", 1 / 8)
        self.mode = self.in_config.get("mode", "All")  ## mode : MulScale,All, Line,Georef
        self.tp_format = self.in_config.get("tp_format", "COSI-Corr")
        self.mm_temp_folder = self.in_config.get("mm_temp_folder", None)

        if self.mm_temp_folder is None:
            geoCfg.CreateTempFolder()
            self.mm_temp_folder = geoCfg.mmTempFolder

    @staticmethod
    def run_mm_tapioca(mm_lib_path, mode, in_imgs_folder, img_size=1000):

        cmd = [mm_lib_path + " Tapioca"]
        cmd.extend([mode])
        cmd.extend([os.path.join(in_imgs_folder, ".*tif")])
        if mode == "All":
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
    def mm_tps(img_i: str, img_j: str, homolDirectory: str, formatCosiCorr: Optional[bool] = False):
        refImgName = os.path.basename(img_i)
        targetName = os.path.basename(img_j)
        fileList = FilesInDirectory(path=homolDirectory, displayFile=True)
        if os.path.join(homolDirectory, "Pastis" + refImgName) in fileList:
            tempPath__ = os.path.join(homolDirectory, "Pastis" + refImgName)
            if not os.path.exists(tempPath__):
                logging.error("Error: " + tempPath__ + "  does not exit !!! ")
                return
            if os.path.exists(os.path.join(tempPath__, targetName + ".txt")):
                data = np.loadtxt(os.path.join(tempPath__, targetName + ".txt"))
                dataDic = {"ref_xPix": data[:, 0], "ref_yPix": data[:, 1], "target_xPix": data[:, 2],
                           "target_yPix": data[:, 3]}
                df = pandas.DataFrame(dataDic)
                filePath_ = os.path.join(tempPath__, Path(img_i).stem + "_VS_" + Path(img_j).stem + ".pts")

                if formatCosiCorr:
                    header = "; COSI-Corr tie points file (from Micmac)\n; base file:" + refImgName + "\n; warp file:" + \
                             targetName + "\n; Base Image (x,y), Warp Image (x,y)\n;\n"
                    with open(filePath_, 'w') as file:
                        file.write(header)
                        df.to_csv(file, header=False, index=False, sep='\t', float_format='%.6f')
                    return {"RefImgPath": img_i, "targetImgPath": img_j, "DataFrame": df, "TpFile": filePath_}
                else:
                    filePath_ = os.path.join(tempPath__, Path(img_i).stem + "_VS_" + Path(img_j).stem + "_matches.pts")
                    np.savetxt(filePath_, data[:, 0:4], delimiter='    ')
                    return {"RefImgPath": img_i, "targetImgPath": img_j, "DataFrame": df, "TpFile": filePath_}
            else:

                logging.error("Error: " + os.path.join(tempPath__, targetName + ".txt") + "  does not exit !!! ")
                return

        else:
            logging.warning("No tie points file !!!! ")
            return 0

    @abstractmethod
    def run_mm_tp(self):
        pass

    def set_img_size(self, img_width, img_height):
        if self.scale_factor != 1:
            return int(min(img_width, img_height) * self.scale_factor)
        else:
            return -1
