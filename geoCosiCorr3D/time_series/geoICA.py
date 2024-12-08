import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import linalg
from sklearn.decomposition import FastICA
from tqdm import tqdm

from geoCosiCorr3D.georoutines.file_cmd_routines import CreateDirectory
from geoCosiCorr3D.georoutines.geo_utils import (WriteRaster, cRasterInfo, geoStat)
from geoCosiCorr3D.georoutines.geoplt_misc import ColorBar_
from geoCosiCorr3D.time_series.base_analyzer import ImageSeriesAnalyzer


class geoICA(ImageSeriesAnalyzer):
    def __init__(self, input_raster_fn, nb_comp, mask_val=20, max_iter=1000, fun="exp", debug=True):
        super().__init__(debug)
        self.raster_fn = input_raster_fn
        self.nb_comp = nb_comp
        self.fun = fun
        self.max_iter = max_iter
        self.mask_val = mask_val
        self.setup()

    def setup(self):
        self.o_folder = CreateDirectory(directoryPath=os.path.dirname(self.raster_fn),
                                        folderName=Path(self.raster_fn).stem + "_geoICA")
        self.raster_info = cRasterInfo(self.raster_fn)
        self.data_cube = self.raster_info.raster_array

    @staticmethod
    def plot_s_component(s_values, o_path: str, s_max=None, debug=False):

        if s_max == None:
            s_values = s_values[:]
        else:
            s_values = s_values[:s_max]

        nb_comps = len(s_values)
        percent = 100 * (s_values ** 2) / sum(s_values ** 2)
        percent = np.round(percent, decimals=2)
        cum_var = np.cumsum(percent)
        if debug:
            print(f"S values: {s_values} - {len(s_values)} ")
            print(f"Percent Var= {percent}")
            print(f"cum_var={cum_var}")

        def millions(x, pos):
            'The two args are the value and tick position'
            return '%1.1f%%' % (x)

        list_xaxis = []
        for i in range(nb_comps):
            list_xaxis.append("PC" + str(i + 1))

        formatter = FuncFormatter(millions)

        x = np.arange(nb_comps)

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(formatter)
        plt.bar(x, percent, color="k", width=0.5, label="Variance")
        plt.plot(cum_var, color="r", label="Cumulative variance")
        plt.ylabel('% Explained Variance', fontsize=8)
        plt.xlabel(' Number of Components', fontsize=8)
        plt.title('geoICA Analysis \n Cumulative sum of variance explained with [%s] components' % (nb_comps),
                  fontsize=8)
        plt.style.context('seaborn-whitegrid')
        plt.xticks(x, list_xaxis, fontsize=6, fontweight='bold', rotation=90)
        plt.yticks(fontsize=6, fontweight='bold')
        plt.legend()
        plt.grid(linestyle='-', linewidth=0.2)
        plt.savefig(o_path, dpi=400)
        plt.close()
        return

    def reverse_mask(self, mask, masked_array):
        """
        Reverses the mask on the masked array to restore the original array shape.
        """
        mask_fl = mask.flatten()
        index_list = np.where(mask_fl > 0)[0]
        if self.debug:
            print(
                f"Valid values:{len(index_list)} :{mask_fl.shape[0]} ===> invalid:{mask_fl.shape[0] - len(index_list)}")

        array = np.full(mask_fl.shape, np.nan)
        array[index_list] = masked_array
        return array

    def reconstruct(self, S_):
        self.mean_reconstructed_fns = []
        for comp_num in tqdm(range(self.nb_comp), desc="Data reconstruction"):
            new_S = np.zeros(S_.shape)
            new_S[:, comp_num] = S_[:, comp_num]
            array_restored = self.ica.inverse_transform(new_S)
            o_rec_arr = []
            for i in range(array_restored.shape[1]):
                tmp_arr = array_restored[:, i]
                array = self.reverse_mask(mask=self.mask_reshaped, masked_array=tmp_arr)
                us_reshaped = array.reshape((self.raster_info.raster_height, self.raster_info.raster_width))
                o_rec_arr.append(np.ma.masked_invalid(us_reshaped))

            WriteRaster(oRasterPath=os.path.join(self.o_folder,
                                                 Path(self.raster_fn).stem + "_geoICA_Rec_IC_" + str(
                                                     comp_num) + ".tif"),
                        geoTransform=self.raster_info.geo_transform,
                        arrayList=o_rec_arr,
                        epsg=self.raster_info.epsg_code)

            mean_reconstructed_fn = os.path.join(self.o_folder,
                                                 Path(self.raster_fn).stem + "_mean_geoICA_Rec_IC_" + str(
                                                     comp_num) + ".tif")
            self.mean_reconstructed_fns.append(mean_reconstructed_fn)
            WriteRaster(oRasterPath=mean_reconstructed_fn,
                        geoTransform=self.raster_info.geo_transform,
                        arrayList=[np.mean(o_rec_arr, axis=0)],
                        epsg=self.raster_info.epsg_code)
            print(f"Reconstructed component {comp_num} saved to {mean_reconstructed_fn}")

        return

    def __call__(self, *args, **kwargs):

        flat_data_cube = self.build_flatten_data_cube(data_cube=self.data_cube)

        mask = self.build_mask_any_nan(input_array=flat_data_cube, mask_value=self.mask_val)

        self.mask_reshaped = mask.reshape((self.raster_info.raster_height, self.raster_info.raster_width))

        masked_array = self.mask_raster(mask=self.mask_reshaped, input_array=flat_data_cube)

        U, S, Vt = linalg.svd(masked_array, full_matrices=False)

        if self.debug:
            print("U:", U.shape, "S:", S.shape, "Vt:", Vt.shape)

        self.plot_s_component(s_values=S,
                              o_path=os.path.join(self.o_folder, Path(self.raster_fn).stem + "_geoICA.png"))

        s_diag = np.diag(S)
        US = np.dot(U, s_diag)

        self.ica = FastICA(n_components=self.nb_comp, fun=self.fun, max_iter=self.max_iter)

        if self.debug:
            print(self.ica.get_params())

        S_ = self.ica.fit_transform(masked_array)  # Reconstruct
        A_ = self.ica.mixing_  # Get estimated mixing matrix

        o_ica_comps = np.zeros((self.nb_comp, self.data_cube.shape[1], self.data_cube.shape[2]))

        for i in tqdm(range(S_.shape[1]), "Extracting components"):
            if self.debug:
                print(f"===> IC:{str(i + 1)}")
            tmp_array = S_[:, i]
            array = self.reverse_mask(mask=self.mask_reshaped, masked_array=tmp_array)
            us_reshaped = array.reshape((self.raster_info.raster_height, self.raster_info.raster_width))
            us_reshapedStat = geoStat(in_array=us_reshaped, display_values=True)

            o_ica_comps[i, :, :] = us_reshaped
            fig1, ax1 = plt.subplots(1, 1)  # , constrained_layout=True)
            factor = 0.5
            cmap = "RdYlBu"
            vMin = float(us_reshapedStat.mean) - factor * float(us_reshapedStat.std)
            vMax = float(us_reshapedStat.mean) + factor * float(us_reshapedStat.std)
            im = ax1.imshow(us_reshaped, vmin=vMin, vmax=vMax, cmap=cmap)
            ColorBar_(ax=ax1, mapobj=im, cmap=cmap, vmin=vMin, vmax=vMin, orientation="vertical")
            ax1.set_title("IC:" + str(i + 1))
            plt.savefig(os.path.join(self.o_folder, "IC" + str(i + 1) + ".png"), dpi=200)
            plt.close(fig1)

        WriteRaster(oRasterPath=os.path.join(self.o_folder, Path(self.raster_fn).stem + "_geoICA_comp.tif"),
                    geoTransform=self.raster_info.geo_transform,
                    arrayList=[array_ for array_ in o_ica_comps],
                    epsg=self.raster_info.epsg_code)
        self.reconstruct(S_=S_)
        return


if __name__ == '__main__':
    path = "dataset"
    ew_fn = os.path.join(path, "EW_WV_Spot_MB_3DDA.tif")
    ns_fn = os.path.join(path, "NS_WV_Spot_MB_3DDA.tif")

    ns_ica = geoICA(input_raster_fn=ns_fn, nb_comp=4)
    ns_ica()
    ew_ica = geoICA(input_raster_fn=ew_fn, nb_comp=4)
    ew_ica()
