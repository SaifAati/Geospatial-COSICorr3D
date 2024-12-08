import os, warnings, sys
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo, WriteRaster, geoStat
from geoCosiCorr3D.georoutines.file_cmd_routines import CreateDirectory
from geoCosiCorr3D.time_series.base_analyzer import ImageSeriesAnalyzer
import geoCosiCorr3D.time_series.utils as utils


@dataclass
class GEO_PCAIM_PARAMS:
    MIN_NUM_ANALYSIS = 0  # Minimum number of valid measurement in the time-series for a given pixel
    TOL = 10e-7
    MAX_ITER = 100
    OBJ_VAL_TOL = 10e-16
    INIT_MISSING_VAL_MEAN = 1
    WEIGHT = 0


class geoPCAIMWarning(UserWarning):
    pass


class geoPCAIM(ImageSeriesAnalyzer):

    def __init__(self,
                 nb_comp: int,
                 in_data_cube_path: str,
                 mask_val=None,
                 init_mean: Optional[int] = None,
                 minim_analysis: Optional[int] = None,
                 weight: Optional[int] = None,
                 debug=False):
        """

        Args:
            nb_comp: maximum number of components to be extracted
            in_data_cube_path: input folder path of the data cube: path (string)
            mask_val:
            init_mean: Initialisation of missing values, either with 0 or with the average of all valid measurements in the time column: int :0 || 1
            minim_analysis: Minimum number of valid measurement in the time-series for a given pixel: int
            weight: The weighting strategy of each correlation (time slice) depends on the ratio of NaN over total number of pixels
            debug: bool
        Notes:
            The weighting strategy of each correlation (time slice) depends on the ratio of NaN over total number of pixels
            pWeight=0 -> no weight, use the arbitrary ones
            pWeight=1 -> proportional to the number of missing data
            pWeight=2 -> more weight on the images with a lot a valid measurements
        """
        super().__init__(debug)

        self.raster_info = cRasterInfo(in_data_cube_path)
        self.in_data_cube_path = in_data_cube_path
        self.init_mean = init_mean
        self.nb_comp = nb_comp
        self.min_num_analysis = minim_analysis
        self.weight = weight
        self.mask_val = mask_val
        self.setup()

    def setup(self):
        self.disp_data_cube = self.raster_info.raster_array
        print(f"disp_data_cube.shape:{self.disp_data_cube.shape}")
        self.o_path = CreateDirectory(os.path.dirname(self.in_data_cube_path),
                                      "gePCAIM_" + Path(self.in_data_cube_path).stem, cal="y")  # savingPath

        self.meta_data = []
        self._set_init_missing_values()
        if self.nb_comp > self.disp_data_cube.shape[0]:
            warnings.warn(
                f"Invalid number of components: {self.nb_comp} ==> nbPC={self.disp_data_cube.shape[0]}",
                geoPCAIMWarning)
            self.nb_comp = self.disp_data_cube.shape[0]

        if self.min_num_analysis is not None:
            if self.min_num_analysis < 0 or self.min_num_analysis > self.disp_data_cube.shape[0]:
                warnings.warn(
                    f"Invalid minimum data: {self.min_num_analysis} ==> minimAnalysis= {self.disp_data_cube.shape[0]})",
                    geoPCAIMWarning)

                self.min_num_analysis = self.disp_data_cube.shape[0]
        else:
            self.min_num_analysis = self.disp_data_cube.shape[0]
            print("Mask All NaNs is selected!!")

        """
        The weighting strategy of each correlation (time slice) depends on the ratio of NaN over total number of pixels
        """
        # pWeight=0 -> no weight, use the arbitrary ones
        # pWeight=1 -> proportional to the number of missing data
        # pWeight=2 -> more weight on the images with a lot a valid measurements
        if self.weight is not None:
            if self.weight not in [0, 1, 2]:
                self.weight = GEO_PCAIM_PARAMS.WEIGHT

        """
          iterative parameters
        """

        self.tol = GEO_PCAIM_PARAMS.TOL
        self.max_iter = GEO_PCAIM_PARAMS.MAX_ITER

        return

    def _set_init_missing_values(self):
        """
            Initialisation of missing values:
            Either with 0 or with the average of all valid measurements in that "time column"
            0: initialization with 0
            1: initialisation with average of temporal pixels values (for each pixel location)
         """
        if self.init_mean is not None:
            if self.init_mean not in [0, 1]:
                msg = "Wrong Initialisation of missing value!!  ==> NaNs initialized with avg "
                warnings.warn(msg)
                self.init_mean = GEO_PCAIM_PARAMS.INIT_MISSING_VAL_MEAN
        else:
            self.init_mean = GEO_PCAIM_PARAMS.INIT_MISSING_VAL_MEAN
        return

    def run_geoPCAIM(self, data, weights, nbOfNaNArr, nbComponents):
        global S, U, V
        if self.debug:
            print("==== Performing geoPCAIM ... ====")
        objectiveValues = np.zeros(self.nb_comp)
        shape = np.shape(data)
        count_valid = shape[0]
        nbImg = shape[1]
        for i in tqdm(range(nbComponents), desc="Extracting components:"):
            if self.debug:
                print("Computing PC:", i + 1)
            objVal = np.sum((data ** 2) * weights)  ## correspond to the reduced  chi2 statistics
            if self.debug:
                print("Chi2 objVal:{:.4f}".format(objVal))
            old_objVal = np.inf
            iter = 0
            dataTmp = np.zeros((count_valid, nbImg))

            while ((abs(objVal - old_objVal) > objVal * self.tol) and (objVal > GEO_PCAIM_PARAMS.OBJ_VAL_TOL)):
                iter += 1
                if iter > self.max_iter:
                    break
                old_objVal = objVal
                A = weights * data + (1 - weights) * dataTmp
                U, S, Vt = np.linalg.svd(A, full_matrices=False)

                del A
                V = Vt.T
                ## Remove unecessary components depending on wich one is currently optimized
                for j in range(i + 1, nbImg):
                    S[j] = 0
                dataTmp = np.dot(np.dot(U, np.diag(S)), Vt)
                AA = ((dataTmp - data) ** 2) * weights
                objVal = np.sum(AA)
                del AA
            if self.debug:
                print("iter=", iter)
            objectiveValues[i] = objVal / (data.size - np.sum(nbOfNaNArr))
        if self.debug:
            print("objectiveValues=", objectiveValues)
            print("S=", S)
            print(data.size, np.sum(nbOfNaNArr))
        return U, S, V, objectiveValues

    def __MaskRaster(self, mask, inputArray):
        maskFl = mask.flatten()
        indexList = list(np.where(maskFl > 0)[0])
        self.count_valid = len(indexList)

        print("Valid values:", len(indexList), ":", maskFl.shape[0], "===>invalid:", maskFl.shape[0] - len(indexList))
        shape = inputArray.shape
        print("inputArray dim:", shape)
        arrayMasked = np.empty((len(indexList), shape[1]))
        for i in range(shape[1]):
            tempArray = inputArray[:, i]
            arrayMasked[:, i] = tempArray.take(indexList)
            del tempArray
        # maskArray = np.tile(maskFl,(shape[1],1)).T
        # tempArray = inputArray * maskArray
        # arrayMasked = tempArray[tempArray!=0.]
        # # arrayMasked = arrayMasked.reshape((int(arrayMasked.shape[0]/shape[1])),shape[1])
        # print(arrayMasked.shape[0])
        print("masked array dim :", arrayMasked.shape)
        return arrayMasked

    def build_mask_any_nan(self, in_arr: np.ndarray, mask_val: Optional[float] = None, visualize_mask=False,
                           save_mask=True):
        # in_arr_fl = self.flatten_data_cube(disp_data_cube=in_arr)
        in_arr_fl = self.build_flatten_data_cube(data_cube=in_arr)
        in_arr_fl = np.ma.masked_invalid(in_arr_fl)
        if mask_val is not None:
            in_arr_fl = np.ma.masked_outside(in_arr_fl, -mask_val, mask_val)
        mask = np.prod(~in_arr_fl.mask, axis=1)
        mask_reshaped = mask.reshape((self.raster_info.raster_height, self.raster_info.raster_width))

        indexList = np.where(mask_reshaped > 0)
        indexListTemp = list(np.where(mask > 0)[0])
        self.count_valid = len(indexListTemp)
        print(f'# valid pix={self.count_valid}')
        self.loc_valid = indexList
        if visualize_mask:
            plt.imshow(mask_reshaped, cmap="gray")
            plt.title("Mask")
            plt.show()
        if save_mask:
            WriteRaster(oRasterPath=os.path.join(self.o_path, "Mask.tif"), geoTransform=self.raster_info.geo_transform,
                        arrayList=[mask_reshaped], epsg=self.raster_info.epsg_code)
        return

    def build_mask_minimum(self, in_arr: np.ndarray, min_val: int, mask_val=None, visualize_mask=False,
                           save_mask=True):

        """
             Locate the pixels which contain minimum measurements in the time column (i.e., not all NaN).
             To do this we'll compute the sum of all the pixels values over the time-series.
             At each location, If the sum = 0.0, that means that all measurement at this location failed (NaN).
        """
        in_array = np.ma.masked_invalid(in_arr)
        # print(iArray.shape)

        if mask_val is not None:
            print(f"Mask Large Values:{mask_val}")
            in_array = np.ma.masked_outside(in_array, -mask_val, mask_val)

        count_validValsArray = np.count_nonzero(~in_array.mask, axis=0)
        mask = count_validValsArray >= min_val
        if visualize_mask:
            plt.imshow(mask, cmap="gray")
            plt.title("Mask")
            plt.show()

        if save_mask:
            WriteRaster(oRasterPath=os.path.join(self.o_path, "Mask.tif"), geoTransform=self.raster_info.geo_transform,
                        arrayList=[mask], epsg=self.raster_info.epsg_code)

        indexList = np.where(mask > 0)
        indexListTemp = list(np.where(mask > 0)[0])
        self.count_valid = len(indexListTemp)
        print('# valid pix=', self.count_valid)
        self.loc_valid = indexList

        return

    def store_extracted_comps(self, S, US):
        arrayList = []
        bandNameList = []
        for i in range(len(S)):
            US1 = US[:, i]
            array = np.empty((self.raster_info.raster_height, self.raster_info.raster_width))
            array[:, :] = np.nan
            index = 0
            for x, y in zip(self.loc_valid[0], self.loc_valid[1]):
                array[x, y] = US1[index]
                index += 1
            us_reshaped = np.copy(array)
            arrayList.append(us_reshaped)
            bandNameList.append("geoPCAIM_PC_" + str(i + 1))

        WriteRaster(
            oRasterPath=os.path.join(self.o_path, Path(self.in_data_cube_path).stem + "_geoPCAIM_Components.tif"),
            geoTransform=self.raster_info.geo_transform,
            arrayList=arrayList,
            epsg=self.raster_info.epsg_code,
            descriptions=bandNameList)
        return

    def geopcaim_based_reconstruction(self, U, S, V):

        NaN_arr = np.empty(shape=(self.disp_data_cube.shape[1], self.disp_data_cube.shape[2]))
        NaN_arr.fill(np.nan)
        nbImg = self.disp_data_cube.shape[0]
        for t in tqdm(range(nbImg), desc="Reconstruction"):
            ## Create the output image and intiate the differnt band with NaN values
            outputArr = np.empty(
                shape=(2 * self.nb_comp + 1, self.disp_data_cube.shape[1], self.disp_data_cube.shape[2]))
            outputArr.fill(np.nan)
            ## Initiate the first band of the output imgae with the original image
            outputArr[0, :, :] = np.array(self.disp_data_cube[t, :, :])
            bandNames = ["Original_Set_Image_" + str(t + 1)]

            ## Loop for reconstruction using only one PC
            for pc in range(self.nb_comp):
                tmpS = np.copy(S)
                for j in range(tmpS.size):
                    ## Take one singular values of S that corresponds to PC number i
                    if pc != j:
                        tmpS[j] = 0.0
                # print(tmpS)
                tmp = np.copy(NaN_arr)  ## NaN image

                ## Reconstruction of image t with the PC number pc
                tmpRecon = np.dot(U, np.dot(np.diag(tmpS), V.T))[:, t]
                # tmpRecon = tmpRecon + centering
                if tmpRecon.size != self.count_valid:
                    sys.exit("Fatal Error !!!")

                for j in range(tmpRecon.size):
                    tmp[self.loc_valid[0][j], self.loc_valid[1][j]] = tmpRecon[j]

                ## save the reconstruction in band
                outputArr[pc + 1, :, :] = np.array(tmp)
                del tmp
                bandNames.append("Reconstructed_with_only_PC_" + str(pc + 1))

                ## At the end of the loop outputArr will contain the reconstruction with each PC of image t
            ## Loop for reconstruction using multiple PCs
            for pc in range(self.nb_comp - 1):
                tmp = np.copy(NaN_arr)  ## Create NaN image
                if pc == 0:
                    tmp = np.array(outputArr[pc + 1, :, :] + outputArr[pc + 2, :, :])
                else:
                    tmp = np.array(outputArr[self.nb_comp + pc, :, :] + outputArr[pc + 2, :, :])
                outputArr[self.nb_comp + pc + 1, :, :] = np.array(tmp)
                del tmp
                bandNames.append("Reconstructed_with_PC_1_to_" + str(pc + 2))

            tmp = np.zeros(np.shape(NaN_arr))
            for j in range(self.count_valid):
                tmp[self.loc_valid[0][j], self.loc_valid[1][j]] = 255
            outputArr[2 * self.nb_comp, :, :] = np.array(tmp)
            del tmp
            bandNames.append("Pixels Considered in geoPCAIM")

            ## Save the output
            # if self.debug:
            #     print("Reconstruct image:", t + 1, " raster.shape=", outputArr.shape)
            listArrays = []
            for i in range(outputArr.shape[0]):
                listArrays.append(outputArr[i])

            WriteRaster(
                oRasterPath=os.path.join(self.o_path,
                                         Path(self.in_data_cube_path).stem + "_" + str(t + 1) + "_geoPCAIM_rec.tif"),
                geoTransform=self.raster_info.geo_transform, arrayList=listArrays,
                epsg=self.raster_info.epsg_code,
                descriptions=bandNames)

    def visualize_pcs(self, S, US, show=False):
        from geoCosiCorr3D.georoutines.geoplt_misc import ColorBar_

        for i in range(len(S)):
            US1 = US[:, i]
            if self.debug:
                print(US1.shape, US1.max(), US1.min())
            # array = geoICA.ReverseMask(mask=mask_reshaped, maskedArray=US1)

            array = np.empty((self.raster_info.raster_height, self.raster_info.raster_width))
            array[:, :] = np.nan
            index = 0
            for x, y in zip(self.loc_valid[0], self.loc_valid[1]):
                array[x, y] = US1[index]
                index += 1

            factor = 1
            cmap = "RdYlBu"

            fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
            us_reshaped = np.copy(array)
            us_reshapedStat = geoStat(us_reshaped, False)
            vMin = float(us_reshapedStat.mean) - factor * float(us_reshapedStat.std)
            vMax = float(us_reshapedStat.mean) + factor * float(us_reshapedStat.std)
            # vMin = -2
            # vMax = 2
            im = ax1.imshow(np.ma.masked_invalid(us_reshaped), vmin=vMin, vmax=vMax, cmap=cmap)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.set_title("US:" + str(i + 1))

            ColorBar_(ax=ax1, mapobj=im, cmap=cmap, vmin=vMin, vmax=vMax, orientation="vertical")

            figSaveFolder = CreateDirectory(self.o_path, "geoPCAIM_PCs_pngs", "n")
            plt.savefig(os.path.join(figSaveFolder, "geoPCAIM_PC" + str(i + 1) + ".png"), dpi=300)
            if show:
                plt.show()
            plt.close(fig)
        return

    def __call__(self, *args, **kwargs):

        """
            @SA: Enable the possibility to choose the mask:
            - option 1: Mask all NaN in all the data cube
            - option 2: Mask pixels if all the pixel in the data cube are NaN, otherwise
                        if a minimum number of valid values exist (minValidVal), replace the NaN pixel with:
                        average, median, ...
            - option 3: predefined mask (shapeFile, or raster )
        """

        print("=================================================================")
        print(f"#PC:{self.nb_comp}")
        print(f"W:{self.weight}")
        print(f"minAnalysis: {self.min_num_analysis}")
        print("mask_vals: [-{},{}]".format(self.mask_val, self.mask_val))
        print(f"Initialisation of missing values:{self.init_mean}")
        print(f"Iterative params:{self.max_iter} - {self.tol}")
        print("=================================================================")

        if self.min_num_analysis == 0:
            self.build_mask_any_nan(in_arr=self.disp_data_cube, mask_val=self.mask_val)
        else:
            self.build_mask_minimum(in_arr=self.disp_data_cube, mask_val=self.mask_val,
                                    min_val=self.min_num_analysis)

        """
            Reformat data for ease of use. 
            Each 2D image is reformatted as a (big) 1D line and only for the pixel with a valid time series
             (identified in previous step).
        """
        shape = self.disp_data_cube.shape
        disp_data_cube2 = np.empty(shape=(shape[0], self.count_valid), dtype=np.float32)
        if self.debug:
            print("disp_data_cube2.shape=", disp_data_cube2.shape)
        imgNb = shape[0]
        # TODO : This loop is very long. --> @SA: future work
        for i in tqdm(range(imgNb), desc="Reformatting data"):
            for j in range(self.count_valid):
                disp_data_cube2[i, j] = self.disp_data_cube[i, self.loc_valid[0][j], self.loc_valid[1][j]]

        ## Compute the average
        mean_arr = np.nanmean(disp_data_cube2, axis=0)

        """
            Compute the number of NaN in the images for the pixel whose time-series is considered valid,
            i.e.,at least min_num_analysis valid measure.
            And Compute the Weight matrix
        """

        weights = np.zeros((imgNb, self.count_valid))
        nbOfNaNArr = np.zeros((1, imgNb))

        for i in range(imgNb):
            #  compute number of NaNs
            locNaN = np.where(np.isnan(disp_data_cube2[i]))
            locNoNaN = np.where(~np.isnan(disp_data_cube2[i]))
            nbNaN = len(locNaN[0])
            if self.debug:
                print("Img=", i, "  #NANs=", nbNaN, "#Val=", len(locNoNaN[0]))

            if len(locNoNaN[0]) == 0:
                if self.debug:
                    print("Error only NaN values in image number:", i + 1)

            """
                Replace NaN values by either:
                    - 0.0 or 
                    - by the average of the "column" (avg of the time series at that pixel).
            """
            if nbNaN != 0:
                if self.init_mean == 1:
                    """
                      If whitening of the data with average is chosen: for each pixel, subtract the
                       average off its time-series
                    """
                    disp_data_cube2[i, locNaN[0]] = mean_arr[locNaN[0]]
                else:
                    disp_data_cube2[i, locNaN[0]] = 0.0
            """
                Compute a weight (confidence) for each measurement. 
            """
            nbOfNaNArr[0, i] = nbNaN
            if self.weight == 0:
                weights[i, locNoNaN[0]] = 1 / 0.1
            else:
                weights[i, locNoNaN[0]] = ((self.count_valid - nbNaN) / self.count_valid) ** self.weight
        ## Normalize the weight
        maxWeight = np.max(weights)
        weights = weights / maxWeight

        """
            Reshape the data cube in order to have in column the nb of images and in rows the locValid values
        """
        disp_data_cube2 = disp_data_cube2.T
        if self.debug:
            print("Data shape: ", disp_data_cube2.shape)
        weights = weights.T
        if self.debug:
            print("normWeight shape: ", weights.shape)

        U, S, V, objectiveValues = self.run_geoPCAIM(data=disp_data_cube2,
                                                     weights=weights,
                                                     nbOfNaNArr=nbOfNaNArr,
                                                     nbComponents=self.nb_comp)
        percentVar = 100 * (S ** 2) / np.sum(S ** 2)
        if self.debug:
            print("PercentVar=", percentVar)
        np.savetxt(os.path.join(self.o_path, "_U.txt"), U, delimiter="\t\t", fmt='%15.25f')
        np.savetxt(os.path.join(self.o_path, "_V.txt"), V, delimiter="\t\t", fmt='%15.25f')
        np.savetxt(os.path.join(self.o_path, "_S.txt"), np.diag(S), delimiter="\t\t", fmt='%15.25f')
        np.savetxt(os.path.join(self.o_path, "_Chi2.txt"), objectiveValues, delimiter="\t\t", fmt='%15.25f')
        locValidArray = np.empty((self.count_valid, 2))
        locValidArray[:, 0] = self.loc_valid[0][:]
        locValidArray[:, 1] = self.loc_valid[1][:]
        np.savetxt(os.path.join(self.o_path, "_LocaValid.txt"), locValidArray, delimiter="\t\t", fmt='%d')

        utils.plot_s(S=S, oPath=os.path.join(self.o_path, Path(self.in_data_cube_path).stem + "_geoPCAIM.png"),
                     show=False)
        utils.chi2_F_test(chi2=objectiveValues)

        sDiag = np.diag(S)
        US = np.dot(U, sDiag)

        self.store_extracted_comps(S, US)
        self.visualize_pcs(S=S, US=US, show=False)
        self.geopcaim_based_reconstruction(U=U, S=S, V=V)
