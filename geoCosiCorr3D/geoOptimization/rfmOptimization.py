"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import numpy as np
import os

from pathlib import Path

import geoRoutines.Plotting.Plotting_Routine as pltRT
import geoRoutines.georoutines as geoRT

from geoCosiCorr3D.geoOptimization.Optimize import Optimize

import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarn

geoWarn.wrIgnoreNotGeoreferencedWarning()


class cRFMRefinement:
    def __init__(self, rfmModel, gcpFile="", gcpArray=np.zeros((1, 1)), oFolder=None, demPath=None, visualize=False,
                 gcpFilter={"flag": True, "factor": 0.75, "Opt_gcp": {"flag": True, "params": []}}):
        """

        Args:
            rfmModel:
            gcpFile:
            oFolder:
            demPath:
            visualize:
            gcpFilter:
        """

        self.rfmModel = rfmModel
        self.gcpFile = gcpFile
        self.demPath = demPath
        self.oFolder = oFolder
        self.gcpFilter = gcpFilter
        if self.demPath == None:
            self.demInfo = None
        else:
            self.demInfo = geoRT.RasterInfo(self.demPath)
        if gcpFile == "" and np.all((gcpArray == 0)):
            ValueError("Error: please provide GCP file or array")
        if gcpFile != "":
            self.gcpArray = np.loadtxt(gcpFile)
        elif np.all((gcpArray == 0)) == False:
            self.gcpArray = gcpArray
        else:
            ValueError("Error: please provide GCP file or array")

        self.nbGCPs_ini = self.gcpArray.shape[0]
        self.ComputeError_ImgSpace(visualize=visualize)
        self.ComputeCorrection()
        if visualize == True:
            self.DisplayErrorAfterRefinement()

    def ComputeError_ImgSpace(self, visualize=False):
        cols, rows = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                  lat=self.gcpArray[:, 1],
                                                  alt=self.gcpArray[:, 2])
        xyPix_with_RFM = np.array([cols, rows]).T
        self.xyPixError = np.array([self.gcpArray[:, 3], self.gcpArray[:, 4]]).T - xyPix_with_RFM
        xStat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 0], displayValue=False)
        yStat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 1], displayValue=False)

        gcpArrayList = []
        if self.gcpFilter["flag"] == True:
            print("___Filtering GCPs____")
            factor = self.gcpFilter["factor"]
            # tqdm(range(len(orthoList)))
            for index in range(self.gcpArray.shape[0]):
                if (self.xyPixError[index, 0] < float(xStat.mean) + factor * float(xStat.std)
                    and self.xyPixError[index, 0] > float(xStat.mean) - factor * float(xStat.std)) or \
                        (self.xyPixError[index, 1] < float(yStat.mean) + factor * float(yStat.std)
                         and self.xyPixError[index, 1] > float(yStat.mean) - factor * float(yStat.std)):
                    gcpArrayList.append(self.gcpArray[index, :])

            self.gcpArray = np.array(gcpArrayList)

            cols, rows = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                      lat=self.gcpArray[:, 1],
                                                      alt=self.gcpArray[:, 2])
            xyPix_with_RFM = np.array([cols, rows]).T
            self.xyPixError = np.array([self.gcpArray[:, 3], self.gcpArray[:, 4]]).T - xyPix_with_RFM

        # if visualize == True:
        #     pltRT.Plot_residuals(sample=self.xyPixError[:, 0], title="xPixError", save=True, oFolder=self.oFolder)
        #     pltRT.Plot_residuals(sample=self.xyPixError[:, 1], title="yPixError", save=True, oFolder=self.oFolder)
        # plt.show()

        # sys.exit()
        colsN, rowsN = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                    lat=self.gcpArray[:, 1],
                                                    alt=self.gcpArray[:, 2],
                                                    normalized=True)
        self.xyPix_with_RFM_N = np.array([colsN, rowsN]).T
        xyPix_Ref_N = np.array([(self.gcpArray[:, 3] - self.rfmModel.colOff) / self.rfmModel.colScale,
                                (self.gcpArray[:, 4] - self.rfmModel.linOff) / self.rfmModel.linScale])

        self.xyPixError_N = xyPix_Ref_N.T - self.xyPix_with_RFM_N

        return

    def ComputeCorrection(self):
        # # # ====================================== Compute correction ===================================================#

        res_N = Optimize(xy_obs_with_model=self.xyPix_with_RFM_N, xy_error=self.xyPixError_N)
        corr_N = np.asarray(res_N.x).reshape(2, 3)
        self.correction_N = np.vstack([corr_N, [0, 0, 1]])
        if self.oFolder == None:
            self.corrModelFile = os.path.join(os.path.dirname(self.rfmModel.RFMFile),
                                              Path(self.rfmModel.RFMFile).stem + "_RFM_correction.txt")
            np.savetxt(self.corrModelFile, self.correction_N)

        else:
            self.corrModelFile = os.path.join(self.oFolder,
                                              Path(self.rfmModel.RFMFile).stem + "_RFM_correction.txt")
            np.savetxt(self.corrModelFile, self.correction_N)

    def DisplayErrorAfterRefinement(self):
        cols_opt, rows_opt = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                          lat=self.gcpArray[:, 1],
                                                          alt=self.gcpArray[:, 2],
                                                          corrModel=self.correction_N)
        xyPix_with_RFM_opt_ = np.array([cols_opt, rows_opt]).T
        xyPixError_opt_ = np.array([self.gcpArray[:, 3], self.gcpArray[:, 4]]).T - xyPix_with_RFM_opt_
        print("")
        pltRT.Plot_residuals_before_after(sampleIni=self.xyPixError[:, 0],
                                          sampleFinal=xyPixError_opt_[:, 0],
                                          title="xPixError_before_after",
                                          yLabel="Error[pix]", save=True, oFolder=self.oFolder)

        pltRT.Plot_residuals_before_after(sampleIni=self.xyPixError[:, 1],
                                          sampleFinal=xyPixError_opt_[:, 1],
                                          title="yPixError_before_after",
                                          yLabel="Error[pix]", save=True, oFolder=self.oFolder)

        stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 0], displayValue=False)
        self.xPixError_mean = stat.mean
        self.xPixError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 1], displayValue=False)
        self.yPixError_mean = stat.mean
        self.yPixError_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=xyPixError_opt_[:, 0], displayValue=False)
        self.xPixError_opt_mean = stat.mean
        self.xPixError_opt_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=xyPixError_opt_[:, 0], displayValue=False)
        self.yPixError_opt_mean = stat.mean
        self.yPixError_opt_RMSE = stat.RMSE
        self.xyPixError_ini_RMSE = np.sqrt(
            float(self.xPixError_RMSE) ** 2 + float(self.yPixError_RMSE) ** 2)
        self.xyPixError_opt_RMSE = np.sqrt(
            float(self.xPixError_opt_RMSE) ** 2 + float(self.yPixError_opt_RMSE) ** 2)

        # ===================================== Compute error in object space ==============================================#
        lon, lat, alt = self.rfmModel.Img2Ground_RFM(col=list(self.gcpArray[:, 3]),
                                                     lin=list(self.gcpArray[:, 4]),
                                                     demInfo=self.demInfo)

        utmProj = geoRT.ComputeEpsg(lon=lon[0], lat=lat[0])
        utmCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=lat, Y=lon, Z=alt, sourceEPSG=4326, targetEPSG=utmProj)
        refCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=self.gcpArray[:, 1], Y=self.gcpArray[:, 0], Z=self.gcpArray[:, 2],
                                                   sourceEPSG=4326,
                                                   targetEPSG=utmProj)
        xMapError = np.array(refCoord[0]) - np.array(utmCoord[0])
        yMapError = np.array(refCoord[1]) - np.array(utmCoord[1])
        altError = np.array(refCoord[2]) - np.array(utmCoord[2])

        lon_op, lat_op, alt_op = self.rfmModel.Img2Ground_RFM(col=list(self.gcpArray[:, 3]),
                                                              lin=list(self.gcpArray[:, 4]),
                                                              corrModel=self.correction_N,
                                                              demInfo=self.demInfo)

        utmProj = geoRT.ComputeEpsg(lon=lon_op[0], lat=lat_op[0])
        utmCoord_op = geoRT.ConvCoordMap1ToMap2_Batch(X=lat_op, Y=lon_op, Z=alt_op, sourceEPSG=4326, targetEPSG=utmProj)
        refCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=self.gcpArray[:, 1], Y=self.gcpArray[:, 0], Z=self.gcpArray[:, 2],
                                                   sourceEPSG=4326,
                                                   targetEPSG=utmProj)
        xMapError_op = np.array(refCoord[0]) - np.array(utmCoord_op[0])
        yMapError_op = np.array(refCoord[1]) - np.array(utmCoord_op[1])
        altError_op = np.array(refCoord[2]) - np.array(utmCoord_op[2])

        pltRT.Plot_residuals_before_after(sampleIni=xMapError, sampleFinal=xMapError_op, title="xMapError",
                                          yLabel="Error[m]", save=True, oFolder=self.oFolder)
        pltRT.Plot_residuals_before_after(sampleIni=yMapError, sampleFinal=yMapError_op, title="yMapError",
                                          yLabel="Error[m]", save=True, oFolder=self.oFolder)
        # pltRT.Plot_residuals_before_after(sampleIni=altError, sampleFinal=altError_op, title="altError",
        #                                   yLabel="Error[m]", save=True, oFolder=self.oFolder)

        stat = geoRT.cgeoStat(inputArray=xMapError, displayValue=False)
        self.xMapError_mean = stat.mean
        self.xMapError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=yMapError, displayValue=False)
        self.yMapError_mean = stat.mean
        self.yMapError_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=xMapError_op, displayValue=False)
        self.xMapError_opt_mean = stat.mean
        self.xMapError_opt_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=yMapError_op, displayValue=False)
        self.yMapError_opt_mean = stat.mean
        self.yMapError_opt_RMSE = stat.RMSE

        self.xyMapError_ini_RMSE = np.sqrt(
            float(self.xMapError_RMSE) ** 2 + float(self.yMapError_RMSE) ** 2)
        self.xyMapError_opt_RMSE = np.sqrt(
            float(self.xMapError_opt_RMSE) ** 2 + float(self.yMapError_opt_RMSE) ** 2)
        print("-------------------------------------------------------")
        print("#GCPs: {} ------> {}".format(self.nbGCPs_ini, self.gcpArray.shape[0]))
        print(
            "xyPix_ini_RMSE:{:.3} ---> xyPix_opt_RMSE:{:.3}  [pix]".format(self.xyPixError_ini_RMSE,
                                                                           self.xyPixError_opt_RMSE))
        print(
            "xyMap_ini_RMSE:{:.3} ---> xyMap_opt_RMSE:{:.3}  [m]".format(self.xyMapError_ini_RMSE,
                                                                         self.xyMapError_opt_RMSE))
        print("-------------------------------------------------------")

        stat = geoRT.cgeoStat(inputArray=altError, displayValue=False)
        self.altError_mean = stat.mean
        self.altError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=altError_op, displayValue=False)
        self.altError_opt_mean = stat.mean
        self.altError_opt_RMSE = stat.RMSE

        reportArray = np.zeros((self.gcpArray.shape[0], 13))

        reportArray[:, 0] = self.gcpArray[:, 0]
        reportArray[:, 1] = self.gcpArray[:, 1]
        reportArray[:, 2] = self.gcpArray[:, 2]
        reportArray[:, 3] = self.gcpArray[:, 3]
        reportArray[:, 4] = self.gcpArray[:, 4]

        reportArray[:, 5] = self.xyPixError[:, 0]
        reportArray[:, 6] = self.xyPixError[:, 1]
        reportArray[:, 7] = xyPixError_opt_[:, 0]
        reportArray[:, 8] = xyPixError_opt_[:, 1]

        reportArray[:, 9] = xMapError
        reportArray[:, 10] = yMapError
        reportArray[:, 11] = xMapError_op
        reportArray[:, 12] = yMapError_op

        self.report = {"lon": reportArray[:, 0],
                       "lat": reportArray[:, 1],
                       "alt": reportArray[:, 2],
                       "xPix": reportArray[:, 3],
                       "yPix": reportArray[:, 4],
                       "xPixEr": reportArray[:, 5],
                       "yPixEr": reportArray[:, 6],
                       "xPixEr_opt": reportArray[:, 7],
                       "yPixEr_opt": reportArray[:, 8],
                       "xMapEr": reportArray[:, 9],
                       "yMapEr": reportArray[:, 10],
                       "xMapEr_opt": reportArray[:, 11],
                       "yMapEr_opt": reportArray[:, 12]}
        import pandas
        df = pandas.DataFrame.from_dict(self.report)
        df.to_csv(os.path.join(os.path.dirname(self.gcpFile), Path(self.gcpFile).stem + "_RFM_opt_report.csv"),
                  index=False, header=True)


class cRFMRefinement_:
    def __init__(self, rfmModel, gcpFile, oFolder=None, demPath=None, visualize=False):

        self.rfmModel = rfmModel
        self.gcpFile = gcpFile
        self.demPath = demPath
        self.oFolder = oFolder
        if self.demPath == None:
            self.demInfo = None
        else:
            self.demInfo = geoRT.RasterInfo(self.demPath)
        self.gcpArray = np.loadtxt(gcpFile)

        self.ComputeError_ImgSpace(visualize=visualize)
        self.ComputeCorrection()
        if visualize == True:
            self.DisplayErrorAfterRefinement()

    def ComputeError_ImgSpace(self, visualize=False):
        cols, rows = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                  lat=self.gcpArray[:, 1],
                                                  alt=self.gcpArray[:, 2])
        xyPix_with_RFM = np.array([cols, rows]).T
        self.xyPixError = np.array([self.gcpArray[:, 3], self.gcpArray[:, 4]]).T - xyPix_with_RFM

        # stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 0], displayValue=False)
        # self.xPixError_mean = stat.mean
        # self.xPixError_RMSE = stat.RMSE
        # stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 1], displayValue=False)
        # self.yPixError_mean = stat.mean
        # self.yPixError_RMSE = stat.RMSE

        if visualize == True:
            pltRT.Plot_residuals(sample=self.xyPixError[:, 0], title="xPixError", save=True, oFolder=self.oFolder)
            pltRT.Plot_residuals(sample=self.xyPixError[:, 1], title="yPixError", save=True, oFolder=self.oFolder)
            # plt.show()

        colsN, rowsN = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                    lat=self.gcpArray[:, 1],
                                                    alt=self.gcpArray[:, 2],
                                                    normalized=True)
        self.xyPix_with_RFM_N = np.array([colsN, rowsN]).T
        xyPix_Ref_N = np.array([(self.gcpArray[:, 3] - self.rfmModel.colOff) / self.rfmModel.colScale,
                                (self.gcpArray[:, 4] - self.rfmModel.linOff) / self.rfmModel.linScale])

        self.xyPixError_N = xyPix_Ref_N.T - self.xyPix_with_RFM_N

        return

    def ComputeCorrection(self):
        # # # ====================================== Compute correction ===================================================#

        res_N = Optimize(xy_obs_with_model=self.xyPix_with_RFM_N, xy_error=self.xyPixError_N)
        corr_N = np.asarray(res_N.x).reshape(2, 3)
        self.correction_N = np.vstack([corr_N, [0, 0, 1]])
        if self.oFolder == None:
            self.corrModelFile = os.path.join(os.path.dirname(self.rfmModel.RFMFile),
                                              Path(self.rfmModel.RFMFile).stem + "_RFM_correction.txt")
            np.savetxt(self.corrModelFile, self.correction_N)

        else:
            self.corrModelFile = os.path.join(self.oFolder,
                                              Path(self.rfmModel.RFMFile).stem + "_RFM_correction.txt")
            np.savetxt(self.corrModelFile, self.correction_N)

    def DisplayErrorAfterRefinement(self):
        cols_opt, rows_opt = self.rfmModel.Ground2Img_RFM(lon=self.gcpArray[:, 0],
                                                          lat=self.gcpArray[:, 1],
                                                          alt=self.gcpArray[:, 2],
                                                          corrModel=self.correction_N)
        xyPix_with_RFM_opt_ = np.array([cols_opt, rows_opt]).T
        xyPixError_opt_ = np.array([self.gcpArray[:, 3], self.gcpArray[:, 4]]).T - xyPix_with_RFM_opt_
        pltRT.Plot_residuals_before_after(sampleIni=self.xyPixError[:, 0],
                                          sampleFinal=xyPixError_opt_[:, 0],
                                          title="xPixError_before_afte",
                                          yLabel="Error[pix]", save=True, oFolder=self.oFolder)

        pltRT.Plot_residuals_before_after(sampleIni=self.xyPixError[:, 1],
                                          sampleFinal=xyPixError_opt_[:, 1],
                                          title="yPixError_before_after",
                                          yLabel="Error[pix]", save=True, oFolder=self.oFolder)

        # ===================================== Compute error in object space ==============================================#
        lon, lat, alt = self.rfmModel.Img2Ground_RFM(col=list(self.gcpArray[:, 3]),
                                                     lin=list(self.gcpArray[:, 4]),
                                                     demInfo=self.demInfo)

        utmProj = geoRT.ComputeEpsg(lon=lon[0], lat=lat[0])
        utmCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=lat, Y=lon, Z=alt, sourceEPSG=4326, targetEPSG=utmProj)
        refCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=self.gcpArray[:, 1], Y=self.gcpArray[:, 0], Z=self.gcpArray[:, 2],
                                                   sourceEPSG=4326,
                                                   targetEPSG=utmProj)
        xMapError = np.array(refCoord[0]) - np.array(utmCoord[0])
        yMapError = np.array(refCoord[1]) - np.array(utmCoord[1])
        altError = np.array(refCoord[2]) - np.array(utmCoord[2])

        lon_op, lat_op, alt_op = self.rfmModel.Img2Ground_RFM(col=list(self.gcpArray[:, 3]),
                                                              lin=list(self.gcpArray[:, 4]),
                                                              corrModel=self.correction_N,
                                                              demInfo=self.demInfo)

        utmProj = geoRT.ComputeEpsg(lon=lon_op[0], lat=lat_op[0])
        utmCoord_op = geoRT.ConvCoordMap1ToMap2_Batch(X=lat_op, Y=lon_op, Z=alt_op, sourceEPSG=4326, targetEPSG=utmProj)
        refCoord = geoRT.ConvCoordMap1ToMap2_Batch(X=self.gcpArray[:, 1], Y=self.gcpArray[:, 0], Z=self.gcpArray[:, 2],
                                                   sourceEPSG=4326,
                                                   targetEPSG=utmProj)
        xMapError_op = np.array(refCoord[0]) - np.array(utmCoord_op[0])
        yMapError_op = np.array(refCoord[1]) - np.array(utmCoord_op[1])
        altError_op = np.array(refCoord[2]) - np.array(utmCoord_op[2])

        pltRT.Plot_residuals_before_after(sampleIni=xMapError, sampleFinal=xMapError_op, title="xMapError",
                                          yLabel="Error[m]", save=True, oFolder=self.oFolder)
        pltRT.Plot_residuals_before_after(sampleIni=yMapError, sampleFinal=yMapError_op, title="yMapError",
                                          yLabel="Error[m]", save=True, oFolder=self.oFolder)
        pltRT.Plot_residuals_before_after(sampleIni=altError, sampleFinal=altError_op, title="altError",
                                          yLabel="Error[m]", save=True, oFolder=self.oFolder)

        stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 0], displayValue=False)
        self.xPixError_mean = stat.mean
        self.xPixError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=self.xyPixError[:, 1], displayValue=False)
        self.yPixError_mean = stat.mean
        self.yPixError_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=xyPixError_opt_[:, 0], displayValue=False)
        self.xPixError_opt_mean = stat.mean
        self.xPixError_opt_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=xyPixError_opt_[:, 0], displayValue=False)
        self.yPixError_opt_mean = stat.mean
        self.yPixError_opt_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=xMapError, displayValue=False)
        self.xMapError_mean = stat.mean
        self.xMapError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=yMapError, displayValue=False)
        self.yMapError_mean = stat.mean
        self.yMapError_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=xMapError_op, displayValue=False)
        self.xMapError_opt_mean = stat.mean
        self.xMapError_opt_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=yMapError_op, displayValue=False)
        self.yMapError_opt_mean = stat.mean
        self.yMapError_opt_RMSE = stat.RMSE

        stat = geoRT.cgeoStat(inputArray=altError, displayValue=False)
        self.altError_mean = stat.mean
        self.altError_RMSE = stat.RMSE
        stat = geoRT.cgeoStat(inputArray=altError_op, displayValue=False)
        self.altError_opt_mean = stat.mean
        self.altError_opt_RMSE = stat.RMSE

        # plt.show()
