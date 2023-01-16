"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""
import numpy as np
import os

from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo, WriteRaster, Convert
from geoCosiCorr3D.geo3DDA.misc import LOS
from geoCosiCorr3D.geo3DDA.triangulation import triangulate


class cCompute3DD:
    # TODO: data_eset = {corr_set:{precorr...}, rsm_set:{--}, trx_Set{}}
    def __init__(self, pre_event_corr,
                 event_corr_1,
                 event_corr_2,
                 dem_path,
                 base_trx,
                 pre_trx,
                 post_trx_1,
                 post_trx2,
                 base_rsm,
                 pre_rsm,
                 post_1_rsm,
                 post_2_rsm,
                 output,
                 inversionFlag=False):
        self.preEventCorr = pre_event_corr
        self.eventCorr1 = event_corr_1
        self.eventCorr2 = event_corr_2
        self.demPath = dem_path
        self.tr1 = base_trx
        self.tr2 = pre_trx
        self.tr3 = post_trx_1
        self.tr4 = post_trx2
        self.filename_ancillary_master_pre = base_rsm
        self.filename_ancillary_slave_pre = pre_rsm
        self.filename_ancillary_post_1 = post_1_rsm
        self.filename_ancillary_post_2 = post_2_rsm
        self.output = output

        self.inversionFlag = inversionFlag
        self.loading_set_data()
        self.run_3DD()

    @staticmethod
    def load_rsm(rsm_file):
        import pickle
        with open(rsm_file, "rb") as input:
            obj_ = pickle.load(input)
        return obj_.interpSatPosition

    def loading_set_data(self):
        # print("## Loading correlation")
        from geoCosiCorr3D.geoImageCorrelation.misc import cCorrelationInfo
        self.pre_corr_info = cCorrelationInfo(self.preEventCorr)
        self.pre_ew_corr = self.pre_corr_info.ew_corr
        self.pre_ns_corr = self.pre_corr_info.ns_corr
        self.pre_sz = self.pre_corr_info.corr_sz
        self.set_epsg = self.pre_corr_info.corr_raster_info.epsg_code

        self.event_1_corr_info = cCorrelationInfo(self.eventCorr1)
        self.event_1_corr_ew = self.event_1_corr_info.ew_corr
        self.event_1_corr_ns = self.event_1_corr_info.ns_corr
        self.event_1_corr_sz = self.event_1_corr_info.corr_sz

        self.event_2_corr_info = cCorrelationInfo(self.eventCorr2)
        self.event_2_corr_ew = self.event_2_corr_info.ew_corr
        self.event_2_corr_ns = self.event_2_corr_info.ns_corr
        self.event_2_corr_sz = self.event_2_corr_info.corr_sz

        ##Fixme: mask invalid values (i.e.,-32767 )
        self.dem_info = cRasterInfo(self.demPath)
        demArray = self.dem_info.raster_array[0]
        demArray = np.ma.masked_invalid(demArray)
        demArray = np.ma.masked_where(demArray == -32767, demArray)
        self.dem_array = demArray.filled(fill_value=0)
        # print(self.dem_array, self.dem_array.shape)

        if self.dem_info.epsg_code != self.set_epsg:
            # TO de reproject DEM to the same resolution
            raise IOError('DEM and correlation have different projection system')

        self.yMat_pre1 = cRasterInfo(self.tr1).image_as_array(2)
        self.yMat_pre2 = cRasterInfo(self.tr2).image_as_array(2)
        self.yMat_post1 = cRasterInfo(self.tr3).image_as_array(2)
        self.yMat_post2 = cRasterInfo(self.tr4).image_as_array(2)

        self.interpSatPos_pre1 = self.load_rsm(self.filename_ancillary_master_pre)
        self.interpSatPos_pre2 = self.load_rsm(self.filename_ancillary_slave_pre)
        self.interpSatPos_post1 = self.load_rsm(self.filename_ancillary_post_1)
        self.interpSatPos_post2 = self.load_rsm(self.filename_ancillary_post_2)

        ##TODO
        ## Note : we assume that all correlations have the same grid and size
        ## Need to deal with NAN values

    def run_3DD(self):

        if not self.pre_sz == self.event_1_corr_sz or not self.pre_sz == self.event_2_corr_sz:
            raise ValueError('Correlations does not have the same size !!')
        nbXcorr = self.pre_sz[1]  ### col
        nbYcorr = self.pre_sz[0]  ### Line
        # print(nbXcorr, nbYcorr)

        ## Setup the output arrays

        ewCorrected_ = np.zeros(shape=(nbYcorr, nbXcorr))
        nsCorrected_ = np.zeros(shape=(nbYcorr, nbXcorr))
        dz_ = np.zeros(shape=(nbYcorr, nbXcorr))
        normPre_ = np.zeros(shape=(nbYcorr, nbXcorr))
        normPost_ = np.zeros(shape=(nbYcorr, nbXcorr))
        masterElevation = np.zeros(shape=(nbXcorr))
        slaveNewElevation = np.zeros(shape=(nbXcorr))
        # print(nbXcorr)
        # nbXcorr = 51
        # nbYcorr = 100
        y = 0
        while y < nbYcorr:
            # for y in range(nbYcorr):  ## line
            # print(y, '/', nbYcorr)
            # for x in range(nbXcorr):  ## col
            # print("x=", x, "  y=", y)
            ##FIXME
            ## is this the best way to look for nan values
            ## how about if nans are sotred as a value (i.e., -32767) !!
            ## how about the DEM has a NaN value or -32767 ??
            nan_index = list(np.where(np.isnan(self.event_1_corr_ew[y, 0:nbXcorr]))[0])
            nan_index.extend(np.where(np.isnan(self.event_1_corr_ns[y, 0:nbXcorr]))[0])
            nan_index.extend(np.where(np.isnan(self.event_2_corr_ew[y, 0:nbXcorr]))[0])
            nan_index.extend(np.where(np.isnan(self.event_2_corr_ns[y, 0:nbXcorr]))[0])
            nan_index.extend(np.where(np.isnan(self.pre_ew_corr[y, 0:nbXcorr]))[0])
            nan_index.extend(np.where(np.isnan(self.pre_ns_corr[y, 0:nbXcorr]))[0])
            index_nan = list(set(nan_index))
            # print(len(index_nan))
            if len(index_nan) > 0:
                # print("nan_exist")
                nbXcorr_valid = nbXcorr - len(index_nan)
                # print("nbXcorr_valid=", nbXcorr_valid)
                ewCorrected = np.empty(nbXcorr)
                nsCorrected = np.empty(nbXcorr)
                dz = np.empty(nbXcorr)
                normPre = np.empty(nbXcorr)
                normPost = np.empty(nbXcorr)
                if nbXcorr_valid != 0:
                    # raise NotImplemented('THIS CODE IS NOT CHECKED YET')
                    # for val in index_nan:
                    (xMap_M, yMap_M) = self.pre_corr_info.corr_raster_info.Pixel2Map(list(range(nbXcorr)),
                                                                                     nbXcorr * [y])
                    # print("len(xMap_M):{}, len(yMap_M):{}".format(len(xMap_M), len(yMap_M)))

                    xMap_M_val = list(np.delete(np.asarray(xMap_M), index_nan))
                    yMap_M_val = list(np.delete(np.asarray(yMap_M), index_nan))
                    # print(xMap_M_val, yMap_M_val)
                    los_arg = [xMap_M_val, yMap_M_val, self.pre_corr_info.corr_raster_info, self.dem_info, self.tr1,
                               self.yMat_pre1, self.interpSatPos_pre1]
                    sightVector_M, XYZ_cart_M = LOS.compute_los(*los_arg)

                    xMap_S = xMap_M + self.pre_ew_corr[y, 0:nbXcorr]
                    yMap_S = yMap_M + self.pre_ns_corr[y, 0:nbXcorr]
                    xMap_S_val = list(np.delete(np.asarray(xMap_S), index_nan))
                    yMap_s_val = list(np.delete(np.asarray(yMap_S), index_nan))
                    los_arg = [xMap_S_val, yMap_s_val, self.pre_corr_info.corr_raster_info, self.dem_info, self.tr2,
                               self.yMat_pre2, self.interpSatPos_pre2]
                    sightVector_S, XYZ_cart_S = LOS.compute_los(*los_arg)

                    diffPre = np.asarray(XYZ_cart_S) - np.asarray(XYZ_cart_M)
                    # print(f'pre_diff:{diffPre}')

                    params, pt1_new, pt2_new, residu, XYZ_cart_PreCorrected = triangulate(XYZ_cart_M=XYZ_cart_M,
                                                                                          XYZ_cart_S=XYZ_cart_S,
                                                                                          sightVector_M=sightVector_M,
                                                                                          sightVector_S=sightVector_S)

                    diffPre_corrected = np.asarray(pt2_new) - np.asarray(pt1_new)

                    diffPre_corrected_Norm2 = np.linalg.norm(diffPre_corrected, axis=1, ord=2)
                    # print(diffPre_corrected_Norm2)
                    # print(XYZ_cart_PreCorrected,"\n",XYZ_cart_PreCorrected[:,1])

                    new_pre_geo = Convert.cartesian_2_geo(x=XYZ_cart_PreCorrected[:, 0],
                                                          y=XYZ_cart_PreCorrected[:, 1],
                                                          z=XYZ_cart_PreCorrected[:, 2])
                    new_pre_geo = np.array(new_pre_geo).T
                    (xMapnew_pre, yMapnew_pre, zMapnew_pre) = Convert.coord_map1_2_map2(X=list(new_pre_geo[:, 1]),
                                                                                        Y=list(new_pre_geo[:, 0]),
                                                                                        Z=list(new_pre_geo[:, 2]),
                                                                                        sourceEPSG=4326,
                                                                                        targetEPSG=self.set_epsg)
                    # print(xMapnew_pre, yMapnew_pre, zMapnew_pre)

                    # ############## Post images ###########################################################################

                    xMap_post1 = xMap_M + self.event_1_corr_ew[y, 0:nbXcorr]
                    yMap_post1 = yMap_M + self.event_1_corr_ns[y, 0:nbXcorr]
                    xMap_post1_val = list(np.delete(np.asarray(xMap_post1), index_nan))
                    yMap_post1_val = list(np.delete(np.asarray(yMap_post1), index_nan))
                    los_arg = [xMap_post1_val, yMap_post1_val, self.event_1_corr_info.corr_raster_info, self.dem_info,
                               self.tr3, self.yMat_post1, self.interpSatPos_post1]
                    sightVector_Post1, XYZ_cart_Post1 = LOS.compute_los(*los_arg)
                    # print(sightVector_Post1)

                    xMap_post2 = xMap_M + self.event_2_corr_ew[y, 0:nbXcorr]
                    yMap_post2 = yMap_M + self.event_2_corr_ns[y, 0:nbXcorr]
                    xMap_post2_val = list(np.delete(np.asarray(xMap_post2), index_nan))
                    yMap_post2_val = list(np.delete(np.asarray(yMap_post2), index_nan))
                    los_arg = [xMap_post2_val, yMap_post2_val, self.event_2_corr_info.corr_raster_info, self.dem_info,
                               self.tr4, self.yMat_post2, self.interpSatPos_post2]
                    sightVector_Post2, XYZ_cart_Post2 = LOS.compute_los(*los_arg)

                    # # print(xMap_post1,yMap_post2)
                    diffPost = np.asarray(XYZ_cart_Post2) - np.asarray(XYZ_cart_Post1)

                    paramsPost, pt1_post_new, pt2_post_new, residu_post, XYZ_cart_PostCorrected = triangulate(
                        XYZ_cart_M=XYZ_cart_Post1,
                        XYZ_cart_S=XYZ_cart_Post2,
                        sightVector_M=sightVector_Post1,
                        sightVector_S=sightVector_Post2)
                    #
                    diffPost_corrected = pt2_post_new - pt1_post_new
                    # # print(diffPost_corrected.shape)
                    diffPost_corrected_Norm2 = np.linalg.norm(diffPost_corrected, axis=1, ord=2)
                    # print(diffPost_corrected_Norm2)
                    # #     normPost[x] = diffPost_corrected_Norm2
                    #     #
                    new_post_geo = Convert.cartesian_2_geo(XYZ_cart_PostCorrected[:, 0],
                                                           XYZ_cart_PostCorrected[:, 1],
                                                           XYZ_cart_PostCorrected[:, 2])
                    new_post_geo = np.array(new_post_geo).T
                    (xMapnew_post, yMapnew_post, zMapnew_post) = Convert.coord_map1_2_map2(
                        X=list(new_post_geo[:, 1]),
                        Y=list(new_post_geo[:, 0]),
                        Z=list(new_post_geo[:, 2]),
                        sourceEPSG=4326,
                        targetEPSG=self.set_epsg)

                    # print(len(xMapnew_post), len(yMapnew_post), len(zMapnew_post))
                    #     #
                    ewCorrected___ = np.array(xMapnew_post) - np.array(xMapnew_pre)
                    nsCorrected___ = np.array(yMapnew_post) - np.array(yMapnew_pre)
                    dz___ = np.array(zMapnew_post) - np.array(zMapnew_pre)
                    normPre___ = diffPre_corrected_Norm2
                    normPost___ = diffPost_corrected_Norm2
                    # print("index_nan=", index_nan, len(index_nan))
                    # print(ewCorrected___.shape)
                    # print(len(len(index_nan) * [np.nan]))

                    tempIndex = 0

                    for i in range(nbXcorr):
                        if i in index_nan:

                            ewCorrected[i] = np.nan
                            nsCorrected[i] = np.nan
                            dz[i] = np.nan
                            normPre = np.nan
                            normPost = np.nan
                        else:
                            # print(i, tempIndex)
                            ewCorrected[i] = ewCorrected___[tempIndex]
                            nsCorrected[i] = nsCorrected___[tempIndex]
                            dz[i] = dz___[tempIndex]
                            normPre = normPre___[tempIndex]
                            normPost = normPost___[tempIndex]
                            tempIndex += 1
                else:
                    ewCorrected[:] = np.nan
                    nsCorrected[:] = np.nan
                    dz[:] = np.nan
                    normPre[:] = np.nan
                    normPost[:] = np.nan

                # nsCorrected = np.insert(nsCorrected, index_nan, len(index_nan) * [np.nan])
                # dz = np.insert(dz, index_nan, len(index_nan) * [np.nan])
                # normPre = np.insert(normPre, index_nan, len(index_nan) * [np.nan])
                # normPost = np.insert(normPost, index_nan, len(index_nan) * [np.nan])

            else:
                (xMap_M, yMap_M) = self.pre_corr_info.corr_raster_info.Pixel2Map(list(range(nbXcorr)), nbXcorr * [y])
                # print("xMap_M:{},yMap_M:{}".format(xMap_M,yMap_M))
                los_arg = [xMap_M, yMap_M, self.pre_corr_info.corr_raster_info, self.dem_info, self.tr1,
                           self.yMat_pre1, self.interpSatPos_pre1]
                sightVector_M, XYZ_cart_M = LOS.compute_los(*los_arg)

                xMap_S = xMap_M + self.pre_ew_corr[y, 0:nbXcorr]
                yMap_S = yMap_M + self.pre_ns_corr[y, 0:nbXcorr]
                los_arg = [xMap_S, yMap_S, self.pre_corr_info.corr_raster_info, self.dem_info, self.tr2,
                           self.yMat_pre2, self.interpSatPos_pre2]
                sightVector_S, XYZ_cart_S = LOS.compute_los(*los_arg)
                diffPre = np.asarray(XYZ_cart_S) - np.asarray(XYZ_cart_M)

                params, pt1_new, pt2_new, residu, XYZ_cart_PreCorrected = triangulate(XYZ_cart_M=XYZ_cart_M,
                                                                                      XYZ_cart_S=XYZ_cart_S,
                                                                                      sightVector_M=sightVector_M,
                                                                                      sightVector_S=sightVector_S)
                diffPre_corrected = np.asarray(pt2_new) - np.asarray(pt1_new)

                diffPre_corrected_Norm2 = np.linalg.norm(diffPre_corrected, axis=1, ord=2)
                # print(diffPre_corrected_Norm2)
                #
                #     normPre[x] = diffPre_corrected_Norm2
                #     #
                # print(XYZ_cart_PreCorrected,"\n",XYZ_cart_PreCorrected[:,1])
                new_pre_geo = Convert.cartesian_2_geo(XYZ_cart_PreCorrected[:, 0],
                                                      XYZ_cart_PreCorrected[:, 1],
                                                      XYZ_cart_PreCorrected[:, 2])
                new_pre_geo = np.array(new_pre_geo).T
                (xMapnew_pre, yMapnew_pre, zMapnew_pre) = Convert.coord_map1_2_map2(X=list(new_pre_geo[:, 1]),
                                                                                    Y=list(new_pre_geo[:, 0]),
                                                                                    Z=list(new_pre_geo[:, 2]),
                                                                                    sourceEPSG=4326,
                                                                                    targetEPSG=self.set_epsg)
                # print(xMapnew_pre, yMapnew_pre, zMapnew_pre)
                ############## Post images ###########################################################################

                xMap_post1 = xMap_M + self.event_1_corr_ew[y, 0:nbXcorr]
                yMap_post1 = yMap_M + self.event_1_corr_ns[y, 0:nbXcorr]
                los_arg = [xMap_post1, yMap_post1, self.event_1_corr_info.corr_raster_info, self.dem_info,
                           self.tr3, self.yMat_post1, self.interpSatPos_post1]
                sightVector_Post1, XYZ_cart_Post1 = LOS.compute_los(*los_arg)

                # print(sightVector_Post1)
                xMap_post2 = xMap_M + self.event_2_corr_ew[y, 0:nbXcorr]
                yMap_post2 = yMap_M + self.event_2_corr_ns[y, 0:nbXcorr]
                los_arg = [xMap_post2, yMap_post2, self.event_2_corr_info.corr_raster_info, self.dem_info,
                           self.tr4, self.yMat_post2, self.interpSatPos_post2]
                sightVector_Post2, XYZ_cart_Post2 = LOS.compute_los(*los_arg)

                # print(xMap_post1,yMap_post2)
                diffPost = np.asarray(XYZ_cart_Post2) - np.asarray(XYZ_cart_Post1)

                paramsPost, pt1_post_new, pt2_post_new, residu_post, XYZ_cart_PostCorrected = triangulate(
                    XYZ_cart_M=XYZ_cart_Post1,
                    XYZ_cart_S=XYZ_cart_Post2,
                    sightVector_M=sightVector_Post1,
                    sightVector_S=sightVector_Post2)

                diffPost_corrected = pt2_post_new - pt1_post_new
                # print(diffPost_corrected.shape)
                diffPost_corrected_Norm2 = np.linalg.norm(diffPost_corrected, axis=1, ord=2)
                # print(diffPost_corrected_Norm2)
                #     normPost[x] = diffPost_corrected_Norm2
                #     #
                new_post_geo = Convert.cartesian_2_geo(XYZ_cart_PostCorrected[:, 0],
                                                       XYZ_cart_PostCorrected[:, 1],
                                                       XYZ_cart_PostCorrected[:, 2])
                new_post_geo = np.array(new_post_geo).T
                (xMapnew_post, yMapnew_post, zMapnew_post) = Convert.coord_map1_2_map2(X=list(new_post_geo[:, 1]),
                                                                                       Y=list(new_post_geo[:, 0]),
                                                                                       Z=list(new_post_geo[:, 2]),
                                                                                       sourceEPSG=4326,
                                                                                       targetEPSG=self.set_epsg)

                #     # # print(xMapnew_post, yMapnew_post, zMapnew_post)

                ewCorrected = np.array(xMapnew_post) - np.array(xMapnew_pre)
                nsCorrected = np.array(yMapnew_post) - np.array(yMapnew_pre)
                dz = np.array(zMapnew_post) - np.array(zMapnew_pre)
                normPre = diffPre_corrected_Norm2
                normPost = diffPost_corrected_Norm2
            #     # print(ewCorrected[x],nsCorrected[x],dz[x])

            # print(ewCorrected.shape)
            ewCorrected_[y, :] = ewCorrected
            nsCorrected_[y, :] = nsCorrected
            dz_[y, :] = dz
            normPre_[y, :] = normPre
            normPost_[y, :] = normPost
            y += 1
        if self.inversionFlag == True or self.inversionFlag == "True":

            WriteRaster(oRasterPath=os.path.join(self.output),
                        geoTransform=self.pre_corr_info.corr_raster_info.geo_transform,
                        arrayList=[-1 * ewCorrected_, -1 * nsCorrected_, -1 * dz_, normPre_, normPost_],
                        epsg=self.set_epsg,
                        descriptions=['East/West', 'North/South', 'Dz', 'norm_pre', 'norm_post'])
        else:
            WriteRaster(oRasterPath=os.path.join(self.output),
                        geoTransform=self.pre_corr_info.corr_raster_info.geo_transform,
                        arrayList=[ewCorrected_, -nsCorrected_, dz_, normPre_, normPost_],
                        epsg=self.set_epsg,
                        descriptions=['East/West', 'North/South', 'Dz', 'norm_pre', 'norm_post'])
        return


def fun_compute3DD(preEventCorr,
                   eventCorr1,
                   eventCorr2,
                   demPath,
                   tr1,
                   tr2,
                   tr3,
                   tr4,
                   filename_ancillary_master_pre,
                   filename_ancillary_slave_pre,
                   filename_ancillary_post_1,
                   filename_ancillary_post_2,
                   output,
                   inversionFlags):
    threeDDObj = cCompute3DD(preEventCorr,
                             eventCorr1,
                             eventCorr2,
                             demPath,
                             tr1,
                             tr2,
                             tr3,
                             tr4,
                             filename_ancillary_master_pre,
                             filename_ancillary_slave_pre,
                             filename_ancillary_post_1,
                             filename_ancillary_post_2,
                             output,
                             inversionFlags)

    del threeDDObj
    return


# def PerformingThreeDD_new(refDEM,
#                           setFile,
#                           workspaceFolder,
#                           corrFolder,
#                           corrEngine,
#                           dataFile,
#                           eventDate="2019-07-04",
#                           tileSize=geoCfg.tileSize,
#                           recomputeRSM=False,
#                           numCpus=40,
#                           debug=False):
#     import pandas
#     from itertools import permutations
#     from tqdm import tqdm
#     import shutil
#     oThreeDDFolder = fileRT.CreateDirectory(workspaceFolder, "o3DDA", "y")
#     pairsData = pandas.read_csv(setFile)
#     data = pandas.read_csv(dataFile)
#     referenceDate = datetime.datetime.strptime(eventDate, '%Y-%m-%d')
#     for setIndex, row in pairsData.iterrows():
#         oSetFolder = fileRT.CreateDirectory(oThreeDDFolder, "3DDA_Set_" + str(setIndex + 1))
#         set = [row["pre_i"], row["pre_j"], row["post_i"], row["post_j"]]
#
#         totalComb = geoThreeDDMisc.ThreeDSetCombinations(preOrthoList=[row["pre_i"], row["pre_j"]],
#                                                          postOrthoList=[row["post_i"], row["post_j"]])
#         for combIndex, comb_ in enumerate(totalComb):
#             corrList = []
#             oCombFolder = fileRT.CreateDirectory(oSetFolder, "Set_Comb" + str(combIndex + 1))
#             if debug:
#                 print("Combination:", len(comb_))
#             _3D_set = {"Base": [], "Pre1": [], "Post1": [], "Post2": []}
#             baseImg = geoThreeDDMisc.ImgInfo()
#             baseImg.imgName = data.loc[data["OrthoPath"] == comb_[0], "Name"].values[0]
#             baseImg.orthoPath = comb_[0]
#             baseImg.imgFolder = data.loc[data["OrthoPath"] == comb_[0], "ImgPath"].values[0]
#             baseImg.rsmFile = data.loc[data["OrthoPath"] == comb_[0], "RSM"].values[0]
#             baseImg.date = data.loc[data["OrthoPath"] == comb_[0], "Date"].values[0]
#             baseImg.warpRaster = data.loc[data["OrthoPath"] == comb_[0], "TransPath"].values[0]
#             if debug:
#                 print(baseImg.__repr__())
#             preEvent = geoThreeDDMisc.ImgInfo()
#             preEvent.imgName = data.loc[data["OrthoPath"] == comb_[1], "Name"].values[0]
#             preEvent.orthoPath = comb_[1]
#             preEvent.imgFolder = data.loc[data["OrthoPath"] == comb_[1], "ImgPath"].values[0]
#             preEvent.rsmFile = data.loc[data["OrthoPath"] == comb_[1], "RSM"].values[0]
#             preEvent.date = data.loc[data["OrthoPath"] == comb_[1], "Date"].values[0]
#             preEvent.warpRaster = data.loc[data["OrthoPath"] == comb_[1], "TransPath"].values[0]
#             if debug:
#                 print(preEvent.__repr__())
#             postEvent1 = geoThreeDDMisc.ImgInfo()
#             postEvent1.imgName = data.loc[data["OrthoPath"] == comb_[2], "Name"].values[0]
#             postEvent1.orthoPath = comb_[2]
#             postEvent1.imgFolder = data.loc[data["OrthoPath"] == comb_[2], "ImgPath"].values[0]
#             postEvent1.rsmFile = data.loc[data["OrthoPath"] == comb_[2], "RSM"].values[0]
#             postEvent1.date = data.loc[data["OrthoPath"] == comb_[2], "Date"].values[0]
#             postEvent1.warpRaster = data.loc[data["OrthoPath"] == comb_[2], "TransPath"].values[0]
#             if debug:
#                 print(postEvent1.__repr__())
#             postEvent2 = geoThreeDDMisc.ImgInfo()
#             postEvent2.imgName = data.loc[data["OrthoPath"] == comb_[3], "Name"].values[0]
#             postEvent2.orthoPath = comb_[3]
#             postEvent2.imgFolder = data.loc[data["OrthoPath"] == comb_[3], "ImgPath"].values[0]
#             postEvent2.rsmFile = data.loc[data["OrthoPath"] == comb_[3], "RSM"].values[0]
#             postEvent2.date = data.loc[data["OrthoPath"] == comb_[3], "Date"].values[0]
#             postEvent2.warpRaster = data.loc[data["OrthoPath"] == comb_[3], "TransPath"].values[0]
#             if debug:
#                 print(postEvent2.__repr__())
#             inversionFlag = "False"
#
#             setNames = [baseImg.imgName, preEvent.imgName, postEvent1.imgName, postEvent2.imgName]
#             orthoList = [baseImg.orthoPath, preEvent.orthoPath, postEvent1.orthoPath, postEvent2.orthoPath]
#             if debug:
#                 print("Set Names:", setNames)
#                 print("Ortho path", orthoList)
#
#             _3D_set["Base"].append(baseImg.rsmFile)
#
#             with open(baseImg.rsmFile, "rb") as input:
#                 obj_ = pickle.load(input)
#             baseImg.date = obj_.date_time_obj
#             if debug:
#                 print("--- baseImg.rsmFile:{} ---".format(baseImg.rsmFile))
#             delta = referenceDate - baseImg.date
#
#             if delta.days < 0:
#                 inversionFlag = "True"
#             else:
#                 inversionFlag = "False"
#
#             rsmList = [baseImg.rsmFile, preEvent.rsmFile, postEvent1.rsmFile, postEvent2.rsmFile]
#             #
#             # print("====================================")
#             # print("             RSM files: ")
#             #
#             # for rsm_ in rsmList:
#             #     print(rsm_)
#             # print("====================================")
#
#             ### Save acquisistion paramters geometry
#             # svgOutput = os.path.join(oCombFolder, corrFolder.split("/")[-1] + "_acq_Geometry.svg")
#             # print(rsmList)
#             # geoThreeDDMisc.Set_Info_Plot(pklList=rsmList, limit=True, savingFolder=svgOutput)
#
#             warpList = []
#
#             warpList = [baseImg.warpRaster, preEvent.warpRaster, postEvent1.warpRaster, postEvent2.warpRaster]
#             if debug:
#                 print("====================================")
#                 print("             Warp Files: ")
#                 print(baseImg.warpRaster)
#                 print(preEvent.warpRaster)
#                 print(postEvent1.warpRaster)
#                 print(postEvent2.warpRaster)
#                 print("====================================")
#
#             threeDDA_folder = fileRT.CreateDirectory(oCombFolder, os.path.basename(oCombFolder) + "_3DDA",
#                                                      cal="y")
#             rsmFolder = fileRT.CreateDirectory(threeDDA_folder, "RSMFiles", cal="y")
#             transformFolder = fileRT.CreateDirectory(threeDDA_folder, "TransformRasters", cal="y")
#             cropCoorFolder = fileRT.CreateDirectory(threeDDA_folder, "Corr", cal="y")
#             rdemFolder = fileRT.CreateDirectory(threeDDA_folder, "rDEM", cal="y")
#
#             # return
#             for rsm_ in rsmList:
#                 fileRT.CopyFile(rsm_, rsmFolder, True)
#
#             geoRT.CropBatch(rasterList=warpList, outputFolder=transformFolder, vrt=True)
#
#             temp = Path(baseImg.orthoPath).stem + "_VS_" + Path(preEvent.orthoPath).stem + "_"
#             corr_base_pre = os.path.join(corrFolder, temp + "geoFqCorr_W" +
#                                          str(corrEngine["windowSizes"][0]) + "_S" +
#                                          str(corrEngine["stepss"][0]) + ".tif")
#             temp = Path(baseImg.orthoPath).stem + "_VS_" + Path(postEvent1.orthoPath).stem + "_"
#             corr_base_post1 = os.path.join(corrFolder, temp + "geoFqCorr_W" +
#                                            str(corrEngine["windowSizes"][0]) + "_S" +
#                                            str(corrEngine["stepss"][0]) + ".tif")
#             temp = Path(baseImg.orthoPath).stem + "_VS_" + Path(postEvent2.orthoPath).stem + "_"
#             corr_base_post2 = os.path.join(corrFolder, temp + "geoFqCorr_W" +
#                                            str(corrEngine["windowSizes"][0]) + "_S" +
#                                            str(corrEngine["steps"][0]) + ".tif")
#             corrList = [corr_base_pre, corr_base_post1, corr_base_post2]
#
#             mapCoord, mapCoordPrj, _ = geoRT.CropBatch(rasterList=corrList, outputFolder=cropCoorFolder, vrt=True)
#
#             demInfo = geoRT.RasterInfo(refDEM)
#             if demInfo.EPSG_Code != mapCoordPrj:
#                 import warnings
#                 msg = "Reproject DEM from {}-->{}".format(demInfo.EPSG_Code, mapCoordPrj)
#                 warnings.warn(msg)
#                 refDEM = geoRT.ReprojectRaster(iRasterPath=demInfo.rasterPath, oPrj=mapCoordPrj, vrt=True)
#
#             oList_ = geoRT.SubsetRasters(rasterList=[refDEM],
#                                          areaCoord=[min(mapCoord[0]), max(mapCoord[1]), max(mapCoord[0]),
#                                                     min(mapCoord[1])],
#                                          outputFolder=rdemFolder, vrt=True, outputType=gdal.GDT_UInt16)
#             refDEM_cropPath = oList_[0]
#             if debug:
#                 print("#######################################")
#                 print("                  Tiling              ")
#                 print("#######################################")
#             tileFolder = fileRT.CreateDirectory(threeDDA_folder, "Tiles", cal="y")
#             inputFolderList = [rdemFolder, cropCoorFolder, transformFolder]
#             imgPath_list_tiles = geoTile.Raster_tiling_batch(inputFolderList=inputFolderList,
#                                                              outputTileFolder=tileFolder,
#                                                              refImg=fileRT.GetFilesBasedOnExtensions(cropCoorFolder)[0],
#                                                              tileSize=tileSize)
#             warpMatricesList = [os.path.basename(item) for item in
#                                 fileRT.GetFilesBasedOnExtensions(transformFolder, ["*.tif", "*.vrt"])]
#
#             for warpMat_ in warpMatricesList:
#                 if baseImg.imgName in Path(warpMat_).stem:
#                     baseImg.warpRaster = os.path.join(transformFolder, warpMat_)
#                 if preEvent.imgName in Path(warpMat_).stem:
#                     preEvent.warpRaster = os.path.join(transformFolder, warpMat_)
#                 if postEvent1.imgName in Path(warpMat_).stem:
#                     postEvent1.warpRaster = os.path.join(transformFolder, warpMat_)
#                 if postEvent2.imgName in Path(warpMat_).stem:
#                     postEvent2.warpRaster = os.path.join(transformFolder, warpMat_)
#             warpMatricesSoted = [baseImg.warpRaster, preEvent.warpRaster, postEvent1.warpRaster, postEvent2.warpRaster]
#
#             rsmFileList = fileRT.GetFilesBasedOnExtensions(rsmFolder, ["*.pkl"])
#             for rsm_ in rsmFileList:
#                 if baseImg.imgName in Path(rsm_).stem:
#                     baseImg.rsmFile = rsm_
#                 if preEvent.imgName in Path(rsm_).stem:
#                     preEvent.rsmFile = rsm_
#                 if postEvent1.imgName in Path(rsm_).stem:
#                     postEvent1.rsmFile = rsm_
#                 if postEvent2.imgName in Path(rsm_).stem:
#                     postEvent2.rsmFile = rsm_
#             rsmFileListSorted = [baseImg.rsmFile, preEvent.rsmFile, postEvent1.rsmFile, postEvent2.rsmFile]
#
#             corrList = fileRT.GetFilesBasedOnExtensions(cropCoorFolder, ["*.tif", "*.vrt"])
#             for corrImg_ in corrList:
#                 if preEvent.imgName in Path(corrImg_).stem.split("_VS_")[1]:
#                     preCorr = corrImg_
#                 if postEvent1.imgName in Path(corrImg_).stem.split("_VS_")[1]:
#                     postCorr1 = corrImg_
#                 if postEvent2.imgName in Path(corrImg_).stem.split("_VS_")[1]:
#                     postCorr2 = corrImg_
#             # print(preCorr)
#             # print(postCorr1)
#             # print(postCorr2)
#             # ###########################################################################################################
#             tileFolder_ = os.path.join(tileFolder, "TempTiles")
#             demName = os.path.basename(refDEM_cropPath)
#
#             preCorrelation = os.path.basename(preCorr)
#             event1Correlation = os.path.basename(postCorr1)
#             event2Corrlation = os.path.basename(postCorr2)
#             warpMatrix1 = os.path.basename(baseImg.warpRaster)
#             warpMatrix2 = os.path.basename(preEvent.warpRaster)
#             warpMatrix3 = os.path.basename(postEvent1.warpRaster)
#             warpMatrix4 = os.path.basename(postEvent2.warpRaster)
#
#             preCorrTileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(preCorrelation).stem + "_Tiles"), ["*.tif", "*.vrt"])
#             event1CorrTileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(event1Correlation).stem + "_Tiles"), ["*.tif", "*.vrt"])
#             event2CorrTileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(event2Corrlation).stem + "_Tiles"), ["*.tif", "*.vrt"])
#
#             warpMat1TileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(warpMatrix1).stem + "_Tiles"), ["*.tif", "*.vrt"])
#             warpMat2TileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(warpMatrix2).stem + "_Tiles"), ["*.tif", "*.vrt"])
#             warpMat3TileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(warpMatrix3).stem + "_Tiles"), ["*.tif", "*.vrt"])
#             warpMat4TileList = fileRT.GetFilesBasedOnExtensions(
#                 os.path.join(tileFolder_, Path(warpMatrix4).stem + "_Tiles"), ["*.tif", "*.vrt"])
#
#             demTileList = fileRT.GetFilesBasedOnExtensions(os.path.join(tileFolder_, Path(demName).stem + "_Tiles"),
#                                                            ["*.tif", "*.vrt"])
#             lenList = [len(preCorrTileList), len(event1CorrTileList), len(event2CorrTileList), len(warpMat1TileList),
#                        len(warpMat2TileList), len(warpMat3TileList), len(warpMat4TileList), len(demTileList)]
#
#             if geoThreeDDMisc.all_equal(lenList):
#                 totalNb_Tiles = lenList[0]
#                 totalProccNb = multiprocessing.cpu_count()
#                 if debug:
#                     print("TotalProccNb=", totalProccNb)
#                 threeDTilesFolder = fileRT.CreateDirectory(threeDDA_folder, "3DDTiles", cal="y")
#
#                 preEventCorr = preCorrTileList
#                 eventCorr1 = event1CorrTileList
#                 eventCorr2 = event2CorrTileList
#                 demPath = demTileList
#                 tr1 = warpMat1TileList
#                 tr2 = warpMat2TileList
#                 tr3 = warpMat3TileList
#                 tr4 = warpMat4TileList
#                 filename_ancillary_master_pre = totalNb_Tiles * [baseImg.rsmFile]
#                 filename_ancillary_slave_pre = totalNb_Tiles * [preEvent.rsmFile]
#                 filename_ancillary_post_1 = totalNb_Tiles * [postEvent1.rsmFile]
#                 filename_ancillary_post_2 = totalNb_Tiles * [postEvent2.rsmFile]
#                 output = []
#                 for i in range(totalNb_Tiles):
#                     output.append(os.path.join(threeDTilesFolder, "3DDisp_" + str(i + 1) + ".tif"))
#                 inversionFlags = totalNb_Tiles * [inversionFlag]
#                 p_map(Fun_cCompute3DD, preEventCorr,
#                       eventCorr1,
#                       eventCorr2,
#                       demPath,
#                       tr1,
#                       tr2,
#                       tr3,
#                       tr4,
#                       filename_ancillary_master_pre,
#                       filename_ancillary_slave_pre,
#                       filename_ancillary_post_1,
#                       filename_ancillary_post_2,
#                       output,
#                       inversionFlags, num_cpus=numCpus)
#                 geoThreeDDMisc.MergeTiles(iFolder=threeDDA_folder, oFolder=os.path.dirname(threeDDA_folder))
#                 # nbTile_ = 0
#                 # for preEventCorr_, eventCorr1_, eventCorr2_, demPath_, tr1_, tr2_, tr3_, tr4_, filename_ancillary_master_pre_, \
#                 #     filename_ancillary_slave_pre_, filename_ancillary_post_1_, filename_ancillary_post_2_, output_, \
#                 #     inversionFlags_ in zip(preEventCorr,
#                 #                            eventCorr1,
#                 #                            eventCorr2,
#                 #                            demPath,
#                 #                            tr1,
#                 #                            tr2,
#                 #                            tr3,
#                 #                            tr4,
#                 #                            filename_ancillary_master_pre,
#                 #                            filename_ancillary_slave_pre,
#                 #                            filename_ancillary_post_1,
#                 #                            filename_ancillary_post_2,
#                 #                            output,
#                 #                            inversionFlags):
#                 #     print("--- Tile :{} \ {}".format(nbTile_, totalNb_Tiles))
#                 #     cCompute3DD(preEventCorr_, eventCorr1_, eventCorr2_, demPath_, tr1_, tr2_, tr3_, tr4_,
#                 #                 filename_ancillary_master_pre_, \
#                 #                 filename_ancillary_slave_pre_, filename_ancillary_post_1_, filename_ancillary_post_2_, output_, \
#                 #                 inversionFlags_)
#                 #     nbTile_ += 1
#                 # geoThreeDDMisc.MergeTiles(iFolder=threeDDA_folder, oFolder=os.path.dirname(threeDDA_folder))
#                 # sys.exit()
#             else:
#                 # TODO: add Error
#                 print("Raster don't have the same tile number")
#
#     return
