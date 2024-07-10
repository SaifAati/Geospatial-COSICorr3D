"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import numpy as np
from geoCosiCorr3D.geoRSM.Ground2Pixel import RSMG2P


class cGCPPatchMapGrid:
    def __init__(self,
                 gridEPSG,
                 gridRes,
                 upleftNS,
                 upleftEW,
                 botrightNS,
                 botrightEW):
        self.gridEPSG = gridEPSG
        self.oRes = gridRes
        self.upleftEW = upleftEW
        self.botrightEW = botrightEW
        self.upleftNS = upleftNS
        self.botrightNS = botrightNS

        self.oUpleftEW = upleftEW
        self.oBotrightEW = botrightEW
        self.oUpleftNS = upleftNS
        self.oBotrightNS = botrightNS

    def __repr__(self):
        return """# Grid Information :
            projEPSG = {}
            res = {}
            upleftNS= {}
            upleftEW = {}
            botrightNS= {}
            botrightEW = {}""".format(self.gridEPSG,
                                      self.oRes,
                                      self.upleftNS,
                                      self.upleftEW,
                                      self.botrightNS,
                                      self.botrightEW)


class cInvereOrthoGCPPatch:
    def GenerateMatriceSatOrtho(self,
                                rsmModel,
                                oGrid,
                                rsmCorrectionArray=np.zeros((3, 3)),
                                demInfo=None,
                                nbColsPix_oMat=None,
                                nbRowsPix_oMat=None):

        """
        Core process of computing the orthorectification matrices of satellite images.

        Args:
            rsmModel: pickl of the RSM model
            oGrid: patch grid of the GCP :object of cGCPPatchMapGrid()
            demInfo: DEM information: object of geoRT.RasterInfo()
            rsmCorrectionArray: correction array : array[3x3]
            nbColsPix_oMat: the number of cols pixel (X-direction) of the oMatrice.
                            If not defined the nb of pixels is determined from the grid structure.
            nbRowsPix_oMat: the number of rows pixel (Y-direction) of the oMatrice.
                            If not defined the nb of pixels is determined from the grid structure.

        Returns:
        Notes:

        """
        ## From the oGrid, define the matrix size
        self.oGrid = oGrid

        oColsNb = round((self.oGrid.oBotrightEW - self.oGrid.oUpleftEW) / self.oGrid.oRes + 1)
        oRowsNb = round((self.oGrid.oUpleftNS - self.oGrid.oBotrightNS) / self.oGrid.oRes + 1)
        if nbColsPix_oMat is not  None:
            oColsNb = nbColsPix_oMat
        if nbRowsPix_oMat is not None:
            oRowsNb = nbRowsPix_oMat
        # print("oColsNb:{} , oRowsNb:{}".format(oColsNb, oRowsNb))

        easting = oGrid.oUpleftEW + np.arange(oColsNb) * oGrid.oRes
        northing = oGrid.oUpleftNS - np.arange(oRowsNb) * oGrid.oRes

        ground2Pix_obj = RSMG2P(rsmModel=rsmModel,
                                     xMap=easting,
                                     yMap=northing,
                                     projEPSG=self.oGrid.gridEPSG,
                                     rsmCorrection=rsmCorrectionArray,
                                     demInfo=demInfo)
        oArray = ground2Pix_obj.oArray

        del ground2Pix_obj

        return oArray, None, None, None
