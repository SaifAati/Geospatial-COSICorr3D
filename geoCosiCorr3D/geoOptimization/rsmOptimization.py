"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import numpy as np

import geoCosiCorr3D.georoutines.geo_utils as geoRT


class cRSMOptimization:
    def __init__(self, rsmModel, gcps, debug=True):
        """

        Args:
            rsmModel:  instance of the physical model (only push-broom @SA)
            gcps: array of gcps [ lon,lat,alt, xPix,yPix, SNR, 1,0,0,0]
        """

        self.rsmModel = rsmModel
        self.gcps = gcps
        self.px = gcps.shape[1]
        self.py = gcps.shape[0]
        self.gcpPixCoords = np.empty((self.py, 2))
        self.gcpPixCoords[:, 0] = gcps[:, 3]
        self.gcpPixCoords[:, 1] = gcps[:, 4]
        self.dU = np.zeros((self.py, 3))
        self.cartCoordPlaneCoefs = np.zeros((3, 3))
        self.debug = debug

    def ComputeLookDirection(self, rsmModel, pixCoord):
        """
        Computes for one given pixel its look direction in geocentric coordinate (cartesian)
        Args:
            rsmModel: instance of the physical model (only push-broom @SA)
            pixCoord:  elements array containing the x, y pixel coordinates

        Returns: 3 elements array representing the pixel look direction in cartesian coordinate system (x, y, z)

        """

        xPix = pixCoord[0]
        yPix = pixCoord[1]

        rotMat = np.array([
            [rsmModel.satToNavMat[int(yPix), 0, 0], rsmModel.satToNavMat[int(yPix), 0, 1],
             rsmModel.satToNavMat[int(yPix), 0, 2]],
            [rsmModel.satToNavMat[int(yPix), 1, 0], rsmModel.satToNavMat[int(yPix), 1, 1],
             rsmModel.satToNavMat[int(yPix), 1, 2]],
            [rsmModel.satToNavMat[int(yPix), 2, 0], rsmModel.satToNavMat[int(yPix), 2, 1],
             rsmModel.satToNavMat[int(yPix), 2, 2]]
        ])

        postMat = np.array([
            [rsmModel.orbitalPos_X[int(yPix), 0], rsmModel.orbitalPos_Y[int(yPix), 0],
             rsmModel.orbitalPos_Z[int(yPix), 0]],
            [rsmModel.orbitalPos_X[int(yPix), 1], rsmModel.orbitalPos_Y[int(yPix), 1],
             rsmModel.orbitalPos_Z[int(yPix), 1]],
            [rsmModel.orbitalPos_X[int(yPix), 2], rsmModel.orbitalPos_Y[int(yPix), 2],
             rsmModel.orbitalPos_Z[int(yPix), 2]]
        ])

        angleX = rsmModel.CCDLookAngle[int(xPix), 0]
        angleY = rsmModel.CCDLookAngle[int(xPix), 1]
        angleZ = rsmModel.CCDLookAngle[int(xPix), 2]
        angles = np.array([angleX, angleY, angleZ])

        ## Compute the look direction
        u2 = np.dot(rotMat, angles)
        u2 = u2 / np.sqrt(np.sum(u2 ** 2))
        ## convert the look angle to terrestrial coordiante system
        u3 = np.dot(postMat, u2)
        return u3

    def ComputeLookDirectionofGCPs(self,
                                   rsmModel,
                                   gcpPixCoords,
                                   gcpCartCoords):
        """

        Args:
            rsmModel:
            gcpPixCoords:
            gcpCartCoords:

        Returns:

        """
        satPosX = rsmModel.interpSatPosition[:, 0][int(gcpPixCoords[1])]
        satPosY = rsmModel.interpSatPosition[:, 1][int(gcpPixCoords[1])]
        satPosZ = rsmModel.interpSatPosition[:, 2][int(gcpPixCoords[1])]
        satPosCoords = np.array([satPosX, satPosY, satPosZ])

        ## Compute the look direction of GPCs
        gcpCorrectedLookDirection = gcpCartCoords - satPosCoords
        gcpCorrectedLookDirection = gcpCorrectedLookDirection / \
                                    np.linalg.norm(gcpCorrectedLookDirection, ord=2)

        return gcpCorrectedLookDirection

    def FitCorrectionPlan(self, x, y, w, obs):
        """
        perform wighted least  square  ==> we could improve this subroutine by an iterative wighted LSQ
        :param x: x -coordinates
        :param y: y -coordinates
        :param w: weight vector, in our case it is the SNR
        :param obs:is the Z value at the corresponding coordinates ==> its the observation

        :return:
        Note: x,y,z should have the same dimensions
        """
        w_square = w ** 2
        alpha1 = np.sum(w_square * x ** 2)
        alpha2 = np.sum(w_square * y * x)
        alpha3 = np.sum(w_square * x)

        beta_2 = np.sum(w_square * y ** 2)
        beta_3 = np.sum(w_square * y)

        gamma_3 = np.sum(w_square)

        delta1 = np.sum(w_square * x * obs)
        delta2 = np.sum(w_square * y * obs)
        delta3 = np.sum(w_square * obs)

        mat = np.array([
            [alpha1, alpha2, alpha3],
            [alpha2, beta_2, beta_3],
            [alpha3, beta_3, gamma_3]])

        delta = np.array([delta1, delta2, delta3]).T
        P = np.dot(np.linalg.inv(mat), delta)

        return P

    def Plot_Error(self):
        import geoCosiCorr3D.georoutines.georoutines as geoRT
        import matplotlib.pyplot as plt
        from matplotlib.ticker import (AutoMinorLocator)
        dU_est = np.dot(self.cartCoordPlaneCoefs.T,
                        np.array([self.gcps[:, 3], self.gcps[:, 4], self.gcps.shape[0] * [1]])).T
        dU_res = self.dU - dU_est
        dU_norm = np.sqrt(self.dU[:, 0] ** 2 + self.dU[:, 1] ** 2 + self.dU[:, 2] ** 2)
        dU_est_norm = np.sqrt(dU_est[:, 0] ** 2 + dU_est[:, 1] ** 2 + dU_est[:, 2] ** 2)

        stat_dU = geoRT.cgeoStat(inputArray=dU_norm, displayValue=False)

        print("Pointing Error Ini :mean:{:.3f} ,median:{:.3f}, std:{:.3f}, RMSE:{:.3f}".format(float(stat_dU.mean),
                                                                                               float(stat_dU.median),
                                                                                               float(stat_dU.std),
                                                                                               float(stat_dU.RMSE)))

        stat_dU_est = geoRT.cgeoStat(inputArray=dU_est_norm, displayValue=False)

        print("Pointing Error opt_est :mean:{:.3f} ,median:{:.3f}, std:{:.3f}, RMSE:{:.3f}".format(
            float(stat_dU_est.mean),
            float(stat_dU_est.median),
            float(stat_dU_est.std),
            float(stat_dU_est.RMSE)))
        fig, ax = plt.subplots()

        # ax.scatter(np.arange(0, sample.shape[0], 1), sample, c="k")
        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU[:, 0], c="k", linestyle="--", marker="o", label="dUx")
        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU[:, 1], c="g", linestyle="--", marker="o", label="dUy")
        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU[:, 2], c="r", linestyle="--", marker="o", label="dUz")
        ax.plot(np.arange(0, self.gcps.shape[0], 1), sorted(dU_norm), c="r", linestyle="--", marker="o", label="dU")

        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU_est[:, 0], c="k", linestyle="-", marker="o", label="dUx_opt")
        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU_est[:, 1], c="g", linestyle="-", marker="o", label="dUy_opt")
        # ax.plot(np.arange(0, self.gcps.shape[0], 1), dU_est[:, 2], c="r", linestyle="-", marker="o", label="dUz_opt")
        ax.plot(np.arange(0, self.gcps.shape[0], 1), sorted(dU_est_norm), c="g", linestyle="-", marker="o",
                label="dU_opt")
        # ax.axhline(y=float(stat.mean), color='r', linewidth=4, linestyle='--')
        ax.grid()

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(which='both', width=2, direction="in")
        ax.set_xlabel('#GCPs')
        # ax.set_ylabel(yLabel)
        plt.show()
        plt.close(fig)
        return

    def __call__(self):
        if self.debug:
            print("Performing RSM refinement ...")
        # gcpCartesianCoordinates = geoRT.ConvertGeo2Cartesian(lon=gcps[0], lat=gcps[1], alt=gcps[2])
        resTemp = []

        self.dU = np.zeros((self.py, 3))
        index = 0
        for pixCoord, gcpItem in zip(list(self.gcpPixCoords), list(self.gcps)):
            # print(gcpItem)
            gcpInitLookDirection = self.ComputeLookDirection(self.rsmModel, pixCoord)
            gcpCartCoords = geoRT.Convert.geo_2_cartesian(Lon=[gcpItem[0]], Lat=[gcpItem[1]], Alt=[gcpItem[2]])

            gcpLookDirection = self.ComputeLookDirectionofGCPs(rsmModel=self.rsmModel,
                                                               gcpPixCoords=pixCoord,
                                                               gcpCartCoords=gcpCartCoords)

            self.dU[index, :] = (gcpLookDirection - gcpInitLookDirection[:])
            index += 1

        for i in range(3):
            corrParams = self.FitCorrectionPlan(x=self.gcps[:, 3],
                                                y=self.gcps[:, 4],
                                                w=self.gcps[:, 5],
                                                obs=self.dU[:, i])
            self.cartCoordPlaneCoefs[:, i] = corrParams
        # self.Plot_Error()
        return self.cartCoordPlaneCoefs
