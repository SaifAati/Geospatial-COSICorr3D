"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import datetime
import logging
import os

import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d, splev, splrep

import geoCosiCorr3D.geoRSM.misc as misc
from geoCosiCorr3D.geoCore.constants import SOFTWARE
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoRSM.geoRSM_metadata.ReadSatMetadata import (
    cGetSpot15Metadata, cGetSpot67Metadata)

geoCosiCorr3DOrientation = SOFTWARE.geoCosiCorr3DOrientation


class cSpot67(RSM):
    """
    This class covers Spot-6-7:
        -1- Read Spot metadata
        -2- Build the RSM
    """

    def __init__(self, dmpXml: str, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.dmpFile = dmpXml
        self.file = self.dmpFile
        self.spotMetadata = cGetSpot67Metadata(self.dmpFile)
        self.date = self.spotMetadata.imagingDate
        self.time = self.spotMetadata.time
        self.platform = self.spotMetadata.instrument + " " + self.spotMetadata.instrumentIndex

        self.nbRows = self.spotMetadata.nbRows
        self.nbCols = self.spotMetadata.nbCols

        self.sunAz = self.spotMetadata.sunAz
        self.sunElev = self.spotMetadata.sunAz
        self.satElev = self.spotMetadata.satElev
        self.satAz = self.spotMetadata.satAz
        self.offNadir_angle = self.spotMetadata.offNadir_angle
        self.satAlt = self.spotMetadata.satAlt
        self.azAngle = self.spotMetadata.azAngle
        self.viewAngle = self.spotMetadata.viewAngle
        self.incidenceAngle = self.spotMetadata.incidenceAngle

        self.avgLineRate = None
        self.scanDirection = None

        self.nbBands = self.spotMetadata.nbBands
        self.gsd_ACT = self.spotMetadata.gsd_ACT
        self.gsd_ALT = self.spotMetadata.gsd_ALT
        self.meanGSD = self.spotMetadata.meanGSD
        self.gsd = self.meanGSD

        self.time = self.spotMetadata.time
        self.startTime = self.spotMetadata.startTime
        self.date_time_obj = self.startTime

        self.focal = self.spotMetadata.focal
        self.szCol = self.spotMetadata.szCol
        self.szRow = self.spotMetadata.szRow

        self.lineOfSight = self.spotMetadata.lineOfSight
        self.eph = self.spotMetadata.eph_tpv
        self.quat_txyzs = self.spotMetadata.quat_txyzs

        self.linePeriod = self.spotMetadata.linePeriod
        if self.debug:
            logging.info("--- Computing " + self.platform + "  RSM:")

        # Date each line of the raw image :right now O-time is at first line
        self.linesDate = np.arange(self.nbRows) * self.linePeriod

        self.interpSatPosition = np.empty((self.nbRows, 3))
        self.interpSatVelocity = np.empty((self.nbRows, 3))

        self.interp_eph()

        self.compute_orbital_reference_system()

        self.compute_look_direction()

        self.interp_attitude()

        self.sat_to_orb_rotation()

        if self.debug:
            logging.info(" Done!\n")

    @staticmethod
    def Quat_2_rot_airbus(quat):
        """
        Convert rotation quaternions into a rotation matrix
        Args:
            quat: A 4 elements array [q1,q2,q3,q4] with q4 the scalar part

        Returns:

        """
        quat_ = quat / np.linalg.norm(quat, ord=2)
        x = quat_[0]
        y = quat_[1]
        z = quat_[2]
        w = quat_[3]  # scalar part

        res = [[(w ** 2 + x ** 2 - y ** 2 - z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
               [2 * (x * y + w * z), (w ** 2 - x ** 2 + y ** 2 - z ** 2), 2 * (y * z - w * x)],
               [2 * (x * z - w * y), 2 * (y * z + w * x), (w ** 2 - x ** 2 - y ** 2 + z ** 2)]]

        return res

    @staticmethod
    def quat_to_rot(quat):
        """
        Convert rotation quaternions into a rotation matrix
        Args:
             quat: An Nx4 array of quaternions with N quaternions and each quaternion consisting of [q1,q2,q3,q4] with q4 as the scalar part.

        Returns:
            An Nx3x3 array of rotation matrices.

        """
        quaternions_normalized = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        rotation = Rotation.from_quat(quaternions_normalized)
        rotation_matrices = rotation.as_matrix()

        return rotation_matrices

    def interp_eph(self):

        def interpolate_data(data):
            return [interp1d(eph_time, data[:, i],
                             kind='quadratic',
                             bounds_error=False,
                             fill_value="extrapolate")(self.linesDate) for i in range(3)]

        eph_time = self.eph[:, 0] - self.startTime
        num_elements_exceeding = np.sum(self.linesDate > eph_time[-1])

        if num_elements_exceeding > 0:
            logging.warning(
                f"Performing EPH extrapolation for {num_elements_exceeding} line exceeding the maximum eph time")

        self.interpSatPosition = np.array(interpolate_data(self.eph[:, 1:4])).T
        self.interpSatVelocity = np.array(interpolate_data(self.eph[:, 4:7])).T

        self.interp_sat_eph_tpv = np.hstack(
            (self.linesDate[:, np.newaxis], self.interpSatPosition, self.interpSatVelocity))
        logging.info(f"{self.__class__.__name__}: interp_sat_eph: {self.interp_sat_eph_tpv.shape}")

        return

    def compute_orbital_reference_system(self):
        """
                Orbital coordinate system definition for each line of the image are defined as follow:
                        Z = satPos/norm(satPos)
                        X = velSat ^ Z /norm( velSat ^ Z)
                        Y = Z ^ X
                """

        self.orbitalPos_Z = misc.normalize_array(self.interpSatPosition)
        self.orbitalPos_X = misc.normalize_array(np.cross(self.interpSatVelocity, self.orbitalPos_Z))
        self.orbitalPos_Y = np.cross(self.orbitalPos_Z, self.orbitalPos_X)

    def compute_look_direction(self):
        """
        Notes:
            Pleiades/Spot orientation is:
                +X along sat track movement
                +Y left of +X
                +Z down, towards Earth center
                -->
                VX = tanPsiY      = line_of_sight[2]
                VY = - tanPsiX    = -(line_of_sight[0] +line_of_sight[1] * col)
                VZ = 1            = 1
            geoCosiCorr3D orientation is:
                +X right or +Y
                +Y along sat track movement
                +Z up, away from Earth center
                --->
                VX = -(line_of_sight[0] +line_of_sight[1] * col)
                VYcosi = line_of_sight[2]
                VZcosi = -1
        """

        self.CCDLookAngle = np.empty((self.nbCols, 3))
        self.CCDLookAngle[:, 0] = -(self.lineOfSight[0] + np.arange(self.nbCols) * self.lineOfSight[1])
        self.CCDLookAngle[:, 1] = (self.lineOfSight[2] + np.arange(self.nbCols) * self.lineOfSight[3])
        self.CCDLookAngle[:, 2] = -1
        self.CCDLookAngle = misc.normalize_array(self.CCDLookAngle)
        logging.info(f"{self.__class__.__name__}: look_angle: {self.CCDLookAngle.shape}")
        return

    def sat_to_orb_rotation(self):
        """
        Build the Matrix that change the reference system from satellite reference system to orbital reference system
        taking into account the satellite attitude: roll, pitch, yaw.

        Notes:
            Computing satellite to orbital systems rotation matrices for each line of the image.
            As follow:
                        R= [[orbitalPos_X[i,0],orbitalPos_X[i,1],orbitalPos_X[i,2]]
                           [orbitalPos_Y[i,0],orbitalPos_Y[i,1],orbitalPos_Y[i,2]]
                           [orbitalPos_Z[i,0],orbitalPos_Z[i,1],orbitalPos_Z[i,2]]]

            Pleiades/Spot orientation is:
                +X along sat track movement
                +Y left of +X
                +Z down, towards Earth center

            geoCosiCorr3D orientation is:
                +X right or +Y
                +Y along sat track movement
                +Z up, away from Earth center

            Therefore, we add a transformation matrix referred to as:
            airbus_2_cosi_rot = np.array([[0, 1, 0], [1., 0, 0], [0, 0, -1]])
            earthRotation_sat[i, :, :] = quat_i @ airbus_2_cosi_rot
            ==> satToNavMat[i, :, :] = R @ earthRotation_sat[i, :, :]
        """
        quat_xyzs = self.interp_quat_txyzs[:, 1:]

        earthRotation_sat = np.empty((self.nbRows, 3, 3))
        airbus_2_cosi_rot = np.copy(geoCosiCorr3DOrientation)
        for i in range(self.nbRows):
            quat_i = np.array(
                [self.interp_quat_txyzs[:, 1][i],
                 self.interp_quat_txyzs[:, 2][i],
                 self.interp_quat_txyzs[:, 3][i],
                 self.interp_quat_txyzs[:, 4][i]])

            earthRotation_sat[i, :, :] = self.Quat_2_rot_airbus(quat=quat_i) @ airbus_2_cosi_rot

        # airbus_2_cosi_rot = np.copy(geoCosiCorr3DOrientation)
        # earthRotation_sat = np.einsum('ij,kjl->kil', airbus_2_cosi_rot, self.quat_to_rot(quat_xyzs))

        self.satToNavMat = np.empty((self.nbRows, 3, 3))
        for i in range(self.nbRows):
            R = [[self.orbitalPos_X[i, 0], self.orbitalPos_X[i, 1], self.orbitalPos_X[i, 2]],
                 [self.orbitalPos_Y[i, 0], self.orbitalPos_Y[i, 1], self.orbitalPos_Y[i, 2]],
                 [self.orbitalPos_Z[i, 0], self.orbitalPos_Z[i, 1], self.orbitalPos_Z[i, 2]]]
            self.satToNavMat[i, :, :] = np.dot(R, earthRotation_sat[i, :, :])

        # R = np.stack((self.orbitalPos_X, self.orbitalPos_Y, self.orbitalPos_Z), axis=-1)
        # self.satToNavMat = np.einsum('ijk,ikl->ijl', R, earthRotation_sat)

    def interp_attitude(self, slerp_method=False):
        """
        Build the Matrix that change the reference system from satellite reference system to orbital reference system
        taking into account the satellite attitude: roll, pitch, yaw.

        Notes:
            Computing satellite to orbital systems rotation matrices for each line of the image.
            As follow:
                        R= [[orbitalPos_X[i,0],orbitalPos_X[i,1],orbitalPos_X[i,2]]
                           [orbitalPos_Y[i,0],orbitalPos_Y[i,1],orbitalPos_Y[i,2]]
                           [orbitalPos_Z[i,0],orbitalPos_Z[i,1],orbitalPos_Z[i,2]]]

            Pleiades/Spot orientation is:
                +X along sat track movement
                +Y left of +X
                +Z down, towards Earth center

            geoCosiCorr3D orientation is:
                +X right or +Y
                +Y along sat track movement
                +Z up, away from Earth center

            Therefore, we add a transformation matrix referred to as:
            airbus_2_cosi_rot = np.array([[0, 1, 0], [1., 0, 0], [0, 0, -1]])
            earthRotation_sat[i, :, :] = quat_i @ airbus_2_cosi_rot
            ==> satToNavMat[i, :, :] = R @ earthRotation_sat[i, :, :]
        """
        interp_times = self.linesDate
        quat_time = self.quat_txyzs[:, 0] - self.startTime
        num_elements_exceeding = np.sum(interp_times > quat_time[-1])

        if num_elements_exceeding > 0:
            logging.warning(
                f"Performing ATT extrapolation for {num_elements_exceeding} line exceeding the maximum quat time")

        interp_quat_components = np.zeros((len(interp_times), 4))
        if not slerp_method:

            for i in range(1, 5):
                f = interp1d(quat_time, self.quat_txyzs[:, i],
                             kind='quadratic', bounds_error=False, fill_value="extrapolate")
                interp_quat_components[:, i - 1] = f(interp_times)
        else:
            quaternions = self.quat_txyzs[:, 1:]
            rotations = Rotation.from_quat(quaternions)
            slerp = Slerp(quat_time, rotations)

            for i, time in enumerate(interp_times):
                if time <= quat_time[-1]:
                    interp_quat_components[i] = slerp(time).as_quat()
                else:
                    # Extrapolation with the last valid quaternion
                    interp_quat_components[i] = rotations.as_quat()[-1]

        self.interp_quat_txyzs = np.hstack((interp_times[:, np.newaxis], interp_quat_components))
        logging.info(f"{self.__class__.__name__}: interp_quat: {self.interp_quat_txyzs.shape}")

        return


class cSpot15(RSM):
    """
    This class covers Spot-1-2-3-4 and -5:
        -1- Read Spot metadata
        -2- Build the RSM
            -Correct for  CCD misalignment of SPOT2_HRV1 and SPOT4_HRV1
    """

    def __init__(self, dmpFile: str, debug: bool = True):
        super().__init__()
        self.debug = debug
        self.dmpFile = dmpFile
        self.spotMetadata = cGetSpot15Metadata(dmpFile=self.dmpFile, debug=self.debug)

        acqTime = self.spotMetadata.date + "T" + self.spotMetadata.time
        self.date_time_obj = datetime.datetime.strptime(acqTime, '%Y-%m-%dT%H:%M:%S')
        self.date = self.spotMetadata.date
        self.data = self.spotMetadata.date
        self.time = self.spotMetadata.time

        ##Spot-MISSION-INSTRUMENTNAME-INSTRUMENTINDEX-MODE
        self.platform = "Spot-" + str(self.spotMetadata.mission) + "-" + self.spotMetadata.instrumentName + "-" + str(
            self.spotMetadata.instrument) + "-" + str(self.spotMetadata.sensorCode)

        # self.productCatalogId = None
        # self.imagLevel = None
        # self.bandId = None
        self.nbRows = self.spotMetadata.nbRows
        self.nbCols = self.spotMetadata.nbCols

        ## Compute Nominal ground resolutionof SPOT1-5 imagery
        self.__GetSpotGSD()
        if self.debug:
            logging.info("GSD:{}".format(self.gsd))
        ##Compute the time for line of the SPOT raw image
        ## Ti = Tcenter - linePeriod(Lcenter-Li +1) <--> Ti = Tcenter + linePeriod(Li-Lcenter +1)
        self.linesDates = self.spotMetadata.sceneCenterTime + (
                np.arange(self.nbRows) + 1 - self.spotMetadata.sceneCenterLine) * self.spotMetadata.linePeriod
        """
        For SPOT1-4 spacecraft velocities are expressed in the Earth Centered Reference System. 
        So no need to correct for earth rotation. satVelocity = ecfSatVelocity  
        """

        ## Before ephemeris interpolation, ephemeris value that are inside the image acquisition are discarded so
        ## that it doesn't weight too  much in the Lagrange interpolation.

        time_, position_, velocity_ = self.discard_eph(inf=self.linesDates[0],
                                                       sup=self.linesDates[-1],
                                                       time=self.spotMetadata.ephTime,
                                                       position=self.spotMetadata.satPosition,
                                                       velocity=self.spotMetadata.satVelocity)
        self.ephTime = time_
        self.satPosition = position_
        self.satVelocity = velocity_

        self.interp_eph()

        self.compute_orbital_reference_system()

        self.compute_look_direction()

        self.interp_attitude()

        self.sat_to_orb_rotation()

        self.sunAz = self.spotMetadata.sunAz
        self.sunElev = self.spotMetadata.sunElev

        self.satElev = 0
        self.satAz = 0
        self.offNadir_angle = 0

        self.satAlt = 0
        self.azAngle = 0
        self.viewAngle = 0
        self.incidenceAngle = self.spotMetadata.incidenceAngle
        # self.focal = None
        # self.szCol = None
        # self.szRow = None

    def __GetSpotGSD(self):

        if self.nbCols == 3000:
            self.gsd = 20
        elif self.nbCols == 6000:
            self.gsd = 10
        elif self.nbCols == 12000:
            self.gsd = 5
        elif self.nbCols == 24000:
            self.gsd = 2.5
        else:
            raise ValueError("Only whole SPOT1-5 scene can be processed (no subset)")

    @staticmethod
    def discard_eph(inf, sup, time, position, velocity):
        """
        AUTHORS:
            Sylvain Barbot (sbarbot@gps.caltech.edu)
            Sebastien Leprince (leprincs@caltech.edu)
            Updated by: Saif Aati saif@caltech.edu
        PURPOSE:
            Discard ephemeris information that are inside image acquisition
            so that it doesn't weight too much in the next Lagrange interpolation
        Args:
            inf: the times squaring the image acquisition
            sup: the times squaring the image acquisition
            time: the time array of the ephemeris values
            position: position of the satellite for each time of time array
            velocity: satellite velocity at each time value

        Returns:

        """

        indexInf = np.where(time < inf)[0]
        indexSup = np.where(time > sup)[0]
        # logging.info(indexInf,indexSup)

        if len(indexInf) < 4 or len(indexSup) < 4:
            raise ValueError("Ephemeris is missing")
        # Selecting the last 4 elements of z1
        indexInf = indexInf[len(indexInf) - 4:len(indexInf)]
        # ;selecting the first 4 elements of z2
        indexSup = indexSup[0: 4]
        time_ = []

        if np.sum(list(indexInf) + list(indexSup)) > -1:
            for index_ in list(indexInf) + list(indexSup):
                time_.append(time[index_])
            position_ = position[list(indexInf) + list(indexSup)]
            velocity_ = velocity[list(indexInf) + list(indexSup)]
            return time_, position_, velocity_
        else:
            raise ValueError("Ephemeris is missing")

    @staticmethod
    def LagrangeInterpolation(linesDates, satPos, ephTime):
        sz = len(linesDates)
        oArray = []
        for scanLine_ in range(sz):
            interpValue = 0
            for j in range(len(ephTime)):
                iterVal = satPos[j]
                for i in range(len(ephTime)):
                    if i != j:
                        iterVal = iterVal * (linesDates[scanLine_] - ephTime[i]) / (ephTime[j] - ephTime[i])
                interpValue += iterVal
            oArray.append(interpValue)
        return oArray

    @staticmethod
    def __CCDCorrection_Spot2_HRV1():
        Spot2_HRV1_correctionFile = os.path.join(os.path.dirname(__file__), "./SPOT2_HRV1_CCD_Correction")
        spot2HRV1CorrectionArray = np.loadtxt(Spot2_HRV1_correctionFile)
        return spot2HRV1CorrectionArray

    @staticmethod
    def __CCDCorrection_Spot4_HRV1():
        Spot4_HRV1_correctionFile = os.path.join(os.path.dirname(__file__), "./SPOT4_HRV1_CCD_Correction")
        spot4HRV1CorrectionArray = np.loadtxt(Spot4_HRV1_correctionFile)
        return spot4HRV1CorrectionArray

    def interp_eph(self):
        if self.debug:
            logging.info("Sat position and velocity interpolation ...")
        self.interpSatPosition = np.empty((self.nbRows, 3))
        self.interpSatVelocity = np.empty((self.nbRows, 3))

        for i in range(3):
            self.interpSatPosition[:, i] = self.LagrangeInterpolation(self.linesDates,
                                                                      self.satPosition[:, i],
                                                                      self.ephTime)
            self.interpSatVelocity[:, i] = self.LagrangeInterpolation(self.linesDates,
                                                                      self.satVelocity[:, i],
                                                                      self.ephTime)

    def compute_orbital_reference_system(self):
        ###Compute for each scan line the coordinates in Orbital coordinate system
        self.orbitalPos_Z = misc.NormalizeArray(self.interpSatPosition)
        self.orbitalPos_X = misc.NormalizeArray(inputArray=np.cross(self.interpSatVelocity, self.orbitalPos_Z))
        self.orbitalPos_Y = np.cross(self.orbitalPos_Z, self.orbitalPos_X)
        return

    def compute_look_direction(self, ):
        """
        Compute the satellite look angle from the on board measures
        Args:
            lookAngles:

        Returns:

        """
        if self.debug:
            logging.info("Computing sat look angles ...")
        lookAngles = self.spotMetadata.lookAngles
        CCDLookAngle = np.empty((lookAngles.shape[0], 3))
        CCDLookAngle[:, 0] = -1 * np.tan(lookAngles[:, 1])  # PsyX
        CCDLookAngle[:, 1] = np.tan(lookAngles[:, 0])  # PsyY
        CCDLookAngle[:, 2] = -1  # the Z direction
        CCDLookAngle = misc.NormalizeArray(inputArray=CCDLookAngle)

        if self.spotMetadata.mission == 5:
            self.CCDLookAngle = CCDLookAngle
        else:
            ## Perform a linear interpolation of all CCD detectors look angles
            self.CCDLookAngle = np.empty((self.nbCols, 3))
            tempArr = np.arange(self.nbCols) / (self.nbCols - 1)
            self.CCDLookAngle[:, 0] = CCDLookAngle[0, 0] + tempArr * (CCDLookAngle[1, 0] - CCDLookAngle[0, 0])
            self.CCDLookAngle[:, 1] = CCDLookAngle[0, 1] + tempArr * (CCDLookAngle[1, 1] - CCDLookAngle[0, 1])
            self.CCDLookAngle[:, 2] = CCDLookAngle[0, 2] + tempArr * (CCDLookAngle[1, 2] - CCDLookAngle[0, 2])
            correction = self.__SpotCCDCorrection()
            if correction.all() != 0:
                mirrorRotation = (self.spotMetadata.mirrorStep - 48) * 0.6 * np.pi / 180
                mcvMatrix = np.array([[np.cos(mirrorRotation), 0, np.sin(mirrorRotation)],
                                      [0, 1, 0],
                                      [-1 * np.sin(mirrorRotation), 0, np.cos(mirrorRotation)]]).T
                for ccd in range(self.nbCols):
                    correction[ccd, :] = np.dot(mcvMatrix, correction[ccd, :].T).T
                self.CCDLookAngle = self.CCDLookAngle + correction
        return

    def __SpotCCDCorrection(self):
        msg = "Unavailable CCD correction for SPOT image (mission:{},instrument:{}). No CCD correction applied.".format(
            self.spotMetadata.mission, self.spotMetadata.instrument)
        if self.spotMetadata.mission == 1 or self.spotMetadata.mission == 3:
            if self.debug:
                logging.info(msg)
            return np.zeros((1, 1))
        elif self.spotMetadata.mission == 2:
            if self.spotMetadata.instrument == 1:
                if self.debug:
                    logging.info(" CCD correction for SPOT2_HRV1 (mission:{},instrument:{}).".format(
                        self.spotMetadata.mission, self.spotMetadata.instrument))
                correction = self.__CCDCorrection_Spot2_HRV1()
                return correction
            else:
                if self.debug:
                    logging.info(msg)
                return np.zeros((1, 1))
        elif self.spotMetadata.mission == 4:
            if self.spotMetadata.instrument == 1:
                if self.debug:
                    logging.info(" CCD correction for SPOT4_HRV1 (mission:{},instrument:{}).".format(
                        self.spotMetadata.mission, self.spotMetadata.instrument))
                correction = self.__CCDCorrection_Spot4_HRV1()
                return correction
            else:
                if self.debug:
                    logging.info(msg)
                return np.zeros((1, 1))
        else:
            if self.debug:
                logging.info(msg)
            return np.zeros((1, 1))

    def interp_attitude(self):
        if self.debug:
            logging.info("Computing Sat Attitude ...")

        if self.spotMetadata.mission != 5:
            ## We need to locate invalid constant attitude and replace them with null-attitude values.
            ## Note: for Spot1-4 the are only 2 measurements, at the beginning and the end of the image acquisition.
            invalidIndex_attAng = []
            validIndex_attAng = []
            for index_, val_ in enumerate(self.spotMetadata.outOfRangeAttAng):
                if val_ == "Y":
                    invalidIndex_attAng.append(index_)
                else:
                    validIndex_attAng.append(index_)

            if len(invalidIndex_attAng) != 0:
                for index_ in invalidIndex_attAng:
                    self.spotMetadata.yawAttAng[index_] = 0
                    self.spotMetadata.pitchAttAng[index_] = 0
                    self.spotMetadata.rollAttAng[index_] = 0

            ## Do the same thing for invalid rotation speed. If all are invalid replace them by null-values.
            ## Otherwise replace them by linearly interpolated values from valid rotation speeds.
            invalidIndex_speedAtt = []
            validIndex_speedAtt = []
            for index_, val_ in enumerate(self.spotMetadata.outOfRangeSpeedAtt):
                if val_ == "Y":
                    invalidIndex_speedAtt.append(index_)
                else:
                    validIndex_speedAtt.append(index_)
            if len(invalidIndex_speedAtt) >= len(self.spotMetadata.outOfRangeSpeedAtt):
                for index_ in invalidIndex_speedAtt:
                    self.spotMetadata.yawSpeedAtt[index_] = 0
                    self.spotMetadata.pitchSpeedAtt[index_] = 0
                    self.spotMetadata.rollSpeedAtt[index_] = 0
            else:
                yawSpeedAttValid = []
                pitchSpeedAtValid = []
                rollSpeedAttValid = []
                timeSpeedAttValid = []

                for index_ in validIndex_speedAtt:
                    yawSpeedAttValid.append(self.spotMetadata.yawSpeedAtt[index_])
                    pitchSpeedAtValid.append(self.spotMetadata.pitchSpeedAtt[index_])
                    rollSpeedAttValid.append(self.spotMetadata.rollSpeedAtt[index_])
                    timeSpeedAttValid.append(self.spotMetadata.timeSpeedAtt[index_])

                fYaw = interp1d(timeSpeedAttValid, yawSpeedAttValid)
                fPitch = interp1d(timeSpeedAttValid, pitchSpeedAtValid)
                fRoll = interp1d(timeSpeedAttValid, rollSpeedAttValid)
                for index_ in invalidIndex_speedAtt:
                    self.spotMetadata.yawSpeedAtt[index_] = fYaw(self.spotMetadata.timeSpeedAtt[index_])
                    self.spotMetadata.pitchSpeedAtt[index_] = fPitch(self.spotMetadata.timeSpeedAtt[index_])
                    self.spotMetadata.rollSpeedAtt[index_] = fRoll(self.spotMetadata.timeSpeedAtt[index_])

            #### Perfrom attitude integration ######
            ## Compute attitude angles from angles velocities.
            ## Add the constant angle measures for the constant integration.

            self.yaw = self.__IntegrateRotationSpeed(const=self.spotMetadata.yawAttAng[0],
                                                     rotationSpeedArray=[0] + self.spotMetadata.yawSpeedAtt,
                                                     time=[self.spotMetadata.timeAttAng[0]]
                                                          + self.spotMetadata.timeSpeedAtt)[1:]

            self.pitch = self.__IntegrateRotationSpeed(const=self.spotMetadata.pitchAttAng[0],
                                                       rotationSpeedArray=[0] + self.spotMetadata.pitchSpeedAtt,
                                                       time=[self.spotMetadata.timeAttAng[0]]
                                                            + self.spotMetadata.timeSpeedAtt)[1:]

            self.roll = self.__IntegrateRotationSpeed(const=self.spotMetadata.rollAttAng[0],
                                                      rotationSpeedArray=[0] + self.spotMetadata.rollSpeedAtt,
                                                      time=[self.spotMetadata.timeAttAng[0]]
                                                           + self.spotMetadata.timeSpeedAtt)[1:]
            self.attitudeTime = self.spotMetadata.timeSpeedAtt


        else:

            self.attitudeTime = self.spotMetadata.attitudeTime
            self.yaw = self.spotMetadata.yaw
            self.pitch = self.spotMetadata.pitch
            self.roll = self.spotMetadata.roll

        ### Check the validity of attitude #####
        self.__SpotAttitudeCheck()

        ## Smoothing the attitudes values
        pitchSmoothed = self.__SpotAttitudeSmooth(vect=self.pitch)
        yawSmoothed = self.__SpotAttitudeSmooth(vect=self.yaw)
        rollSmoothed = self.__SpotAttitudeSmooth(vect=self.roll)

        ## Interpolate the attitudes for each line of the SPOT image

        fPitch = splrep(self.attitudeTime, pitchSmoothed, k=3)
        pitchInterp = splev(self.linesDates, fPitch, der=0)
        fYaw = splrep(self.attitudeTime, yawSmoothed, k=3)
        yawInterp = splev(self.linesDates, fYaw, der=0)
        fRoll = splrep(self.attitudeTime, rollSmoothed, k=3)
        rollInterp = splev(self.linesDates, fRoll, der=0)
        self.interp_rot_angles = np.vstack((pitchInterp, yawInterp, rollInterp)).T

    def sat_to_orb_rotation(self):
        ## Computing satellite to orbital systems rotation matrices
        self.satToNavMat = self.__SpotSat2NavMatrix(pitch=self.interp_rot_angles[:, 0],
                                                    yaw=self.interp_rot_angles[:, 1],
                                                    roll=self.interp_rot_angles[:, 2])
        if self.debug:
            logging.info("satToNavMat:{}".format(self.satToNavMat.shape))
            logging.info("Attitude:{}".format(self.satToNavMat.shape))

        return

    @staticmethod
    def __IntegrateRotationSpeed(const, rotationSpeedArray, time):
        """
        Compute satellite attitude fro the attitude velocities
        Args:
            const: integration constant
            rotationSpeedArray: rotation velocity
            time: date at each line

        Returns:
            array containing rotation instead of rotation velocities
        References:
             SPOT Imagery: p34-35, eq (8)
        """

        sz1 = len(rotationSpeedArray)
        sz2 = len(time)

        sx = np.min([sz1, sz2])
        output = sx * [0]
        output[0] = const
        for i in range(1, sx):
            val_ = rotationSpeedArray[i] * (time[i] - time[i - 1])
            output[i] = output[i - 1] + val_
        return output

    def __SpotAttitudeCheck(self):
        """
        Attitudes provided by SPOT are not always covering the scene acquisition.
        The basic method is to interpolate the data when data is missing.
        To be able to interpolate, one adds data at the beginning and at the end of the acquisition.
        Adding data at the beginning and at the end of the acquisition when data is missing to be able to interpolat
        Returns:

        """

        dt = 1e-3
        meanInterval = 1

        attTimeBound = [self.attitudeTime[0], self.attitudeTime[-1]]
        if attTimeBound[0] > self.linesDates[0]:
            begYawAve = np.mean(self.yaw[0: meanInterval + 1])
            begPitchAve = np.mean(self.pitch[0: meanInterval + 1])
            begRollAve = np.mean(self.roll[0: meanInterval + 1])

            newTime = self.linesDates[0] - dt
            self.attitudeTime = [newTime] + self.attitudeTime
            self.yaw = [begYawAve] + self.yaw
            self.pitch = [begPitchAve] + self.pitch
            self.roll = [begRollAve] + self.roll

        yx = len(self.yaw)
        if attTimeBound[1] < self.linesDates[-1]:
            endYawAve = np.mean(self.yaw[yx - 1 - meanInterval:])
            endPitchAve = np.mean(self.pitch[yx - 1 - meanInterval:])
            endRollAve = np.mean(self.roll[yx - 1 - meanInterval:])

            newTime = self.linesDates[- 1] + dt
            self.attitudeTime = self.attitudeTime + [newTime]
            self.yaw = self.yaw + [endYawAve]
            self.pitch = self.pitch + [endPitchAve]
            self.roll = self.roll + [endRollAve]

        return

    @staticmethod
    def __SpotAttitudeSmooth(vect):
        """
        Applying bspline interpolation
        Args:
            vect:

        Returns:

        """

        res = len(vect) * [0]
        n = len(vect)
        res[0] = vect[0]
        res[-1] = vect[-1]
        for x in range(1, n - 1):
            res[x] = (1 / 6) * vect[x - 1] + (2 / 3) * vect[x] + (1 / 6) * vect[x + 1]

        return res

    def __SpotSat2NavMatrix(self, pitch, yaw, roll):
        """
        Build the Matrix that change the reference system from satellite reference system to orbital reference system
        taking into account the satellite attitude: roll, pitch, yaw.
        Satellite interior look angles are added to satellite attitude.

        Args:
            pitch: list in [rad]
            yaw: list in [rad]
            roll: list in [rad]

        Returns:
        Notes:
            For historical reasons, attitudes values (rotation speed or absolute angles) are not expressed within the
            Navigation Reference Coordinate System (O1,X1,Y1,Z1), but with its inverse system (O1,-X1,-Y1,Z1).
            Sign of Roll and Pith values (rotation speed or absolute values) found in metadata will therefore be
             multiplied by -1 except the yaw values.
        References:
             SPOT123-4-5 Geometry Handbook
        """

        nb = len(pitch)

        oArray = np.empty((self.nbRows, 3, 3))
        for i in range(nb):
            oArray[i, :, :] = misc.cSpatialRotations().R_opk_xyz(omg=-pitch[i], phi=-roll[i], kpp=yaw[i])

        return oArray
