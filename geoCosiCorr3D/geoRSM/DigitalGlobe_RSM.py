"""
AUTHOR: Saif Aati (saif@caltech.edu)
PURPOSE: Read QuickBird/WorldView1-2-3-4 Image MetaData ASCII & XML files and Build RSM.
"""
import logging

import geoCosiCorr3D.geoRSM.misc as geoRSMMisc
import numpy as np
from geoCosiCorr3D.geoCore.constants import SOFTWARE
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoRSM.geoRSM_metadata.ReadSatMetadata import cGetDGMetadata
from geoCosiCorr3D.geoRSM.Interpol import custom_linear_interpolation


class cDigitalGlobe(RSM):
    __cosiCorrOrientation = SOFTWARE.geoCosiCorr3DOrientation

    def __init__(self, dgFile: str, debug: bool = False):
        """

        Args:
            dgFile:
        """
        super().__init__()
        self.file = dgFile
        self.debug = debug
        self.dgMetadata = cGetDGMetadata(dgFile=dgFile, debug=self.debug)
        self.platform = self.dgMetadata.platform
        self.date = self.dgMetadata.date
        self.time = self.dgMetadata.time
        self.productCatalogId = self.dgMetadata.productCatalogId
        self.imgLevel = self.dgMetadata.imgLevel
        self.bandId = self.dgMetadata.bandId
        self.nbRows = self.dgMetadata.nbRows
        self.nbCols = self.dgMetadata.nbCols
        self.startTime = self.dgMetadata.startTime
        self.avgLineRate = self.dgMetadata.avgLineRate
        self.sunAz = self.dgMetadata.sunAz
        self.sunElev = self.dgMetadata.sunElev
        self.scanDirection = self.dgMetadata.scanDirection
        self.meanGSD = self.dgMetadata.meanGSD
        self.gsd = self.meanGSD
        self.satElev = self.dgMetadata.satElev
        self.satAz = self.dgMetadata.satAz
        self.offNadir_angle = self.dgMetadata.offNadir_angle
        self.date_time_obj = self.dgMetadata.date_time_obj

        self.startTimeEph = self.dgMetadata.startTimeEph
        self.dtEph = self.dgMetadata.dtEph
        self.nbPoints = self.dgMetadata.nbPoints
        self.ephemeris = self.dgMetadata.ephemeris

        self.startTimeAtt = self.dgMetadata.startTimeAtt
        self.dtAtt = self.dgMetadata.dtAtt
        self.nbPointsAtt = self.dgMetadata.nbPointsAtt
        self.attitude = self.dgMetadata.attitude

        self.principalDistance = self.dgMetadata.principalDistance
        self.detPitch = self.dgMetadata.detPitch
        self.detOrigin = self.dgMetadata.detOrigin
        self.cameraAttitude = self.dgMetadata.cameraAttitude

        self.interpSatPosition = None
        self.CCDLookAngle = None

        self.orbitalPos_Z = None
        self.orbitalPos_X = None
        self.orbitalPos_Y = None
        self.satToNavMat = None
        if self.debug:
            print(f"--- Computing DG RSM:{self.__class__.__name__}")

        temp = (1 / self.avgLineRate)
        if self.scanDirection.lower() == "FORWARD".lower():
            loc_img = np.arange(0, self.nbRows, 1) * temp
        else:
            loc_img = np.arange(0, self.nbRows, 1) * -1 * temp
        self.loc_img = loc_img

        self.interp_eph()

        self.interp_attitude()

        self.compute_look_direction()

        self.compute_orbital_reference_system()

        self.sat_to_orb_rotation()

        if self.debug:
            print(" Done!\n", end="")
        return

    def interp_eph(self):
        """

        Returns:
        Notes:
            Interpolation of ephemeris and attitude for each image line, we use a linear interpolation as
            ephemeris and attitude measures frequency is high enough.
        """

        loc_eph = np.arange(0, self.ephemeris.shape[0], 1) * self.dtEph
        line_dates = self.startTime - self.startTimeEph + self.loc_img

        self.interpSatPosition = np.zeros((self.nbRows, 3))
        for i in range(3):
            self.interpSatPosition[:, i] = custom_linear_interpolation(VV=self.ephemeris[:, i],
                                                    XX=loc_eph,
                                                    x_out=line_dates)

        self.interpSatVelocity = np.zeros((self.nbRows, 3))
        for i in range(3):
            self.interpSatVelocity[:, i] = custom_linear_interpolation(VV=self.ephemeris[:, i + 3],
                                                    XX=loc_eph,
                                                    x_out=line_dates)
        self.interp_sat_eph_tpv = np.hstack((line_dates[:, np.newaxis], self.interpSatPosition, self.interpSatVelocity))
        logging.info(f"{self.__class__.__name__}: interp_sat_eph: {self.interp_sat_eph_tpv.shape}")

    def interp_attitude(self):
        loc_att = np.arange(0, self.attitude.shape[0], 1) * self.dtAtt

        ## Attitude (quaternions interpolation)
        line_dates = self.startTime - self.startTimeAtt + self.loc_img
        self.att_quat = np.zeros((self.nbRows, 4))
        for i in range(4):
            self.att_quat[:, i] = custom_linear_interpolation(VV=self.attitude[:, i], XX=loc_att, x_out=line_dates)

        self.interp_quat_txyzs = np.hstack((line_dates[:, np.newaxis], self.att_quat))
        logging.info(f"{self.__class__.__name__}: interp_quat: {self.interp_quat_txyzs.shape}")
        return

    def compute_look_direction(self):
        """

        Returns:
        Notes:
            Step 1:
                Compute each CCD coordinates in the camera reference system (centered on the perspective center)
            Step 2:
                Convert quaternion into a rotation matrix from Camera ref to Satellite ref
            Step 3 :
                Convert each CCD look angle in Camera ref into Satellite ref
            Step 4:
                Orient the CCD look angle into the COSI-corr orientation system (SPOT)
        """

        ccd_coord = np.zeros((self.nbCols, 3))

        ccd_coord[:, 0] = self.detOrigin[0]  # X component. Identical for each CCD

        tempList = [self.detOrigin[1] - item * self.detPitch for item in range(self.nbCols)]
        ccd_coord[:, 1] = np.asarray(tempList)  # Y component, varies for each CCD
        ccd_coord[:, 2] = self.principalDistance  # Z component=  Principal distance. Identical for each CCD

        # normalize the CCDs coordinates -> Definition of the look angle of each CCD in the camera geometry
        look_angle_in_camera = geoRSMMisc.NormalizeArray(inputArray=ccd_coord)

        ##Convert quaternion into a rotation matrix from Camera ref to Satellite ref
        rotMatCamToSat = np.asarray(geoRSMMisc.Qaut2Rot(quat_=self.cameraAttitude, axis=0, order=2)).T

        ## Convert each CCD look angle in Camera ref into Satellite ref

        self.CCDLookAngle = np.zeros((self.nbCols, 3))  ## CCD Look Angle in Satellite reference system

        for i in range(self.nbCols):
            self.CCDLookAngle[i, :] = np.dot(rotMatCamToSat, look_angle_in_camera[i, :].T)

        # Orient the CCD look angle into the COSI-corr orientation system (SPOT)
        ##TODO: we can use:  self.CCDLookAngle=cosiOri @ self.CCDLookAngle
        tmp = np.copy(self.CCDLookAngle[:, 0])
        self.CCDLookAngle[:, 0] = self.CCDLookAngle[:, 1]
        self.CCDLookAngle[:, 1] = tmp
        self.CCDLookAngle[:, 2] = -self.CCDLookAngle[:, 2]

        return

    def compute_orbital_reference_system(self):
        """
        ORBITAL REFERENCE SYSTEM DEFINITION
        Express orbital reference system for each line of the image in the ECF frame
        Returns:
        Notes:
            Coordinates directly expressed in the COSI-corr orientation

        """

        self.orbitalPos_Z = geoRSMMisc.normalize_array(self.interpSatPosition)  # normalize_array(interpSatPosition)
        self.orbitalPos_X = geoRSMMisc.normalize_array(np.cross(self.interpSatVelocity, self.orbitalPos_Z))
        self.orbitalPos_Y = np.cross(self.orbitalPos_Z, self.orbitalPos_X)

        return

    def sat_to_orb_rotation(self):
        """
        DEFINITION OF THE SATELLITE TO ORBITAL REFERENCE ROTATION MATRIX
        Returns:

        """

        ## Convert quaternions into rotation matrix
        ## Define a rotation matrix describing the satellite reference frame attitude relative to the ECF frame for each image line
        rotMatSatToECF = np.zeros((self.nbRows, 3, 3))

        for i in range(self.nbRows):
            rotMatSatToECF[i, :, :] = geoRSMMisc.Qaut2Rot(self.att_quat[i, :], axis=0, order=2)
        # Orient the satellite reference frame rotation matrix into the COSI-corr orientation
        for i in range(self.nbRows):
            rotMatSatToECF[i, :, :] = np.dot(self.__cosiCorrOrientation, rotMatSatToECF[i, :, :])

        ##Compute the Satellite to orbital rotation matrices (for each line)
        # MAT_SatToECF = MAT_OrbitalToECF . MAT_SatToOrbital
        ##so : MAT_SatToOrbital = INVERT(MAT_OrbitalToECF)## MAT_SatToECF
        self.satToNavMat = np.zeros((self.nbRows, 3, 3))
        for i in range(self.nbRows):
            temp = np.array([self.orbitalPos_X[i, :], self.orbitalPos_Y[i, :], self.orbitalPos_Z[i, :]])
            invTemp = np.linalg.inv(temp.T)
            self.satToNavMat[i, :, :] = np.dot(invTemp, rotMatSatToECF[i, :, :].T)  ##
        return

    def __repr__(self):
        pass

    def __str__(self):
        pass
