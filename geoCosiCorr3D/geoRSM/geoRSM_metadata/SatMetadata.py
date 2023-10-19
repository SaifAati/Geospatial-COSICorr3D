"""
Author: Saif Aati (saif@caltech.edu)
Read QuickBird/WorldView1-2-3-4 Image MetaData ASCII and XML files
"""
import datetime
import logging

import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from astropy.time import Time
from abc import ABC, abstractmethod
from geoCosiCorr3D.geoCore.constants import SatScanDirection
import geoCosiCorr3D.geoRSM.utils as utils
from geoCosiCorr3D.geoCore.constants import SOFTWARE
from shapely.geometry import Polygon
import geopandas


class SatAncError(Exception):
    """Custom exception class for cGetDGMetadata errors."""

    def __init__(self, message="An error occurred while processing Sat metadata"):
        self.message = message
        super().__init__(self.message)


# TODO: move to geocore base
class SatAnc(ABC):
    def __init__(self, anc_file):
        self.anc_file = anc_file
        self.platform = None
        self.nb_rows = None
        self.nb_cols = None
        self.scan_direction = SatScanDirection.FORWARD.lower()
        self.interp_eph = None
        self.interp_att_quat = None
        self.look_angles = None
        self.sat_pos_orbital = None
        self.att_rot_sat_to_orbital = None
        self.time_str = None

    @abstractmethod
    def parse_anc(self):
        return

    @abstractmethod
    def parse_att(self):
        return

    @abstractmethod
    def parse_eph(self):
        return

    @abstractmethod
    def parse_img_info(self):
        return

    @abstractmethod
    def parse_camera_geometry(self):
        return

    @abstractmethod
    def compute_interp_att(self) -> np.ndarray:
        pass

    @abstractmethod
    def compute_interp_eph(self) -> np.ndarray:
        pass

    @abstractmethod
    def compute_interp_look_angles(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_interp_att_rot_orbital(self,
                                   att_quat_ecef: np.ndarray,
                                   sat_pos_orbital: np.ndarray,
                                   geoCosiCorr=True) -> np.ndarray:
        pass

    @abstractmethod
    def get_anc_footprint(self) -> geopandas.GeoDataFrame:
        pass


class cGetDGMetadata(SatAnc):
    def __init__(self, anc_file: str, debug: bool = False):

        super().__init__(anc_file)
        self.debug = debug

        file_extension = Path(self.anc_file).suffix.lower()
        self.parse_file_based_on_extension(file_extension)

        if self.debug:
            self.log_debug_information()

        self.interp_eph = self.compute_interp_eph()
        self.interp_att_quat = self.compute_interp_att()
        self.look_angles = self.compute_interp_look_angles()
        self.sat_pos_orbital = utils.Convert.ecef_to_orbital(self.interp_eph)
        self.att_rot_sat_to_orbital = self.get_interp_att_rot_orbital(self.interp_att_quat, self.sat_pos_orbital)
        return

    def parse_file_based_on_extension(self, file_extension: str):
        """Process the file based on its extension."""
        if file_extension == ".xml":
            self.parse_anc()

        else:
            raise SatAncError(
                'Invalid DG metadata file extension, '
                'Expected: .xml. Deprecated support for [.ATT, .EPH, .GEO, .IMD]')

    def log_debug_information(self):
        """Log debug information."""
        logging.info(f"Platform: {self.platform}, "
                     # f"bandID: {self.bandId}, "
                     f"nbRows: {self.nb_rows}, "
                     f"nbCols: {self.nb_cols}, "
                     # f"startTime: {self.startTime}, "
                     # f"self.avgLineRate: {self.avgLineRate}"
                     )

        # logging.info(f"sunAz: {self.sunAz}, "
        #              f"sunElev: {self.sunElev}, "
        #              f"satElev: {self.satElev}, "
        #              f"sunAz: {self.sunAz}, "
        #              f"offNadir_angle: {self.offNadir_angle}, "
        #              f"scanDirection: {self.scanDirection}, "
        #              f"ID: {self.productCatalogId}, "
        #              f"meanGSD: {self.meanGSD}")

        logging.info("--- SAT Camera Geometry [mm] ---")
        logging.info(f"f: {self.principal_distance}, "
                     f"cam_att: {self.camera_att}, "
                     f"orig: {self.det_origin}, "
                     f"pitch: {self.det_pitch}")

    def parse_anc(self):
        if self.debug:
            logging.info("--- Computing DG ANC from XML:")

        tree = ET.parse(self.anc_file)
        self.root = tree.getroot()

        self.parse_img_info()

        first_line_start_timestamp_str = [elem.text for elem in self.root.findall(".//FIRSTLINETIME")][0]
        self.time_str = first_line_start_timestamp_str
        first_line_start_time_dt = datetime.datetime.strptime(first_line_start_timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        start_time = Time(first_line_start_time_dt, format='datetime',
                          scale='utc')  # the number of seconds since the Unix Epoch (1970-01-01 00:00:00 UTC) not including leap seconds.
        self.first_line_start_time_unix = start_time.unix
        # print(f'first line time:{first_line_start_timestamp_str}')

        self.anc_eph, self.eph_time_interval, self.eph_start_time_unix = self.parse_eph()
        self.anc_quat_att, self.att_time_interval, self.att_start_time_unix = self.parse_att()
        self.principal_distance, self.camera_att, self.det_pitch, self.det_origin = self.parse_camera_geometry()

        return

    def parse_eph(self):
        """
        retrieve ephemeris data:
        start_eph: date UTC (in second) of first ephemeris
        dt_eph: sampling period (in second) of ephemeris measurment
        ephemeris: ephemeris values (position and velocity) - 2D array ([X, Y, Z, Vx, Vy, Vz] x NB measures)
        expressed in ECF (Earth Centered Fixed reference)

        """

        eph_start_time_str = [elem.text for elem in self.root.findall(".//EPH//STARTTIME")][0]
        eph_start_time_str_dt = datetime.datetime.strptime(eph_start_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        start_time = Time(eph_start_time_str_dt, format='datetime',
                          scale='utc')  # the number of seconds since the Unix Epoch (1970-01-01 00:00:00 UTC) not including leap seconds.
        eph_start_time_unix = start_time.unix

        eph_time_interval = [float(elem.text) for elem in self.root.findall(".//EPH//TIMEINTERVAL")][
            0]  # sampling period [s]
        eph_nb_samples = [int(elem.text) for elem in self.root.findall(".//EPH//NUMPOINTS")][0]
        eph = np.zeros((eph_nb_samples, 6))
        eph_list = [elem.text for elem in self.root.findall(".//EPH//EPHEMLIST")]

        for line_ in eph_list:
            index, X, Y, Z, VX, VY, VZ, _, _, _, _, _, _ = line_.split()
            eph[int(float(index)) - 1, 0] = float(X)
            eph[int(float(index)) - 1, 1] = float(Y)
            eph[int(float(index)) - 1, 2] = float(Z)
            eph[int(float(index)) - 1, 3] = float(VX)
            eph[int(float(index)) - 1, 4] = float(VY)
            eph[int(float(index)) - 1, 5] = float(VZ)
        return eph, eph_time_interval, eph_start_time_unix

    def parse_att(self):
        """
            RETRIEVE SATELLITE ATTITUDE DATA:
            start_att: date UTC (in second) of first attitude
            dt_att: sampling period (in second) of attitude measurment
            attitude: satellite attitude values (quaternions) - 2D array ([q1, q2, q3, q4] x NB measures).
            Note: q4 scalar part
            Describe the rotation of the satellite coordinate system relative to ECF.
        """

        att_start_time_str = [elem.text for elem in self.root.findall(".//ATT//STARTTIME")][0]
        att_start_time_dt = datetime.datetime.strptime(att_start_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        start_time = Time(att_start_time_dt, format='datetime', scale='utc')
        att_start_time_unix = start_time.unix
        att_time_interval = [float(elem.text) for elem in self.root.findall(".//ATT//TIMEINTERVAL")][0]
        att_nb_samples = [int(elem.text) for elem in self.root.findall(".//ATT//NUMPOINTS")][0]
        quat_att = np.zeros((att_nb_samples, 4))
        att_list = [elem.text for elem in self.root.findall(".//ATT//ATTLIST")]

        for line_ in att_list:
            index, q1, q2, q3, q4, _, _, _, _, _, _, _, _, _, _ = line_.split()
            quat_att[int(float(index)) - 1, 0] = float(q1)
            quat_att[int(float(index)) - 1, 1] = float(q2)
            quat_att[int(float(index)) - 1, 2] = float(q3)
            quat_att[int(float(index)) - 1, 3] = float(q4)
        # print(f'att start time:{att_start_time_str}')
        return quat_att, att_time_interval, att_start_time_unix

    def parse_camera_geometry(self):
        """
        RETRIEVE CAMERA GEOMETRY INFO:
        detOrigin: two elements array representing the location of the first CCD in the camera reference system (in millimeters)
        detPitch: distance, in millimeter, between two adjacent CCD centers
        principalDist: Principal distance of the camera in millimeter
        geoQuat: 4 elements array. Quaternions representing the rotation of the camera reference system relative to the satellite ref
          """
        principal_distance = [float(elem.text) for elem in self.root.findall(".//GEO/PRINCIPAL_DISTANCE//PD")][0]
        qcs1 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS1")][0]
        qcs2 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS2")][0]
        qcs3 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS3")][0]
        qcs4 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS4")][0]
        camera_att = [qcs1, qcs2, qcs3, qcs4]
        xOrigin = [float(elem.text) for elem in self.root.findall(".//GEO//DETORIGINX")][0]
        yOrigin = [float(elem.text) for elem in self.root.findall(".//GEO//DETORIGINY")][0]

        det_pitch = [float(elem.text) for elem in self.root.findall(".//GEO//DETPITCH")][0]

        det_origin = [xOrigin, yOrigin]  # principal point

        return principal_distance, camera_att, det_pitch, det_origin

    def parse_img_info(self):
        self.platform = [elem.text for elem in self.root.findall(".//SATID")][0]
        self.nb_rows = [int(elem.text) for elem in self.root.findall(".//NUMROWS")][0]
        self.nb_cols = [int(elem.text) for elem in self.root.findall(".//NUMCOLUMNS")][0]
        self.nb_bands = None
        self.scan_direction = [elem.text for elem in self.root.findall(".//SCANDIRECTION")][0]
        self.line_rate = [float(elem.text) for elem in self.root.findall(".//AVGLINERATE")][0]

        return

    def compute_interp_eph(self):
        """

        Returns:
        Notes:
            Interpolation of ephemeris and attitude for each image line, we use a linear interpolation as
            ephemeris and attitude measures frequency is high enough.
        """

        if self.scan_direction.lower() == SatScanDirection.FORWARD.lower():
            loc_img = np.arange(0, self.nb_rows, 1) * (1 / self.line_rate)
        else:
            loc_img = np.arange(0, self.nb_rows, 1) * -1 * (1 / self.line_rate)

        eph_nb_samples = np.shape(self.anc_eph)[0]
        location_eph = np.arange(0, eph_nb_samples, 1) * self.eph_time_interval
        loc_img0 = self.first_line_start_time_unix - self.eph_start_time_unix + loc_img

        # Ephemeris
        interp_sat_pos = np.zeros((self.nb_rows, 3))
        interp_sat_vel = np.zeros((self.nb_rows, 3))

        for i in range(3):
            interp_sat_pos[:, i] = utils.Interpolate.custom_linear_interpolation(VV=self.anc_eph[:, i],
                                                                                 XX=location_eph,
                                                                                 x_out=loc_img0)

            interp_sat_vel[:, i] = utils.Interpolate.custom_linear_interpolation(VV=self.anc_eph[:, i + 3],
                                                                                 XX=location_eph,
                                                                                 x_out=loc_img0)
        interp_eph = np.hstack((interp_sat_pos, interp_sat_vel))

        return interp_eph

    def compute_interp_att(self):
        """

        Returns:
        Notes:
            Interpolation of ephemeris and attitude for each image line, we use a linear interpolation as
            ephemeris and attitude measures frequency is high enough.
        """

        if self.scan_direction.lower() == SatScanDirection.FORWARD.lower():
            loc_img = np.arange(0, self.nb_rows, 1) * (1 / self.line_rate)
        else:
            loc_img = np.arange(0, self.nb_rows, 1) * -1 * (1 / self.line_rate)

        att_nb_samples = np.shape(self.anc_quat_att)[0]
        loc_att = np.arange(0, att_nb_samples, 1) * self.att_time_interval
        # FIXME: using linear interpolation for quaternion ??
        ## Attitude (quaternions interpolation)
        loc_img0 = self.first_line_start_time_unix - self.att_start_time_unix + loc_img
        interp_att_quat = np.zeros((self.nb_rows, 4))
        for i in range(4):
            interp_att_quat[:, i] = utils.Interpolate.custom_linear_interpolation(VV=self.anc_quat_att[:, i],
                                                                                  XX=loc_att,
                                                                                  x_out=loc_img0)

        return interp_att_quat

    def compute_interp_look_angles(self):
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

        ccd_coord = np.zeros((self.nb_cols, 3))
        ccd_coord[:, 0] = self.det_origin[0]  # X component. Identical for each CCD.
        ccd_coord[:, 1] = [self.det_origin[1] - item * self.det_pitch for item in
                           range(self.nb_cols)]  # Y component, varies for each CCD
        ccd_coord[:, 2] = self.principal_distance  # Z component=  Principal distance. Identical for each CCD

        # Normalize the CCDs coordinates -> Definition of the look angle of each CCD in the camera geometry
        look_angle_camera_body = utils.normalize_array(input_array=ccd_coord)
        # Convert quaternion into a rotation matrix
        cam_to_sat_rot = utils.Convert.custom_quat_to_rotation(self.camera_att)
        # Convert each CCD look angle in Camera ref into Satellite ref
        look_angles = np.dot(look_angle_camera_body, cam_to_sat_rot)
        return look_angles @ SOFTWARE.geoCosiCorr3DOrientation

    def get_interp_att_rot_orbital(self,
                                   att_quat_ecef: np.ndarray,
                                   sat_pos_orbital: np.ndarray,
                                   geoCosiCorr=True):
        """
        For every line we define attitude rotation matrix expressed in orbital coordinates

        """

        # Define a rotation matrix describing the satellite reference frame attitude relative to the ECF
        # frame for each image line.
        att_rot_ecef = np.zeros((self.nb_rows, 3, 3))

        for i in range(self.nb_rows):
            att_rot_ecef[i, :, :] = utils.Convert.custom_quat_to_rotation(att_quat_ecef[i, :])

        # Orient the satellite reference frame rotation matrix into the COSI-corr orientation
        if geoCosiCorr:
            for i in range(self.nb_rows):
                att_rot_ecef[i, :, :] = np.dot(SOFTWARE.geoCosiCorr3DOrientation, att_rot_ecef[i, :, :])

        # Compute the Satellite to orbital rotation matrices (for each line)
        att_rot_sat_to_orbital = np.zeros((self.nb_rows, 3, 3))
        for i in range(self.nb_rows):
            temp = np.array(
                [sat_pos_orbital[:, 0:3][i, :], sat_pos_orbital[:, 3:6][i, :], sat_pos_orbital[:, 6:9][i, :]])
            invTemp = np.linalg.inv(temp.T)
            att_rot_sat_to_orbital[i, :, :] = np.dot(invTemp, att_rot_ecef[i, :, :].T)

        return att_rot_sat_to_orbital

    def get_anc_footprint(self) -> geopandas.GeoDataFrame:
        ulLon = [float(elem.text) for elem in self.root.findall(".//IMD//ULLON")][0]
        ulLat = [float(elem.text) for elem in self.root.findall(".//IMD//ULLAT")][0]
        urLon = [float(elem.text) for elem in self.root.findall(".//IMD//URLON")][0]
        urLat = [float(elem.text) for elem in self.root.findall(".//IMD//URLAT")][0]
        lrLon = [float(elem.text) for elem in self.root.findall(".//IMD//LRLON")][0]
        lrLat = [float(elem.text) for elem in self.root.findall(".//IMD//LRLAT")][0]
        llLon = [float(elem.text) for elem in self.root.findall(".//IMD//LLLON")][0]
        llLat = [float(elem.text) for elem in self.root.findall(".//IMD//LLLAT")][0]
        polygon = Polygon([(ulLon, ulLat), (urLon, urLat), (lrLon, lrLat), (llLon, llLat)])
        gdf = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])
        gdf.to_file(f'{self.platform}_{self.time_str}_anc_fp.geojson', driver='GeoJSON')
        return gdf

    def parse_img_info_to_DEPRECATE(self):

        imgLevel = [elem.text for elem in self.root.findall(".//IMAGEDESCRIPTOR")][0]

        bandId = [elem.text for elem in self.root.findall(".//BANDID")][0]

        avgLineRate = [float(elem.text) for elem in self.root.findall(".//AVGLINERATE")][0]

        sun_az = [float(elem.text) for elem in self.root.findall(".//MEANSUNAZ")][0]
        sun_elev = [float(elem.text) for elem in self.root.findall(".//MEANSUNEL")][0]

        sat_elev = [float(elem.text) for elem in self.root.findall(".//MEANSATEL")][0]
        sat_az = [float(elem.text) for elem in self.root.findall(".//MEANSATAZ")][0]

        off_nadir_angle = [float(elem.text) for elem in self.root.findall(".//MEANOFFNADIRVIEWANGLE")][0]

        # self.productCatalogId = [elem.text for elem in root.findall(".//PRODUCTCATALOGID")][0]
        try:
            meanGSD = [float(elem.text) for elem in self.root.findall(".//MEANPRODUCTGSD")][0]
        except:
            meanGSD = [float(elem.text) for elem in self.root.findall(".//MEANCOLLECTEDCOLGSD")][0]
        return


# class cGetSpot67Metadata(SatAnc):
#     def __init__(self, anc_file: str):
#         # create element tree object
#         super().__init__(anc_file)
#         self.parse_anc()
#
#     def parse_anc(self):
#         tree = ET.parse(self.anc_file)
#         self.root = tree.getroot()
#
#         self.platform = None
#         self.nb_cols = [int(elem.text) for elem in self.root.findall(".//NCOLS")][0]
#         self.nb_rows = [int(elem.text) for elem in self.root.findall(".//NROWS")][0]
#         self.nb_bands = [int(elem.text) for elem in self.root.findall(".//NBANDS")][0]
#
#         self.anc_eph = self.parse_eph()
#         self.anc_quat_att = self.parse_att()
#         # self.principal_distance, self.camera_att, self.det_pitch, self.det_origin = self.parse_camera_geometry()
#         principal_distance, detector_col_sz, detector_row_sz = self.parse_camera_geometry()
#
#         return
#
#     def parse_eph(self):
#         logging.info("--- Get Look angles ----")
#
#         # self.lineOfSight[0, 0] = float(
#         #     [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/XLOS_0")][0])
#         # self.lineOfSight[1, 0] = float(
#         #     [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/XLOS_1")][0])
#         # self.lineOfSight[2, 0] = float(
#         #     [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/YLOS_0")][0])
#         # self.lineOfSight[3, 0] = float(
#         #     [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/YLOS_1")][0])
#         # logging.info("Look_Angles:\n{}".format(self.lineOfSight))
#
#         sat_pos = np.array([np.array(elem.text.split(" "), dtype=float) for elem in
#                             self.root.findall(".//Refined_Model/Ephemeris//Point/LOCATION_XYZ")])
#         sat_vel = np.array([np.array(elem.text.split(" "), dtype=float) for elem in
#                             self.root.findall(".//Refined_Model/Ephemeris//Point/VELOCITY_XYZ")])
#         eph_nb_samples = sat_pos.shape[0]
#         eph_times_str = [elem.text for elem in self.root.findall(".//Refined_Model/Ephemeris//Point/TIME")]
#
#         eph_times_dt = [datetime.datetime.strptime(eph_time_str, '%Y-%m-%dT%H:%M:%S.%fZ') for eph_time_str in
#                         eph_times_str]
#         eph_times_unix = [Time(time_dt, format='datetime', scale='utc').unix for time_dt in eph_times_dt]
#         sat_eph = np.hstack((sat_pos, sat_vel))
#         return sat_eph
#
#     def parse_att(self):
#         q1 = [float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q0")]
#         q2 = [float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q1")]
#         q3 = [float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q2")]
#         q4 = [float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q3")]
#         att_times_str = [elem.text for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/TIME")]
#         att_times_dt = [datetime.datetime.strptime(att_time_str, '%Y-%m-%dT%H:%M:%S.%fZ') for att_time_str in
#                         att_times_str]
#         att_times_unix = [Time(time_dt, format='datetime', scale='utc').unix for time_dt in att_times_dt]
#         quat_att = np.array([q1, q2, q3, q4]).T
#
#         return quat_att
#
#     def parse_img_info(self):
#         pass
#
#     def parse_camera_geometry(self):
#         principal_distance = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//FOCAL_LENGTH")][0])
#         detector_col_sz = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_COL")][0])
#         detector_row_sz = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_ROW")][0])
#         return principal_distance, detector_col_sz, detector_row_sz
#
#     # def GetSourceInformation(self):
#     #     if 'Dataset_Sources' in self.childTags:
#     #         self.instrument = [elem.text for elem in self.root.findall(".//INSTRUMENT")][0]
#     #         self.instrumentIndex = [elem.text for elem in self.root.findall(".//INSTRUMENT_INDEX")][0]
#     #         self.imagingTime = [elem.text for elem in self.root.findall(".//IMAGING_TIME")][0]
#     #         self.imagingDate = [elem.text for elem in self.root.findall(".//IMAGING_DATE")][0]
#     #     logging.info("mission:{},instrument:{},imagingDate:{}, imagingTime:{},".format(self.instrument,
#     #                                                                                    self.instrumentIndex,
#     #                                                                                    self.imagingDate,
#     #                                                                                    self.imagingTime))
#     #
#     #     self.sunAz = float([elem.text for elem in self.root.findall(".//Use_Area//SUN_AZIMUTH")][0])
#     #     self.sunElev = float([elem.text for elem in self.root.findall(".//Use_Area//SUN_ELEVATION")][0])
#     #     self.satElev = self.sunElev
#     #     ## Taking only across_track gsd for both directions
#     #     self.gsd_ACT = float([elem.text for elem in self.root.findall(".//Use_Area//GSD_ACROSS_TRACK")][0])
#     #     self.gsd_ALT = float([elem.text for elem in self.root.findall(".//Use_Area//GSD_ALONG_TRACK")][0])
#     #     self.meanGSD = np.mean([[self.gsd_ALT, self.gsd_ACT]])
#     #     self.time = [elem.text for elem in self.root.findall(".//Use_Area//TIME")][0]
#     #     self.satAlt = float([elem.text for elem in self.root.findall(".//Use_Area//SATELLITE_ALTITUDE")][0])
#     #     self.azAngle = float([elem.text for elem in self.root.findall(".//Use_Area//AZIMUTH_ANGLE")][0])
#     #     self.satAz = self.azAngle
#     #     self.viewAngle = float([elem.text for elem in self.root.findall(".//Use_Area//VIEWING_ANGLE")][0])
#     #     self.offNadir_angle = self.viewAngle
#     #     self.incidenceAngle = float([elem.text for elem in self.root.findall(".//Use_Area//INCIDENCE_ANGLE")][0])
#     #     logging.info(
#     #         "sunAz:{:.6},"
#     #         "sunElev:{:.6},"
#     #         " meanGSD:{:.6}, "
#     #         "time:{}, "
#     #         "satAlt:{:.6},"
#     #         " azAngle:{:.6}, "
#     #         "viewAngle:{:.6}, "
#     #         "incidenceAngle:{:.6}".format(
#     #             self.sunAz,
#     #             self.sunElev,
#     #             self.meanGSD,
#     #             self.time,
#     #             self.satAlt,
#     #             self.azAngle,
#     #             self.viewAngle,
#     #             self.incidenceAngle))
#     #     return
#     #
#     # def GetSpotSensorConfig(self):
#     #     logging.info("--- Get Spot Sensor Configuration ---")
#     #
#     #     startTime = [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Range/START")][0]
#     #     startTime = datetime.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%fZ')
#     #     self.startTime = ConvertTime2Second(startTime)
#     #     endTime = [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Range/END")][0]
#     #     endTime = datetime.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%S.%fZ')
#     #     self.endTime = ConvertTime2Second(endTime)
#     #     ##linePeriod in microsecond, need to convert it to second
#     #     self.linePeriod = float(
#     #         [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Stamp/LINE_PERIOD")][0]) / 1000000
#     #     # logging.info(startTime, endTime, linePeriod)
#     #     logging.info("linePeriod:{},startTime:{},endTimeLine:{}".format(self.linePeriod,
#     #                                                                     self.startTime,
#     #                                                                     self.endTime))
#     #
#     #     self.focal = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//FOCAL_LENGTH")][0])
#     #     self.szCol = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_COL")][0])
#     #     self.szRow = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_ROW")][0])
#     #
#     #     logging.info("focal[m]:{:.4}, detectorSize[m]:(col:{:.4},row:{:.4})".format(self.focal, self.szCol, self.szRow))
#     #
#     #     return
#
#
# class cGetSpot15Metadata:
#     def __init__(self, dmpFile: str, debug: bool = False, band: int = 0):
#
#         self.debug = debug
#         self.dmpFile = dmpFile
#         self.band = band  # 0:panchromatic
#         # create element tree object
#         self.tree = ET.parse(self.dmpFile)
#
#         # get root element
#         self.root = self.tree.getroot()
#         self.childTags = [elem.tag for elem in self.root]
#         if self.debug:
#             logging.info("____START PARSING SPOT METADATA______")
#             logging.info(self.childTags)
#
#         self.lookAngles = np.zeros((2, 2))
#         self.mirrorStep = None
#         self.GetSpotSensor()
#
#         self.sunAz = None
#         self.sunElev = None
#         self.mission = None
#         self.instrument = None
#         self.GetSourceInformation()
#
#         self.linePeriod = None
#         self.sceneCenterTime = None
#         self.sceneCenterLine = None
#         self.GetSpotSensorConfig()
#
#         self.satPosition = None
#         self.satVelocity = None
#         self.ephTime = []
#         self.GetSpotEphemeris()
#
#         self.nbCols = None
#         self.nbRows = None
#         self.nbBands = None
#         self.GetSpotRasterDimensions()
#
#         if self.mission == 5:
#             self.attitudeTime = None
#             self.yaw = None
#             self.pitch = None
#             self.roll = None
#             self.outOfRange = None
#             self.starTrackerUsed = None
#             if self.debug:
#                 logging.info("--- SPOT-5: get Corrected Attitude ---")
#             self.GetSpotCorrectedAttitude()
#         else:
#             if self.debug:
#                 logging.info("--- SPOT-1-4: get Raw Speed Attitude ---")
#             self.timeAttAng = []
#             self.yawAttAng = []
#             self.rollAttAng = []
#             self.pitchAttAng = []
#             self.outOfRangeAttAng = []
#             self.timeSpeedAtt = []
#             self.yawSpeedAtt = []
#             self.pitchSpeedAtt = []
#             self.rollSpeedAtt = []
#             self.outOfRangeSpeedAtt = []
#             self.GetSpotRawSpeedAttitude()
#         if self.debug:
#             logging.info("____END PARSING SPOT METADATA______")
#
#     def GetSpotSensor(self):
#         if self.debug:
#             logging.info("--- Get Spot Sensor --- ")
#
#         PSI_X = [float(elem.text) for elem in self.root.findall(".//Look_Angles_List//Look_Angles//PSI_X")]
#         PSI_Y = [float(elem.text) for elem in self.root.findall(".//Look_Angles_List//Look_Angles//PSI_Y")]
#         # print(self.lookAngles.shape)
#         # self.lookAngles[:, 0] = PSI_X
#         # self.lookAngles[:, 1] = PSI_Y
#         self.lookAngles = np.array([PSI_X, PSI_Y]).T
#         try:
#             self.mirrorStep = [int(elem.text) for elem in self.root.findall(".//Mirror_Position//STEP_COUNT")][0]
#         except:
#             self.mirrorStep = 0
#         if self.debug:
#             logging.info("lookAngles:{}".format(self.lookAngles.shape))
#             logging.info("mirrorStep:{}".format(self.mirrorStep))
#
#     def GetSourceInformation(self):
#         if 'Dataset_Sources' in self.childTags:
#             self.sunAz = float([elem.text for elem in
#                                 self.root.findall("Dataset_Sources/Source_Information/Scene_Source/SUN_AZIMUTH")][0])
#             self.sunElev = float([elem.text for elem in
#                                   self.root.findall("Dataset_Sources/Source_Information/Scene_Source/SUN_ELEVATION")][
#                                      0])
#             self.mission = int([elem.text for elem in
#                                 self.root.findall("Dataset_Sources/Source_Information/Scene_Source/MISSION_INDEX")][
#                                    0])
#             self.instrument = int([elem.text for elem in
#                                    self.root.findall(
#                                        "Dataset_Sources/Source_Information/Scene_Source/INSTRUMENT_INDEX")][
#                                       0])
#             self.instrumentName = [elem.text for elem in
#                                    self.root.findall(
#                                        "Dataset_Sources/Source_Information/Scene_Source/INSTRUMENT")][0]
#             self.sensorCode = [elem.text for elem in
#                                self.root.findall(
#                                    "Dataset_Sources/Source_Information/Scene_Source/SENSOR_CODE")][0]
#             self.incidenceAngle = [float(elem.text) for elem in self.root.findall(".//INCIDENCE_ANGLE")][0]
#
#             self.date = [elem.text for elem in self.root.findall(".//IMAGING_DATE")][0]
#             self.time = [elem.text for elem in self.root.findall(".//IMAGING_TIME")][0]
#             if self.debug:
#                 logging.info("sunAz:{},sunElev:{},mission:{},instrument:{}, incidenceAngle:{}".format(self.sunAz,
#                                                                                                       self.sunElev,
#                                                                                                       self.mission,
#                                                                                                       self.instrument,
#                                                                                                       self.incidenceAngle))
#
#     def GetSpotSensorConfig(self):
#         """
#         linePeriod: scalar
#         sceneCenterTime:  time in seconds at the scene center
#         sceneCenterLine: line at the scene center
#         :return:
#         """
#         if self.debug:
#             logging.info("--- Get Spot Sensor Configuration ---")
#         self.linePeriod = \
#             [float(elem.text) for elem in self.root.findall(".//Sensor_Configuration//Time_Stamp//LINE_PERIOD")][0]
#         sceneCenterTime = \
#             [elem.text for elem in self.root.findall(".//Sensor_Configuration//Time_Stamp//SCENE_CENTER_TIME")][0]
#         if self.debug:
#             logging.info(sceneCenterTime)
#         try:
#             sceneCenterTime = datetime.datetime.strptime(sceneCenterTime, '%Y-%m-%dT%H:%M:%S.%fZ')
#         except:
#             sceneCenterTime = datetime.datetime.strptime(sceneCenterTime, '%Y-%m-%dT%H:%M:%S.%f')
#
#         self.sceneCenterTime = ConvertTime2Second(sceneCenterTime)
#         self.sceneCenterLine = [int(elem.text) for elem in
#                                 self.root.findall(".//Sensor_Configuration//Time_Stamp//SCENE_CENTER_LINE")][0]
#         if self.debug:
#             logging.info("linePeriod:{},sceneCenterTime:{},sceneCenterLine:{}".format(self.linePeriod,
#                                                                                       self.sceneCenterTime,
#                                                                                       self.sceneCenterLine))
#
#         return
#
#     def GetSpotEphemeris(self):
#         """
#         time: array containing the time when ephemeris are measured
#         position: satellite center of mass position in cartesian coordinate system
#         velocity: satellite velocity
#         :return:
#         """
#         if self.debug:
#             logging.info("--- Get Spot Ephemeris ---")
#         self.dorisUsed = [elem.text for elem in self.root.findall(".//DORIS_USED")][0]
#
#         X = [float(elem.text) for elem in self.root.findall(".//Points//Location//X")]
#         Y = [float(elem.text) for elem in self.root.findall(".//Points//Location//Y")]
#         Z = [float(elem.text) for elem in self.root.findall(".//Points//Location//Z")]
#         self.satPosition = np.array([X, Y, Z]).T
#
#         vX = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//X")]
#         vY = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//Y")]
#         vZ = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//Z")]
#         self.satVelocity = np.array([vX, vY, vZ]).T
#
#         time = [elem.text for elem in self.root.findall(".//Points//TIME")]
#         for t_ in time:
#             try:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
#             except:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
#             self.ephTime.append(ConvertTime2Second(t__))
#         if self.debug:
#             logging.info("dorisUsed:{},satPosition:{} ,satVelocity:{},ephTime:{}".format(self.dorisUsed,
#                                                                                          self.satPosition.shape,
#                                                                                          self.satVelocity.shape,
#                                                                                          len(self.ephTime)))
#         return
#
#     def GetSpotRasterDimensions(self):
#         self.nbCols = [int(elem.text) for elem in self.root.findall(".//NCOLS")][0]
#         self.nbRows = [int(elem.text) for elem in self.root.findall(".//NROWS")][0]
#         self.nbBands = [int(elem.text) for elem in self.root.findall(".//NBANDS")][0]
#         if self.debug:
#             logging.info("nbCols:{},nbRows:{},nbBands:{}".format(self.nbCols, self.nbRows, self.nbBands))
#
#         return
#
#     def GetSpotRawSpeedAttitude(self):
#         """
#         timeCst: an array containing time when absolute attitudes are measured
#          yawCst : array containing yaw absolute values
#          pitchCst: array containing pitch absolute values
#          rollCst: array containing roll absolute values
#
#          timeVel: an array containing time when speed attitudes are measured
#          yawVel : array containing yaw speed values
#          pitchVel : array containing pitch speed values
#          rollVel : array containing roll speed values
#
#         :return:
#         """
#         if self.debug:
#             logging.info("--- Get SPOT {} Speed Attitude ---".format(self.mission))
#
#         #### Angles_List  ####
#         self.yawAttAng = [float(elem.text) for elem in
#                           self.root.findall(
#                               ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//YAW")]
#         self.pitchAttAng = [float(elem.text) for elem in
#                             self.root.findall(
#                                 ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//PITCH")]
#         self.rollAttAng = [float(elem.text) for elem in
#                            self.root.findall(
#                                ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//ROLL")]
#
#         timeAtt = [elem.text for elem in
#                    self.root.findall(".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//TIME")]
#         for t_ in timeAtt:
#             try:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
#             except:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
#             self.timeAttAng.append(ConvertTime2Second(t__))
#         self.outOfRangeAttAng = [elem.text for elem in
#                                  self.root.findall(
#                                      ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//OUT_OF_RANGE")]
#
#         if self.debug:
#             logging.info("yawAttAng:{} , pitchAttAng:{}, rollAttAng:{} ,timeAttAng:{},outOfRangeAttAng:{} ".format(
#                 len(self.yawAttAng),
#                 len(self.pitchAttAng),
#                 len(self.rollAttAng),
#                 len(self.timeAttAng),
#                 len(self.outOfRangeAttAng)))
#
#         #######  Angular_Speeds_List ##########
#         self.yawSpeedAtt = [float(elem.text) for elem in
#                             self.root.findall(
#                                 ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//YAW")]
#         self.pitchSpeedAtt = [float(elem.text) for elem in
#                               self.root.findall(
#                                   ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//PITCH")]
#
#         self.rollSpeedAtt = [float(elem.text) for elem in
#                              self.root.findall(
#                                  ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//ROLL")]
#
#         timeSpeed = [elem.text for elem in
#                      self.root.findall(
#                          ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//TIME")]
#         for t_ in timeSpeed:
#             try:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
#             except:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
#             self.timeSpeedAtt.append(ConvertTime2Second(t__))
#         self.outOfRangeSpeedAtt = [elem.text for elem in
#                                    self.root.findall(
#                                        ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//OUT_OF_RANGE")]
#
#         if self.debug:
#             logging.info(
#                 "yawSpeedAtt:{} ,pitchSpeedAtt:{},rollSpeedAtt:{} ,timeSpeedAtt:{},outOfRangeSpeedAtt:{} ".format(
#                     len(self.yawSpeedAtt),
#                     len(self.pitchSpeedAtt),
#                     len(self.rollSpeedAtt),
#                     len(self.timeSpeedAtt),
#                     len(self.outOfRangeSpeedAtt)))
#
#         return
#
#     def GetSpotCorrectedAttitude(self):
#
#         """
#         time -- an array containing time when attitude is measured
#         yaw -- array containing yaw absolute values
#         pitch -- array containing pitch absolute values
#         roll -- array containing roll absolute values
#         :return:
#         """
#         self.starTrackerUsed = [elem.text for elem in self.root.findall(".//STAR_TRACKER_USED")][0]
#         # self.correctedAtt = [elem.text for elem in self.root.findall(".//Corrected_Attitude")]
#         self.yaw = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//YAW")]
#         self.pitch = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//PITCH")]
#         self.roll = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//ROLL")]
#         self.outOfRange = [elem.text for elem in self.root.findall(".//Corrected_Attitude//OUT_OF_RANGE")]
#         time = [elem.text for elem in self.root.findall(".//Corrected_Attitude//TIME")]
#
#         self.attitudeTime = []
#         for t_ in time:
#             try:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
#             except:
#                 t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
#             self.attitudeTime.append(ConvertTime2Second(t__))
#
#         if self.debug:
#             logging.info("starTrackerUsed:{} , yaw:{} , pitch:{} , roll:{} , outOfRange:{} , attitudeTime:{}".format(
#                 self.starTrackerUsed, len(self.yaw), len(self.pitch), len(self.roll), len(self.outOfRange),
#                 len(self.attitudeTime)))
#         return


class cGetHiriseMetadata:
    pass


if __name__ == '__main__':
    cGetDGMetadata(anc_file='/home/saif/PycharmProjects/Geospatial-COSICorr3D/tests/test_dataset/WV2.XML', debug=True)
    # cGetSpot67Metadata(anc_file='/home/saif/PycharmProjects/Geospatial-COSICorr3D/tests/test_dataset/SPOT6.XML')

    pass
