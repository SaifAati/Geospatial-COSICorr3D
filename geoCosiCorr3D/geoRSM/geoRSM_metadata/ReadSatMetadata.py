"""
Author: Saif Aati (saif@caltech.edu)
Read QuickBird/WorldView1-2-3-4 Image MetaData ASCII and XML files
"""
import logging

import numpy as np
import datetime, os
import xml.etree.ElementTree as ET
from decimal import *
from pathlib import Path

from geoCosiCorr3D.georoutines.file_cmd_routines import FilesInDirectory
import geoCosiCorr3D.geoRSM.misc as misc


class SatMetadata:
    def __init__(self, md_fn: str, debug: bool = False):
        self.debug = debug
        self.md_fn = md_fn

        self.platform = None
        self.imaging_time = None
        self.imaging_date = None
        self.sun_az = None
        self.sun_elev = None

        pass


class cGetSpot15Metadata:
    def __init__(self, dmpFile: str, debug: bool = False, band: int = 0):

        self.debug = debug
        self.dmpFile = dmpFile
        self.band = band  # 0:panchromatic
        # create element tree object
        self.tree = ET.parse(self.dmpFile)

        # get root element
        self.root = self.tree.getroot()
        self.childTags = [elem.tag for elem in self.root]
        if self.debug:
            logging.info("____START PARSING SPOT METADATA______")
            logging.info(self.childTags)

        self.lookAngles = np.zeros((2, 2))
        self.mirrorStep = None
        self.GetSpotSensor()

        self.sunAz = None
        self.sunElev = None
        self.mission = None
        self.instrument = None
        self.GetSourceInformation()

        self.linePeriod = None
        self.sceneCenterTime = None
        self.sceneCenterLine = None
        self.GetSpotSensorConfig()

        self.satPosition = None
        self.satVelocity = None
        self.ephTime = []
        self.GetSpotEphemeris()

        self.nbCols = None
        self.nbRows = None
        self.nbBands = None
        self.GetSpotRasterDimensions()

        if self.mission == 5:
            self.attitudeTime = None
            self.yaw = None
            self.pitch = None
            self.roll = None
            self.outOfRange = None
            self.starTrackerUsed = None
            if self.debug:
                logging.info("--- SPOT-5: get Corrected Attitude ---")
            self.GetSpotCorrectedAttitude()
        else:
            if self.debug:
                logging.info("--- SPOT-1-4: get Raw Speed Attitude ---")
            self.timeAttAng = []
            self.yawAttAng = []
            self.rollAttAng = []
            self.pitchAttAng = []
            self.outOfRangeAttAng = []
            self.timeSpeedAtt = []
            self.yawSpeedAtt = []
            self.pitchSpeedAtt = []
            self.rollSpeedAtt = []
            self.outOfRangeSpeedAtt = []
            self.GetSpotRawSpeedAttitude()
        if self.debug:
            logging.info("____END PARSING SPOT METADATA______")

    def GetSpotSensor(self):
        if self.debug:
            logging.info("--- Get Spot Sensor --- ")

        PSI_X = [float(elem.text) for elem in self.root.findall(".//Look_Angles_List//Look_Angles//PSI_X")]
        PSI_Y = [float(elem.text) for elem in self.root.findall(".//Look_Angles_List//Look_Angles//PSI_Y")]
        # print(self.lookAngles.shape)
        # self.lookAngles[:, 0] = PSI_X
        # self.lookAngles[:, 1] = PSI_Y
        self.lookAngles = np.array([PSI_X, PSI_Y]).T
        try:
            self.mirrorStep = [int(elem.text) for elem in self.root.findall(".//Mirror_Position//STEP_COUNT")][0]
        except:
            self.mirrorStep = 0
        if self.debug:
            logging.info("lookAngles:{}".format(self.lookAngles.shape))
            logging.info("mirrorStep:{}".format(self.mirrorStep))

    def GetSourceInformation(self):
        if 'Dataset_Sources' in self.childTags:
            self.sunAz = float([elem.text for elem in
                                self.root.findall("Dataset_Sources/Source_Information/Scene_Source/SUN_AZIMUTH")][0])
            self.sunElev = float([elem.text for elem in
                                  self.root.findall("Dataset_Sources/Source_Information/Scene_Source/SUN_ELEVATION")][
                                     0])
            self.mission = int([elem.text for elem in
                                self.root.findall("Dataset_Sources/Source_Information/Scene_Source/MISSION_INDEX")][
                                   0])
            self.instrument = int([elem.text for elem in
                                   self.root.findall(
                                       "Dataset_Sources/Source_Information/Scene_Source/INSTRUMENT_INDEX")][
                                      0])
            self.instrumentName = [elem.text for elem in
                                   self.root.findall(
                                       "Dataset_Sources/Source_Information/Scene_Source/INSTRUMENT")][0]
            self.sensorCode = [elem.text for elem in
                               self.root.findall(
                                   "Dataset_Sources/Source_Information/Scene_Source/SENSOR_CODE")][0]
            self.incidenceAngle = [float(elem.text) for elem in self.root.findall(".//INCIDENCE_ANGLE")][0]

            self.date = [elem.text for elem in self.root.findall(".//IMAGING_DATE")][0]
            self.time = [elem.text for elem in self.root.findall(".//IMAGING_TIME")][0]
            if self.debug:
                logging.info("sunAz:{},sunElev:{},mission:{},instrument:{}, incidenceAngle:{}".format(self.sunAz,
                                                                                                      self.sunElev,
                                                                                                      self.mission,
                                                                                                      self.instrument,
                                                                                                      self.incidenceAngle))

    def GetSpotSensorConfig(self):
        """
        linePeriod: scalar
        sceneCenterTime:  time in seconds at the scene center
        sceneCenterLine: line at the scene center
        :return:
        """
        if self.debug:
            logging.info("--- Get Spot Sensor Configuration ---")
        self.linePeriod = \
            [float(elem.text) for elem in self.root.findall(".//Sensor_Configuration//Time_Stamp//LINE_PERIOD")][0]
        sceneCenterTime = \
            [elem.text for elem in self.root.findall(".//Sensor_Configuration//Time_Stamp//SCENE_CENTER_TIME")][0]
        if self.debug:
            logging.info(sceneCenterTime)
        try:
            sceneCenterTime = datetime.datetime.strptime(sceneCenterTime, '%Y-%m-%dT%H:%M:%S.%fZ')
        except:
            sceneCenterTime = datetime.datetime.strptime(sceneCenterTime, '%Y-%m-%dT%H:%M:%S.%f')

        self.sceneCenterTime = misc.time_to_second(sceneCenterTime)
        self.sceneCenterLine = [int(elem.text) for elem in
                                self.root.findall(".//Sensor_Configuration//Time_Stamp//SCENE_CENTER_LINE")][0]
        if self.debug:
            logging.info("linePeriod:{},sceneCenterTime:{},sceneCenterLine:{}".format(self.linePeriod,
                                                                                      self.sceneCenterTime,
                                                                                      self.sceneCenterLine))

        return

    def GetSpotEphemeris(self):
        """
        time: array containing the time when ephemeris are measured
        position: satellite center of mass position in cartesian coordinate system
        velocity: satellite velocity
        :return:
        """
        if self.debug:
            logging.info("--- Get Spot Ephemeris ---")
        self.dorisUsed = [elem.text for elem in self.root.findall(".//DORIS_USED")][0]

        X = [float(elem.text) for elem in self.root.findall(".//Points//Location//X")]
        Y = [float(elem.text) for elem in self.root.findall(".//Points//Location//Y")]
        Z = [float(elem.text) for elem in self.root.findall(".//Points//Location//Z")]
        self.satPosition = np.array([X, Y, Z]).T

        vX = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//X")]
        vY = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//Y")]
        vZ = [float(elem.text) for elem in self.root.findall(".//Points//Velocity//Z")]
        self.satVelocity = np.array([vX, vY, vZ]).T

        time = [elem.text for elem in self.root.findall(".//Points//TIME")]
        for t_ in time:
            try:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
            self.ephTime.append(misc.time_to_second(t__))
        if self.debug:
            logging.info("dorisUsed:{},satPosition:{} ,satVelocity:{},ephTime:{}".format(self.dorisUsed,
                                                                                         self.satPosition.shape,
                                                                                         self.satVelocity.shape,
                                                                                         len(self.ephTime)))
        return

    def GetSpotRasterDimensions(self):
        self.nbCols = [int(elem.text) for elem in self.root.findall(".//NCOLS")][0]
        self.nbRows = [int(elem.text) for elem in self.root.findall(".//NROWS")][0]
        self.nbBands = [int(elem.text) for elem in self.root.findall(".//NBANDS")][0]
        if self.debug:
            logging.info("nbCols:{},nbRows:{},nbBands:{}".format(self.nbCols, self.nbRows, self.nbBands))

        return

    def GetSpotRawSpeedAttitude(self):
        """
        timeCst: an array containing time when absolute attitudes are measured
         yawCst : array containing yaw absolute values
         pitchCst: array containing pitch absolute values
         rollCst: array containing roll absolute values

         timeVel: an array containing time when speed attitudes are measured
         yawVel : array containing yaw speed values
         pitchVel : array containing pitch speed values
         rollVel : array containing roll speed values

        :return:
        """
        if self.debug:
            logging.info("--- Get SPOT {} Speed Attitude ---".format(self.mission))

        #### Angles_List  ####
        self.yawAttAng = [float(elem.text) for elem in
                          self.root.findall(
                              ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//YAW")]
        self.pitchAttAng = [float(elem.text) for elem in
                            self.root.findall(
                                ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//PITCH")]
        self.rollAttAng = [float(elem.text) for elem in
                           self.root.findall(
                               ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//ROLL")]

        timeAtt = [elem.text for elem in
                   self.root.findall(".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//TIME")]
        for t_ in timeAtt:
            try:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
            self.timeAttAng.append(misc.time_to_second(t__))
        self.outOfRangeAttAng = [elem.text for elem in
                                 self.root.findall(
                                     ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angles_List//Angles//OUT_OF_RANGE")]

        if self.debug:
            logging.info("yawAttAng:{} , pitchAttAng:{}, rollAttAng:{} ,timeAttAng:{},outOfRangeAttAng:{} ".format(
                len(self.yawAttAng),
                len(self.pitchAttAng),
                len(self.rollAttAng),
                len(self.timeAttAng),
                len(self.outOfRangeAttAng)))

        #######  Angular_Speeds_List ##########
        self.yawSpeedAtt = [float(elem.text) for elem in
                            self.root.findall(
                                ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//YAW")]
        self.pitchSpeedAtt = [float(elem.text) for elem in
                              self.root.findall(
                                  ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//PITCH")]

        self.rollSpeedAtt = [float(elem.text) for elem in
                             self.root.findall(
                                 ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//ROLL")]

        timeSpeed = [elem.text for elem in
                     self.root.findall(
                         ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//TIME")]
        for t_ in timeSpeed:
            try:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
            self.timeSpeedAtt.append(misc.time_to_second(t__))
        self.outOfRangeSpeedAtt = [elem.text for elem in
                                   self.root.findall(
                                       ".//Satellite_Attitudes//Raw_Attitudes//Aocs_Attitude//Angular_Speeds_List//Angular_Speeds//OUT_OF_RANGE")]

        if self.debug:
            logging.info(
                "yawSpeedAtt:{} ,pitchSpeedAtt:{},rollSpeedAtt:{} ,timeSpeedAtt:{},outOfRangeSpeedAtt:{} ".format(
                    len(self.yawSpeedAtt),
                    len(self.pitchSpeedAtt),
                    len(self.rollSpeedAtt),
                    len(self.timeSpeedAtt),
                    len(self.outOfRangeSpeedAtt)))

        return

    def GetSpotCorrectedAttitude(self):

        """
        time -- an array containing time when attitude is measured
        yaw -- array containing yaw absolute values
        pitch -- array containing pitch absolute values
        roll -- array containing roll absolute values
        :return:
        """
        self.starTrackerUsed = [elem.text for elem in self.root.findall(".//STAR_TRACKER_USED")][0]
        # self.correctedAtt = [elem.text for elem in self.root.findall(".//Corrected_Attitude")]
        self.yaw = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//YAW")]
        self.pitch = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//PITCH")]
        self.roll = [float(elem.text) for elem in self.root.findall(".//Corrected_Attitude//ROLL")]
        self.outOfRange = [elem.text for elem in self.root.findall(".//Corrected_Attitude//OUT_OF_RANGE")]
        time = [elem.text for elem in self.root.findall(".//Corrected_Attitude//TIME")]

        self.attitudeTime = []
        for t_ in time:
            try:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                t__ = datetime.datetime.strptime(t_, '%Y-%m-%dT%H:%M:%S.%f')
            self.attitudeTime.append(misc.time_to_second(t__))

        if self.debug:
            logging.info("starTrackerUsed:{} , yaw:{} , pitch:{} , roll:{} , outOfRange:{} , attitudeTime:{}".format(
                self.starTrackerUsed, len(self.yaw), len(self.pitch), len(self.roll), len(self.outOfRange),
                len(self.attitudeTime)))
        return


class cGetSpot67Metadata():
    def __init__(self, dmpXml: str):
        # create element tree object
        self.dmpFile = dmpXml
        self.tree = ET.parse(self.dmpFile)

        # get root element
        self.root = self.tree.getroot()
        self.childTags = [elem.tag for elem in self.root]
        logging.info(f"{self.__class__.__name__}: {self.childTags}")

        self.instrument = None
        self.instrumentIndex = None
        self.imagingTime = None
        self.imagingDate = None
        self.sunAz = None
        self.sunElev = None
        self.mission = None
        self.GetSourceInformation()

        self.nbCols = None
        self.nbRows = None
        self.nbBands = None
        self.GetSpotRasterDimensions()

        self.linePeriod = None
        self.startTime = None
        self.endTime = None
        self.focal = None
        self.szCol = None
        self.szRow = None
        self.GetSpotSensorConfig()

        self.lineOfSight = np.empty((4, 1))
        # self.position = None
        # self.velocity = None
        # self.ephTime = None
        self.get_eph()

        self.lookAngles = np.copy(self.lineOfSight)
        # self.Q0 = None
        # self.Q1 = None
        # self.Q2 = None
        # self.Q3 = None
        # self.QTime = None

        self.get_attitude()

    def GetSourceInformation(self):
        if 'Dataset_Sources' in self.childTags:
            self.instrument = [elem.text for elem in self.root.findall(".//INSTRUMENT")][0]
            self.instrumentIndex = [elem.text for elem in self.root.findall(".//INSTRUMENT_INDEX")][0]
            self.imagingTime = [elem.text for elem in self.root.findall(".//IMAGING_TIME")][0]
            self.imagingDate = [elem.text for elem in self.root.findall(".//IMAGING_DATE")][0]
        logging.info("mission:{},instrument:{},imagingDate:{}, imagingTime:{},".format(self.instrument,
                                                                                       self.instrumentIndex,
                                                                                       self.imagingDate,
                                                                                       self.imagingTime))

        self.sunAz = float([elem.text for elem in self.root.findall(".//Use_Area//SUN_AZIMUTH")][0])
        self.sunElev = float([elem.text for elem in self.root.findall(".//Use_Area//SUN_ELEVATION")][0])
        self.satElev = self.sunElev
        ## Taking only across_track gsd for both directions
        self.gsd_ACT = float([elem.text for elem in self.root.findall(".//Use_Area//GSD_ACROSS_TRACK")][0])
        self.gsd_ALT = float([elem.text for elem in self.root.findall(".//Use_Area//GSD_ALONG_TRACK")][0])
        self.meanGSD = np.mean([[self.gsd_ALT, self.gsd_ACT]])
        self.time = [elem.text for elem in self.root.findall(".//Use_Area//TIME")][0]
        self.satAlt = float([elem.text for elem in self.root.findall(".//Use_Area//SATELLITE_ALTITUDE")][0])
        self.azAngle = float([elem.text for elem in self.root.findall(".//Use_Area//AZIMUTH_ANGLE")][0])
        self.satAz = self.azAngle
        self.viewAngle = float([elem.text for elem in self.root.findall(".//Use_Area//VIEWING_ANGLE")][0])
        self.offNadir_angle = self.viewAngle
        self.incidenceAngle = float([elem.text for elem in self.root.findall(".//Use_Area//INCIDENCE_ANGLE")][0])
        logging.info(
            "sunAz:{:.6},"
            "sunElev:{:.6},"
            " meanGSD:{:.6}, "
            "time:{}, "
            "satAlt:{:.6},"
            " azAngle:{:.6}, "
            "viewAngle:{:.6}, "
            "incidenceAngle:{:.6}".format(
                self.sunAz,
                self.sunElev,
                self.meanGSD,
                self.time,
                self.satAlt,
                self.azAngle,
                self.viewAngle,
                self.incidenceAngle))
        return

    def GetSpotRasterDimensions(self):
        self.nbCols = [int(elem.text) for elem in self.root.findall(".//NCOLS")][0]
        self.nbRows = [int(elem.text) for elem in self.root.findall(".//NROWS")][0]
        self.nbBands = [int(elem.text) for elem in self.root.findall(".//NBANDS")][0]
        logging.info("nbCols:{},nbRows:{},nbBands:{}".format(self.nbCols, self.nbRows, self.nbBands))

        return

    def GetSpotSensorConfig(self):
        logging.info("--- Get Spot Sensor Configuration ---")

        startTime = [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Range/START")][0]
        startTime = datetime.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%fZ')
        self.startTime = misc.time_to_second(startTime)
        endTime = [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Range/END")][0]
        endTime = datetime.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%S.%fZ')
        self.endTime = misc.time_to_second(endTime)
        ##linePeriod in microsecond, need to convert it to second
        self.linePeriod = float(
            [elem.text for elem in self.root.findall(".//Refined_Model/Time/Time_Stamp/LINE_PERIOD")][0]) / 1000000
        # logging.info(startTime, endTime, linePeriod)
        logging.info("linePeriod:{},startTime:{},endTimeLine:{}".format(self.linePeriod,
                                                                        self.startTime,
                                                                        self.endTime))

        self.focal = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//FOCAL_LENGTH")][0])
        self.szCol = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_COL")][0])
        self.szRow = Decimal([elem.text for elem in self.root.findall(".//Refined_Model//DETECTOR_SIZE_ROW")][0])

        logging.info("focal[m]:{:.4}, detectorSize[m]:(col:{:.4},row:{:.4})".format(self.focal, self.szCol, self.szRow))

        return

    def get_eph(self):
        logging.info("--- Get Look angles ----")
        self.lineOfSight[0, 0] = float(
            [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/XLOS_0")][0])
        self.lineOfSight[1, 0] = float(
            [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/XLOS_1")][0])
        self.lineOfSight[2, 0] = float(
            [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/YLOS_0")][0])
        self.lineOfSight[3, 0] = float(
            [elem.text for elem in self.root.findall(".//Refined_Model//Polynomial_Look_Angles/YLOS_1")][0])
        logging.info("Look_Angles:\n{}".format(self.lineOfSight))

        temp_XYZ = np.array(
            [elem.text.split(" ") for elem in self.root.findall(".//Refined_Model/Ephemeris//Point/LOCATION_XYZ")],
            dtype=float)
        temp_V_XYZ = np.array(
            [elem.text.split(" ") for elem in self.root.findall(".//Refined_Model/Ephemeris//Point/VELOCITY_XYZ")],
            dtype=float)
        temp_ephTime = np.array([elem.text for elem in self.root.findall(".//Refined_Model/Ephemeris//Point/TIME")])

        eph_time = np.array(
            [(datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.datetime(1950, 1, 1)).total_seconds()
             for t in temp_ephTime])
        self.eph_tpv = np.hstack((eph_time[:, np.newaxis], temp_XYZ, temp_V_XYZ))

        logging.info(f"{self.__class__.__name__}: input ephemeris: {self.eph_tpv.shape}")
        return

    def get_attitude(self):
        quat_time_str = np.array(
            [elem.text for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/TIME")])

        quat_time = np.array(
            [(datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.datetime(1950, 1, 1)).total_seconds()
             for t in quat_time_str])

        Q0 = np.array([float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q0")])
        Q1 = np.array([float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q1")])
        Q2 = np.array([float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q2")])
        Q3 = np.array([float(elem.text) for elem in self.root.findall(".//Refined_Model/Attitudes//Quaternion/Q3")])

        self.quat_txyzs = np.vstack([quat_time, Q1, Q2, Q3, Q0]).T

        logging.info(f"{self.__class__.__name__}: Optimized input attitude: {self.quat_txyzs.shape}")


class cGetDGMetadata():
    def __init__(self, dgFile: str, debug: bool = False):
        """

        Args:
            dgFile: could be the *.xml file or one of these [*.ATT,*.EPH,*.GEO,*.IMD] files


        Returns:
        Notes:
            when one of these [*.ATT,*.EPH,*.GEO,*.IMD] files is passed as an argument,
            the other files should be in the same folder and having the same file name.

        """

        self.dgFile = dgFile
        self.debug = debug
        self.platform = None
        self.productCatalogId = None

        self.imgLevel = None

        self.bandId = None

        self.nbRows = None

        self.nbCols = None

        self.startTime = None

        self.avgLineRate = None

        self.sunAz = None

        self.sunElev = None

        self.scanDirection = None

        self.meanGSD = None

        self.satElev = None

        self.satAz = None

        self.offNadir_angle = None

        self.date_time_obj = None
        self.date = None
        self.startTimeEph = None

        self.dtEph = None

        self.nbPoints = None

        self.ephemeris = None

        self.startTimeAtt = None

        self.dtAtt = None

        self.nbPointsAtt = None

        self.cameraAttitude = []

        self.principalDistance = None

        self.detPitch = None
        self.detOrigin = []
        fileExtension = Path(self.dgFile).suffix
        if fileExtension in [".xml", ".XML"]:
            self.getDGFromXML()
        elif fileExtension in [".ATT", ".EPH", ".GEO", ".IMD"]:
            self.getDGFromASCII()
        else:
            raise ValueError("DG metadata file not :*.XML, *.ATT,*.EPH,*.GEO,*.IMD")
        if self.debug:
            logging.info("Platform:{}, "
                         "bandID:{}, "
                         "nbRows:{},"
                         "nbCols:{},"
                         "startTime:{},"
                         "self.avgLineRate:{}".format(self.platform,
                                                      self.bandId,
                                                      self.nbRows,
                                                      self.nbCols,
                                                      self.startTime,
                                                      self.avgLineRate))
            logging.info("sunAz:{},"
                         "sunElev:{},"
                         "satElev:{},"
                         "sunAz:{},"
                         "offNadir_angle:{},"
                         "scanDirection:{},"
                         "ID:{},"
                         "meanGSD:{} ".format(self.sunAz,
                                              self.sunElev,
                                              self.satElev,
                                              self.sunAz,
                                              self.offNadir_angle,
                                              self.scanDirection,
                                              self.productCatalogId,
                                              self.meanGSD))
            logging.info("--- Read Ephemeris ---")
            logging.info("--- Read Attitude ---")
            # logging.info(self.attitude)
            logging.info("--- Read Camera Geometry ---")
            logging.info("f[mm]:{},"
                         "camAttitude:{},"
                         "detectorOrig:{},"
                         "detectorPitch:{}".format(self.principalDistance,
                                                   self.cameraAttitude,
                                                   self.detOrigin,
                                                   self.detPitch))
        return

    def getDGFromXML(self):
        if self.debug:
            logging.info("--- Computing DG RSM from XML:")
        self.dmpFile = self.dgFile
        self.tree = ET.parse(self.dgFile)

        # get root element
        self.root = self.tree.getroot()
        self.childTags = [elem.tag for elem in self.root]
        if self.debug:
            logging.info(self.childTags)
        self.imgFootprint = []
        ulLon = [float(elem.text) for elem in self.root.findall(".//IMD//ULLON")][0]
        ulLat = [float(elem.text) for elem in self.root.findall(".//IMD//ULLAT")][0]

        urLon = [float(elem.text) for elem in self.root.findall(".//IMD//URLON")][0]
        urLat = [float(elem.text) for elem in self.root.findall(".//IMD//URLAT")][0]

        lrLon = [float(elem.text) for elem in self.root.findall(".//IMD//LRLON")][0]
        lrLat = [float(elem.text) for elem in self.root.findall(".//IMD//LRLAT")][0]

        llLon = [float(elem.text) for elem in self.root.findall(".//IMD//LLLON")][0]
        llLat = [float(elem.text) for elem in self.root.findall(".//IMD//LLLAT")][0]

        self.imgFootprint = [ulLon, ulLat, urLon, urLat, lrLon, lrLat, llLon, llLat]
        self.imgLevel = [elem.text for elem in self.root.findall(".//IMAGEDESCRIPTOR")][0]
        self.platform = [elem.text for elem in self.root.findall(".//SATID")][0]
        self.bandId = [elem.text for elem in self.root.findall(".//BANDID")][0]
        self.nbRows = [int(elem.text) for elem in self.root.findall(".//NUMROWS")][0]
        self.nbCols = [int(elem.text) for elem in self.root.findall(".//NUMCOLUMNS")][0]
        item_ = [elem.text for elem in self.root.findall(".//FIRSTLINETIME")][0]
        self.date = item_.split("T")[0]
        self.time = item_.split("T")[1]
        self.date_time_obj = datetime.datetime.strptime(item_, '%Y-%m-%dT%H:%M:%S.%fZ')

        self.startTime = misc.time_to_second(self.date_time_obj, 2000)
        self.avgLineRate = [float(elem.text) for elem in self.root.findall(".//AVGLINERATE")][0]

        self.sunAz = [float(elem.text) for elem in self.root.findall(".//MEANSUNAZ")][0]
        self.sunElev = [float(elem.text) for elem in self.root.findall(".//MEANSUNEL")][0]
        self.satElev = [float(elem.text) for elem in self.root.findall(".//MEANSATEL")][0]
        self.sunAz = [float(elem.text) for elem in self.root.findall(".//MEANSATAZ")][0]
        self.offNadir_angle = [float(elem.text) for elem in self.root.findall(".//MEANOFFNADIRVIEWANGLE")][0]

        self.scanDirection = [elem.text for elem in self.root.findall(".//SCANDIRECTION")][0]

        self.productCatalogId = [elem.text for elem in self.root.findall(".//PRODUCTCATALOGID")][0]
        try:
            self.meanGSD = [float(elem.text) for elem in self.root.findall(".//MEANPRODUCTGSD")][0]
        except:
            self.meanGSD = [float(elem.text) for elem in self.root.findall(".//MEANCOLLECTEDCOLGSD")][0]

        item_ = [elem.text for elem in self.root.findall(".//EPH//STARTTIME")][0]
        date_time_obj = datetime.datetime.strptime(item_, '%Y-%m-%dT%H:%M:%S.%fZ')
        self.startTimeEph = misc.time_to_second(date_time_obj, 2000)
        self.dtEph = [float(elem.text) for elem in self.root.findall(".//EPH//TIMEINTERVAL")][0]
        self.nbPoints = [int(elem.text) for elem in self.root.findall(".//EPH//NUMPOINTS")][0]
        self.ephemeris = np.zeros((self.nbPoints, 6))
        ephList = [elem.text for elem in self.root.findall(".//EPH//EPHEMLIST")]
        """
             retrieve ephemeris data:
             start_eph: date UTC (in second) of first ephemeris
             dt_eph: sampling period (in second) of ephemeris measurment
             ephemeris: ephemeris values (position and velocity) - 2D array ([X, Y, Z, Vx, Vy, Vz] x NB measures)
             expressed in ECF (Earth Centered Fixed reference)
         """
        self.ephemeris = np.zeros((self.nbPoints, 6))
        for line_ in ephList:
            index, X, Y, Z, VX, VY, VZ, _, _, _, _, _, _ = line_.split()
            self.ephemeris[int(float(index)) - 1, 0] = float(X)
            self.ephemeris[int(float(index)) - 1, 1] = float(Y)
            self.ephemeris[int(float(index)) - 1, 2] = float(Z)
            self.ephemeris[int(float(index)) - 1, 3] = float(VX)
            self.ephemeris[int(float(index)) - 1, 4] = float(VY)
            self.ephemeris[int(float(index)) - 1, 5] = float(VZ)

        # self.dtEph = [float(elem.text) for elem in self.root.findall(".//EPH//TIMEINTERVAL")][0]
        item_ = [elem.text for elem in self.root.findall(".//ATT//STARTTIME")][0]
        date_time_obj = datetime.datetime.strptime(item_, '%Y-%m-%dT%H:%M:%S.%fZ')
        self.startTimeAtt = misc.time_to_second(date_time_obj, 2000)
        self.dtAtt = [float(elem.text) for elem in self.root.findall(".//ATT//TIMEINTERVAL")][0]
        self.nbPointsAtt = [int(elem.text) for elem in self.root.findall(".//ATT//NUMPOINTS")][0]
        self.attitude = np.zeros((self.nbPointsAtt, 4))
        attList = [elem.text for elem in self.root.findall(".//ATT//ATTLIST")]

        """
            RETRIEVE SATELLITE ATTITUDE DATA:
            start_att: date UTC (in second) of first attitude
            dt_att: sampling period (in second) of attitude measurment
            attitude: satellite attitude values (quaternions) - 2D array ([q1, q2, q3, q4] x NB measures). 
            Note: q4 scalar part
            Describe the rotation of the satellite coordinate system relative to ECF.
        """

        for line_ in attList:
            index, q1, q2, q3, q4, _, _, _, _, _, _, _, _, _, _ = line_.split()
            self.attitude[int(float(index)) - 1, 0] = float(q1)
            self.attitude[int(float(index)) - 1, 1] = float(q2)
            self.attitude[int(float(index)) - 1, 2] = float(q3)
            self.attitude[int(float(index)) - 1, 3] = float(q4)

        """
            RETRIEVE CAMERA GEOMETRY INFO:
            detOrigin: two elements array representing the location of the first CCD in the camera reference system (in millimeters)
            detPitch: distance, in millimeter, between two adjacents CCD centers
            principalDist: Principal distance of the camera in millimeter
            geoQuat: 4 elements array. Quaternions representing the rotation of the camera reference system relative to the satellite ref
        """
        self.principalDistance = [float(elem.text) for elem in self.root.findall(".//GEO/PRINCIPAL_DISTANCE//PD")][0]
        qcs1 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS1")][0]
        qcs2 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS2")][0]
        qcs3 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS3")][0]
        qcs4 = [float(elem.text) for elem in self.root.findall(".//GEO/CAMERA_ATTITUDE//QCS4")][0]
        self.cameraAttitude = [qcs1, qcs2, qcs3, qcs4]
        xOrigin = [float(elem.text) for elem in self.root.findall(".//GEO//DETORIGINX")][0]
        yOrigin = [float(elem.text) for elem in self.root.findall(".//GEO//DETORIGINY")][0]
        self.detPitch = [float(elem.text) for elem in self.root.findall(".//GEO//DETPITCH")][0]
        self.detOrigin = [xOrigin, yOrigin]

        return

    def getDGFromASCII(self):
        if self.debug:
            logging.info("--- Computing DG RSM from ASCII:")
        dirPath = os.path.dirname(self.dgFile)
        files = FilesInDirectory(path=dirPath)
        filesExtentions = [item_.split('.')[-1] for item_ in files]
        imdFileFlag = False
        attFileFlag = False
        ephFileFlag = False
        geoFileFlag = False
        for file_ in files:
            if file_.split('.')[-1] == "IMD" in filesExtentions and os.path.exists(os.path.join(dirPath, file_)):
                self.imdFilePath = os.path.join(dirPath, file_)
                imdFileFlag = True
            if file_.split('.')[-1] == "ATT" and os.path.exists(os.path.join(dirPath, file_)):
                self.attFilePath = os.path.join(dirPath, file_)
                attFileFlag = True

            if file_.split('.')[-1] == "EPH" and os.path.exists(os.path.join(dirPath, file_)):
                self.ephFilePath = os.path.join(dirPath, file_)
                ephFileFlag = True

            if file_.split('.')[-1] == "GEO" and os.path.exists(os.path.join(dirPath, file_)):
                self.geoFilePath = os.path.join(dirPath, file_)
                geoFileFlag = True
        if imdFileFlag == False:
            raise ValueError("DG IMD metadata file not found")
        if attFileFlag == False:
            raise ValueError("DG ATT metadata file not found")
        if ephFileFlag == False:
            raise ValueError("DG EPH metadata file not found")
        if geoFileFlag == False:
            raise ValueError("DG GEO metadata file not found")

        self.__Read_IMD()
        self.__Read_EPH()
        self.__Read_ATT()
        self.__Read_GEO()

    def __LookupInfile(self, file, lookup):
        line_ = None
        num_ = None
        with open(file, 'r') as myFile:
            for num, line in enumerate(myFile, 1):
                if lookup in line:
                    line_ = line
                    num_ = num
                    item_ = line_.split("=")[1].split(";")[0]
                    return line_, num_, item_

        return line_, num_, 0

    def __Read_IMD(self):
        """
        Author: Saif Aati (saif@caltech.edu)
        read the QuickBird/WorldView Image Meta Data (IMD) ASCII file
        Returns:
        Notes:
            imgLevel: image processing Level (Basic1B, Level0, Metadata, Standard2A, ORStandard2A, OrthoRectified3, Stereo1B
            bandId: Spectral band description (P, Multi, RGB, NRG, BGRN)
            ncols: sample number of the image
            nrows: line number of the image
            resx_anc: nominal average ground resolution in column direction (in meter)
            resy_anc: nominal average ground resolution in line direction (in meter)
            sunAzimuth: sun azimuth (degree) at the time of acquisition
            sunElevation: sun elevation (degree) at the time of acquisition
            start_img: time (in second since 1/1/2000) of the acquisition of the first image line
            avgLineRate: number of line acquired per second
            scanDirection: "Forward" or "Reverse". Indicate whether the top line of the image was acquired first or last
        """
        lookupList = ["imageDescriptor", "bandId", "numRows", "numColumns", "firstLineTime", "avgLineRate", "sunAz",
                      "meanSunAz", "sunEl", "meanSunEl", "scanDirection", "meanProductGSD", "satId", "meanSatEl",
                      "meanSatAz", "meanOffNadirViewAngle", "productCatalogId"]
        for lookup_ in lookupList:
            line, num, item_ = self.__LookupInfile(file=self.imdFilePath, lookup=lookup_)

            if not line == None:
                if lookup_ == "imageDescriptor":
                    self.imgLevel = item_
                if lookup_ == "satId":
                    itemTemp = item_.replace(" ", "")
                    self.platform = itemTemp.replace('"', '')

                if lookup_ == "bandId":
                    self.bandId = item_
                if lookup_ == "numRows":
                    self.nbRows = int(item_)
                if lookup_ == "numColumns":
                    self.nbCols = int(item_)
                if lookup_ == "firstLineTime":
                    self.date_time_obj = datetime.datetime.strptime(item_, ' %Y-%m-%dT%H:%M:%S.%fZ')
                    self.startTime = misc.time_to_second(self.date_time_obj, 2000)
                if "avgLineRate" == lookup_:
                    self.avgLineRate = float(item_)
                if "sunAz" == lookup_:
                    self.sunAz = item_
                if "meanSunAz" == lookup_:
                    self.sunAz = item_
                if "sunEl" == lookup_:
                    self.sunElev = item_
                if "meanSunEl" == lookup_:
                    self.sunElev = item_
                if "scanDirection" == lookup_:
                    itemTemp = item_.replace(" ", "")
                    self.scanDirection = itemTemp.replace('"', '')
                if "meanSatEl" == lookup_:
                    self.satElev = item_

                if "meanSatAz" == lookup_:
                    self.satAz = item_
                if "meanOffNadirViewAngle" in lookup_:
                    itemTemp = item_.replace(" ", "")
                    self.offNadir_angle = itemTemp.replace('"', '')
                if "productCatalogId" in lookup_:
                    itemTemp = item_.replace(" ", "")
                    self.productCatalogId = itemTemp.replace('"', '')
                if "meanProductGSD" in lookup_:
                    itemTemp = item_.replace(" ", "")
                    self.meanGSD = itemTemp.replace('"', '')

        return

    def __Read_EPH(self):
        """
        Author: Saif Aati (saif@caltech.edu)
        rRad the QuickBird/WorldView Ephemeris (EPH) ASCII file and retrieve the needed data
        Returns:
        Notes:
            start_eph: time (in second since 1/1/2000) of the first ephemeris measured
            dt_eph: time interval between ephemeris measures
            ephemeris: ephemeris measurements : Array(n,6) [X, Y, Z, Vx, Vy, Vz] in ECF

        """
        lookupList = ["startTime", "timeInterval", "numPoints", "ephemList"]
        ephemNum = None
        for lookup_ in lookupList:
            line, num, item_ = self.__LookupInfile(file=self.ephFilePath, lookup=lookup_)
            if lookup_ == "startTime":
                date_time_obj = datetime.datetime.strptime(item_, ' %Y-%m-%dT%H:%M:%S.%fZ')
                self.startTimeEph = misc.time_to_second(date_time_obj, 2000)
            if lookup_ == lookupList[1]:
                self.dtEph = float(item_)
            if lookup_ == lookupList[2]:
                self.nbPoints = int(item_)
            if lookup_ == lookupList[3]:
                ephemNum = int(num)

        self.ephemeris = np.zeros((self.nbPoints, 6))
        i = 0
        with open(self.ephFilePath, 'r') as myFile:
            for num_, line_ in enumerate(myFile, 1):
                if num_ > ephemNum and num_ <= self.nbPoints + ephemNum:
                    tempList = line_.split("(")[1].split(",")
                    self.ephemeris[i, 0] = float(tempList[1])
                    self.ephemeris[i, 1] = float(tempList[2])
                    self.ephemeris[i, 2] = float(tempList[3])
                    self.ephemeris[i, 3] = float(tempList[4])
                    self.ephemeris[i, 4] = float(tempList[5])
                    self.ephemeris[i, 5] = float(tempList[6])
                    i += 1

        return

    def __Read_ATT(self):
        """
        Author: Saif Aati (saif@caltech.edu)
        Read the QuickBird/WorldView Attitude (ATT) ASCII file and retrieve the needed data
        Returns:
        Notes:
            - start_att: time (in second since 1/1/2000) of the first attitude measured
            - dt_att: time interval between attitude measures
            - attitude: attitude measurements quaternions: Array(n, 4) [q1, q2, q3, q4] in ECF
        """
        lookupList = ["startTime", "timeInterval", "numPoints", "attList"]
        attNum = None
        for lookup_ in lookupList:
            line, num, item_ = self.__LookupInfile(file=self.attFilePath, lookup=lookup_)
            if lookup_ == "startTime":
                date_time_obj = datetime.datetime.strptime(item_, ' %Y-%m-%dT%H:%M:%S.%fZ')
                self.startTimeAtt = misc.time_to_second(date_time_obj, 2000)
            if lookup_ == lookupList[1]:
                self.dtAtt = float(item_)
            if lookup_ == lookupList[2]:
                self.nbPointsAtt = int(item_)
            if lookup_ == lookupList[3]:
                attNum = int(num)
        self.attitude = np.zeros((self.nbPointsAtt, 4))
        i = 0
        with open(self.attFilePath, 'r') as myFile:
            for num_, line_ in enumerate(myFile, 1):
                if num_ > attNum and num_ <= self.nbPointsAtt + attNum:
                    tempList = line_.split("(")[1].split(",")
                    self.attitude[i, 0] = float(tempList[1])
                    self.attitude[i, 1] = float(tempList[2])
                    self.attitude[i, 2] = float(tempList[3])
                    self.attitude[i, 3] = float(tempList[4])
                    i += 1
        return

    def __Read_GEO(self):
        """
        Author: Saif Aati (saif@caltech.edu)
        Read the QuickBird/WorldView Geometric calibration (GEO) ASCII file and retrieve the needed data
        Returns:
        Notes:

            - detOrigin: X and Y coordinates of the pixel 0 of the linear detector array in the camera coordinate system,
             in mm
            - detPitch: The pitch or pixel spacing of the detector in the detector Y direction, in mm.
            This is the distance between centers of adjacent pixels in the array.
            - principalDist: Principal distance of the camera (in mm) ~ focal length of the telescope
            - geoQuat: The unit quaternion for the attitude of the camera coordinate system in the spacecraft body
            system, i.e., the quaternion for the rotation of the spacecraft body frame into the virtual frame.
        """

        lookupList = ["PD =", "qcs1 =", "qcs2 =", "qcs3 =", "qcs4 =", "detOriginX =", "detOriginY =", "detPitch = "]

        for lookup_ in lookupList:
            line, num, item_ = self.__LookupInfile(file=self.geoFilePath, lookup=lookup_)

            if not line == None:
                if lookup_ == lookupList[0]:
                    self.principalDistance = float(item_)
                if lookup_ == lookupList[1]:
                    self.cameraAttitude.append(float(item_))
                if lookup_ == lookupList[2]:
                    self.cameraAttitude.append(float(item_))
                if lookup_ == lookupList[3]:
                    self.cameraAttitude.append(float(item_))
                if lookup_ == lookupList[4]:
                    self.cameraAttitude.append(float(item_))
                if lookup_ == lookupList[5]:
                    self.detOrigin.append(float(item_))
                if lookup_ == lookupList[6]:
                    self.detOrigin.append(float(item_))
                if lookup_ == lookupList[7]:
                    self.detPitch = float(item_)
