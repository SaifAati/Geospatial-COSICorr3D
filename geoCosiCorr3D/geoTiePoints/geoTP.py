"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import dataclasses
import logging
import os, sys, ctypes, rasterio

import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from pathlib import Path
from geoCosiCorr3D.geoConfig import cgeoCfg

from typing import Dict, Optional, List, Any
import cv2

geoCfg = cgeoCfg()

LIB_GEOSIFT = ctypes.CDLL(geoCfg.geoSIFTLib)
LIB_RANSAC = ctypes.CDLL(geoCfg.geoRansacLib)


@dataclasses.dataclass
class Cv2Features:
    features: np.ndarray
    kps: tuple
    des: np.ndarray


class SiftKpDistributionCv2:

    def __init__(self,
                 input_img: str,
                 band_idx: Optional[int] = 1,
                 max_kps: Optional[int] = None,
                 offset=None):

        self.input_img = input_img
        self.raster_info = geoRT.cRasterInfo(self.input_img)
        self.band_idx = band_idx
        self.max_kps = max_kps
        if offset is None:
            self.offset = [0, 0]
        else:
            self.offset = offset

    def __call__(self) -> Cv2Features:
        self.cv2_sift_extractor()
        self.features[:, 0] += self.offset[0]
        self.features[:, 1] += self.offset[0]
        return Cv2Features(self.features, self.kps, self.des)

    def cv2_sift_extractor(self, mask_path=None):
        """
        Detect SIFT keypoints in a single input grayscale image using OpenCV
        Args:
            mask_path (optional): path to npy binary mask, to restrict the search of keypoints to a certain area,
                                  parts of the mask with 0s are not explored
        Returns:
            features: Nx132 array, where N is the number of SIFT keypoints detected in image i
                      each row/keypoint is represented by 132 values:
                      (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
            n_kp: integer, number of keypoints detected

        """

        img = self.raster_info.image_as_array(self.band_idx)
        self.img = img.astype(np.uint8)

        sift = cv2.SIFT_create()

        mask = None if mask_path is None else np.load(mask_path, mmap_mode='r').astype(np.uint8)
        kps, des = sift.detectAndCompute(self.img, mask)
        max_sift_kps_per_img = 10000
        if kps:
            keypoints = sorted(kps, key=lambda x: (x.size, x.response), reverse=True)
            top_keypoints = keypoints[:max_sift_kps_per_img]
            kps, des = sift.compute(self.img, top_keypoints)

        self.features = np.array([[*k.pt, k.size, k.angle, *d] for k, d in zip(kps, des)])
        self.kps = kps
        self.des = des
        return


class Cv2DescriptorMatcher:
    def __init__(self, features_1: Cv2Features, features_2: Cv2Features):
        # FLANN matcher: https://www.fit.vutbr.cz/~ibarina/pub/VGE/reading/flann_manual-1.6.pdf
        # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        self.features_1 = features_1
        self.features_2 = features_2

    def __call__(self) -> np.ndarray:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=200)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        corresp = flann.knnMatch(self.features_1.features[:, 4:].astype(np.float32),
                                 self.features_2.features[:, 4:].astype(np.float32), k=2)
        matches_ij = np.array([[m.queryIdx, m.trainIdx] for m, n in corresp])
        matches = np.zeros((matches_ij.shape[0], 4))
        for index, match in enumerate(matches_ij):
            matches[index, 0] = self.features_1.features[match[0], 0]
            matches[index, 1] = self.features_1.features[match[0], 1]
            matches[index, 2] = self.features_2.features[match[1], 0]
            matches[index, 3] = self.features_2.features[match[1], 1]

        return matches




#######################################################################################################################
class Matching:

    def __init__(self, tpFile_i: str, tpFile_j: str):
        self.tp_i = np.load(tpFile_i)["arr_0"]
        self.tp_j = np.load(tpFile_j)["arr_0"]

    def openCV_Matching(self, ratioTh=0.8, matcher="flann"):
        """

        Args:
            ratioTh:
            matcher:

        Returns:

        """

        import cv2
        descriptors_i = self.tp_i[:, 4:].astype(np.float32)
        descriptors_j = self.tp_j[:, 4:].astype(np.float32)
        print(descriptors_i.shape, descriptors_j.shape)
        matcher = 'bruteforce'
        print(f'----matcher : {matcher}')
        if matcher == "bruteforce":
            # Bruteforce matcher
            bf = cv2.BFMatcher()
            # matches = bf.match(descriptors_i, descriptors_j)  # , k=2)
            # Sort them in the order of their distance.
            # matches = sorted(matches, key=lambda x: x.distance)

            # matches_ij = np.array([[match_obj.queryIdx, match_obj.trainIdx] for match_obj in matches ])
            # print(matches_ij.shape)
            # sys.exit()
            matches = bf.knnMatch(descriptors_i, descriptors_j, k=2)

        elif matcher == "flann":
            # FLANN matcher: https://www.fit.vutbr.cz/~ibarina/pub/VGE/reading/flann_manual-1.6.pdf
            # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=200)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_i, descriptors_j, k=2)
        else:
            logging.error('ERROR: OpenCV matcher is not recognized ! Valid values are "flann" or "bruteforce"')
            sys.exit('ERROR: OpenCV matcher is not recognized ! Valid values are "flann" or "bruteforce"')

        # NOTE the Ratio can be applied only when we use knnmatch
        # Apply ratio test as in Lowe's paper
        logging.info("ratio test:{}".format(ratioTh))
        matches_ij = np.array([[m.queryIdx, m.trainIdx] for m, n in matches if m.distance < ratioTh * n.distance])
        # n_matches_after_ratio_test = matches_ij.shape[0]

        self.matches = np.zeros((matches_ij.shape[0], 4))
        for index, match in enumerate(matches_ij):
            self.matches[index, 0] = self.tp_i[match[0], 0]
            self.matches[index, 1] = self.tp_i[match[0], 1]
            self.matches[index, 2] = self.tp_j[match[1], 0]
            self.matches[index, 3] = self.tp_j[match[1], 1]

        return

    def geoTpMatching(self, method='relative', sift_threshold=0.6, epi_threshold=10, F=None):
        """

        Args:
            method:
            sift_threshold:
            epi_threshold:
            F:

        Returns:

        """
        # Set expected args and return types
        kp1 = self.tp_i
        kp2 = self.tp_j
        LIB_GEOSIFT.matching.argtypes = (ndpointer(dtype=ctypes.c_float, shape=kp1.shape),
                                         ndpointer(dtype=ctypes.c_float, shape=kp2.shape),
                                         ctypes.c_uint,
                                         ctypes.c_uint,
                                         ctypes.c_uint,
                                         ctypes.c_uint,
                                         ctypes.c_float,
                                         ctypes.c_float,
                                         ndpointer(dtype=ctypes.c_double, shape=(5,)),
                                         ctypes.c_bool,
                                         ctypes.c_bool,
                                         ctypes.POINTER(ctypes.c_uint))

        LIB_GEOSIFT.matching.restype = ctypes.POINTER(ctypes.c_float)

        # Get info of descriptor size
        nb_sift_k1, descr = kp1.shape
        sift_offset = 4
        length_descr = descr - sift_offset

        # Transform information of method into boolean
        use_relative_method = (method == 'relative')

        # Format fundamental matrix
        use_fundamental_matrix = False
        coeff_mat = np.zeros(5)
        if F is not None:
            coeff_mat = np.asarray([F[0, 2], F[1, 2], F[2, 0], F[2, 1], F[2, 2]])
            use_fundamental_matrix = True

        # Create variables to be updated by function call
        nb_matches = ctypes.c_uint()

        # Call sift fonction from sift4ctypes.so
        matches_ptr = LIB_GEOSIFT.matching(kp1.astype('float32'),
                                           kp2.astype('float32'),
                                           length_descr,
                                           sift_offset,
                                           len(kp1),
                                           len(kp2),
                                           sift_threshold,
                                           epi_threshold,
                                           coeff_mat,
                                           use_fundamental_matrix,
                                           use_relative_method,
                                           ctypes.byref(nb_matches))

        # Transform result into a numpy array
        matches = np.asarray([matches_ptr[i] for i in range(nb_matches.value * 4)])
        self.matches = matches.reshape((nb_matches.value, 4))

        return


class cTpMatching(Matching):

    def __init__(self, tpFile_i: str, tpFile_j: str, oFolder: str, img_i: Optional[str] = None,
                 img_j: Optional[str] = None, matchKernel="openCV", svg=True, matchFilter=False):
        """

        Args:
            tpFile_i:
            tpFile_j:
            oFolder:
            img_i:  Optional(str): path to images for plotting purpose
            img_j: Optional(str): path to images for plotting purpose
            matchKernel: "openCV", "SIMD"
            matchFilter:
        """

        # self.tp_i =
        # self.tp_j =
        super().__init__(tpFile_i, tpFile_j)
        self.img_i = img_i
        self.img_j = img_j
        self.oFolder = oFolder

        # self.matchKernel = {"kernel": "openCV", "params": {"ratioTh": 0.8, "matcher": "flann"}}
        self.matchKernel = {"kernel": "openCV", "params": {"ratioTh": 0.8, "matcher": "bruteforce"}}

        if matchKernel == "SIMD":
            self.matchKernel = {"kernel": "SIMD",
                                "params": {"method": 'relative', "sift_threshold": 0.6, "epi_threshold": 10, "F": None}}
        if self.matchKernel["kernel"] == "openCV":
            logging.info("openCV matching:{}".format(self.matchKernel))
            self.openCV_Matching(self.matchKernel["params"]["ratioTh"], self.matchKernel["params"]["matcher"])

        elif self.matchKernel["kernel"] == "SIMD":
            self.geoTpMatching()
        nbTp_beforeFiltering = self.matches.shape[0]
        if matchFilter:
            # Geometric filtering using the Fundamental matrix
            logging.info("Geometric filtering using RANSAC + F-matrix ")
            # self.matches = TPFilter.TPFilter_F_RANSAC_CV(matches=self.matches)
            self.matches = TPFilter.TPFilter_F_RANSAC_SIMD(matches=self.matches)
            logging.info(
                "Tp before and after filtering :{} ----> {}".format(nbTp_beforeFiltering, self.matches.shape[0]))

        self.matchFile = os.path.join(self.oFolder, Path(tpFile_i).stem + "_VS_" + Path(tpFile_j).stem + "_matches.pts")
        np.savetxt(self.matchFile, self.matches)
        logging.info("#Matches:{}".format(self.matches.shape[0]))
        # if svg:
        self.DispalyMatches()
        return

    def DispalyMatches(self):
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        fig.suptitle("# Matches:" + str(self.matches.shape[0]))
        if self.img_i != None:
            src1 = rasterio.open(self.img_i)
            axs[0].imshow(src1.read(1), cmap='gray')
        if self.img_j is not None:
            src2 = rasterio.open(self.img_j)
            axs[1].imshow(src2.read(1), cmap='gray')

        from geoCosiCorr3D.georoutines.geoplt_misc import GenerateColors
        colors = GenerateColors(self.matches.shape[0])
        for index, match_ in enumerate(self.matches):
            axs[0].scatter(match_[0], match_[1], marker="+", s=80, color=colors[index])
            axs[1].scatter(match_[2], match_[3], marker="+", s=80, color=colors[index])
        axs[0].axes.xaxis.set_visible(False)
        axs[0].axes.yaxis.set_visible(False)
        axs[1].axes.xaxis.set_visible(False)
        axs[1].axes.yaxis.set_visible(False)
        axs[0].set_title("Ref image")
        axs[1].set_title("Target image")
        fig.savefig(os.path.join(self.oFolder, Path(self.matchFile).stem + ".svg"), dpi=100)
        plt.close(fig)
        return


class TPFilter:

    @staticmethod
    def TPFilter_F_RANSAC_CV(matches, ransac_th: Optional[float] = 0.3):
        """

        Args:
            matches:  Mx2 array representing M matches between features_i and features_j
            ransac_th: RANSAC outlier rejection threshold

        Returns:
        Notes:
            Given a series of pairwise matches, use OpenCV to fit a fundamental matrix using RANSAC to filter outliers
            The 7-point algorithm is used to derive the fundamental matrix
            https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findfundamentalmat

        """

        import cv2
        logging.info("Filter matches with OPENCV-RANSAC-F-matrix")

        F, inliers_mask = cv2.findFundamentalMat(matches[:, :2], matches[:, 2:4], cv2.FM_RANSAC, ransac_th)

        inliers_mask = inliers_mask.ravel().astype(bool)

        matches = matches[inliers_mask, :] if inliers_mask is not None else None
        logging.info("F-matrix using CV- RANSAC::{}".format(F))
        return matches

    @staticmethod
    def TPFilter_F_RANSAC_SIMD(matches, nb_trials: Optional[int] = 1000, ransac_th: Optional[float] = 0.3):
        """
             Estimate a fundamental matrix from a list of point matches, with RANSAC.
        Args:
            matches: list of point matches, each match being represented by a list or tuple (x1, y1, x2, y2) containing the x, y
                    coordinates of two matching points
            nb_trials:
            ransac_th:

        Returns:  inliers_mask fundamental_matrix (numpy array): array of shape (3, 3) representing
        the fundamental matrix

        """

        # filter matches with ransac
        global inliers_mask
        if matches.shape[0] > 7:
            logging.info("Filter matches with SIMD-RANSAC-F-matrix")
            n = len(matches)
            LIB_RANSAC.find_fundamental_matrix_by_ransac.argtypes = (
                np.ctypeslib.ndpointer(dtype=ctypes.c_bool, shape=(n,)),  # inliers
                np.ctypeslib.ndpointer(dtype=ctypes.c_float, shape=(9,)),  # F
                np.ctypeslib.ndpointer(dtype=ctypes.c_float, shape=(n, 4)),  # matches
                ctypes.c_int, ctypes.c_int, ctypes.c_float
            )
            LIB_RANSAC.find_fundamental_matrix_by_ransac.restype = ctypes.c_int

            inliers_mask = np.zeros(n, dtype=np.bool)
            F = np.zeros(9, dtype=np.float32)
            LIB_RANSAC.find_fundamental_matrix_by_ransac(inliers_mask,
                                                         F,
                                                         np.asarray(matches).astype(np.float32),
                                                         n,
                                                         nb_trials,
                                                         ransac_th)
            logging.info("F-matrix using RANSAC::{}".format(F))

        return matches[inliers_mask]