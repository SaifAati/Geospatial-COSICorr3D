"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""

import logging
from functools import partial

import numpy as np
import pandas
from scipy.optimize import least_squares

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoOptimization.model_refinement.func_models as func_model
import geoCosiCorr3D.georoutines.geo_utils as geoRT


class SatModelRefinement:
    def __init__(self, sat_model, gcps: pandas.DataFrame, debug=False):
        self.sat_model = sat_model
        self.gcps = gcps
        self.debug = debug

    def refine(self):
        pass


class RsmRefinement(SatModelRefinement):
    def __init__(self, sat_model,
                 gcps: pandas.DataFrame,
                 refinement_model=C.RsmRefinementModels.LINEAR,
                 solver: str = C.Solvers.SP_LM,
                 debug=False):
        super().__init__(sat_model, gcps, debug)
        self.rsm_model = sat_model
        self.refinement_model = refinement_model
        self.solver = solver
        self.nb_gcps = gcps.shape[0]
        self.dU = np.zeros((self.nb_gcps, 3))

    def _setup(self):
        if self.refinement_model == C.RsmRefinementModels.LINEAR:
            self.X_0 = np.zeros(3)
            self.corr_model = np.zeros((3, 3))
        elif self.refinement_model == C.RsmRefinementModels.QUADRATIC:
            self.X_0 = np.zeros(6)
            self.corr_model = np.zeros((6, 3))
        else:
            raise ValueError(f"{self.__class__.__name__}: Model {self.refinement_model} is not supported")

    def refine(self):
        logging.info(f"{self.__class__.__name__}: Performing RSM refinement ...")
        self._setup()
        self.compute_look_direction_error()
        if self.solver == C.Solvers.W_GAUSS_MARKOV:
            self.corr_model = np.zeros((3, 3))
            for i in range(3):
                corr_params = func_model.wlsq_gauss_markov(x=self.gcps[C.GCPKeys.COL],
                                                           y=self.gcps[C.GCPKeys.LIN],
                                                           w=self.gcps[C.GCPKeys.WEIGHT],
                                                           error=self.dU[:, i])
                self.corr_model[:, i] = corr_params
        elif self.solver == C.Solvers.SP_LM:
            # create observations list
            observations = []
            for index, gcp in self.gcps.iterrows():
                obs = C.Observation(COL=gcp[C.GCPKeys.COL],
                                    LIN=gcp[C.GCPKeys.LIN],
                                    DU=self.dU[index, :])
                observations.append(obs)

            for dim in range(3):
                cost_args = [observations, dim, func_model.FuncModel(self.refinement_model)]
                func = partial(func_model.cost_func, *cost_args)

                res = least_squares(func, self.X_0, method='lm', verbose=0, tr_solver='lsmr')
                self.corr_model[:, dim] = res.x
        else:
            raise ValueError(f"{self.__class__.__name__}:Solver {self.solver} is not supported")
        logging.info(f'{self.__class__.__name__}: Correction model: \n{self.corr_model}')

    @staticmethod
    def compute_look_direction(rsm_model, gcp: C.GCP):
        """
        Computes for one given pixel its look direction in geocentric coordinate (cartesian)
        Args:
            rsm_model: instance of the physical model (only push-broom @SA)
            gcp:  elements array containing the x, y pixel coordinates

        Returns: 3 elements array representing the pixel look direction in cartesian coordinate system (x, y, z)

        """

        col = gcp.COL
        line = gcp.LIN

        rotMat = np.array([[rsm_model.satToNavMat[int(line), 0, 0],
                            rsm_model.satToNavMat[int(line), 0, 1],
                            rsm_model.satToNavMat[int(line), 0, 2]],

                           [rsm_model.satToNavMat[int(line), 1, 0],
                            rsm_model.satToNavMat[int(line), 1, 1],
                            rsm_model.satToNavMat[int(line), 1, 2]],

                           [rsm_model.satToNavMat[int(line), 2, 0],
                            rsm_model.satToNavMat[int(line), 2, 1],
                            rsm_model.satToNavMat[int(line), 2, 2]]])

        postMat = np.array([[rsm_model.orbitalPos_X[int(line), 0],
                             rsm_model.orbitalPos_Y[int(line), 0],
                             rsm_model.orbitalPos_Z[int(line), 0]],

                            [rsm_model.orbitalPos_X[int(line), 1],
                             rsm_model.orbitalPos_Y[int(line), 1],
                             rsm_model.orbitalPos_Z[int(line), 1]],

                            [rsm_model.orbitalPos_X[int(line), 2],
                             rsm_model.orbitalPos_Y[int(line), 2],
                             rsm_model.orbitalPos_Z[int(line), 2]]])

        angleX = rsm_model.CCDLookAngle[int(col), 0]
        angleY = rsm_model.CCDLookAngle[int(col), 1]
        angleZ = rsm_model.CCDLookAngle[int(col), 2]
        angles = np.array([angleX, angleY, angleZ])

        ## Compute the look direction
        u2 = np.dot(rotMat, angles)
        u2 = u2 / np.sqrt(np.sum(u2 ** 2))
        ## convert the look angle to terrestrial coordinate system
        u3 = np.dot(postMat, u2)
        return u3

    @staticmethod
    def compute_gcp_los(rsm_model, gcp: C.GCP):

        gcp_cart_coords = geoRT.Convert.geo_2_cartesian(Lon=[gcp.LON],
                                                        Lat=[gcp.LAT],
                                                        Alt=[gcp.ALT])

        sat_pos_x = rsm_model.interpSatPosition[:, 0][int(gcp.LIN)]
        sat_pos_y = rsm_model.interpSatPosition[:, 1][int(gcp.LIN)]
        sat_pos_z = rsm_model.interpSatPosition[:, 2][int(gcp.LIN)]
        sat_pos = np.array([sat_pos_x, sat_pos_y, sat_pos_z])

        # Compute the look direction of GPCs  (line of sight)
        gcp_los = gcp_cart_coords - sat_pos
        gcp_los = gcp_los / np.linalg.norm(gcp_los, ord=2)

        return gcp_los

    def compute_look_direction_error(self):
        # TODO vectorise this function
        index_ = 0
        for index, gcp in self.gcps.iterrows():
            GCP = C.GCP(ID=gcp[C.GCPKeys.ID],
                        LON=gcp[C.GCPKeys.LON],
                        LAT=gcp[C.GCPKeys.LAT],
                        ALT=gcp[C.GCPKeys.ALT],
                        COL=gcp[C.GCPKeys.COL],
                        LIN=gcp[C.GCPKeys.LIN])
            gcp_init_look_dir = self.compute_look_direction(rsm_model=self.rsm_model,
                                                            gcp=GCP)
            gcp_look_dir = self.compute_gcp_los(rsm_model=self.rsm_model,
                                                gcp=GCP)

            gcp_error_vect = (gcp_look_dir - gcp_init_look_dir[:])

            if self.debug:
                logging.info(f"{self.__class__.__name__}: GCP: {gcp[C.GCPKeys.ID]}: error vect :{gcp_error_vect}")
            self.dU[index, :] = gcp_error_vect
            index_ += 1
        return


class RfmRefinement:
    def __init__(self, sat_model, rfm_model):
        self.sat_model = sat_model
        self.rfm_model = rfm_model

    def refine(self):
        pass

# if __name__ == '__main__':
#     logging.basicConfig(filename='app.log',
#                         filemode='w', format='%(name)s - %(levelname)s - %(message)s',
#                         level=logging.INFO)
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)
#
#     from geoCosiCorr3D.geoCore.core_RSM import RSM
#
#     sat_model_metadata = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/NEW/SP-2.DIM'
#     sensor = 'Spot2'
#     debug = True
#     gcps_df = pandas.read_csv('/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/NEW/input_GCPs.csv')
#     sat_model = RSM.build_RSM(metadata_file=sat_model_metadata,
#                               sensor_name=sensor,
#                               debug=debug)
#
#     rsm_refinement = RsmRefinement(sat_model=sat_model, gcps=gcps_df, debug=debug)
#     rsm_refinement.refine()
#
#     expected_correction = np.array([[5.72835604e-07, 1.68738324e-07, 5.61578545e-07],
#                                     [1.66133358e-08, -6.20805542e-10, -1.89712215e-08],
#                                     [-1.26942529e-02, 6.70568661e-02, -4.98777828e-03]])
#     assert np.allclose(rsm_refinement.corr_model, expected_correction, atol=1e-6)
#     logging.info(f"Test passed: {rsm_refinement.__class__.__name__}")
