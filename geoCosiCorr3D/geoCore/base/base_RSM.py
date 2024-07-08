"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC, abstractmethod


class BaseRSM(ABC):

    @abstractmethod
    def interp_attitude(self):
        pass

    @abstractmethod
    def interp_eph(self):
        pass

    @abstractmethod
    def compute_orbital_reference_system(self):
        pass

    @abstractmethod
    def compute_look_direction(self):
        pass

    @abstractmethod
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
        pass
