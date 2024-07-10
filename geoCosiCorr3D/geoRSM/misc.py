"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import datetime

import numpy as np


def time_to_second(time_in, year_origin=1950):
    date_time_origin = datetime.datetime.strptime(str(year_origin), '%Y')
    return (time_in - date_time_origin).total_seconds()


def NormalizeArray(inputArray, order=2):
    """
    normalize the given 2D array by line
    :param inputArray:
    :param order:
    :return:
    """

    norm = np.linalg.norm(inputArray, axis=1, ord=order)
    normalizedArray = np.empty(inputArray.shape)
    for i in range(normalizedArray.shape[0]):
        normalizedArray[i, :] = inputArray[i, :] / norm[i]
    return normalizedArray


def normalize_array(in_arr, order=2):
    """
    Normalize the given 2D array by line.
    :param inputArray: 2D NumPy array to be normalized
    :param order: Order of the norm (e.g., 2 for Euclidean)
    :return: Normalized 2D array
    """
    norm = np.linalg.norm(in_arr, axis=1, ord=order, keepdims=True)
    return in_arr / norm


def Qaut2Rot(quat_, axis=1, order=2):
    """
     Convert rotation quaternions onto a rotation matrix
    Args:
        quat_: A 4 elts array [q1,q2,q3,q4] with q4 the scalar
        axis:
        order:

    Returns:

    """

    norm = np.linalg.norm(quat_)  # , axis=axis, ord=order)
    quat = quat_ / norm
    # print("Norm:{} | quat::{}".format(norm,quat_))
    ii = quat[0]
    jj = quat[1]
    kk = quat[2]
    aa = quat[3]  # scalar     part

    xs = ii * 2.0
    ys = jj * 2.0
    zs = kk * 2.0
    wx = aa * xs
    wy = aa * ys
    wz = aa * zs
    xx = ii * xs
    xy = ii * ys
    xz = ii * zs
    yy = jj * ys
    yz = jj * zs
    zz = kk * zs

    temp = [[(1 - yy - zz), (xy + wz), (xz - wy)],
            [(xy - wz), (1 - xx - zz), (yz + wx)],
            [(xz + wy), (yz - wx), (1 - xx - yy)]]
    return temp


def GetRot_Geographic2LocalRefSystem(ptLon, ptLat, unit="degree"):
    """
    Define the rotation matrix to transform the geographic reference system to a local reference system
    Args:
        ptLon: input Longitude of the point
        ptLat: input latitude of the point
        unit: default degree, it could be radian

    Returns: rotation matrix shape=(3,3)

    """
    if unit == "degree":
        ##Convert to radian
        ptLon = ptLon * np.pi / 180
        ptLat = ptLat * np.pi / 180
    elif unit != "radian":
        raise ValueError("Unsupported unit to define the rotation matrix ")
    rotMatrix = np.array([[-np.sin(ptLon), np.cos(ptLon), 0],
                          [-np.sin(ptLat) * np.cos(ptLon), -np.sin(ptLat) * np.sin(ptLon), np.cos(ptLat)],
                          [np.cos(ptLat) * np.cos(ptLon), np.cos(ptLat) * np.sin(ptLon), np.sin(ptLat)]], dtype=float)
    return rotMatrix


def RotMatrix_RollPitchYaw(iOmega, iPhi, iKappa):
    """
    Function that will return the rotation matrix due to the attitude roll pitch and yaw, (omega,phi,kappa)
    Args:
        iOmega:omega angle in radian
        iPhi: phi angle in radian
        iKappa: kappa angle in radian

    Returns: attitude matrix [3,3]

    """
    m11 = np.cos(iPhi) * np.cos(iKappa)
    m12 = np.sin(iOmega) * np.sin(iPhi) * np.cos(iKappa) + np.cos(iOmega) * np.sin(iKappa)
    m13 = -np.cos(iOmega) * np.sin(iPhi) * np.cos(iKappa) + np.sin(iOmega) * np.sin(iKappa)
    m21 = -np.cos(iPhi) * np.sin(iKappa)
    m22 = -np.sin(iOmega) * np.sin(iPhi) * np.sin(iKappa) + np.cos(iOmega) * np.cos(iKappa)
    m23 = np.cos(iOmega) * np.sin(iPhi) * np.sin(iKappa) + np.sin(iOmega) * np.cos(iKappa)
    m31 = np.sin(iPhi)
    m32 = -np.sin(iOmega) * np.cos(iPhi)
    m33 = np.cos(iOmega) * np.cos(iPhi)

    return np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])


class cSpatialRotations:

    def __init__(self):
        return

    def RotNormalization(self, rot):
        """
        If the coef of a rotation matrix are not explicitly derived from 3 rotation values,
        but instead are the result of a calculation process such as the determination
        of exterior orientation or spatial similarity transformation,
         then rot matrix can show departures from orthogonality and orthonormality ==> det(R.T * R)!=1
         In this case the matrix can be orthonormalized using Gram-Schimit procedure
        Args:
            rot:

        Returns:

        """
        u = rot[:, 0]
        v = rot[:, 1]
        w = rot[:, 2]

        # uNorm = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        # u_ = rot[:, 0] / uNorm
        # from numpy.linalg import norm
        u_ = u / np.linalg.norm(u)
        s = v - (np.dot(u, u_) / u_)
        v_ = s / np.linalg.norm(s)
        w_ = np.cross(u, v_)
        # rot_ = np.array([u_, v_, w_]).T
        # print(rot_)
        rot_, r = np.linalg.qr(rot)
        return rot_

    def R_opk_xyz(self, omg, phi, kpp, unit="rad"):
        """

        Args:
            omg:
            phi:
            kpp:
            unit:

        Returns:

        """
        if unit == "deg":
            omg = omg * np.pi / 180
            phi = phi * np.pi / 180
            kpp = kpp * np.pi / 180
        # r11 = np.cos(phi) * np.cos(kpp)
        # r12 = -np.cos(phi) * np.sin(kpp)
        # r13 = np.sin(phi)
        #
        # r21 = np.cos(omg) * np.sin(kpp) + np.sin(omg) * np.sin(phi) * np.cos(kpp)
        # r22 = np.cos(omg) * np.cos(kpp) - np.sin(omg) * np.sin(phi) * np.sin(kpp)
        # r23 = -np.sin(omg) * np.cos(phi)
        #
        # r31 = np.sin(omg) * np.sin(kpp) - np.cos(omg) * np.sin(phi) * np.cos(kpp)
        # r32 = np.sin(omg) * np.cos(kpp) + np.cos(omg) * np.sin(phi) * np.sin(kpp)
        # r33 = np.cos(omg) * np.cos(phi)
        # R_opk = np.array([[r11, r12, r13],
        #                   [r21, r22, r23],
        #                   [r31, r32, r33]])
        R_opk = np.dot(self._R_omega_X(omg), np.dot(self._R_phi_Y(phi), self._R_kappa_Z(kpp)))
        return R_opk

    def R_opk_angles(self, rot_opk, unit="deg"):
        # if unit == "deg":
        #     omg = omg * np.pi / 180
        #     phi = phi * np.pi / 180
        #     kpp = kpp * np.pi / 180
        phi = np.arcsin(rot_opk[0, 2])  # sin(phi)
        omg = -np.arctan(rot_opk[1, 2] / rot_opk[2, 2])
        kpp = -np.arctan(rot_opk[0, 1] / rot_opk[0, 0])
        if unit == "deg":
            omg = omg * 180 / np.pi
            phi = phi * 180 / np.pi
            kpp = kpp * 180 / np.pi
        return [omg, phi, kpp]

    def R_pok_yxz(self, omg, phi, kpp, unit="rad"):
        if unit == "deg":
            omg = omg * np.pi / 180
            phi = phi * np.pi / 180
            kpp = kpp * np.pi / 180
        # r11 = np.cos(phi) * np.cos(kpp) + np.sin(phi) * np.sin(omg) * np.sin(kpp)
        # r12 = -np.cos(phi) * np.sin(kpp) + np.sin(phi) * np.sin(omg) * np.cos(kpp)
        # r13 = np.sin(phi) * np.cos(omg)
        #
        # r21 = np.cos(omg) * np.sin(kpp)
        # r22 = np.cos(omg) * np.cos(kpp)
        # r23 = -np.sin(omg)
        #
        # r31 = -np.sin(phi) * np.cos(kpp) + np.cos(phi) * np.sin(omg) * np.sin(kpp)
        # r32 = np.sin(phi) * np.sin(kpp) + np.cos(phi) * np.sin(omg) * np.cos(kpp)
        # r33 = np.cos(phi) * np.cos(omg)
        # R_pok = np.array([[r11, r12, r13],
        #                   [r21, r22, r23],
        #                   [r31, r32, r33]])
        R_pok = np.dot(self._R_phi_Y(phi), np.dot(self._R_omega_X(omg), self._R_kappa_Z(kpp)))
        return R_pok

    def R_kpok_zyx(self, omg, phi, kpp, unit="rad"):
        if unit == "deg":
            omg = omg * np.pi / 180
            phi = phi * np.pi / 180
            kpp = kpp * np.pi / 180

        R_kpo = np.dot(self._R_kappa_Z(kpp), np.dot(self._R_phi_Y(phi), self._R_omega_X(omg)))
        return R_kpo

    def _R_kappa_Z(self, kpp):
        """

        Args:
            kpp: [rad]

        Returns:

        """

        return np.array([[np.cos(kpp), -np.sin(kpp), 0],
                         [np.sin(kpp), np.cos(kpp), 0],
                         [0, 0, 1]])

    def _R_phi_Y(self, phi):
        return np.array([[np.cos(phi), 0, np.sin(phi)],
                         [0, 1, 0],
                         [-np.sin(phi), 0, np.cos(phi)]])

    def _R_omega_X(self, omg):
        return np.array([[1, 0, 0],
                         [0, np.cos(omg), -np.sin(omg)],
                         [0, np.sin(omg), np.cos(omg)]])


def SpatialRotations():
    # R_omega_phi_kappa = [35, 60, 30]  ## degrees
    # R_omega_phi_kappa = [-13.0567, -4.4369, 0.7782]  ## degrees
    R_omega_phi_kappa = [-13.0577, -4.4387, 0.7791]  ## degrees p287
    spatialRot = cSpatialRotations()
    R_opk = np.around(
        spatialRot.R_opk_xyz(R_omega_phi_kappa[0], R_omega_phi_kappa[1], R_omega_phi_kappa[2], unit="deg"), 6)
    print(R_opk)

    # angles = spatialRot.R_opk_angles(rot_opk=R_opk)
    # print(np.around(angles, 4))
    #
    # R_pok = np.around(
    #     spatialRot.R_pok_yxz(omg=0, phi=90, kpp=90, unit="deg"), 6)
    # print(R_pok)
    # R_omega_phi_kappa_rad = np.radians(R_omega_phi_kappa)
    # omg, phi, kpp = R_omega_phi_kappa_rad[0], R_omega_phi_kappa_rad[1], R_omega_phi_kappa_rad[2]

    # comb = [[omg, phi, kpp],
    #         [omg, kpp, phi],
    #         [phi, omg, kpp],
    #         [phi, kpp, omg],
    #         [kpp, phi, omg],
    #         [kpp, omg, phi]]
    # for comb_ in comb:
    #     r = R.from_rotvec(rotvec=comb_)
    #     rot = r.as_matrix()
    #     print(rot)
    #     print("-----------------")
    return
