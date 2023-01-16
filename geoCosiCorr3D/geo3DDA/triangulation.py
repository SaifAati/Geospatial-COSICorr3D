"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""

import numpy as np


def triangulate(XYZ_cart_M, XYZ_cart_S, sightVector_M, sightVector_S):
    # print("_________")
    # print(list(XYZ_cart_M))
    # print(list(XYZ_cart_S))
    # print(list(sightVector_M))
    # print(list(sightVector_S))
    # print("_________")
    params = []
    pt1_new = []
    pt2_new = []
    residu = []
    XYZ_cart_corr = []

    for XYZ_cart_M_, XYZ_cart_S_, sightVector_M_, sightVector_S_ in zip(list(XYZ_cart_M), list(XYZ_cart_S),
                                                                        list(sightVector_M), list(sightVector_S)):
        params_, pt1_new_, pt2_new_, residu_, XYZ_cart_corrected_ = Intersect_Leaq(
            p1=np.asarray(XYZ_cart_M_),
            p2=np.asarray(XYZ_cart_S_),
            s1=sightVector_M_,
            s2=sightVector_S_)

        params.append(params_)
        pt1_new.append(pt1_new_)
        pt2_new.append(pt2_new_)
        residu.append(residu_)
        XYZ_cart_corr.append(XYZ_cart_corrected_)

    return np.asarray(params), np.asarray(pt1_new), np.asarray(pt2_new), np.asarray(residu), np.asarray(XYZ_cart_corr)


def SkewIntersect(s1, s2, pt2, pt1):
    diffPre = np.asarray(pt2) - np.asarray(pt1)
    A_pre = [[np.sum(s1 ** 2), -1 * np.sum(s1 * s2)],
             [np.sum(s1 * s2), -1 * np.sum(s2 ** 2)]]
    A_pre = np.asarray(A_pre)

    B_pre = [np.sum(s1 * diffPre), np.sum(s2 * diffPre)]
    B_pre = np.asarray(B_pre)
    temp_1 = np.linalg.inv(np.dot(A_pre.T, A_pre))
    temp_2 = np.dot(A_pre.T, B_pre)
    scale_pre = np.dot(temp_1, temp_2)

    xyz_pre1Corrected = pt1 + scale_pre[0] * s1
    xyz_pre2Corrected = pt2 + scale_pre[1] * s2
    pt_intersect = (xyz_pre1Corrected + xyz_pre2Corrected) / 2

    return scale_pre, xyz_pre1Corrected, xyz_pre2Corrected, pt_intersect


def SkewIntersect_Batch_V1(s1, s2, pt2, pt1):
    nb = s1.shape[0]
    scale_pre_Array = np.zeros((nb, 2))
    pt1_corrected = np.zeros(s1.shape)
    pt2_corrected = np.zeros(s1.shape)
    diffPre = np.asarray(pt2) - np.asarray(pt1)
    for i in range(nb):
        A_pre = [[np.sum(s1[i, :] ** 2), -1 * np.sum(s1[i, :] * s2[i, :])],
                 [np.sum(s1[i, :] * s2[i, :]), -1 * np.sum(s2[i, :] ** 2)]]
        A_pre = np.asarray(A_pre)

        B_pre = [np.sum(s1[i, :] * diffPre[i, :]), np.sum(s2[i, :] * diffPre[i, :])]
        B_pre = np.asarray(B_pre)
        temp_1 = np.linalg.inv(np.dot(A_pre.T, A_pre))
        temp_2 = np.dot(A_pre.T, B_pre)
        scale_pre = np.dot(temp_1, temp_2)
        scale_pre_Array[i, 0] = scale_pre[0]
        scale_pre_Array[i, 1] = scale_pre[1]
        pt1_corrected[i, :] = pt1[i, :] + scale_pre_Array[i, 0] * s1[i, :]
        pt2_corrected[i, :] = pt2[i, :] + scale_pre_Array[i, 1] * s2[i, :]

    pt_intersect = (pt1_corrected + pt2_corrected) / 2
    # print(pt_intersect)
    return scale_pre_Array, pt1_corrected, pt2_corrected, pt_intersect


def SkewIntersect_Batch_V2(s1, s2, pt2, pt1):
    diffPre = np.asarray(pt2) - np.asarray(pt1)
    A_pre = [[np.sum(s1 ** 2), -1 * np.sum(s1 * s2)],
             [np.sum(s1 * s2), -1 * np.sum(s2 ** 2)]]
    A_pre = np.asarray(A_pre)

    B_pre = [np.sum(s1 * diffPre), np.sum(s2 * diffPre)]
    B_pre = np.asarray(B_pre)
    temp_1 = np.linalg.inv(np.dot(A_pre.T, A_pre))
    temp_2 = np.dot(A_pre.T, B_pre)
    scale_pre = np.dot(temp_1, temp_2)

    xyz_pre1Corrected = pt1 + scale_pre[0] * s1
    xyz_pre2Corrected = pt2 + scale_pre[1] * s2
    pt_intersect = (xyz_pre1Corrected + xyz_pre2Corrected) / 2
    print(pt_intersect)
    return scale_pre


def AnalyticComputing(s1, s2, pt1, pt2):
    ## There is a problem withi in this function
    normS1 = np.sqrt(s1[0] ** 2 + s1[1] ** 2 + s1[2] ** 2)
    normS2 = np.sqrt(s2[0] ** 2 + s2[1] ** 2 + s2[2] ** 2)

    A1 = s1  # / normS1
    A2 = s2  # / normS2

    a = np.linalg.det(np.array([[A1[0], A1[1]], [A2[0], A2[1]]]))
    b = np.linalg.det(np.array([[A1[1], A1[2]], [A2[1], A2[2]]]))
    c = np.linalg.det(np.array([[A1[2], A1[0]], [A2[2], A2[0]]]))

    denum = np.linalg.det(np.array([
        [A1[0], A1[1], A1[2]],
        [A2[0], A2[1], A2[2]],
        [a, b, c]]))

    lamda = -np.linalg.det(np.array([
        [pt1[0] - pt2[0], pt1[1] - pt2[1], pt1[2] - pt2[2]],
        [a, b, c],
        [A2[0], A2[1], A2[2]]])) / denum

    mu = -np.linalg.det(np.array([
        [pt1[0] - pt2[0], pt1[1] - pt2[1], pt1[2] - pt2[2]],
        [a, b, c],
        [A1[0], A1[1], A1[2]]])) / denum

    print(lamda, mu)

    pt1_new = pt1 + lamda * s1
    pt2_new = pt2 + mu * s2
    ptIntersect = (pt1_new + pt2_new) / 2
    # print(u)
    # print(ptIntersect)

    return lamda, mu, pt1_new, pt2_new, ptIntersect


def check_NaNs(array):
    array_sum = np.sum(array)
    return np.isnan(array_sum)


def Intersect_Leaq(p1, p2, s1, s2):
    # print("p1=",check_NaNs(p1))
    # print("p2=",check_NaNs(p2))
    # print("s1=",check_NaNs(s1))
    # print("s2=",check_NaNs(s2))
    X = [np.nan, np.nan]
    pt1_new = [np.nan, np.nan, np.nan]
    pt2_new = [np.nan, np.nan, np.nan]
    residu = [np.nan]
    ptIntersect = [np.nan, np.nan, np.nan]
    if check_NaNs(p1) == False and check_NaNs(p2) == False and check_NaNs(s1) == False and check_NaNs(s2) == False:
        # X = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
        A = np.column_stack((s1, -s2))
        b = p2 - p1
        # print("A:", A, "b:", b)
        results = np.linalg.lstsq(A, b, rcond=None)
        X = results[0]
        residu = results[1]
        rank = results[2]
        sinfularValues = results[3]

        pt1_new = p1 + X[0] * s1
        pt2_new = p2 + X[1] * s2
        ptIntersect = (pt1_new + pt2_new) / 2

    return X, pt1_new, pt2_new, residu, ptIntersect


def Plot3DLine(p1, p2, s1, s2, pt1_new, pt2_new, ptIntersect, xyz_pre1Corrected, xyz_pre2Corrected, ptIntersect2):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    l = -8
    u = -8
    ax.plot([p1[0], p1[0] + l * s1[0]], [p1[1], p1[1] + l * s1[1]], [p1[2], p1[2] + l * s1[2]], c="r")
    ax.plot([p2[0], p2[0] + u * s2[0]], [p2[1], p2[1] + u * s2[1]], [p2[2], p2[2] + u * s2[2]], )
    ax.scatter([pt1_new[0], pt2_new[0], ptIntersect[0]], [pt1_new[1], pt2_new[1], ptIntersect[1]],
               [pt1_new[2], pt2_new[2], ptIntersect[2]], c="r")
    ax.scatter([xyz_pre1Corrected[0], xyz_pre2Corrected[0], ptIntersect2[0]],
               [xyz_pre1Corrected[1], xyz_pre2Corrected[1], ptIntersect2[1]],
               [xyz_pre1Corrected[2], xyz_pre2Corrected[2], ptIntersect2[2]], c='k')
    # xyz_pre1Corrected, xyz_pre2Corrected, ptIntersect
    plt.show()
    return


def SkewIntersect_Plucker():
    return


if __name__ == '__main__':
    # preXYZ = np.asarray([-2401041.2517469465, -4590274.826117556, 3709495.3869768754])
    # pre2XYZ = np.asarray([-2401041.692476227, -4590274.9331094185, 3709494.972106257])
    # sightVector_pre1 = np.asarray([70539.71727394, -569890.15395742, 642395.36012483])
    # sightVector_pre2 = np.asarray([1209.94504468, -371108.40130538, 588199.15505255])
    # # scale_pre = SkewIntersect(s1=sightVector_pre1, s2=sightVector_pre2, pt2=pre2XYZ, pt1=preXYZ)
    # # print(scale_pre)
    # # AnalyticComputing(s1=sightVector_pre1, s2=sightVector_pre2, pt1=preXYZ, pt2=pre2XYZ)
    #
    # lamda, mu, pt1_new, pt2_new, ptIntersect = AnalyticComputing(s1=sightVector_pre1, s2=sightVector_pre2,
    #                                                              pt1=sightVector_pre1, pt2=pre2XYZ)
    # print("ptIntersect:", ptIntersect)
    # scale_pre, xyz_pre1Corrected, xyz_pre2Corrected, ptIntersect2 = SkewIntersect(s1=sightVector_pre1,
    #                                                                               s2=sightVector_pre2,
    #                                                                               pt1=sightVector_pre1, pt2=pre2XYZ)
    # print("lsqure intersec:", ptIntersect2)
    # Plot3DLine(p1=preXYZ, p2=pre2XYZ, s1=sightVector_pre1, s2=sightVector_pre2, pt1_new=pt1_new, pt2_new=pt2_new,
    #            ptIntersect=ptIntersect,
    #            xyz_pre1Corrected=xyz_pre1Corrected,
    #            xyz_pre2Corrected=xyz_pre2Corrected, ptIntersect2=ptIntersect2)

    p1 = np.array([0, 2, -1])
    p2 = np.array([1, 0, -1])
    s1 = np.array([1, 1, 2])
    s2 = np.array([1, 1, 3])
    # lamda, mu, pt1_new, pt2_new, ptIntersect = AnalyticComputing(s1=s1, s2=s2, pt1=p1, pt2=p2)
    # print("ptIntersect:", ptIntersect)
    # print(lamda, mu, "\n")
    scale_pre, xyz_pre1Corrected, xyz_pre2Corrected, ptIntersect2 = SkewIntersect(s1=s1, s2=s2, pt2=p2, pt1=p1)
    print("lsqure intersec_IDL:", ptIntersect2)
    print(scale_pre)

    X, pt1_new, pt2_new, residu, ptIntersect = Intersect_Leaq(p1, p2, s1, s2)
    Plot3DLine(p1=p1, p2=p2, s1=s1, s2=s2, pt1_new=pt1_new, pt2_new=pt2_new, ptIntersect=ptIntersect,
               xyz_pre1Corrected=xyz_pre1Corrected,
               xyz_pre2Corrected=xyz_pre2Corrected, ptIntersect2=ptIntersect2)
    # [  1.41597717  90.45256174 -94.76107573]
