"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os.path

import numpy as np
import pandas


def opt_report(reportPath, image_path, snrTh=0.9, debug=False, plotError=True):
    df = pandas.read_csv(reportPath)

    totalNbLoop = list(df["nbLoop"])[-1]
    # print(totalNbLoop)
    loopList = []
    rmseList = []
    avgErrorList = []
    for loop_ in range(totalNbLoop + 1):
        if debug:
            print("------ Loop:{} -------".format(loop_))
        itemList = []
        dxPixList = []
        dyPixList = []
        snrList = []
        dxList = []
        dyList = []
        for item, dxPix_, dyPix_, snr_ in zip(list(df["nbLoop"]), list(df["dxPix"]), list(df["dyPix"]),
                                              list(df["SNR"])):
            if item == loop_:
                itemList.append(item)
                dxPixList.append(dxPix_)
                dyPixList.append(dyPix_)
                snrList.append(snr_)

        nanList = [item_ for item_ in snrList if item_ == 0]
        snrThList = [item_ for item_ in snrList if item_ > snrTh]
        dxPixAvg = np.nanmean(np.asarray(dxPixList))
        dyPixAvg = np.nanmean(np.asarray(dyPixList))

        dxPixRMSE = np.nanstd(np.asarray(dxPixList))
        dyPixRMSE = np.nanstd(np.asarray(dyPixList))

        xyErrorAvg = np.sqrt(dxPixAvg ** 2 + dyPixAvg ** 2)
        xyRMSE = np.sqrt(dxPixRMSE ** 2 + dyPixRMSE ** 2)
        if debug:
            print("#GCPs:{} --> #NaNs:{} ; #snrTh >{}:{}".format(len(itemList), len(nanList), snrTh, len(snrThList)))
            print("dxPixAvg:{}  , xRMSE:{}".format("{0:.4f}".format(dxPixAvg),

                                                   "{0:.2f}".format(dxPixRMSE)))
            print("dyPixAvg:{}  , yRMSE:{}".format("{0:.4f}".format(dyPixAvg),

                                                   "{0:.2f}".format(dyPixRMSE)))
            print("xyErrorAvg:{}  , xyRMSE:{}".format("{0:.4f}".format(xyErrorAvg),

                                                      "{0:.2f}".format(xyRMSE)))
        loopList.append(loop_)
        rmseList.append(xyRMSE)
        avgErrorList.append(xyErrorAvg)
    indexMin = np.argmin(avgErrorList)
    # if debug:
    print("Loop of Min Error:{} --> RMSE:{:.3f} , avgErr:{:.3f}".format(loopList[indexMin], np.min(rmseList),
                                                                        np.min(avgErrorList)))
    if plotError:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        fig, ax = plt.subplots()

        ax.plot(loopList, rmseList, c="r", linestyle="--", marker="o", label="RMSE [pix]")

        ax.plot(loopList, avgErrorList, c="g", linestyle="-", marker="o", label="meanErr [pix]")

        ax.grid()
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(which='both', width=2, direction="in")
        ax.set_xlabel('#iterations')
        ax.set_ylabel("Error [pix]")
        # plt.show()
        fig.savefig(os.path.join(os.path.dirname(reportPath), f'{image_path}_CoregistrationError.png'), dpi=400)

    return loopList[indexMin], totalNbLoop, np.min(avgErrorList)


def parse_opt_report(opt_report_path):
    df = pandas.read_csv(opt_report_path)

    nb_loops = list(df["nbLoop"])[-1]
    loopList = []
    rmse = []
    avg_error = []
    for loop_ in range(nb_loops + 1):

        itemList = []
        dxPixList = []
        dyPixList = []
        snrList = []
        for item, dxPix_, dyPix_, snr_ in zip(list(df["nbLoop"]), list(df["dxPix"]), list(df["dyPix"]),
                                              list(df["SNR"])):
            if item == loop_:
                itemList.append(item)
                dxPixList.append(dxPix_)
                dyPixList.append(dyPix_)
                snrList.append(snr_)
        dxPixAvg = np.nanmean(np.asarray(dxPixList))
        dyPixAvg = np.nanmean(np.asarray(dyPixList))

        dxPixRMSE = np.nanstd(np.asarray(dxPixList))
        dyPixRMSE = np.nanstd(np.asarray(dyPixList))

        xyErrorAvg = np.sqrt(dxPixAvg ** 2 + dyPixAvg ** 2)
        xyRMSE = np.sqrt(dxPixRMSE ** 2 + dyPixRMSE ** 2)
        loopList.append(loop_)
        rmse.append(xyRMSE)
        avg_error.append(xyErrorAvg)
    idx_min = np.argmin(avg_error)
    loop_min_err = loopList[idx_min]
    # print("Loop of Min Error:{} --> RMSE:{:.3f} , avgErr:{:.3f}".format(loopList[indexMin], np.min(rmse),
    #                                                                     np.min(avg_error)))

    return rmse, avg_error, loop_min_err
