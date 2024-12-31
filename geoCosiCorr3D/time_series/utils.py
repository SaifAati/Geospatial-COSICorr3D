"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def chi2_F_test(chi2):
    print(chi2)
    print(np.sum(chi2))
    chi2_norm = chi2 / np.sum(chi2)
    print(chi2_norm)
    # compute F_test
    F = []
    for i in range(len(chi2) - 1):
        F_i = np.abs(chi2[i] ** 2 - chi2[i + 1] ** 2) / chi2[i] ** 2
        F.append(F_i)
    print(F)

    return


def plot_s(S, oPath, fontSize=12, show=False):
    sValues = S[S != 0]
    sValues = sValues[:20]
    print("sValues=", sValues, len(sValues))

    nbComponents = len(sValues)
    # sValues = np.sqrt(sValues)
    percent = 100 * (sValues ** 2) / sum(sValues ** 2)
    percent = np.round(percent, decimals=2)
    print("PercentVar= ", percent)

    x = np.arange(nbComponents)

    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1f%%' % (x)

    listXaxis = []
    for i in range(nbComponents):
        listXaxis.append("PC" + str(i + 1))

    formatter = FuncFormatter(millions)
    x = np.arange(nbComponents)
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    ax.yaxis.set_major_formatter(formatter)
    plt.bar(x, percent, color="k", width=0.5, label="Variance")

    ## Cumulative sum variance
    cumVar = np.cumsum(percent)
    print("cumVar=", cumVar)
    plt.plot(cumVar, color="r", label="Cumulative variance")

    plt.ylabel('% Explained Variance', fontsize=fontSize)
    plt.xlabel(' Number of Components', fontsize=fontSize)
    plt.title('geoPCAIM Analysis \n Cumulative sum of variance explained with [%s] components' % (nbComponents),
              fontsize=fontSize)

    plt.style.context('seaborn-whitegrid')
    plt.xticks(x, listXaxis, fontsize=fontSize, rotation=90)  # , fontweight='bold'
    plt.yticks(fontsize=fontSize)  # , fontweight='bold')
    plt.legend()
    # plt.grid(False)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.savefig(oPath, dpi=400)
    if show:
        plt.show()
    plt.close()

    return
