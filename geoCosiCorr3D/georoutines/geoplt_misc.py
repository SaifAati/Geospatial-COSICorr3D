"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import random


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def GenerateColors(N, pastel_factor=0.9):
    colors = []
    #TODO use a multi-thread
    for i in range(0, N):
        colors.append(generate_new_color(colors, pastel_factor=pastel_factor))

    return colors

def ColorBar_(ax, mapobj, cmap, vmin, vmax, label="Disp.[m]", width="3%", height="50%",
              orientation='vertical',  # horizontal',
              bounds=None,
              extend="neither", size=8):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # fig, ax = plt.subplots(figsize=(6, 1))
    # fig.subplots_adjust(bottom=0.5)

    # cmap = mpl.cm.cool
    axins = inset_axes(ax,
                       width=width,  # width = 5% of parent_bbox width
                       height=height,  # height : 50%
                       loc=6,
                       bbox_to_anchor=(1, 0, 1, 1),  # [x,y,height,width]
                       bbox_transform=ax.transAxes,
                       borderpad=0.3)
    if bounds == None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend=extend)

    cbar = plt.colorbar(mapobj,
                        cax=axins, orientation=orientation, label=label, spacing='uniform')
    cbar.set_label(label=label, size=size)
    cbar.ax.tick_params(labelsize=size)

    return
