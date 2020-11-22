#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


def plot_u(x, y, color, lw, lb,
           ti="Plot", xl="X", yl="Y", legendloc=4,
           xlim=(0, 1), ylim=(0, 1), ylog=False,
           fn="plot2d.pdf", sa=False):
    """
        Plot u on x-y coordinate system
        :param x:
        :param y:
        :param color: The color of each line
        :param lw: The width of each line
        :param lb: The label of each line
        :param ti: The title of plot
        :param xl: The label of x axis
        :param yl: The label of y axis
        :param legendloc: The location of legend
        :param xlim: The range of x axis
        :param ylim: The range of y axis
        :param ylog: Using logarithmic y axis or not
        :param fn:  The saved file name
        :param sa:  Saving the file or not
        :return: None
        """

    plt.figure()
    plt.plot(x, y, color=color, linewidth=lw, label=lb)

    plt.title(ti)

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if ylog:
        plt.yscale('log')

    plt.legend(shadow=True, loc=legendloc)

    if sa:
        plt.savefig(fn)
    plt.show()
    plt.close()


def plot_2d(n, x, y, color, lw, lb,
            ti="Plot", xl="X", yl="Y", legendloc=4,
            xlim=(0, 1), ylim=(0, 1), ylog=False,
            fn="plot2d.pdf", sa=False):
    """
    Plot n lines on x-y coordinate system
    :param n: The number of the plot line
    :param x:
    :param y:
    :param color: The color of each line
    :param lw: The width of each line
    :param lb: The label of each line
    :param ti: The title of plot
    :param xl: The label of x axis
    :param yl: The label of y axis
    :param legendloc: The location of legend
    :param xlim: The range of x axis
    :param ylim: The range of y axis
    :param ylog: Using logarithmic y axis or not
    :param fn:  The saved file name
    :param sa:  Saving the file or not
    :return: None
    """

    plt.figure()
    for i in range(n):
        plt.plot(x, y[i], color=color[i], linewidth=lw[i], label=lb[i])

    plt.title(ti)

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if ylog:
        plt.yscale('log')

    plt.legend(shadow=True, loc=legendloc)

    if sa:
        plt.savefig(fn)
    plt.show()
    plt.close()


def plot_contour(x, y, z, xl=r"$X$", yl=r"$Y$", steps=1):
    xx, yy = np.meshgrid(x, y)
    plt.contourf(xx, yy, z.T, cmap='seismic', levels=np.linspace(-1.0, 1.0, 32),
                 extend='both')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title('t = {0:.1f} s'.format(steps * 0.1))
    cb = plt.colorbar()
    cb.set_ticks([-1, -0.8, -0.6, -0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label(r'$\zeta$', rotation=0)
    plt.show()
    plt.close()


def plot_contour_ani(x, y, z, xl=r"$X$", yl=r"$Y$", steps=100, fn="test.mp4"):
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    xx, yy = np.meshgrid(x, y)
    fig = plt.figure()
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.contourf(xx, yy, z[0, :, :].T, cmap='seismic', levels=np.linspace(-1.0, 1.0, 32),
                 extend='both')
    cb = plt.colorbar()
    cb.set_ticks([-1, -0.8, -0.6, -0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label(r'$\zeta$', rotation=0)

    with writer.saving(fig, fn, 100):
        for i in range(steps):
            if i % 10 == 0:
                plt.contourf(xx, yy, z[i, :, :].T,
                             cmap='seismic', levels=np.linspace(-1.0, 1.0, 32),
                             extend='both')
                plt.title('t = {0:.1f} s'.format(i * 0.1))
                print("i = ", i)
                writer.grab_frame()

    plt.close()
