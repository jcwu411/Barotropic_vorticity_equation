#!/usr/bin/env python3.8
"""
Created on 16/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import plot_data
import ode_solver2


def init(nx, ny, turb=False):
    # Initializing x-y grid
    x_grid = generate_grid(xb, nx)
    y_grid = generate_grid(yb, ny)

    # Initializing zeta
    zeta_i = np.zeros((nx, ny))
    center_x = int(nx / 2)
    center_y = int(ny / 2)
    width = 10
    zeta_i[:, center_y:(center_y + width)] = 1.0
    zeta_i[:, (center_y - width):center_y] = -1.0

    if turb:
        intensity = 0.1
        zeta_i[center_x, center_y] += intensity
        zeta_i[center_x - 1, center_y] += intensity
        zeta_i[center_x, center_y - 1] -= intensity
        zeta_i[center_x - 1, center_y - 1] -= intensity

    return x_grid, y_grid, zeta_i


def generate_grid(zb, n):
    """
    Generate n points within 2pi
    :param zb: The boundary of x
    :param n: The number of point within a giving range
    :return: x
    """
    d = zb/n
    dhalf = d/2
    z_grid = np.linspace(dhalf, zb-dhalf, n)
    return z_grid


def lapace_tran(z):
    c = np.fft.fft2(z)
    c = K2 * c
    lapace = np.fft.ifft2(c)
    return lapace.real


def ilapace_tran(z):
    c = np.fft.fft2(z)
    c = iK2 * c
    ilapace = np.fft.ifft2(c)
    return ilapace.real


def pzpx(z):
    c = np.fft.fft2(z)
    c = kx * c
    pzpx = np.fft.ifft2(c)
    return pzpx.real


def pzpy(z):
    c = np.fft.fft2(z)
    c = ky * c
    pzpy = np.fft.ifft2(c)
    return pzpy.real


def tend_bve(z):
    phi = ilapace_tran(z)
    u = -1 * pzpy(phi)
    v = pzpx(phi)
    pzpt = -1 * (u * pzpx(z) + v * pzpy(z)) + nu * lapace_tran(z)
    return pzpt


if __name__ == '__main__':
    # Initial conditions
    Nx = 128
    Ny = 128
    kmax = int(Nx / 2)
    lmax = int(Ny / 2)
    yb = 2*np.pi        # The boundary of y
    xb = 2*np.pi        # The boundary of x
    dt = 0.1            # unit: second
    steps = 5000
    ode_scheme = "rk4"
    nu = 1e-4           # The vaule of nu

    # Optional Setting
    save = False
    plt = False
    ani = True

    x, y, zeta = init(Nx, Ny, turb=True)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx_temp = complex(0.0, Nx * dx) * np.fft.fftfreq(Nx, d=dx)
    ky_temp = complex(0.0, Ny * dy) * np.fft.fftfreq(Ny, d=dy)
    kx = kx_temp[:, np.newaxis]
    ky = ky_temp[np.newaxis, :]
    K2 = kx ** 2 + ky ** 2

    K2_temp = K2
    K2_temp[0, 0] = 1.0
    iK2 = 1 / K2_temp
    iK2[0, 0] = 0

    BVE = ode_solver2.Ode(zeta, tend_bve, dt, steps, debug=1)
    BVE.integrate(ode_scheme)

    if plt:
        plot = np.array((100, 200, 300, 400, 500, 1000, 2000, 5000))
        for i in range(len(plot)):
            plot_data.plot_contour(x, y, BVE.trajectory[plot[i] - 1, :, :],
                                   xl=r"$Longitude$", yl=r"$Latitude$", steps=plot[i])

    filename = 'bve_ani_center_turbulence_{0}_steps.mp4'.format(steps)
    if ani:
        print("\nPlot Process")
        plot_data.plot_contour_ani(x, y, BVE.trajectory,
                                   xl=r"$Longitude$", yl=r"$Latitude$", steps=steps, fn=filename)

