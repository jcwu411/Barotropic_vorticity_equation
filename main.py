#!/usr/bin/env python3.8
"""
Created on 16/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import ode_solver
import plot_data


def init(nx, ny):
    x_grid = generate_grid(xb, nx)
    y_grid = generate_grid(yb, ny)
    zeta_i = np.zeros((nx, ny))
    zeta_i[:, 64:74] = 1
    zeta_i[:, 53:64] = -1
    u_test = np.zeros((nx, ny))
    ih = 64 + 10
    il = 64 - 10
    for j in range(nx):
        for i in range(64, ih):
            u_test[j, i] = -1 * y_grid[i] + y_grid[ih - 1]
        for i in range(il, 64):
            u_test[j, i] = y_grid[i] - y_grid[il]

    return x_grid, y_grid, zeta_i, u_test


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


def tend_psm(y):
    c = np.fft.fft(y)
    for k in range(kmax):
        c[k] = c[k] * jj * k
        c[-k] = c[-k] * jj * (-k)
    c[kmax] = c[kmax] * jj * kmax
    pypx = np.fft.ifft(c)
    return


def lapace_tran(z):
    c = np.fft.fft2(z)
    for ly in range(lmax):
        for k in range(kmax):
            if (k == 0) & (ly == 0):
                c[k, ly] = c[k, ly]
            else:
                c[k, ly] = c[k, ly] * ((jj * k) ** 2 + (jj * ly) ** 2)
                c[k, -ly] = c[k, -ly] * ((jj * k) ** 2 + (jj * (-ly)) ** 2)
                c[-k, ly] = c[-k, ly] * ((jj * (-k)) ** 2 + (jj * ly) ** 2)
                c[-k, -ly] = c[-k, -ly] * ((jj * (-k)) ** 2 + (jj * (-ly)) ** 2)
        c[kmax, ly] = c[kmax, ly] * ((jj * kmax) ** 2 + (jj * ly) ** 2)
        c[kmax, -ly] = c[kmax, -ly] * ((jj * kmax) ** 2 + (jj * (-ly)) ** 2)
    c[kmax, lmax] = c[kmax, lmax] * ((jj * kmax) ** 2 + (jj * lmax) ** 2)

    for k in range(kmax):
        c[k, lmax] = c[k, lmax] * ((jj * k) ** 2 + (jj * lmax) ** 2)
        c[-k, lmax] = c[-k, lmax] * ((jj * (-k)) ** 2 + (jj * lmax) ** 2)

    lapace = np.fft.ifft2(c)
    return lapace.real


def inv_lapace_tran(z):
    c = np.fft.fft2(z)
    for ly in range(lmax):
        for k in range(kmax):
            if (k == 0) & (ly == 0):
                c[k, ly] = c[k, ly]
            else:
                c[k, ly] = c[k, ly] / ((jj * k) ** 2 + (jj * ly) ** 2)
                c[k, -ly] = c[k, -ly] / ((jj * k) ** 2 + (jj * (-ly)) ** 2)
                c[-k, ly] = c[-k, ly] / ((jj * (-k)) ** 2 + (jj * ly) ** 2)
                c[-k, -ly] = c[-k, -ly] / ((jj * (-k)) ** 2 + (jj * (-ly)) ** 2)
        c[kmax, ly] = c[kmax, ly] / ((jj * kmax) ** 2 + (jj * ly) ** 2)
        c[kmax, -ly] = c[kmax, -ly] / ((jj * kmax) ** 2 + (jj * (-ly)) ** 2)
    c[kmax, lmax] = c[kmax, lmax] / ((jj * kmax) ** 2 + (jj * lmax) ** 2)

    for k in range(kmax):
        c[k, lmax] = c[k, lmax] / ((jj * k) ** 2 + (jj * lmax) ** 2)
        c[-k, lmax] = c[-k, lmax] / ((jj * (-k)) ** 2 + (jj * lmax) ** 2)

    phi = np.fft.ifft2(c)
    return phi.real


def pzpx_2d(z):
    cx = np.fft.fft2(z)
    for ly in range(lmax):
        for k in range(kmax):
            cx[k, ly] = cx[k, ly] * (jj * k)
            cx[k, -ly] = cx[k, -ly] * (jj * k)
            cx[-k, ly] = cx[-k, ly] * (jj * (-k))
            cx[-k, -ly] = cx[-k, -ly] * (jj * (-k))
        cx[kmax, ly] = cx[kmax, ly] * (jj * kmax)
        cx[kmax, -ly] = cx[kmax, -ly] * (jj * kmax)

    for k in range(kmax):
        cx[k, lmax] = cx[k, lmax] * (jj * k)
        cx[-k, lmax] = cx[-k, lmax] * (jj * (-k))

    cx[kmax, lmax] = cx[kmax, lmax] * (jj * kmax)

    pzpx = np.fft.ifft2(cx)
    return pzpx.real


def pzpy_2d(z):
    cy = np.fft.fft2(z)
    for k in range(kmax):
        for ly in range(lmax):
            cy[k, ly] = cy[k, ly] * (jj * ly)
            cy[k, -ly] = cy[k, -ly] * (jj * (-ly))
            cy[-k, ly] = cy[-k, ly] * (jj * ly)
            cy[-k, -ly] = cy[-k, -ly] * (jj * (-ly))
        cy[k, lmax] = cy[k, lmax] * (jj * lmax)
        cy[-k, lmax] = cy[-k, lmax] * (jj * lmax)

    for ly in range(lmax):
        cy[kmax, ly] = cy[kmax, ly] * (jj * ly)
        cy[kmax, -ly] = cy[kmax, -ly] * (jj * (-ly))

    cy[kmax, lmax] = cy[kmax, lmax] * (jj * lmax)

    pzpy = np.fft.ifft2(cy)
    return pzpy.real


def pzpx_2d_c(cx):
    for ly in range(lmax):
        for k in range(kmax):
            cx[k, ly] = cx[k, ly] * (jj * k) ** 2
            cx[k, -ly] = cx[k, -ly] * (jj * k) ** 2
            cx[-k, ly] = cx[-k, ly] * (jj * (-k)) ** 2
            cx[-k, -ly] = cx[-k, -ly] * (jj * (-k)) ** 2
        cx[kmax, ly] = cx[kmax, ly] * (jj * kmax) ** 2
        cx[kmax, -ly] = cx[kmax, -ly] * (jj * kmax) ** 2

    for k in range(kmax):
        cx[k, lmax] = cx[k, lmax] * (jj * k) ** 2
        cx[-k, lmax] = cx[-k, lmax] * (jj * (-k)) ** 2

    cx[kmax, lmax] = cx[kmax, lmax] * (jj * kmax) ** 2
    return cx


def pzpy_2d_c(cy):
    for k in range(kmax):
        for ly in range(lmax):
            cy[k, ly] = cy[k, ly] * (jj * ly) ** 2
            cy[k, -ly] = cy[k, -ly] * (jj * (-ly)) ** 2
            cy[-k, ly] = cy[-k, ly] * (jj * ly) ** 2
            cy[-k, -ly] = cy[-k, -ly] * (jj * (-ly)) ** 2
        cy[k, lmax] = cy[k, lmax] * (jj * lmax) ** 2
        cy[-k, lmax] = cy[-k, lmax] * (jj * lmax) ** 2

    for ly in range(lmax):
        cy[kmax, ly] = cy[kmax, ly] * (jj * ly) ** 2
        cy[kmax, -ly] = cy[kmax, -ly] * (jj * (-ly)) ** 2

    cy[kmax, lmax] = cy[kmax, lmax] * (jj * lmax) ** 2

    return cy


if __name__ == '__main__':
    Nx = 128
    Ny = 128
    yb = 2*np.pi
    xb = 2*np.pi
    save = False
    x, y, zeta, u_t = init(Nx, Ny)
    # print("\nx = ", x)
    # print("\ny = ", y)
    dx = x[1] - x[0]
    dt = 0.1       # unit: second
    steps = 100
    kmax = int(Nx / 2)
    lmax = int(Ny / 2)
    jj = (0+1j)     # The imaginary unit
    ode_scheme = "rk4"

    phi = inv_lapace_tran(zeta)
    zeta_r = lapace_tran(phi)
    print("\n L(phi) = ", zeta_r[1, 63])
    v = pzpx_2d(phi)
    u = -1 * pzpy_2d(phi)
    zeta_rr = pzpx_2d(v) - pzpy_2d(u)
    print("\n The curl of velocity = ", zeta_rr[1, 62:66])
    cr = np.fft.fft2(phi)
    zeta_rrr = np.fft.ifft2(pzpx_2d_c(cr) + pzpy_2d_c(cr)).real
    print("\n (p2px2 + p2px2)(phi) = ", zeta_rrr[1, 63])

    # print("\n Method 2: Pseudo Spectral Method")
    # PSM = ode_solver.Ode(y0, tend_psm, dt, steps, debug=0)
    # PSM.integrate(ode_scheme)

    plot_data.plot_u(u[63, :], y, color='black', lw=2, lb="U", ti="Plot", xl="U", yl="Y", legendloc=4,
                     xlim=(-1, 2*np.pi), ylim=(0, yb), ylog=False, fn="plot2d.pdf", sa=False)
    #  plot_data.plot_contour(x, y, zeta, xl="X", yl="Y")
    # plot_data.plot_contour(x, y, phi, xl="X", yl="Y")
