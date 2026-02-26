"""
CI_2 utility functions for clumping index and gap fraction calculations.

These functions are used for pre-processing structural data (gap fractions
and clumping indices) and are not required for the main BRF simulation.
They are included here for completeness and reproducibility.
"""

import numpy as np
from scipy.integrate import quad


# =========================================================================
# get_gFun (get_gFun.m)
# =========================================================================
def get_gfun(iorien, theta_L):
    """
    Calculate leaf angle distribution g_L(theta_L).

    Parameters
    ----------
    iorien : int
        Type of leaf normal orientation:
        1 - planophile, 2 - erectophile, 3 - plagiophile,
        4 - extremophile, 5 - uniform, 6 - spherical.
    theta_L : float
        Leaf inclination angle in radians.

    Returns
    -------
    g_L : float
        Leaf angle distribution value.
    """
    if iorien == 6:
        g_L = np.sin(theta_L)
    else:
        params = {1: (1, 2), 2: (-1, -2), 3: (-1, 4), 4: (1, 4), 5: (0, 0)}
        a, b = params[iorien]
        g_L = (2.0 / np.pi) * (1.0 + a * np.cos(b * theta_L))
    return g_L


# =========================================================================
# get_G (get_G.m)
# =========================================================================
def get_G(iorien, theta, fi):
    """
    Estimate integral G(Omega): the projection function.

    Parameters
    ----------
    iorien : int
        Type of leaf normal orientation (1-6).
    theta : float
        Zenith angle of direction Omega [degrees].
    fi : float
        Azimuth angle of direction Omega [degrees].

    Returns
    -------
    G_Fun : float
        G-function value.
    """
    theta = np.deg2rad(theta)
    fi = np.deg2rad(fi)

    n = 30
    m = 4 * n
    h_theta = 0.5 * np.pi / n
    h_fi = 2.0 * np.pi / m
    theta_i = 0.5 * h_theta
    fi_1 = 0.5 * h_fi
    G_Fun = 0.0

    for i in range(n):
        fi_j = fi_1
        c_i = np.cos(theta_i)
        s_i = np.sin(theta_i)
        xx = 0.0
        for j in range(m):
            yy = np.cos(theta) * c_i + np.sin(theta) * s_i * np.cos(fi - fi_j)
            xx += abs(yy)
            fi_j += h_fi
        xx *= h_fi
        yy = get_gfun(iorien, theta_i)
        G_Fun += yy * xx
        theta_i += h_theta

    G_Fun *= h_theta
    G_Fun /= (2.0 * np.pi)

    return G_Fun


# =========================================================================
# get_CI (get_CI.m)
# =========================================================================
def get_CI(VZA, Gap, LAI, G):
    """
    Calculate clumping index from gap fraction.

    Parameters
    ----------
    VZA : float
        View zenith angle [degrees].
    Gap : float
        Gap fraction.
    LAI : float
        Leaf area index.
    G : float
        G-function value.

    Returns
    -------
    CI : float
        Clumping index.
    """
    theta = np.deg2rad(VZA)
    CI = (-np.cos(theta) * np.log(Gap)) / (G * LAI)
    return CI


# =========================================================================
# get_Gap (get_Gap.m)
# =========================================================================
def get_Gap(VZA, CI, LAI, G):
    """
    Calculate gap fraction from clumping index.

    Parameters
    ----------
    VZA : float
        View zenith angle [degrees].
    CI : float
        Clumping index.
    LAI : float
        Leaf area index.
    G : float
        G-function value.

    Returns
    -------
    Gap : float
        Gap fraction.
    """
    theta = np.deg2rad(VZA)
    Gap = np.exp(-CI * G * LAI / np.cos(theta))
    return Gap


# =========================================================================
# volscat2 (volscat2.m) - single direction version
# =========================================================================
def volscat2(tts, ttli):
    """
    Volume scattering (single direction, solar only).

    Parameters
    ----------
    tts : float
        Sun zenith angle [degrees].
    ttli : np.ndarray
        Leaf inclination array [degrees].

    Returns
    -------
    chi_s : np.ndarray
        Extinction cross section in solar direction.
    """
    deg2rad = np.pi / 180.0

    cos_ttli = np.cos(ttli * deg2rad)
    sin_ttli = np.sin(ttli * deg2rad)

    cos_tts = np.cos(tts * deg2rad)
    sin_tts = np.sin(tts * deg2rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    As = np.maximum(Ss, Cs)

    bts = np.arccos(-Cs / As)

    chi_s = (2.0 / np.pi) * ((bts - np.pi / 2.0) * Cs + np.sin(bts) * Ss)

    return chi_s


# =========================================================================
# PHASE2 (PHASE2.m) - single direction version
# =========================================================================
def phase2(tts, lidf):
    """
    Calculate G-function and extinction coefficient (single direction).

    Parameters
    ----------
    tts : float
        Sun zenith angle [degrees].
    lidf : np.ndarray
        Leaf inclination distribution function (13 elements).

    Returns
    -------
    Gs : float
        G-function in solar direction.
    k : float
        Extinction coefficient in solar direction.
    """
    deg2rad = np.pi / 180.0
    litab = np.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])

    cos_tts = np.cos(tts * deg2rad)

    chi_s = volscat2(tts, litab)

    ksli = chi_s / cos_tts

    k = np.dot(ksli, lidf)

    Gs = k * cos_tts

    return Gs, k
