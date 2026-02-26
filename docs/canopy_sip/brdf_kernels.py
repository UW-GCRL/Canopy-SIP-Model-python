"""
Kernel-driven BRDF model fitting and albedo computation.

Implements the RossThick-LiSparseR (RTLSR) kernel model, the MODIS MCD43
standard for BRDF characterization (Lucht et al., 2000; Schaaf et al., 2002).

BRF(theta_s, theta_v, phi) = f_iso + f_vol * K_vol + f_geo * K_geo
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════
# Kernel functions
# ═════════════════════════════════════════════════════════════════════════

def ross_thick(sza, vza, raa):
    """
    RossThick volumetric scattering kernel.

    Parameters
    ----------
    sza, vza : float or np.ndarray
        Sun / view zenith angle in degrees.
    raa : float or np.ndarray
        Relative azimuth angle in degrees.

    Returns
    -------
    K_vol : float or np.ndarray
    """
    sza_r = np.deg2rad(np.asarray(sza, dtype=np.float64))
    vza_r = np.deg2rad(np.asarray(vza, dtype=np.float64))
    raa_r = np.deg2rad(np.asarray(raa, dtype=np.float64))

    cos_xi = (np.cos(sza_r) * np.cos(vza_r)
              + np.sin(sza_r) * np.sin(vza_r) * np.cos(raa_r))
    cos_xi = np.clip(cos_xi, -1.0, 1.0)
    xi = np.arccos(cos_xi)

    denom = np.cos(sza_r) + np.cos(vza_r)
    denom = np.maximum(denom, 1e-10)

    K_vol = ((np.pi / 2.0 - xi) * cos_xi + np.sin(xi)) / denom - np.pi / 4.0
    return K_vol


def li_sparse_r(sza, vza, raa, h_b=2.0, b_r=1.0):
    """
    LiSparseR (reciprocal) geometric-optical kernel.

    MODIS standard parameters: h/b=2, b/r=1.

    Parameters
    ----------
    sza, vza : float or np.ndarray
        Sun / view zenith angle in degrees.
    raa : float or np.ndarray
        Relative azimuth angle in degrees.
    h_b : float
        Crown height-to-width ratio.
    b_r : float
        Crown shape ratio.

    Returns
    -------
    K_geo : float or np.ndarray
    """
    sza_r = np.deg2rad(np.asarray(sza, dtype=np.float64))
    vza_r = np.deg2rad(np.asarray(vza, dtype=np.float64))
    raa_r = np.deg2rad(np.asarray(raa, dtype=np.float64))

    # Transform angles for crown shape
    sza_p = np.arctan(b_r * np.tan(sza_r))
    vza_p = np.arctan(b_r * np.tan(vza_r))

    cos_sza_p = np.maximum(np.cos(sza_p), 1e-10)
    cos_vza_p = np.maximum(np.cos(vza_p), 1e-10)
    sec_sza_p = 1.0 / cos_sza_p
    sec_vza_p = 1.0 / cos_vza_p

    tan_sza_p = np.tan(sza_p)
    tan_vza_p = np.tan(vza_p)

    # Distance term
    D_sq = (tan_sza_p ** 2 + tan_vza_p ** 2
            - 2.0 * tan_sza_p * tan_vza_p * np.cos(raa_r))
    D = np.sqrt(np.maximum(D_sq, 0.0))

    # Overlap area
    cos_t = (h_b * np.sqrt(D ** 2 + (tan_sza_p * tan_vza_p * np.sin(raa_r)) ** 2)
             / (sec_sza_p + sec_vza_p))
    cos_t = np.clip(cos_t, -1.0, 1.0)
    t = np.arccos(cos_t)

    O = (1.0 / np.pi) * (t - np.sin(t) * np.cos(t)) * (sec_sza_p + sec_vza_p)
    O = np.maximum(O, 0.0)

    # Phase angle for transformed angles
    cos_xi_p = (np.cos(sza_p) * np.cos(vza_p)
                + np.sin(sza_p) * np.sin(vza_p) * np.cos(raa_r))
    cos_xi_p = np.clip(cos_xi_p, -1.0, 1.0)

    K_geo = (O - sec_sza_p - sec_vza_p
             + 0.5 * (1.0 + cos_xi_p) * sec_sza_p * sec_vza_p)
    return K_geo


# ═════════════════════════════════════════════════════════════════════════
# Model fitting
# ═════════════════════════════════════════════════════════════════════════

def fit_brdf_kernels(sza, vza_array, raa_array, brf_array):
    """
    Fit RossThick-LiSparseR kernel model to BRF observations.

    Parameters
    ----------
    sza : float
        Sun zenith angle in degrees.
    vza_array, raa_array : np.ndarray
        View zenith and relative azimuth angles in degrees.
    brf_array : np.ndarray
        Observed BRF values.

    Returns
    -------
    f_iso, f_vol, f_geo : float
        Kernel weights.
    r_squared : float
        Goodness of fit.
    brf_predicted : np.ndarray
        Predicted BRF from the fitted model.
    """
    K_vol = ross_thick(sza, vza_array, raa_array)
    K_geo = li_sparse_r(sza, vza_array, raa_array)

    A = np.column_stack([np.ones_like(vza_array, dtype=np.float64), K_vol, K_geo])
    result, _, _, _ = np.linalg.lstsq(A, brf_array, rcond=None)
    f_iso, f_vol, f_geo = result

    brf_predicted = A @ result
    ss_res = np.sum((brf_array - brf_predicted) ** 2)
    ss_tot = np.sum((brf_array - np.mean(brf_array)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return f_iso, f_vol, f_geo, r_squared, brf_predicted


def predict_brdf(f_iso, f_vol, f_geo, sza, vza, raa):
    """Predict BRF at arbitrary view geometry using fitted kernel weights."""
    return f_iso + f_vol * ross_thick(sza, vza, raa) + f_geo * li_sparse_r(sza, vza, raa)


# ═════════════════════════════════════════════════════════════════════════
# Albedo computation
# ═════════════════════════════════════════════════════════════════════════

def compute_bsa(f_iso, f_vol, f_geo, sza):
    """
    Black-sky albedo (directional-hemispherical reflectance).

    Polynomial integration weights from Lucht et al. (2000).

    Parameters
    ----------
    f_iso, f_vol, f_geo : float
        Kernel weights.
    sza : float
        Sun zenith angle in degrees.

    Returns
    -------
    bsa : float
    """
    s = np.deg2rad(sza)
    s2 = s * s
    s3 = s2 * s

    g_iso = 1.0
    g_vol = -0.007574 + (-0.070987) * s2 + 0.307588 * s3
    g_geo = -1.284909 + (-0.166314) * s2 + 0.041840 * s3

    return f_iso * g_iso + f_vol * g_vol + f_geo * g_geo


def compute_wsa(f_iso, f_vol, f_geo):
    """
    White-sky albedo (bihemispherical reflectance).

    Integration constants from Schaaf et al. (2002).

    Parameters
    ----------
    f_iso, f_vol, f_geo : float
        Kernel weights.

    Returns
    -------
    wsa : float
    """
    return f_iso * 1.0 + f_vol * 0.189184 + f_geo * (-1.377622)


# ═════════════════════════════════════════════════════════════════════════
# Hemisphere prediction grid
# ═════════════════════════════════════════════════════════════════════════

def generate_hemisphere_brf(f_iso, f_vol, f_geo, sza,
                            vza_max=60, vza_step=2, raa_step=5):
    """
    Generate a 2-D BRF prediction grid over the viewing hemisphere.

    Returns
    -------
    vza_grid, raa_grid : np.ndarray (2-D)
        Meshgrid arrays for VZA and RAA.
    brf_grid : np.ndarray (2-D)
        Predicted BRF at each grid point.
    """
    vza_1d = np.arange(0, vza_max + vza_step, vza_step)
    raa_1d = np.arange(0, 360 + raa_step, raa_step)
    raa_grid, vza_grid = np.meshgrid(raa_1d, vza_1d)

    brf_grid = predict_brdf(f_iso, f_vol, f_geo, sza, vza_grid, raa_grid)
    return vza_grid, raa_grid, brf_grid


# ═════════════════════════════════════════════════════════════════════════
# Polar plot
# ═════════════════════════════════════════════════════════════════════════

def plot_polar_brf(sza, vza_obs, raa_obs, brf_obs,
                   vza_grid=None, raa_grid=None, brf_grid=None):
    """
    Create a polar BRF plot with observed data and hemisphere heatmap.

    Parameters
    ----------
    sza : float
        Sun zenith angle for annotation.
    vza_obs, raa_obs : np.ndarray
        Observed absolute VZA and RAA in degrees.
    brf_obs : np.ndarray
        Observed BRF values.
    vza_grid, raa_grid, brf_grid : np.ndarray, optional
        Pre-computed hemisphere prediction grids.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    # Hemisphere heatmap from kernel model
    if brf_grid is not None:
        raa_rad = np.deg2rad(raa_grid)
        mesh = ax.pcolormesh(raa_rad, vza_grid, brf_grid,
                             cmap='YlOrRd', shading='auto', alpha=0.85)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.75)
        cbar.set_label('BRF (kernel model)', fontsize=9)

    # Observed principal-plane data points
    raa_obs_rad = np.deg2rad(raa_obs)
    ax.scatter(raa_obs_rad, vza_obs, c=brf_obs, cmap='YlOrRd',
               edgecolors='black', s=70, zorder=5, linewidths=1.2,
               vmin=brf_grid.min() if brf_grid is not None else None,
               vmax=brf_grid.max() if brf_grid is not None else None)

    # Sun position marker
    ax.plot(np.deg2rad(0), sza, '*', color='gold', markersize=14,
            markeredgecolor='black', markeredgewidth=0.8, zorder=10,
            label=f'Sun (SZA={sza}\u00b0)')

    # Configure polar axis
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_rlim(0, 65)
    ax.set_title(f'Hemisphere BRF  (SZA = {sza}\u00b0)', pad=18, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.08), fontsize=8)

    fig.tight_layout()
    return fig
