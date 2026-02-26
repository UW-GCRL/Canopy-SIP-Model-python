"""
Canopy-SIP Model — JAX backend.

Fully JIT-compilable, auto-differentiable, and GPU-accelerable.
Provides `jax.grad` for parameter sensitivity and inverse problems,
and `jax.vmap` for efficient batch simulations.

Usage:
    from canopy_sip.jax_backend import simulate_brf, simulate_brf_jit
    from canopy_sip.jax_backend import grad_brf, batch_simulate

    # Single simulation (JIT-compiled)
    BRF = simulate_brf_jit(params, gap_data, lidf)

    # Gradient of BRF w.r.t. parameters
    grads = grad_brf(params, gap_data, lidf)

    # Batch simulation over multiple SZA values
    BRFs = batch_simulate(param_batch, gap_data, lidf)
"""

import jax
import jax.numpy as jnp
from functools import partial

from .campbell import campbell
from .dladgen import dladgen, dcum  # noqa: F401
from .cixy import cixy
from .phase import phase
from .get_hsf_go import get_hsf_go
from .sunshade_h import sunshade_h
from .sunshade_kt_he import sunshade_kt_he


# ═════════════════════════════════════════════════════════════════════════
# Data structures (as JAX-friendly dicts / arrays)
# ═════════════════════════════════════════════════════════════════════════

def make_params(SZA=0.0, SAA=0.0, Crowndeepth=12.8675, Height=20.0,
                Height_c=6.634, dthr=0.41234, bl=0.1, iD=0.58073,
                LAI=5.0, D=0.0, rho=0.4957, tau=0.4409, rg=0.159):
    """
    Create a parameter dictionary for the JAX simulation.

    All values are JAX-traced floats so the simulation is differentiable
    with respect to any of them.
    """
    return {
        'SZA': jnp.float64(SZA),
        'SAA': jnp.float64(SAA),
        'Crowndeepth': jnp.float64(Crowndeepth),
        'Height': jnp.float64(Height),
        'Height_c': jnp.float64(Height_c),
        'dthr': jnp.float64(dthr),
        'bl': jnp.float64(bl),
        'iD': jnp.float64(iD),
        'LAI': jnp.float64(LAI),
        'D': jnp.float64(D),
        'rho': jnp.float64(rho),
        'tau': jnp.float64(tau),
        'rg': jnp.float64(rg),
    }


def make_gap_data(gap_tot_vals, gap_within_vals, gap_betw_vals, CI_within_vals):
    """
    Create gap data arrays for the JAX simulation.

    Parameters
    ----------
    gap_tot_vals : array-like, shape (13,)
        Total gap fraction (column 3 of gap_tot.csv).
    gap_within_vals : array-like, shape (13,)
        Within-crown gap fraction.
    gap_betw_vals : array-like, shape (13,)
        Between-crown gap fraction.
    CI_within_vals : array-like, shape (13,)
        Within-crown clumping index.

    Returns
    -------
    dict of jnp.ndarray
    """
    return {
        'gap_tot': jnp.array(gap_tot_vals, dtype=jnp.float64),
        'gap_within': jnp.array(gap_within_vals, dtype=jnp.float64),
        'gap_betw': jnp.array(gap_betw_vals, dtype=jnp.float64),
        'CI_within': jnp.array(CI_within_vals, dtype=jnp.float64),
    }


# ═════════════════════════════════════════════════════════════════════════
# View angle geometry (fixed, 13 angles in principal plane)
# ═════════════════════════════════════════════════════════════════════════

# Forward VZA: 60, 50, 40, 30, 20, 10, 0
# Backward VZA: 10, 20, 30, 40, 50, 60
_VA_VZA = jnp.array([60., 50., 40., 30., 20., 10., 0., 10., 20., 30., 40., 50., 60.])
_VA_VAA = jnp.array([0., 0., 0., 0., 0., 0., 0., 180., 180., 180., 180., 180., 180.])
_SIGNED_VZA = jnp.array([-60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60.])


# ═════════════════════════════════════════════════════════════════════════
# Core BRF computation for a SINGLE viewing angle
# ═════════════════════════════════════════════════════════════════════════

def _compute_brf_single(t, params, gap_data, lidf,
                        gap_S, gap_S_within, gap_S_tot,
                        c, HotSpotPar, go_par, w):
    """
    Compute BRF for a single viewing angle index t.

    This function is designed to be vmapped over t.
    """
    SZA = params['SZA']
    SAA = params['SAA']
    LAI = params['LAI']
    rho_leaf = params['rho']
    tau_leaf = params['tau']
    rg = params['rg']
    iD = params['iD']
    D = params['D']
    Height = params['Height']
    Crowndeepth = params['Crowndeepth']
    Height_c = params['Height_c']

    tto = _VA_VZA[t]
    psi = _VA_VAA[t]

    # Ensure relative azimuth angle is symmetric
    psi = jnp.where(psi > 180.0, psi - 360.0, psi)
    psi = jnp.abs(psi)
    psi = jnp.abs(psi - 360.0 * jnp.round(psi / 360.0))

    CIy1 = 1.0
    CIy2 = 1.0
    CIs = cixy(CIy1, CIy2, SZA)
    CIo = cixy(CIy1, CIy2, tto)

    # 2.1 Directional gap extraction
    gap_V_tot = gap_data['gap_tot'][t]
    gap_V_within = gap_data['gap_within'][t]
    gap_V_betw = gap_data['gap_betw'][t]

    Ps_dir_go = gap_S
    Pv_dir_go = gap_V_betw

    # 2.2 Four GO components
    Kg = (Ps_dir_go * Pv_dir_go +
          get_hsf_go(go_par, SZA, SAA, tto, psi, Ps_dir_go, Pv_dir_go, Height_c))
    Kz = Pv_dir_go - Kg
    Kct = 1.0 - Pv_dir_go

    delta_angle = (jnp.cos(jnp.deg2rad(SZA)) * jnp.cos(jnp.deg2rad(tto)) +
                   jnp.sin(jnp.deg2rad(SZA)) * jnp.sin(jnp.deg2rad(tto)) *
                   jnp.cos(jnp.deg2rad(psi - SAA)))
    phi = jnp.rad2deg(jnp.arccos(jnp.clip(delta_angle, -1.0, 1.0)))
    delta_val = jnp.cos(jnp.deg2rad(phi * (1.0 - jnp.sin(jnp.pi * c / 2.0))))

    is_continuous = ((Height - Crowndeepth) < Crowndeepth) & (tto > SZA) & (SAA == psi)
    Kc = jnp.where(is_continuous, Kct, 0.5 * (1.0 + delta_val) * Kct)
    Kt = Kct - Kc

    # 2.3 Vegetation single scattering
    Ps_dir_inKz = gap_S_within

    Gs, Go, k_ext, K_ext, sob, sof = phase(SZA, tto, psi, lidf)

    kc_val, kg_val = sunshade_h(SZA, tto, psi, Gs, Go, CIs, CIo, LAI, HotSpotPar)
    kc_kt, kg_kt = sunshade_kt_he(SZA, tto, psi, Gs, Go, CIs, CIo, LAI)

    wso = sob * rho_leaf + sof * tau_leaf

    BRF_v1 = wso * kc_val / K_ext
    BRF_v1_kt = jnp.sqrt(Ps_dir_inKz) * wso * kc_kt / K_ext

    # 2.4 Soil contribution
    BRFsc = kg_val * rg
    BRFs_kt = kg_kt * rg

    # 2.5 Multiple scattering (p-theory)
    i0 = 1.0 - gap_S_tot
    i0 = D * iD + (1.0 - D) * i0
    iv = 1.0 - gap_V_tot
    t0 = 1.0 - i0
    tv = 1.0 - iv

    p = 1.0 - iD / LAI
    rho2 = iv / 2.0 / LAI
    rho_hemi2 = iD / 2.0 / LAI

    Tdn = t0 + i0 * w * rho_hemi2 / (1.0 - p * w)
    Tup_o = tv + iD * w * rho2 / (1.0 - p * w)
    Rdn = iD * w * rho_hemi2 / (1.0 - p * w)

    BRF_vm = i0 * w**2 * p * rho2 / (1.0 - p * w)
    BRFm = rg * Tdn * Tup_o / (1.0 - rg * Rdn) - t0 * rg * tv

    # 3. Total BRF
    brf = (Kc * (BRFsc + BRF_v1) +
           Kt * (BRFs_kt + BRF_v1_kt) +
           (Kg + Kz * Ps_dir_inKz) * rg +
           BRFm + BRF_vm)

    return brf


# ═════════════════════════════════════════════════════════════════════════
# Main simulation function
# ═════════════════════════════════════════════════════════════════════════

def simulate_brf(params, gap_data, lidf):
    """
    Run the full BRF simulation over 13 principal-plane viewing angles (JAX).

    This is a pure function: no side effects, fully differentiable.

    Parameters
    ----------
    params : dict
        Model parameters (from `make_params`).
    gap_data : dict
        Gap fraction data (from `make_gap_data`).
    lidf : jnp.ndarray (13,)
        Leaf inclination distribution function.

    Returns
    -------
    BRF3 : jnp.ndarray (13,)
        BRF values for 13 viewing angles.
    """
    # Derived quantities (nadir = index 6)
    gap_H = gap_data['gap_betw'][6]
    gap_H_within = gap_data['gap_within'][6]
    CI_H_within = gap_data['CI_within'][6]

    gap_S = gap_data['gap_betw'][6]
    gap_S_within = gap_data['gap_within'][6]
    gap_S_tot = gap_S + gap_S_within

    HotSpotPar = params['bl'] / params['Height']
    c1 = CI_H_within
    c2 = 1.0 - gap_H
    c = c1 * c2
    w = params['rho'] + params['tau']
    go_par = params['dthr'] * params['Crowndeepth']

    # Vectorize over 13 viewing angles using vmap
    t_indices = jnp.arange(13)

    BRF3 = jax.vmap(
        lambda t: _compute_brf_single(
            t, params, gap_data, lidf,
            gap_S, gap_S_within, gap_S_tot,
            c, HotSpotPar, go_par, w
        )
    )(t_indices)

    return BRF3


# ═════════════════════════════════════════════════════════════════════════
# JIT-compiled version
# ═════════════════════════════════════════════════════════════════════════

simulate_brf_jit = jax.jit(simulate_brf)


# ═════════════════════════════════════════════════════════════════════════
# Gradient functions
# ═════════════════════════════════════════════════════════════════════════

def grad_brf(params, gap_data, lidf):
    """
    Compute the gradient of the mean BRF with respect to all parameters.

    Useful for sensitivity analysis and inverse problems.

    Returns
    -------
    grads : dict
        Dictionary with the same keys as `params`, each containing the
        partial derivative of mean(BRF) with respect to that parameter.
    """
    def mean_brf(p):
        return jnp.mean(simulate_brf(p, gap_data, lidf))

    return jax.grad(mean_brf)(params)


def jacobian_brf(params, gap_data, lidf):
    """
    Compute the full Jacobian: d(BRF_i) / d(param_j) for all i, j.

    Returns
    -------
    jac : dict
        Dictionary with the same keys as `params`, each containing a
        (13,) array of partial derivatives.
    """
    return jax.jacfwd(lambda p: simulate_brf(p, gap_data, lidf))(params)


# ═════════════════════════════════════════════════════════════════════════
# Batch simulation (vectorized over parameter sets)
# ═════════════════════════════════════════════════════════════════════════

def batch_simulate(params_batch, gap_data, lidf):
    """
    Run simulations for a batch of parameter sets.

    Parameters
    ----------
    params_batch : dict
        Each value is a (N,) array of parameter values.
    gap_data : dict
        Gap fraction data (shared across batch).
    lidf : jnp.ndarray (13,)
        LIDF (shared across batch).

    Returns
    -------
    BRF_batch : jnp.ndarray (N, 13)
        BRF values for each parameter set.
    """
    return jax.vmap(lambda p: simulate_brf(p, gap_data, lidf))(params_batch)


# ═════════════════════════════════════════════════════════════════════════
# Convenience: high-level run function matching numpy API
# ═════════════════════════════════════════════════════════════════════════

def run_simulation(gap_tot_vals=None, gap_within_vals=None,
                   gap_betw_vals=None, CI_within_vals=None,
                   TypeLidf=2, LIDFa=57.3, LIDFb=0.0,
                   **param_kwargs):
    """
    High-level simulation function matching the numpy API.

    Parameters
    ----------
    gap_*_vals : array-like, shape (13,), optional
        Gap fraction data. If None, loaded from CSV files.
    TypeLidf : int
        1 = Two-parameter, 2 = Campbell.
    LIDFa, LIDFb : float
        Leaf angle distribution parameters.
    **param_kwargs
        Any parameters accepted by `make_params`.

    Returns
    -------
    result : dict
        'BRF3': jnp.ndarray (13,), 'signed_vza': jnp.ndarray (13,).
    """
    import os
    import numpy as np

    # Load gap data if not provided
    if gap_tot_vals is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        gap_tot_vals = np.loadtxt(os.path.join(data_dir, 'gap_tot.csv'), delimiter=',', skiprows=1)[:, 2]
        gap_within_vals = np.loadtxt(os.path.join(data_dir, 'gap_within.csv'), delimiter=',', skiprows=1)[:, 2]
        gap_betw_vals = np.loadtxt(os.path.join(data_dir, 'gap_betw.csv'), delimiter=',', skiprows=1)[:, 2]
        CI_within_vals = np.loadtxt(os.path.join(data_dir, 'CI_within.csv'), delimiter=',', skiprows=1)[:, 2]

    gap_data = make_gap_data(gap_tot_vals, gap_within_vals, gap_betw_vals, CI_within_vals)
    params = make_params(**param_kwargs)

    # Compute LIDF
    if TypeLidf == 2:
        lidf, _ = campbell(jnp.float64(LIDFa))
    else:
        lidf, _ = dladgen(jnp.float64(LIDFa), jnp.float64(LIDFb))

    BRF3 = simulate_brf(params, gap_data, lidf)

    return {
        'BRF3': BRF3,
        'signed_vza': _SIGNED_VZA,
    }
