"""
Sunlit foliage and soil probabilities without hotspot — JAX backend.
"""

import jax
import jax.numpy as jnp
from .quadrature import gauss_legendre_quad

_NL = 20
_DX = 1.0 / _NL
_X = jnp.arange(-_DX, -1.0 - 1e-10, -_DX)
_XL = jnp.concatenate([jnp.array([0.0]), _X])


def _pso_integrand_independent(xl_points, K, k, CIs, CIo, LAI):
    """Independent joint probability: Pso = Ps * Po."""
    return jnp.exp((K * CIo + k * CIs) * LAI * xl_points)


def _integrate_layer_independent(j, K, k, CIs, CIo, LAI):
    """Integrate independent Pso over a single canopy layer j."""
    xl_j = _XL[j]
    lower = xl_j - _DX
    upper = xl_j

    def integrand(pts):
        return _pso_integrand_independent(pts, K, k, CIs, CIo, LAI)

    return gauss_legendre_quad(integrand, lower, upper) / _DX


def sunshade_kt_he(tts, tto, psi, Gs, Go, CIs, CIo, LAI):
    """
    Calculate sunlit foliage and soil probabilities without hotspot (JAX).

    Parameters
    ----------
    tts, tto, psi : float
        Sun zenith, view zenith, relative azimuth [degrees].
    Gs, Go : float
        G-function values.
    CIs, CIo : float
        Clumping indices.
    LAI : float
        Leaf Area Index.

    Returns
    -------
    kc, kg : float
        Sunlit foliage and soil probabilities.
    """
    deg2rad = jnp.pi / 180.0

    cos_tts = jnp.cos(tts * deg2rad)
    tan_tto = jnp.tan(tto * deg2rad)
    cos_tto = jnp.cos(tto * deg2rad)
    tan_tts = jnp.tan(tts * deg2rad)

    psi = jnp.abs(psi - 360.0 * jnp.round(psi / 360.0))

    # ── Case 1: exact overlap ───────────────────────────────────────
    kc_case1 = 1.0 - jnp.exp(-Gs * CIs / cos_tts * LAI)
    kg_case1 = jnp.exp(-Gs * CIs / cos_tts * LAI)

    # ── Case 2: general calculation ─────────────────────────────────
    iLAI = LAI / _NL

    k = Gs / cos_tts
    K = Go / cos_tto

    Ps = jnp.exp(k * _XL * CIs * LAI)
    Po = jnp.exp(K * _XL * CIo * LAI)

    corr_s = (1.0 - jnp.exp(-k * CIs * LAI * _DX)) / (k * CIs * LAI * _DX + 1e-30)
    corr_o = (1.0 - jnp.exp(-K * CIo * LAI * _DX)) / (K * CIo * LAI * _DX + 1e-30)
    Ps_corr = Ps.at[:_NL].set(Ps[:_NL] * corr_s)
    Po_corr = Po.at[:_NL].set(Po[:_NL] * corr_o)

    layer_indices = jnp.arange(_NL + 1)
    Pso = jax.vmap(
        lambda j: _integrate_layer_independent(j, K, k, CIs, CIo, LAI)
    )(layer_indices)

    Pso = jnp.where(Pso > Po_corr, jnp.minimum(Po_corr, Ps_corr), Pso)
    Pso = jnp.where(Pso > Ps_corr, jnp.minimum(Po_corr, Ps_corr), Pso)

    kc_case2 = iLAI * CIo * K * jnp.sum(Pso[:_NL])
    kg_case2 = Pso[_NL]

    # ── Select case ─────────────────────────────────────────────────
    is_overlap = (tts == tto) & (psi == 0.0)
    kc = jnp.where(is_overlap, kc_case1, kc_case2)
    kg = jnp.where(is_overlap, kg_case1, kg_case2)

    return kc, kg
