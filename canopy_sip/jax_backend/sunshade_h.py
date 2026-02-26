"""
Sunlit foliage and soil probabilities with hotspot — JAX backend.

Uses Gauss-Legendre quadrature instead of scipy.integrate.quad.
All branches use jnp.where for JIT compatibility.
"""

import jax
import jax.numpy as jnp
from .quadrature import gauss_legendre_quad

# Pre-compute canopy layer structure (nl=20 is fixed)
_NL = 20
_DX = 1.0 / _NL
_X = jnp.arange(-_DX, -1.0 - 1e-10, -_DX)  # shape (20,)
_XL = jnp.concatenate([jnp.array([0.0]), _X])  # shape (21,)


def _pso_integrand_hotspot(xl_points, K, k, CIs, CIo, LAI, q, dso):
    """
    Kuusk hotspot correlation integrand (JAX, vectorized over xl_points).

    Handles dso == 0 case using jnp.where (no Python branch).
    """
    alf = (dso / (q + 1e-30)) * 2.0 / (k + K + 1e-30)
    sqrt_kK = jnp.sqrt(K * CIo * k * CIs)

    # dso != 0 branch
    pso_nonzero = jnp.exp(
        (K * CIo + k * CIs) * LAI * xl_points +
        sqrt_kK * LAI / (alf + 1e-30) * (1.0 - jnp.exp(xl_points * alf))
    )

    # dso == 0 branch
    pso_zero = jnp.exp(
        (K * CIo + k * CIs) * LAI * xl_points -
        sqrt_kK * LAI * xl_points
    )

    return jnp.where(jnp.abs(dso) > 1e-15, pso_nonzero, pso_zero)


def _integrate_layer_hotspot(j, K, k, CIs, CIo, LAI, q, dso):
    """Integrate the Pso function over a single canopy layer j."""
    xl_j = _XL[j]
    lower = xl_j - _DX
    upper = xl_j

    def integrand(pts):
        return _pso_integrand_hotspot(pts, K, k, CIs, CIo, LAI, q, dso)

    return gauss_legendre_quad(integrand, lower, upper) / _DX


def sunshade_h(tts, tto, psi, Gs, Go, CIs, CIo, LAI, hotspot):
    """
    Calculate sunlit foliage and soil probabilities with hotspot (JAX).

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
    hotspot : float
        Hotspot parameter.

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

    # ── Case 1: exact hotspot ───────────────────────────────────────
    kc_case1 = 1.0 - jnp.exp(-Gs * CIs / cos_tts * LAI)
    kg_case1 = jnp.exp(-Gs * CIs / cos_tts * LAI)

    # ── Case 2: general calculation ─────────────────────────────────
    iLAI = LAI / _NL
    q = hotspot

    dso = jnp.sqrt(tan_tts**2 + tan_tto**2 - 2.0 * tan_tts * tan_tto * jnp.cos(psi * deg2rad))

    k = Gs / cos_tts
    K = Go / cos_tto

    # Independent probabilities
    Ps = jnp.exp(k * _XL * CIs * LAI)
    Po = jnp.exp(K * _XL * CIo * LAI)

    # Correct for finite layer thickness
    corr_s = (1.0 - jnp.exp(-k * CIs * LAI * _DX)) / (k * CIs * LAI * _DX + 1e-30)
    corr_o = (1.0 - jnp.exp(-K * CIo * LAI * _DX)) / (K * CIo * LAI * _DX + 1e-30)
    Ps_corr = Ps.at[:_NL].set(Ps[:_NL] * corr_s)
    Po_corr = Po.at[:_NL].set(Po[:_NL] * corr_o)

    # Vectorized integration over all 21 layers
    layer_indices = jnp.arange(_NL + 1)
    Pso = jax.vmap(
        lambda j: _integrate_layer_hotspot(j, K, k, CIs, CIo, LAI, q, dso)
    )(layer_indices)

    # Rounding error correction
    Pso = jnp.where(Pso > Po_corr, jnp.minimum(Po_corr, Ps_corr), Pso)
    Pso = jnp.where(Pso > Ps_corr, jnp.minimum(Po_corr, Ps_corr), Pso)

    kc_case2 = iLAI * CIo * K * jnp.sum(Pso[:_NL])
    kg_case2 = Pso[_NL]

    # ── Select case ─────────────────────────────────────────────────
    is_hotspot = (tts == tto) & (psi == 0.0)
    kc = jnp.where(is_hotspot, kc_case1, kc_case2)
    kg = jnp.where(is_hotspot, kg_case1, kg_case2)

    return kc, kg
