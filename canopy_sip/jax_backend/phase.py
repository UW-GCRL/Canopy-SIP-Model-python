"""
Scattering phase function calculation — JAX backend.
"""

import jax.numpy as jnp
from .volscat import volscat

_LITAB = jnp.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])


def phase(tts, tto, psi, lidf):
    """
    Calculate the scattering phase function (JAX).

    Parameters
    ----------
    tts, tto, psi : float
        Sun zenith, view zenith, relative azimuth [degrees].
    lidf : jnp.ndarray (13,)
        Leaf inclination distribution function.

    Returns
    -------
    Gs, Go, k, K, sob, sof : float
    """
    deg2rad = jnp.pi / 180.0

    cos_tts = jnp.cos(tts * deg2rad)
    cos_tto = jnp.cos(tto * deg2rad)

    psi = jnp.abs(psi - 360.0 * jnp.round(psi / 360.0))

    chi_s, chi_o, frho, ftau = volscat(tts, tto, psi, _LITAB)

    ksli = chi_s / cos_tts
    koli = chi_o / cos_tto

    sobli = frho * jnp.pi / (cos_tts * cos_tto)
    sofli = ftau * jnp.pi / (cos_tts * cos_tto)

    k = jnp.dot(ksli, lidf)
    K = jnp.dot(koli, lidf)
    sob = jnp.dot(sobli, lidf)
    sof = jnp.dot(sofli, lidf)

    Go = K * cos_tto
    Gs = k * cos_tts

    return Gs, Go, k, K, sob, sof
