"""
Bidirectional between-crown gap probability covariance (hotspot effect) — JAX backend.
"""

import jax.numpy as jnp


def get_hsf_go(par, SZA, SAA, VZA, VAA, Ps_dir_go, Pv_dir_go, z):
    """
    Calculate the hotspot enhancement term (JAX).

    Parameters
    ----------
    par : float
        Crown size or hotspot parameter [m].
    SZA, SAA, VZA, VAA : float
        Sun/view zenith/azimuth angles [degrees].
    Ps_dir_go, Pv_dir_go : float
        Directional between-crown gap probabilities.
    z : float
        Crown center height [m].

    Returns
    -------
    bg : float
        Covariance term of bidirectional gap probability.
    """
    SZA_rad = jnp.deg2rad(SZA)
    SAA_rad = jnp.deg2rad(SAA)
    VZA_rad = jnp.deg2rad(VZA)
    VAA_rad = jnp.deg2rad(VAA)

    f1 = jnp.sqrt(Ps_dir_go * Pv_dir_go * (1.0 - Ps_dir_go) * (1.0 - Pv_dir_go))

    cosgamma = (jnp.cos(SZA_rad) * jnp.cos(VZA_rad) +
                jnp.sin(SZA_rad) * jnp.sin(VZA_rad) * jnp.cos(VAA_rad - SAA_rad))

    delta = jnp.sqrt(
        1.0 / jnp.cos(SZA_rad)**2 +
        1.0 / jnp.cos(VZA_rad)**2 -
        2.0 * cosgamma / (jnp.cos(SZA_rad) * jnp.cos(VZA_rad))
    )

    # Numerical stability: clamp delta to a small positive minimum
    delta = jnp.maximum(delta, 1e-5)

    Y = jnp.exp(-delta / par * z)
    bg = f1 * Y

    return bg
