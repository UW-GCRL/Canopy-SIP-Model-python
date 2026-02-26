"""
Angular dependence of the clumping index — JAX backend.
"""

import jax.numpy as jnp


def cixy(CIy1, CIy2, tts):
    """
    Calculate the angular dependence of the clumping index (JAX).

    Parameters
    ----------
    CIy1, CIy2 : float
        Clumping indices at nadir and 75 degrees.
    tts : float
        Target zenith angle [degrees].

    Returns
    -------
    CIs : float
        Estimated clumping index.
    """
    CIs_interp = (CIy2 - CIy1) / 75.0 * tts + CIy1
    CIs = jnp.where(tts > 75.0, CIy2, CIs_interp)
    return CIs
