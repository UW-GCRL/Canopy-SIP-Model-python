"""
Campbell leaf angle distribution function — JAX backend.
"""

import jax.numpy as jnp


# Static angle bin edges (constants, safe for JIT)
_TX1 = jnp.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
_TX2 = jnp.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])
_LITAB = (_TX2 + _TX1) / 2.0


def campbell(ala):
    """
    Compute leaf angle distribution using the Campbell (1986) ellipsoidal model (JAX).

    Parameters
    ----------
    ala : float
        Average leaf inclination angle in degrees.

    Returns
    -------
    freq0 : jnp.ndarray (13,)
        Leaf angle distribution frequencies.
    litab : jnp.ndarray (13,)
        Leaf inclination angle centers.
    """
    tl1 = _TX1 * (jnp.pi / 180.0)
    tl2 = _TX2 * (jnp.pi / 180.0)
    excent = jnp.exp(-1.6184e-5 * ala**3 + 2.1145e-3 * ala**2 - 1.2390e-1 * ala + 3.2491)

    x1 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl1)**2)
    x2 = excent / jnp.sqrt(1.0 + excent**2 * jnp.tan(tl2)**2)

    # Branch: excent == 1 (uniform)
    freq_eq1 = jnp.abs(jnp.cos(tl1) - jnp.cos(tl2))

    # Branch: excent != 1
    alpha = excent / jnp.sqrt(jnp.abs(1.0 - excent**2) + 1e-30)
    alpha2 = alpha**2
    x12 = x1**2
    x22 = x2**2

    # Sub-branch: excent > 1
    alpx1_gt = jnp.sqrt(alpha2 + x12)
    alpx2_gt = jnp.sqrt(alpha2 + x22)
    dum_gt = x1 * alpx1_gt + alpha2 * jnp.log(x1 + alpx1_gt + 1e-30)
    freq_gt1 = jnp.abs(dum_gt - (x2 * alpx2_gt + alpha2 * jnp.log(x2 + alpx2_gt + 1e-30)))

    # Sub-branch: excent < 1
    almx1_lt = jnp.sqrt(jnp.maximum(alpha2 - x12, 0.0))
    almx2_lt = jnp.sqrt(jnp.maximum(alpha2 - x22, 0.0))
    dum_lt = x1 * almx1_lt + alpha2 * jnp.arcsin(jnp.clip(x1 / (alpha + 1e-30), -1.0, 1.0))
    freq_lt1 = jnp.abs(dum_lt - (x2 * almx2_lt + alpha2 * jnp.arcsin(jnp.clip(x2 / (alpha + 1e-30), -1.0, 1.0))))

    # Select branch based on excent value
    freq_ne1 = jnp.where(excent > 1.0, freq_gt1, freq_lt1)
    freq = jnp.where(jnp.abs(excent - 1.0) < 1e-10, freq_eq1, freq_ne1)

    sum0 = jnp.sum(freq)
    freq0 = freq / sum0

    return freq0, _LITAB
