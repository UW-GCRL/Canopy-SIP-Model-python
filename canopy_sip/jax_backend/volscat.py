"""
Volume scattering functions — JAX backend.
Fully JIT-compilable and differentiable.
"""

import jax.numpy as jnp


def volscat(tts, tto, psi, ttli):
    """
    Volume scattering function (JAX).

    Parameters
    ----------
    tts, tto, psi : float
        Sun zenith, view zenith, relative azimuth [degrees].
    ttli : jnp.ndarray
        Leaf inclination array [degrees], shape (13,).

    Returns
    -------
    chi_s, chi_o, frho, ftau : jnp.ndarray, each shape (13,)
    """
    deg2rad = jnp.pi / 180.0
    nli = ttli.shape[0]

    psi_rad = psi * deg2rad * jnp.ones(nli)
    cos_psi = jnp.cos(psi * deg2rad)

    cos_ttli = jnp.cos(ttli * deg2rad)
    sin_ttli = jnp.sin(ttli * deg2rad)

    cos_tts = jnp.cos(tts * deg2rad)
    sin_tts = jnp.sin(tts * deg2rad)
    cos_tto = jnp.cos(tto * deg2rad)
    sin_tto = jnp.sin(tto * deg2rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts
    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = jnp.maximum(Ss, Cs)
    Ao = jnp.maximum(So, Co)

    bts = jnp.arccos(jnp.clip(-Cs / As, -1.0, 1.0))
    bto = jnp.arccos(jnp.clip(-Co / Ao, -1.0, 1.0))

    chi_o = (2.0 / jnp.pi) * ((bto - jnp.pi / 2.0) * Co + jnp.sin(bto) * So)
    chi_s = (2.0 / jnp.pi) * ((bts - jnp.pi / 2.0) * Cs + jnp.sin(bts) * Ss)

    delta1 = jnp.abs(bts - bto)
    delta2 = jnp.pi - jnp.abs(bts + bto - jnp.pi)

    Tot = psi_rad + delta1 + delta2

    bt1 = jnp.minimum(psi_rad, delta1)
    bt3 = jnp.maximum(psi_rad, delta2)
    bt2 = Tot - bt1 - bt3

    T1 = 2.0 * Cs * Co + Ss * So * cos_psi
    T2 = jnp.sin(bt2) * (2.0 * As * Ao + Ss * So * jnp.cos(bt1) * jnp.cos(bt3))

    Jmin = bt2 * T1 - T2
    Jplus = (jnp.pi - bt2) * T1 + T2

    frho = jnp.maximum(0.0, Jplus / (2.0 * jnp.pi**2))
    ftau = jnp.maximum(0.0, -Jmin / (2.0 * jnp.pi**2))

    return chi_s, chi_o, frho, ftau
