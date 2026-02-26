"""
Two-parameter leaf angle distribution generation — JAX backend.
"""

import jax
import jax.numpy as jnp

_LITAB = jnp.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])


def dcum(a, b, t):
    """
    Cumulative leaf angle distribution function (JAX).

    Uses jax.lax.while_loop for the iterative solver.
    """
    rd = jnp.pi / 180.0

    # Branch a >= 1: simple formula
    f_simple = 1.0 - jnp.cos(rd * t)

    # Branch a < 1: iterative solver
    p = 2.0 * rd * t
    x0 = p  # initial x

    def cond_fn(state):
        _, delx = state
        return delx >= 1e-8

    def body_fn(state):
        x, _ = state
        y = a * jnp.sin(x) + 0.5 * b * jnp.sin(2.0 * x)
        dx = 0.5 * (y - x + p)
        x = x + dx
        return (x, jnp.abs(dx))

    x_final, _ = jax.lax.while_loop(cond_fn, body_fn, (x0, jnp.float64(1.0)))
    y_final = a * jnp.sin(x_final) + 0.5 * b * jnp.sin(2.0 * x_final)
    f_iter = (2.0 * y_final + p) / jnp.pi

    return jnp.where(a >= 1.0, f_simple, f_iter)


def dladgen(a, b):
    """
    Generate leaf angle distribution using two-parameter model (JAX).

    Parameters
    ----------
    a, b : float
        Distribution parameters. Requirement: |a| + |b| < 1.

    Returns
    -------
    freq : jnp.ndarray (13,)
        Leaf angle distribution frequencies.
    litab : jnp.ndarray (13,)
        Leaf inclination angle centers.
    """
    # Angles to evaluate: 10,20,...,80,82,84,86,88
    angles = jnp.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])

    # Vectorized dcum over angles
    cum_vals = jax.vmap(lambda t: dcum(a, b, t))(angles)

    # freq[0:12] = cum_vals, freq[12] = 1.0
    freq_cum = jnp.concatenate([cum_vals, jnp.array([1.0])])

    # Differentiate: freq[i] = freq_cum[i] - freq_cum[i-1] for i>=1
    freq = jnp.concatenate([freq_cum[:1], jnp.diff(freq_cum)])

    return freq, _LITAB
