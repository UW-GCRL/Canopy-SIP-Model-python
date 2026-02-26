"""
JAX-compatible Gauss-Legendre quadrature.

Replaces scipy.integrate.quad for use within JIT-compiled functions.
Uses 20-point Gauss-Legendre rule for high accuracy.
"""

import jax.numpy as jnp

# Pre-computed 20-point Gauss-Legendre nodes and weights on [-1, 1]
# These are static constants, safe for JIT.
_GL_NODES_20 = jnp.array([
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
    -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
    -0.0765265211334973,
     0.0765265211334973,  0.2277858511416451,  0.3737060887154195,
     0.5108670019508271,  0.6360536807265150,  0.7463319064601508,
     0.8391169718222188,  0.9122344282513259,  0.9639719272779138,
     0.9931285991850949,
])

_GL_WEIGHTS_20 = jnp.array([
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
    0.1527533871307258,
    0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
    0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
    0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
    0.0176140071391521,
])


def gauss_legendre_quad(f, a, b):
    """
    Integrate f from a to b using 20-point Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable
        Function to integrate. Must accept a JAX array of points and return
        an array of the same shape (vectorized).
    a : float
        Lower integration limit.
    b : float
        Upper integration limit.

    Returns
    -------
    result : float
        Approximate integral of f from a to b.
    """
    # Transform from [-1, 1] to [a, b]
    mid = 0.5 * (b + a)
    half_len = 0.5 * (b - a)
    x = mid + half_len * _GL_NODES_20
    return half_len * jnp.dot(_GL_WEIGHTS_20, f(x))
