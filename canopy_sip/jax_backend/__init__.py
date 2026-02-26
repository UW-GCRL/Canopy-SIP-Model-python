"""
Canopy-SIP Model — JAX Backend.

Fully JIT-compilable, auto-differentiable, and GPU-accelerable implementation
of the Canopy-SIP BRF simulation model.

Key features over the NumPy backend:
  - `jax.jit`:  JIT compilation for 10-50x speedup after first call
  - `jax.grad`: Automatic differentiation for parameter sensitivity / inversion
  - `jax.vmap`: Vectorized batch simulations (e.g., sweep over SZA, LAI, etc.)
  - GPU/TPU:    Transparent acceleration on accelerator hardware

Quick start:
    >>> from canopy_sip.jax_backend import run_simulation
    >>> result = run_simulation(SZA=0, LAI=5, rho=0.4957, tau=0.4409, rg=0.159)
    >>> print(result['BRF3'])

Advanced (gradients):
    >>> from canopy_sip.jax_backend import simulate_brf, make_params, make_gap_data, grad_brf
    >>> params = make_params(SZA=0, LAI=5)
    >>> grads = grad_brf(params, gap_data, lidf)  # d(mean BRF)/d(param)

Advanced (batch):
    >>> from canopy_sip.jax_backend import batch_simulate
    >>> import jax.numpy as jnp
    >>> # Sweep LAI from 1 to 8
    >>> params_batch = {k: jnp.full(8, v) for k, v in params.items()}
    >>> params_batch['LAI'] = jnp.linspace(1, 8, 8)
    >>> BRFs = batch_simulate(params_batch, gap_data, lidf)  # (8, 13)
"""

from .model import (
    simulate_brf,
    simulate_brf_jit,
    grad_brf,
    jacobian_brf,
    batch_simulate,
    run_simulation,
    make_params,
    make_gap_data,
)
from .campbell import campbell
from .dladgen import dladgen

__all__ = [
    'simulate_brf',
    'simulate_brf_jit',
    'grad_brf',
    'jacobian_brf',
    'batch_simulate',
    'run_simulation',
    'make_params',
    'make_gap_data',
    'campbell',
    'dladgen',
]
