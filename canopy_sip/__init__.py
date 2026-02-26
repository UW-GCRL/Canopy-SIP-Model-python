"""
Canopy-SIP Model (Optical BRF Simulation) - Python Implementation.

This package simulates the optical Bidirectional Reflectance Factor (BRF)
for discrete vegetation canopies using Geometric-Optical (GO) theory and
Spectral Invariants Theory (p-theory).

Two backends are available:
  - NumPy/SciPy (default): ``from canopy_sip import run_simulation``
  - JAX (GPU, JIT, autodiff): ``from canopy_sip.jax_backend import run_simulation``

Authors: Yelu Zeng, Min Chen, Dalei Hao, Yachang He
Python translation by: AI Assistant
"""

from .model import CanopySIPModel, run_simulation
