# Canopy-SIP Model — Python Version

Python implementation of the **Canopy-SIP Model** for simulating canopy optical Bidirectional Reflectance Factor (BRF).

> Translated from the original MATLAB implementation: [YachangHe/Canopy-SIP-Model](https://github.com/YachangHe/Canopy-SIP-Model).  
> Results match the MATLAB reference output to within ~2×10⁻⁷.

## Online Demo

**Try it in your browser** (no installation needed):
**[https://uw-gcrl.github.io/Canopy-SIP-Model-python/](https://uw-gcrl.github.io/Canopy-SIP-Model-python/)**

The online version runs entirely in the browser using WebAssembly (NumPy/SciPy backend). For JAX acceleration (JIT, GPU, autodiff), download the code and run locally.

## Overview

This model simulates BRF for **discrete vegetation canopies** by integrating:

1. **Geometric-Optical (GO) Theory** — calculates area fractions of four scene components (sunlit/shaded crown, sunlit/shaded soil) and directional gap probabilities.
2. **Spectral Invariants Theory (p-theory)** — efficiently simulates multiple scattering within the canopy.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `scipy`, `matplotlib`

### JAX Backend (optional — GPU, JIT, autodiff)

```bash
pip install "jax[cpu]"        # CPU-only
pip install "jax[cuda12]"     # NVIDIA GPU (CUDA 12)
```

## Usage

### Web UI (local)

```bash
streamlit run app.py     # Opens interactive web UI with NumPy/JAX backend selection
```

### Command Line

```bash
python main.py           # NumPy backend
python main_jax.py       # JAX backend (JIT, grad, batch demo)
```

### As a Python Library

```python
# ── NumPy backend (default) ──────────────────────────────────────
from canopy_sip import run_simulation

result = run_simulation(SZA=30, LAI=3, rho=0.4957, tau=0.4409, rg=0.159)
BRF = result['BRF3']           # (13,) array of BRF values
vza = result['signed_vza']     # (13,) array of signed view zenith angles

# ── JAX backend ──────────────────────────────────────────────────
from canopy_sip.jax_backend import run_simulation

result = run_simulation(SZA=30, LAI=3, rho=0.4957, tau=0.4409, rg=0.159)
```

### JAX: JIT Compilation

After the first call (which compiles), subsequent calls are 10-50× faster:

```python
from canopy_sip.jax_backend import simulate_brf_jit, make_params, make_gap_data, campbell

params = make_params(SZA=0, LAI=5, rho=0.4957, tau=0.4409, rg=0.159)
lidf, _ = campbell(57.3)
BRF = simulate_brf_jit(params, gap_data, lidf)   # ~0.1 ms after compilation
```

### JAX: Automatic Differentiation

Compute gradients of BRF with respect to any parameter — useful for sensitivity analysis, parameter inversion, and data assimilation:

```python
from canopy_sip.jax_backend import grad_brf, jacobian_brf

# Gradient of mean BRF w.r.t. all parameters
grads = grad_brf(params, gap_data, lidf)
print(grads['LAI'])    # ∂(mean BRF) / ∂LAI
print(grads['rho'])    # ∂(mean BRF) / ∂ρ_leaf

# Full Jacobian: ∂BRF_i / ∂param_j for all 13 angles
jac = jacobian_brf(params, gap_data, lidf)
print(jac['LAI'])      # (13,) array: ∂BRF / ∂LAI for each angle
```

### JAX: Batch Simulation (vmap)

Run thousands of parameter combinations simultaneously via vectorization:

```python
import jax.numpy as jnp
from canopy_sip.jax_backend import batch_simulate

# Sweep LAI from 1 to 10
N = 100
params_batch = {k: jnp.full(N, v) for k, v in params.items()}
params_batch['LAI'] = jnp.linspace(1.0, 10.0, N)

BRFs = batch_simulate(params_batch, gap_data, lidf)  # shape (100, 13)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SZA` | Sun Zenith Angle [°] | 0 |
| `SAA` | Sun Azimuth Angle [°] | 0 |
| `LAI` | Leaf Area Index [m²/m²] | 5 |
| `rho` | Leaf reflectance | 0.4957 |
| `tau` | Leaf transmittance | 0.4409 |
| `rg` | Soil reflectance | 0.159 |
| `Height` | Canopy height [m] | 20 |
| `Crowndeepth` | Crown depth [m] | 12.8675 |
| `TypeLidf` | LIDF type (1=two-param, 2=Campbell) | 2 |
| `LIDFa` | Leaf angle param (57.3=spherical) | 57.3 |
| `D` | Diffuse fraction | 0 |

## Project Structure

```
├── main.py                       # CLI entry point (NumPy)
├── main_jax.py                   # CLI entry point (JAX demo: JIT, grad, batch)
├── app.py                        # Streamlit web UI (auto-selects backend)
├── canopy_sip/
│   ├── __init__.py
│   ├── model.py                  # Core simulation engine (NumPy)
│   ├── phase.py                  # Scattering phase function
│   ├── volscat.py                # Volume scattering
│   ├── campbell.py               # Campbell LIDF
│   ├── dladgen.py                # Two-parameter LIDF
│   ├── cixy.py                   # Clumping index angular dependence
│   ├── get_hsf_go.py             # Hotspot effect / GO gap
│   ├── sunshade_h.py             # Sunlit fractions with hotspot
│   ├── sunshade_kt_he.py         # Sunlit fractions without hotspot
│   ├── ci_utils.py               # CI/Gap preprocessing utilities
│   └── jax_backend/              # ⚡ JAX-accelerated backend
│       ├── __init__.py           #    Public API
│       ├── model.py              #    Core engine (JIT, vmap, differentiable)
│       ├── quadrature.py         #    Gauss-Legendre quadrature
│       ├── volscat.py            #    Volume scattering (JAX)
│       ├── campbell.py           #    Campbell LIDF (JAX)
│       ├── dladgen.py            #    Two-parameter LIDF (JAX)
│       ├── cixy.py               #    Clumping index (JAX)
│       ├── phase.py              #    Phase function (JAX)
│       ├── get_hsf_go.py         #    Hotspot effect (JAX)
│       ├── sunshade_h.py         #    Sunlit fractions with hotspot (JAX)
│       └── sunshade_kt_he.py     #    Sunlit fractions without hotspot (JAX)
├── data/                         # Structural look-up tables (CSV)
├── docs/                         # GitHub Pages (stlite browser app)
│   └── index.html                #    Entry point for online demo
├── requirements.txt
├── LICENSE
└── README.md
```

### MATLAB → Python File Mapping

| MATLAB (original repo) | Python (this repo) |
|---|---|
| `main.m` | `canopy_sip/model.py` + `main.py` |
| `PHASE.m` | `canopy_sip/phase.py` |
| `volscat.m` | `canopy_sip/volscat.py` |
| `campbell.m` | `canopy_sip/campbell.py` |
| `dladgen.m` | `canopy_sip/dladgen.py` |
| `CIxy.m` | `canopy_sip/cixy.py` |
| `get_HSF_go.m` | `canopy_sip/get_hsf_go.py` |
| `sunshade_H.m` | `canopy_sip/sunshade_h.py` |
| `sunshade_Kt_He.m` | `canopy_sip/sunshade_kt_he.py` |
| `CI_2/*.m` | `canopy_sip/ci_utils.py` |
| `CI_2/*.mat` | `data/*.csv` |

## Citation

If you use this code in your research, please cite:

> **He, Y.**, Zeng, Y., Hao, D., Shabanov, N. V., Huang, J., Yin, G., Biriukova, K., Lu, W., Gao, Y., Celesti, M., Xu, B., Gao, S., Migliavacca, M., Li, J., & Rossini, M. (2025). Combining geometric-optical and spectral invariants theories for modeling canopy fluorescence anisotropy. *Remote Sensing of Environment*, 323, 114716. [https://doi.org/10.1016/j.rse.2025.114716](https://doi.org/10.1016/j.rse.2025.114716)

## Acknowledgments

- **Original MATLAB implementation**: [YachangHe/Canopy-SIP-Model](https://github.com/YachangHe/Canopy-SIP-Model)
- **Original SIP Model**: Zeng et al. (2018), *Remote Sensing*, 10(10), 1508.
- **PATH_RT Model**: Li et al. (2024), *Remote Sensing of Environment*, 303, 113985.
- **LESS Model**: Qi et al. (2019), *Remote Sensing of Environment*, 221, 695-706.

## License

MIT License
