#!/usr/bin/env python3
"""
Canopy-SIP Model — JAX Backend Demo.

Demonstrates JIT compilation, automatic differentiation, and batch simulation.

USAGE:
    pip install "jax[cpu]"   # or jax[cuda12] for GPU
    python main_jax.py
"""

import time
import numpy as np

import jax
import jax.numpy as jnp

# Enable 64-bit precision (important for matching MATLAB/NumPy results)
jax.config.update("jax_enable_x64", True)

from canopy_sip.jax_backend import (
    run_simulation,
    simulate_brf,
    simulate_brf_jit,
    make_params,
    make_gap_data,
    campbell,
    grad_brf,
    jacobian_brf,
    batch_simulate,
)


def main():
    print("=" * 70)
    print("Canopy-SIP Model — JAX Backend Demo")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 70)

    # ── 1. Basic simulation (matches NumPy backend) ─────────────────────
    print("\n[1] Basic BRF Simulation")
    print("-" * 40)

    result = run_simulation(
        SZA=0, LAI=5, rho=0.4957, tau=0.4409, rg=0.159,
    )
    BRF3 = result['BRF3']
    signed_vza = result['signed_vza']

    print(f"{'VZA':>8s}  {'BRF':>12s}")
    for i in range(13):
        print(f"{float(signed_vza[i]):8.1f}  {float(BRF3[i]):12.8f}")

    # Load reference for comparison
    import os
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ref = np.loadtxt(os.path.join(data_dir, 'BRF_SIP_SZA00_Nir2_reference.csv'),
                     delimiter=',', skiprows=1).flatten()
    max_diff = float(jnp.max(jnp.abs(BRF3 - ref)))
    print(f"\nMax difference from MATLAB reference: {max_diff:.2e}")

    # ── 2. JIT Compilation Benchmark ────────────────────────────────────
    print("\n[2] JIT Compilation Benchmark")
    print("-" * 40)

    gap_data = make_gap_data(
        np.loadtxt(os.path.join(data_dir, 'gap_tot.csv'), delimiter=',', skiprows=1)[:, 2],
        np.loadtxt(os.path.join(data_dir, 'gap_within.csv'), delimiter=',', skiprows=1)[:, 2],
        np.loadtxt(os.path.join(data_dir, 'gap_betw.csv'), delimiter=',', skiprows=1)[:, 2],
        np.loadtxt(os.path.join(data_dir, 'CI_within.csv'), delimiter=',', skiprows=1)[:, 2],
    )
    params = make_params(SZA=0, LAI=5, rho=0.4957, tau=0.4409, rg=0.159)
    lidf, _ = campbell(jnp.float64(57.3))

    # First call (includes compilation)
    t0 = time.perf_counter()
    _ = simulate_brf_jit(params, gap_data, lidf).block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"First call (compile + run): {t_compile*1000:.1f} ms")

    # Subsequent calls (compiled)
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        _ = simulate_brf_jit(params, gap_data, lidf).block_until_ready()
    t_run = (time.perf_counter() - t0) / N
    print(f"Compiled call (avg of {N}): {t_run*1000:.3f} ms")
    print(f"Speedup: {t_compile/t_run:.0f}x")

    # ── 3. Automatic Differentiation ────────────────────────────────────
    print("\n[3] Automatic Differentiation (Sensitivity Analysis)")
    print("-" * 40)

    grads = grad_brf(params, gap_data, lidf)
    print("∂(mean BRF) / ∂(parameter):")
    for key, val in sorted(grads.items()):
        print(f"  {key:>15s}: {float(val):+.6f}")

    # ── 4. Full Jacobian ────────────────────────────────────────────────
    print("\n[4] Jacobian (per-angle sensitivities)")
    print("-" * 40)

    jac = jacobian_brf(params, gap_data, lidf)
    print("∂(BRF_nadir) / ∂(parameter):")
    for key, val in sorted(jac.items()):
        print(f"  {key:>15s}: {float(val[6]):+.6f}")  # Index 6 = nadir

    # ── 5. Batch Simulation (LAI sweep) ─────────────────────────────────
    print("\n[5] Batch Simulation (LAI sweep with vmap)")
    print("-" * 40)

    n_batch = 20
    lai_values = jnp.linspace(1.0, 10.0, n_batch)

    # Build batched parameter dict
    params_batch = {k: jnp.full(n_batch, v) for k, v in params.items()}
    params_batch['LAI'] = lai_values

    t0 = time.perf_counter()
    BRF_batch = batch_simulate(params_batch, gap_data, lidf)
    BRF_batch.block_until_ready()
    t_batch = time.perf_counter() - t0

    print(f"Batch of {n_batch} simulations: {t_batch*1000:.1f} ms "
          f"({t_batch/n_batch*1000:.2f} ms per simulation)")
    print(f"BRF_batch shape: {BRF_batch.shape}")
    print(f"Nadir BRF at LAI=1: {float(BRF_batch[0, 6]):.6f}")
    print(f"Nadir BRF at LAI=10: {float(BRF_batch[-1, 6]):.6f}")

    # ── 6. Save plot ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: BRF curves for different LAI
        ax = axes[0]
        for i in range(0, n_batch, 4):
            ax.plot(np.array(signed_vza), np.array(BRF_batch[i]),
                    'o-', label=f'LAI={float(lai_values[i]):.1f}', markersize=4)
        ax.set_xlabel('View Zenith Angle (°)')
        ax.set_ylabel('BRF')
        ax.set_title('BRF vs LAI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Sensitivity (gradient) bar chart
        ax = axes[1]
        keys = ['LAI', 'rho', 'tau', 'rg', 'iD', 'D']
        vals = [float(grads[k]) for k in keys]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
        ax.barh(keys, vals, color=colors)
        ax.set_xlabel('∂(mean BRF) / ∂(parameter)')
        ax.set_title('Parameter Sensitivity')
        ax.axvline(0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Panel 3: Nadir BRF vs LAI
        ax = axes[2]
        ax.plot(np.array(lai_values), np.array(BRF_batch[:, 6]), 'ro-', markersize=4)
        ax.set_xlabel('LAI')
        ax.set_ylabel('Nadir BRF')
        ax.set_title('Nadir BRF vs LAI')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('jax_demo_results.png', dpi=150)
        print(f"\nPlot saved to jax_demo_results.png")
    except ImportError:
        print("\nMatplotlib not available, skipping plot.")

    print("\n" + "=" * 70)
    print("JAX demo completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
