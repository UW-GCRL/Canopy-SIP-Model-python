"""
Canopy-SIP Model — Web Interface (Streamlit / stlite).

This version runs in the browser via stlite (NumPy/SciPy backend only).
For JAX acceleration, download the code and run locally.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Canopy-SIP Model", page_icon="🌳", layout="wide")

st.title("🌳 Canopy-SIP Model")
st.markdown(
    "Optical Bidirectional Reflectance Factor (BRF) simulation for discrete "
    "vegetation canopies using **Geometric-Optical (GO)** theory and "
    "**Spectral Invariants (p-theory)**."
)

# ── Sidebar: Parameters ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    st.subheader("☀️ Sun Geometry")
    SZA = st.slider("Sun Zenith Angle (SZA) [°]", 0, 75, 0, 5)
    SAA = st.slider("Sun Azimuth Angle (SAA) [°]", 0, 360, 0, 10)

    st.subheader("🌿 Leaf Optical Properties")
    rho = st.slider("Leaf Reflectance (ρ)", 0.01, 0.60, 0.4957, 0.01)
    tau = st.slider("Leaf Transmittance (τ)", 0.01, 0.60, 0.4409, 0.01)
    rg = st.slider("Soil Reflectance", 0.01, 0.50, 0.159, 0.01)

    st.subheader("🏗️ Canopy Structure")
    LAI = st.slider("Leaf Area Index (LAI)", 0.5, 10.0, 5.0, 0.5)
    Height = st.slider("Canopy Height [m]", 5.0, 40.0, 20.0, 1.0)
    Crowndeepth = st.slider("Crown Depth [m]", 2.0, 25.0, 12.8675, 0.5)
    Height_c = st.slider("Crown Center Height [m]", 2.0, 20.0, 6.634, 0.5)
    iD = st.slider("Hemispherical Interceptance (iD)", 0.1, 0.9, 0.58073, 0.01)
    D = st.slider("Diffuse Fraction (D)", 0.0, 1.0, 0.0, 0.05)
    CIy1 = st.slider("Clumping Index at Nadir (CIy1)", 0.0, 1.0, 1.0, 0.05,
                       help="1.0 = random distribution (no clumping); < 1.0 = clumped foliage")
    CIy2 = st.slider("Clumping Index at 75° (CIy2)", 0.0, 1.0, 1.0, 0.05,
                       help="1.0 = random distribution (no clumping); < 1.0 = clumped foliage")

    st.subheader("🍃 Leaf Angle Distribution")
    TypeLidf = st.selectbox("LIDF Type", [2, 1],
                            format_func=lambda x: "Campbell (single-param)" if x == 2 else "Two-parameter")
    if TypeLidf == 2:
        LIDFa = st.slider("Average Leaf Angle [°]", 10.0, 80.0, 57.3, 1.0,
                           help="57.3° = spherical, ~27° = planophile, ~63° = erectophile")
        LIDFb = 0.0
    else:
        LIDFa = st.slider("LIDFa", -1.0, 1.0, -0.35, 0.05)
        LIDFb = st.slider("LIDFb", -1.0, 1.0, -0.15, 0.05)

# ── Run simulation ──────────────────────────────────────────────────────
st.caption("Note: Each simulation may take a few seconds in the browser environment.")
if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running BRF simulation..."):
        import time
        t0 = time.perf_counter()

        from canopy_sip import run_simulation
        result = run_simulation(
            SZA=SZA, SAA=SAA,
            Crowndeepth=Crowndeepth, Height=Height, Height_c=Height_c,
            iD=iD, LAI=LAI, D=D,
            TypeLidf=TypeLidf, LIDFa=LIDFa, LIDFb=LIDFb,
            rho=rho, tau=tau, rg=rg,
            CIy1=CIy1, CIy2=CIy2,
        )

        elapsed = time.perf_counter() - t0

    st.session_state['result'] = result
    st.session_state['params'] = {
        'SZA': SZA, 'LAI': LAI, 'rho': rho, 'tau': tau, 'rg': rg,
    }
    st.session_state['elapsed'] = elapsed

# ── Display results ─────────────────────────────────────────────────────
if 'result' in st.session_state:
    result = st.session_state['result']
    params = st.session_state['params']
    BRF3 = result['BRF3']
    signed_vza = result['signed_vza']

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("BRF in the Principal Plane")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=signed_vza, y=BRF3,
            mode='lines+markers',
            marker=dict(size=8, color='red', line=dict(width=1, color='black')),
            line=dict(color='black', width=1.5),
            name='BRF'
        ))
        fig.update_layout(
            xaxis_title="View Zenith Angle (°)",
            yaxis_title="Bidirectional Reflectance Factor (BRF)",
            title=f"Canopy BRF — SZA={params['SZA']}°, LAI={params['LAI']}",
            xaxis=dict(range=[-65, 65], dtick=20),
            yaxis=dict(range=[max(0, BRF3.min() - 0.05), BRF3.max() + 0.05]),
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("BRF Values")
        import pandas as pd
        df = pd.DataFrame({
            'VZA (°)': signed_vza.astype(int),
            'BRF': np.round(BRF3, 6),
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "BRF_results.csv", "text/csv")

    # Summary metrics
    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nadir BRF", f"{BRF3[6]:.4f}")
    c2.metric("Min BRF", f"{BRF3.min():.4f}")
    c3.metric("Max BRF", f"{BRF3.max():.4f}")
    hotspot_idx = np.argmin(np.abs(signed_vza - params['SZA']))
    c4.metric("Hotspot BRF", f"{BRF3[hotspot_idx]:.4f}")
    c5.metric("Runtime", f"{st.session_state.get('elapsed', 0)*1000:.0f} ms")

    # ── BRDF Kernel Analysis ──────────────────────────────────────────
    st.divider()
    st.subheader("BRDF Kernel Analysis (RossThick-LiSparseR)")

    from canopy_sip.brdf_kernels import (
        fit_brdf_kernels, compute_bsa, compute_wsa,
        generate_hemisphere_brf, plot_polar_brf,
    )

    # Derive absolute VZA and RAA from signed VZA
    vza_abs = np.abs(signed_vza)
    raa_obs = np.where(signed_vza <= 0, 0.0, 180.0)

    # Fit kernel model
    f_iso, f_vol, f_geo, r_squared, brf_fitted = fit_brdf_kernels(
        params['SZA'], vza_abs, raa_obs, BRF3,
    )

    # Kernel fit metrics
    st.markdown("**Kernel Weights (RTLSR Model)**")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("f_iso (isotropic)", f"{f_iso:.6f}")
    kc2.metric("f_vol (volumetric)", f"{f_vol:.6f}")
    kc3.metric("f_geo (geometric)", f"{f_geo:.6f}")
    kc4.metric("R\u00b2", f"{r_squared:.6f}")

    # Albedo
    sza_val = params['SZA']
    bsa = compute_bsa(f_iso, f_vol, f_geo, sza_val)
    wsa = compute_wsa(f_iso, f_vol, f_geo)

    st.markdown("**Albedo Estimates**")
    ac1, ac2 = st.columns(2)
    ac1.metric("Black-Sky Albedo (BSA)", f"{bsa:.6f}",
               help=f"Directional-hemispherical reflectance at SZA={sza_val}\u00b0")
    ac2.metric("White-Sky Albedo (WSA)", f"{wsa:.6f}",
               help="Bihemispherical reflectance (diffuse illumination)")

    # Two-column: comparison chart + polar plot
    pcol1, pcol2 = st.columns([1, 1])

    with pcol1:
        st.markdown("**Principal Plane: Simulated vs Kernel Fit**")
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=signed_vza, y=BRF3,
            mode='lines+markers',
            marker=dict(size=8, color='red', line=dict(width=1, color='black')),
            line=dict(color='black', width=1.5),
            name='Canopy-SIP (simulated)',
        ))
        fig_fit.add_trace(go.Scatter(
            x=signed_vza, y=brf_fitted,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='RTLSR kernel fit',
        ))
        fig_fit.update_layout(
            xaxis_title="View Zenith Angle (\u00b0)",
            yaxis_title="BRF",
            xaxis=dict(range=[-65, 65], dtick=20),
            yaxis=dict(range=[max(0, min(BRF3.min(), brf_fitted.min()) - 0.05),
                              max(BRF3.max(), brf_fitted.max()) + 0.05]),
            template="plotly_white",
            height=500,
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_fit, use_container_width=True)

    with pcol2:
        st.markdown("**Hemisphere BRF (Kernel Model Prediction)**")
        vza_grid, raa_grid, brf_grid = generate_hemisphere_brf(
            f_iso, f_vol, f_geo, sza_val,
        )
        polar_fig = plot_polar_brf(
            sza=sza_val,
            vza_obs=vza_abs, raa_obs=raa_obs, brf_obs=BRF3,
            vza_grid=vza_grid, raa_grid=raa_grid, brf_grid=brf_grid,
        )
        st.pyplot(polar_fig)
        plt.close(polar_fig)

else:
    st.info("👈 Set parameters in the sidebar and click **Run Simulation** to generate results.")

# ── Footer ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**Citation**: He, Y., Zeng, Y., et al. (2025). *Remote Sensing of Environment*, 323, 114716. "
    "[DOI](https://doi.org/10.1016/j.rse.2025.114716)"
)
st.markdown(
    "For **JAX acceleration** (JIT, GPU, autodiff), download the code from "
    "[GitHub](https://github.com/UW-GCRL/Canopy-SIP-Model-python) and run locally."
)
