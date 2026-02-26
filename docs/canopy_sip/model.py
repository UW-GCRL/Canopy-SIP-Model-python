"""
Canopy-SIP Model - Core Simulation Engine.

This module contains the main BRF simulation logic, translated from main.m.

AUTHORS:
    Yelu Zeng, Min Chen, Dalei Hao, Yachang He
    Updated by: Yachang He (Email: akhyc13@gmail.com)
    Date: 2024-08-01 / Refined for GitHub Open Source: 2026-02

REFERENCE:
    Please cite the corresponding publications related to the SIP model
    modifications for discrete canopies.
"""

import os
import numpy as np

from .campbell import campbell
from .dladgen import dladgen
from .cixy import cixy
from .phase import phase
from .get_hsf_go import get_hsf_go
from .sunshade_h import sunshade_h
from .sunshade_kt_he import sunshade_kt_he


def _load_data(data_dir=None):
    """
    Load structural data (gap fractions and clumping index) from CSV files.

    Parameters
    ----------
    data_dir : str, optional
        Path to the data directory. Defaults to the 'data' folder alongside
        this package.

    Returns
    -------
    gap_tot, gap_within, gap_betw, CI_within : np.ndarray
        Each is a (13, 3) array with columns [angle, azimuth, value].
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    gap_tot = np.loadtxt(os.path.join(data_dir, 'gap_tot.csv'), delimiter=',', skiprows=1)
    gap_within = np.loadtxt(os.path.join(data_dir, 'gap_within.csv'), delimiter=',', skiprows=1)
    gap_betw = np.loadtxt(os.path.join(data_dir, 'gap_betw.csv'), delimiter=',', skiprows=1)
    CI_within = np.loadtxt(os.path.join(data_dir, 'CI_within.csv'), delimiter=',', skiprows=1)

    return gap_tot, gap_within, gap_betw, CI_within


class CanopySIPModel:
    """
    Canopy-SIP Model for optical BRF simulation of discrete vegetation canopies.

    This model integrates Geometric-Optical (GO) theory for four scene components
    and Spectral Invariants Theory (p-theory) for multiple scattering.

    Parameters
    ----------
    SZA : float
        Sun Zenith Angle [degrees]. Default: 0.
    SAA : float
        Sun Azimuth Angle [degrees]. Default: 0.
    Crowndeepth : float
        Average crown depth [m]. Default: 12.8675.
    Height : float
        Canopy height [m]. Default: 20.
    Height_c : float
        Crown center height [m]. Default: 6.634.
    dthr : float
        Diameter To Height Ratio. Default: 0.41234.
    bl : float
        Leaf width [m]. Default: 0.1.
    iD : float
        Scene hemispherical interceptance. Default: 0.58073.
    LAI : float
        Single crown sphere LAI [m2/m2]. Default: 5.
    FAVD : float
        Foliage Area Volume Density. Default: 0.375.
    D : float
        Ratio of diffuse to incoming irradiance. Default: 0.
    TypeLidf : int
        1 = Two-parameter, 2 = Single-parameter (Campbell). Default: 2.
    LIDFa : float
        Leaf angle distribution parameter a. Default: 57.3 (spherical).
    LIDFb : float
        Leaf angle distribution parameter b. Default: 0.
    rho : float
        Leaf reflectance. Default: 0.4957.
    tau : float
        Leaf transmittance. Default: 0.4409.
    rg : float
        Soil reflectance. Default: 0.159.
    CIy1 : float
        Clumping index at nadir (0 degrees). Default: 1.0 (random, no clumping).
    CIy2 : float
        Clumping index at 75 degrees. Default: 1.0 (random, no clumping).
    pathAngle : int
        0 = homogeneous canopy, 1 = discrete canopy. Default: 1.
    data_dir : str, optional
        Path to the data directory.
    """

    def __init__(self, SZA=0, SAA=0, Crowndeepth=12.8675, Height=20,
                 Height_c=6.634, dthr=0.41234, bl=0.1, iD=0.58073,
                 LAI=5, FAVD=0.375, D=0, TypeLidf=2, LIDFa=57.3,
                 LIDFb=0, rho=0.4957, tau=0.4409, rg=0.159,
                 CIy1=1.0, CIy2=1.0,
                 pathAngle=1, data_dir=None):
        self.SZA = SZA
        self.SAA = SAA
        self.Crowndeepth = Crowndeepth
        self.Height = Height
        self.Height_c = Height_c
        self.dthr = dthr
        self.bl = bl
        self.iD = iD
        self.LAI = LAI
        self.FAVD = FAVD
        self.D = D
        self.TypeLidf = TypeLidf
        self.LIDFa = LIDFa
        self.LIDFb = LIDFb
        self.rho = rho
        self.tau = tau
        self.rg = rg
        self.CIy1 = CIy1
        self.CIy2 = CIy2
        self.pathAngle = pathAngle
        self.data_dir = data_dir

    def run(self):
        """
        Run the BRF simulation.

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'BRF3': np.ndarray of shape (13,), the BRF values.
            - 'signed_vza': np.ndarray of shape (13,), the signed view zenith angles.
            - 'va': np.ndarray of shape (13, 4), the view angle array.
        """
        # =====================================================================
        # 1. Prepare for Simulation
        # =====================================================================

        # Load structural data
        gap_tot, gap_within, gap_betw, CI_within_data = _load_data(self.data_dir)

        # 1.1 Sun-Sensor Geometry
        # Sensor geometry: 13 angles in the principal plane
        va = np.zeros((13, 4))
        for t in range(7):
            va[t, 0] = 10.0 * (6 - t)  # Forward view zenith angles (60,50,...,0)
            va[t, 1] = 0.0             # Forward view azimuth angle
        for t in range(6):
            va[t + 7, 0] = 10.0 * (t + 1)  # Backward view zenith angles (10,...,60)
            va[t + 7, 1] = 180.0            # Backward view azimuth angle

        SZA = self.SZA
        SAA = self.SAA

        # 1.2 Vegetation Type and Structural Parameters
        if self.pathAngle != 0:
            # Gap fractions for discrete canopy at Nadir (Index 6, 0-based)
            gap_H = gap_betw[6, 2]           # Nadir between-crown gap fraction
            gap_H_within = gap_within[6, 2]  # Nadir within-crown gap fraction
            gap_H_tot = gap_H + gap_H_within # Nadir total gap fraction
            CI_H_within = CI_within_data[6, 2]

            gap_S = gap_betw[6, 2]           # Solar direction between-crown gap fraction
            gap_S_within = gap_within[6, 2]
            gap_S_tot = gap_S + gap_S_within # Solar direction total gap fraction

        HotSpotPar = self.bl / self.Height

        c1 = CI_H_within  # Nadir within-crown clumping index
        c2 = 1.0 - gap_H
        c = c1 * c2       # Canopy-level continuity parameter

        w = self.rho + self.tau  # Leaf single scattering albedo

        go_par = self.dthr * self.Crowndeepth  # Hotspot parameter at crown scale

        # 1.3 Leaf angle distribution
        if self.TypeLidf == 1:
            lidf, litab = dladgen(self.LIDFa, self.LIDFb)
        elif self.TypeLidf == 2:
            lidf, litab = campbell(self.LIDFa)
        # In MATLAB: lidf = lidf'; makes it a column vector.
        # In our Python code, lidf is already a 1D array (13,).

        # =====================================================================
        # 2. Start BRF Simulation
        # =====================================================================
        BRF3 = np.ones(13)

        for t in range(13):
            tts = SZA
            tto = va[t, 0]
            psi = va[t, 1]

            # Ensure relative azimuth angle is symmetric
            if psi > 180:
                psi = psi - 360.0
            psi = abs(psi)
            psi = abs(psi - 360.0 * round(psi / 360.0))

            CIy1 = self.CIy1
            CIy2 = self.CIy2
            CIs = cixy(CIy1, CIy2, tts)
            CIo = cixy(CIy1, CIy2, tto)

            # 2.1 Directional gap extraction
            gap_V_tot = gap_tot[t, 2]
            gap_V_within = gap_within[t, 2]
            gap_V_betw = gap_betw[t, 2]

            Ps_dir_go = gap_S        # Between-crown gap fraction in solar direction
            Pv_dir_go = gap_V_betw   # Between-crown gap fraction in view direction

            # 2.2 Calculate four GO components (Kc, Kt, Kg, Kz)
            Kg = (Ps_dir_go * Pv_dir_go +
                  get_hsf_go(go_par, SZA, SAA, tto, psi, Ps_dir_go, Pv_dir_go, self.Height_c))
            Kz = Pv_dir_go - Kg
            Kct = 1.0 - Pv_dir_go

            delta_angle = (np.cos(np.deg2rad(SZA)) * np.cos(np.deg2rad(tto)) +
                           np.sin(np.deg2rad(SZA)) * np.sin(np.deg2rad(tto)) *
                           np.cos(np.deg2rad(psi - SAA)))
            phi = np.rad2deg(np.arccos(delta_angle))
            delta_val = np.cos(np.deg2rad(phi * (1.0 - np.sin(np.pi * c / 2.0))))

            if ((self.Height - self.Crowndeepth) < self.Crowndeepth and
                    tto > SZA and SAA == psi):
                Kc = Kct  # Continuous canopy
            else:
                Kc = 0.5 * (1.0 + delta_val) * Kct  # Discrete canopy
            Kt = Kct - Kc

            # 2.3 Calculate BRF_L: Vegetation single scattering contribution
            Ps_dir_inKz = gap_S_within

            Gs, Go, k_ext, K_ext, sob, sof = phase(tts, tto, psi, lidf)

            kc_val, kg_val = sunshade_h(tts, tto, psi, Gs, Go, CIs, CIo, self.LAI, HotSpotPar)
            kc_kt, kg_kt = sunshade_kt_he(tts, tto, psi, Gs, Go, CIs, CIo, self.LAI)

            wso = sob * self.rho + sof * self.tau  # Bidirectional scattering coefficient

            BRF_v1 = wso * kc_val / K_ext                             # Sunlit crown
            BRF_v1_kt = np.sqrt(Ps_dir_inKz) * wso * kc_kt / K_ext   # Shaded crown

            # 2.4 Calculate BRF_S: Soil contribution
            BRFsc = kg_val * self.rg       # Sunlit crown visible sunlit soil
            BRFs_kt = kg_kt * self.rg      # Shaded crown visible sunlit soil

            # 2.5 Calculate BRF_M: Multiple scattering contribution (p-theory)
            i0 = 1.0 - gap_S_tot
            i0 = self.D * self.iD + (1.0 - self.D) * i0
            iv = 1.0 - gap_V_tot
            t0 = 1.0 - i0
            tv = 1.0 - iv

            p = 1.0 - self.iD / self.LAI
            rho2 = iv / 2.0 / self.LAI
            rho_hemi2 = self.iD / 2.0 / self.LAI

            Tdn = t0 + i0 * w * rho_hemi2 / (1.0 - p * w)
            Tup_o = tv + self.iD * w * rho2 / (1.0 - p * w)
            Rdn = self.iD * w * rho_hemi2 / (1.0 - p * w)

            BRF_vm = i0 * w**2 * p * rho2 / (1.0 - p * w)  # Vegetation multiple scattering
            BRFm = (self.rg * Tdn * Tup_o / (1.0 - self.rg * Rdn) -
                    t0 * self.rg * tv)  # Vegetation-Soil multiple interactions

            # =================================================================
            # 3. Final: Calculate total BRF (Discrete Canopy)
            # =================================================================
            BRF3[t] = (Kc * (BRFsc + BRF_v1) +
                       Kt * (BRFs_kt + BRF_v1_kt) +
                       (Kg + Kz * Ps_dir_inKz) * self.rg +
                       BRFm + BRF_vm)

        # Signed VZA for plotting
        signed_vza = np.zeros(13)
        signed_vza[:7] = -va[:7, 0]    # Forward: -60, -50, -40, -30, -20, -10, 0
        signed_vza[7:] = va[7:, 0]     # Backward: 10, 20, 30, 40, 50, 60

        return {
            'BRF3': BRF3,
            'signed_vza': signed_vza,
            'va': va,
        }


def run_simulation(**kwargs):
    """
    Convenience function to run the Canopy-SIP simulation with given parameters.

    Parameters
    ----------
    **kwargs
        Any keyword arguments accepted by CanopySIPModel.__init__.

    Returns
    -------
    result : dict
        Simulation results (see CanopySIPModel.run).
    """
    model = CanopySIPModel(**kwargs)
    return model.run()
