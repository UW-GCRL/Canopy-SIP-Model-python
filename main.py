#!/usr/bin/env python3
"""
Canopy-SIP Model (Optical BRF Simulation) - Main Script.

This script demonstrates the pure optical bidirectional reflectance factor
(BRF) simulation for a discrete vegetation canopy.

AUTHORS:
    Yelu Zeng, Min Chen, Dalei Hao, Yachang He
    Updated by: Yachang He (Email: akhyc13@gmail.com)
    Python translation: 2026-02

USAGE:
    python main.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canopy_sip import run_simulation


def main():
    # Run the BRF simulation with default parameters
    result = run_simulation(
        SZA=0,            # Sun Zenith Angle [degrees]
        SAA=0,            # Sun Azimuth Angle [degrees]
        Crowndeepth=12.8675,
        Height=20,
        Height_c=6.634,
        dthr=0.41234,
        bl=0.1,
        iD=0.58073,
        LAI=5,
        FAVD=0.375,
        D=0,
        TypeLidf=2,       # Campbell single-parameter
        LIDFa=57.3,       # Spherical distribution
        LIDFb=0,
        rho=0.4957,       # Leaf reflectance (NIR)
        tau=0.4409,        # Leaf transmittance (NIR)
        rg=0.159,          # Soil reflectance
        pathAngle=1,       # Discrete canopy
    )

    BRF3 = result['BRF3']
    signed_vza = result['signed_vza']

    # Save results
    np.savetxt('BRF_SIP_SZA00_Nir2.csv', BRF3, delimiter=',',
               header='BRF3', comments='')
    print('Simulation completed successfully. Results saved to BRF_SIP_SZA00_Nir2.csv')

    # Print results
    print('\nBRF Results:')
    print(f'{"VZA":>8s}  {"BRF":>12s}')
    print('-' * 22)
    for i in range(13):
        print(f'{signed_vza[i]:8.1f}  {BRF3[i]:12.8f}')

    # Plot results
    print('\nGenerating BRF plot in the principal plane...')

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')
    ax.plot(signed_vza, BRF3, 'ko-', linewidth=1.5, markersize=6,
            markerfacecolor='r')

    ax.set_xlabel('View Zenith Angle (°)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bidirectional Reflectance Factor (BRF)', fontsize=12, fontweight='bold')
    ax.set_title(f'Canopy BRF in Principal Plane (NIR, SZA = {int(result["va"][0, 0]) if False else 0}°)',
                 fontsize=14)

    ax.grid(True)
    ax.set_xlim([-65, 65])
    ax.set_ylim([0.1, 0.5])
    ax.set_xticks(np.arange(-60, 61, 20))
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig('BRF_SIP_SZA00_Nir2.png', dpi=150, bbox_inches='tight')
    print('Plot saved to BRF_SIP_SZA00_Nir2.png')


if __name__ == '__main__':
    main()
