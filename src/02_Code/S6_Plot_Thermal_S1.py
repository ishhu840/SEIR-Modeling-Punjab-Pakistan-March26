"""
S6_Plot_Thermal_S1.py
=====================
Generates Figure S1 (Supplementary): Thermal Response Curves

Shows the fitted weather-driven transmission function β(T) for each city,
plotted against temperature (°C). Demonstrates that:
  1. All cities show an inverted-U shaped response (bT2 < 0)
  2. Transmission peaks within the biologically validated 20-35°C window
  3. The city-specific curves reflect local climate and vector differences

β(T) = κ·exp(b₀ + bT·T_z + bT²·T_z²)   [standardized form]
     = exp(b0 + bT·T + bT2·T²)            [simplified illustrative form]

Inputs:  03_Results/V4_Optimal_Lags_ByCity.xlsx
Outputs: 04_Figures/FigS1_ThermalCurves.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', '03_Results')
FIG_DIR    = os.path.join(BASE_DIR, '..', '04_Figures')
os.makedirs(FIG_DIR, exist_ok=True)

OPTIMAL_FILE = os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx')

CITIES_ORDER = [
    'Lahore', 'Rawalpindi', 'Islamabad', 'Gujranwala',
    'Faisalabad', 'Multan', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]

COLORS = [
    '#1565C0','#E53935','#2E7D32','#6A1B9A',
    '#E65100','#00838F','#558B2F','#AD1457','#4E342E'
]

def figS1_thermal():
    print("Generating FigS1_ThermalCurves.png (Faceted Layout)...")

    df = pd.read_excel(OPTIMAL_FILE)
    T_range = np.linspace(10, 45, 500)  # High resolution for smooth curves

    # Create a 3x4 grid (12 subplots)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    # Shared settings
    xlim = (15, 40)
    ylim = (0, 2.5)

    # 1. Individual City Panels (1-9)
    for i, city in enumerate(CITIES_ORDER):
        ax = axes[i]
        row = df[df['City'] == city]
        if row.empty:
            ax.set_visible(False)
            continue
        row = row.iloc[0]

        b0  = float(row['b0'])
        bT  = float(row['bT'])
        bT2 = float(row['bT2'])

        # Calculate relative beta (vs T=28) using standardized scaling /5
        log_beta_rel = bT * (T_range - 28) / 5 + bT2 * ((T_range - 28) / 5) ** 2
        beta_rel = np.exp(log_beta_rel)

        # Clip transmission to zero above 35°C (biological upper limit)
        beta_rel_clipped = beta_rel.copy()
        beta_rel_clipped[T_range > 35] = 0.0

        # Plot city curve
        ax.plot(T_range, beta_rel_clipped, color=COLORS[i], linewidth=3, label=city)
        
        # Shade biological window
        ax.axvspan(20, 35, alpha=0.1, color='green')
        ax.axvline(28, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(35, color='#E53935', linestyle='--', alpha=0.4, linewidth=1)

        # Formatting
        ax.set_title(city, fontsize=14, fontweight='bold', color=COLORS[i])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.15, linestyle='--')
        if i >= 4:
            ax.set_xlabel('Temp (°C)', fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel('Rel. Transmission', fontsize=10)

    # 2. Consolidated Comparison Panel (Panel 10)
    ax_all = axes[9]
    for i, city in enumerate(CITIES_ORDER):
        row = df[df['City'] == city].iloc[0]
        bT, bT2 = float(row['bT']), float(row['bT2'])
        log_beta_rel = bT * (T_range - 28) / 5 + bT2 * ((T_range - 28) / 5) ** 2
        beta_rel = np.exp(log_beta_rel)
        beta_rel[T_range > 35] = 0.0
        ax_all.plot(T_range, beta_rel, color=COLORS[i], linewidth=1.5, alpha=0.7)
    
    ax_all.axvspan(20, 35, alpha=0.1, color='green')
    ax_all.axvline(28, color='orange', linestyle=':', alpha=0.8)
    ax_all.axvline(35, color='#E53935', linestyle='--', alpha=0.4, linewidth=1)
    ax_all.set_title('All Cities Comparison', fontsize=14, fontweight='bold')
    ax_all.set_xlim(xlim)
    ax_all.set_ylim(ylim)
    ax_all.set_xlabel('Temp (°C)', fontsize=10)
    ax_all.grid(alpha=0.2, linestyle='--')

    # 3. Legend Panel (Panel 11)
    ax_leg = axes[10]
    ax_leg.axis('off')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=COLORS[i], lw=3) for i in range(len(CITIES_ORDER))]
    ax_leg.legend(custom_lines, CITIES_ORDER, loc='center', fontsize=11, frameon=False, title='Cities')

    # 4. Math/Notes Panel (Panel 12)
    ax_notes = axes[11]
    ax_notes.axis('off')
    notes_text = (
        "$\mathbf{Model\ Details:}$\n"
        "$\\beta(T) = \kappa \cdot \exp(b_0 + b_T T_z + b_{T2} T_z^2)$\n"
        "where $T_z = (T - 28)/5$.\n\n"
        "• $Rel.\ Transmission$ is normalized\n"
        "  to 1.0 at $T = 28^\circ C$.\n"
        "• Green shading: Biological\n"
        "  window (20–35$^\circ$C).\n"
        "• Dotted line: $T = 28^\circ C$ baseline.\n"
        "• Red dashed: 35$^\circ$C upper\n"
        "  biological limit (Mordecai 2017)."
    )
    ax_notes.text(0.1, 0.5, notes_text, fontsize=11, va='center', linespacing=1.6)

    plt.suptitle('Figure S1: City-Specific Thermal Response Curves — β(T)\nTransmission relative to 28°C optimum (Mordecai et al. 2017)',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(FIG_DIR, 'FigS1_ThermalCurves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

if __name__ == '__main__':
    print("=" * 60)
    print("S6: Generating Figure S1 (Thermal Response Curves)")
    print("=" * 60)
    figS1_thermal()
    print("\n✓ S6 complete. All figures generated!")
    print("✓ Check 04_Figures/ for all output files.")
