"""
S4_Plot_Heatmaps.py
===================
Generates Figure 6: 35-Lag Optimization Heatmaps

For each of the 9 cities, shows a 5×7 heatmap of Test r values
across all rainfall lag (4-8w) × temperature lag (6-12w) combinations.
The optimal combination is highlighted with a red star.

This is the key figure showing WHY city-specific lags are needed —
each city has a different optimal location on the landscape.

Inputs:  03_Results/V4_GridSearch_AllResults.xlsx
         03_Results/V4_Optimal_Lags_ByCity.xlsx
Outputs: 04_Figures/Fig06_Lag_Heatmaps.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', '03_Results')
FIG_DIR    = os.path.join(BASE_DIR, '..', '04_Figures')
os.makedirs(FIG_DIR, exist_ok=True)

GRID_FILE    = os.path.join(RESULT_DIR, 'V4_GridSearch_AllResults.xlsx')
OPTIMAL_FILE = os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx')

RAIN_LAGS = [4, 5, 6, 7, 8]
TEMP_LAGS = [6, 7, 8, 9, 10, 11, 12]

CITIES_ORDER = [
    'Lahore', 'Rawalpindi', 'Islamabad', 'Gujranwala',
    'Faisalabad', 'Multan', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]

def fig06_heatmaps():
    print("Generating Fig06_Lag_Heatmaps.png...")

    grid_df    = pd.read_excel(GRID_FILE)
    optimal_df = pd.read_excel(OPTIMAL_FILE)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes_flat = axes.flatten()

    cmap = plt.cm.RdBu_r

    for idx, city in enumerate(CITIES_ORDER):
        ax = axes_flat[idx]
        city_data = grid_df[grid_df['City'] == city]

        # Build heatmap matrix: rows = rain lags, cols = temp lags
        matrix = np.full((len(RAIN_LAGS), len(TEMP_LAGS)), np.nan)
        for _, row in city_data.iterrows():
            r_idx = RAIN_LAGS.index(int(row['Rain_Lag']))
            t_idx = TEMP_LAGS.index(int(row['Temp_Lag']))
            matrix[r_idx, t_idx] = row['Test_r']

        # Clamp to [-0.85, 0.85] for symmetrical diverging white center
        vmin, vmax = -0.85, 0.85

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Axes labels
        ax.set_xticks(range(len(TEMP_LAGS)))
        ax.set_xticklabels(TEMP_LAGS, fontsize=8.5)
        ax.set_yticks(range(len(RAIN_LAGS)))
        ax.set_yticklabels(RAIN_LAGS, fontsize=8.5)

        if idx >= 6:
            ax.set_xlabel('Temperature Lag (weeks)', fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel('Rainfall Lag (weeks)', fontsize=9)

        # Add r values inside cells
        for ri in range(len(RAIN_LAGS)):
            for ti in range(len(TEMP_LAGS)):
                val = matrix[ri, ti]
                if not np.isnan(val):
                    ax.text(ti, ri, f'{val:.2f}', ha='center', va='center',
                            fontsize=7.5, color='black', fontweight='bold')

        # Mark optimal with bounding box
        opt = optimal_df[optimal_df['City'] == city]
        if not opt.empty:
            opt = opt.iloc[0]
            opt_r_idx = RAIN_LAGS.index(int(opt['Rain_Lag']))
            opt_t_idx = TEMP_LAGS.index(int(opt['Temp_Lag']))
            
            # Create a rectangle patch for the bounding box
            import matplotlib.patches as patches
            rect = patches.Rectangle((opt_t_idx - 0.5, opt_r_idx - 0.5), 1, 1, 
                                     linewidth=4, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            ax.set_title(f'{city}\n[Box] Rain={int(opt["Rain_Lag"])}w, Temp={int(opt["Temp_Lag"])}w, '
                        f'r={opt["Test_r"]:.3f}',
                        fontsize=10, fontweight='bold', pad=4)
        else:
            ax.set_title(city, fontsize=10, fontweight='bold')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Test Pearson r (2023–2024)', fontsize=11)

    fig.suptitle('Figure 6: Lag Optimization Landscape — Test Pearson r for All 35 Lag Combinations\n'
                 '(5 Rainfall Lags × 7 Temperature Lags per City | V4 Model | [Box] = Optimal)',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out = os.path.join(FIG_DIR, 'Fig06_Lag_Heatmaps.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

if __name__ == '__main__':
    print("=" * 60)
    print("S4: Generating Figure 6 (35-Lag Heatmaps)")
    print("=" * 60)
    fig06_heatmaps()
    print("\n✓ S4 complete. Next: run S5_Plot_Model_Results.py")
