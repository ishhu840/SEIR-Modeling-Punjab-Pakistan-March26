"""
S3_Plot_Spearman_NB.py
======================
Generates Figure 3 and Figure 4 for the manuscript.

Figure 3 — Spearman Correlation Validation:
  Shows non-parametric Spearman ρ between lagged weather variables
  (rainfall at optimal rain lag, temperature at optimal temp lag)
  and dengue cases for each city. Validates that using weather variables
  as SEIR model inputs is biologically and statistically justified.

Figure 4 — Fixed vs Optimal Lag Performance:
  Grouped bar chart comparing Test r (2023-2024) for:
  - Fixed uniform lag (7w rain, 7w temp) — previous approach
  - V4 city-specific optimal lags — our method
  Shows the benefit of optimization for each city.

Inputs:
  01_Data/D1_Weekly_Cases_Weather_AllCities.xlsx
  03_Results/V4_Optimal_Lags_ByCity.xlsx
  03_Results/V4_StatTests.xlsx

Outputs:
  04_Figures/Fig03_Spearman_Valid.png
  04_Figures/Fig04_Fixed_vs_Optimal.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', '01_Data')
RESULT_DIR  = os.path.join(BASE_DIR, '..', '03_Results')
FIG_DIR     = os.path.join(BASE_DIR, '..', '04_Figures')
os.makedirs(FIG_DIR, exist_ok=True)

DATA_FILE    = os.path.join(DATA_DIR,   'D1_Weekly_Cases_Weather_AllCities.xlsx')
OPTIMAL_FILE = os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx')
STATS_FILE   = os.path.join(RESULT_DIR, 'V4_StatTests.xlsx')

CITIES_ORDER = [
    'Islamabad', 'Rawalpindi', 'Gujranwala', 'Faisalabad',
    'Lahore', 'Multan', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]

# Publication style
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150
})

# ============================================================================
# FIGURE 3 — Spearman Correlation Validation
# ============================================================================
def fig03_spearman():
    print("Generating Fig03_Spearman_Valid.png...")

    df_all  = pd.read_excel(DATA_FILE)
    opt_df  = pd.read_excel(OPTIMAL_FILE)

    results = []
    for city in CITIES_ORDER:
        opt = opt_df[opt_df['City'] == city]
        if opt.empty:
            continue
        opt = opt.iloc[0]
        rl = int(opt['Rain_Lag'])
        tl = int(opt['Temp_Lag'])

        df = df_all[df_all['City'] == city].copy().reset_index(drop=True)
        df['Rain_L'] = df['Rainfall'].shift(rl)
        df['Temp_L'] = df['Temperature'].shift(tl)
        df = df.dropna(subset=['Rain_L', 'Temp_L'])

        y = df['Number of Dengue Cases'].values
        rho_rain, p_rain = spearmanr(df['Rain_L'].values, y)
        rho_temp, p_temp = spearmanr(df['Temp_L'].values, y)

        results.append({
            'City': city, 'Rain_Lag': rl, 'Temp_Lag': tl,
            'Rain_rho': rho_rain, 'Rain_p': p_rain,
            'Temp_rho': rho_temp, 'Temp_p': p_temp
        })

    df_sp = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Figure 3: Spearman Correlation — Weather Variables vs Dengue Cases\n'
                 '(at city-specific optimal lags)', fontsize=14, fontweight='bold', y=1.02)

    cities   = df_sp['City'].tolist()
    y_pos    = np.arange(len(cities))
    bar_h    = 0.6

    for i, (ax, (col, pcol, xlabel, title), label) in enumerate(zip(axes, [
        ('Rain_rho', 'Rain_p', 'Spearman ρ (Rainfall at optimal lag)', 'Rainfall → Dengue'),
        ('Temp_rho', 'Temp_p', 'Spearman ρ (Temperature at optimal lag)', 'Temperature → Dengue')
    ], ['A', 'B'])):
        vals   = df_sp[col].values
        pvals  = df_sp[pcol].values
        colors = ['#2196F3' if p < 0.001 else '#64B5F6' if p < 0.05 else '#BDBDBD'
                  for p in pvals]

        bars = ax.barh(y_pos, vals, height=bar_h, color=colors,
                       edgecolor='white', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f"{label}) {title}", fontsize=12, fontweight='bold', loc='left')
        ax.set_xlim(-0.6, 0.9)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

        for j, (val, p) in enumerate(zip(vals, pvals)):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ha = 'left' if val >= 0 else 'right'
            xoff = 0.02 if val >= 0 else -0.02
            ax.text(val + xoff, j, f'{val:.2f}{sig}', va='center', ha=ha, fontsize=9.5)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(cities, fontsize=11)

    # Legend
    p1 = mpatches.Patch(color='#2196F3', label='p < 0.001 (***)')
    p2 = mpatches.Patch(color='#64B5F6', label='p < 0.05 (*)')
    p3 = mpatches.Patch(color='#BDBDBD', label='Not significant')
    axes[1].legend(handles=[p1, p2, p3], loc='upper right', fontsize=9.5)

    fig.text(0.5, -0.04,
             '*** p<0.001 | ** p<0.01 | * p<0.05 | ns = not significant\n'
             'Rainfall lag = optimal rain lag; Temperature lag = optimal temp lag',
             ha='center', fontsize=9.5, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(FIG_DIR, 'Fig03_Spearman_Valid.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
# FIGURE 4 — Fixed vs Optimal Lag Performance Comparison
# ============================================================================
def fig04_fixed_vs_optimal():
    print("Generating Fig04_Fixed_vs_Optimal.png...")

    df = pd.read_excel(STATS_FILE)
    df = df[df['City'].isin(CITIES_ORDER)].copy()
    df['City_Order'] = df['City'].map({c: i for i, c in enumerate(CITIES_ORDER)})
    df = df.sort_values('City_Order')

    cities      = df['City'].tolist()
    opt_r       = df['Test_r'].values
    fixed_r     = df['Fixed_Lag_7_7_r'].values
    improvement = df['Improvement'].values
    pvals       = df['Improvement_p'].values

    x     = np.arange(len(cities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_fixed = ax.bar(x - width/2, fixed_r, width, label='Fixed Lag (7w rain, 7w temp)',
                        color='#9E9E9E', edgecolor='white', linewidth=0.8, alpha=0.85)
    bars_opt   = ax.bar(x + width/2, opt_r,   width, label='City-Specific Optimal Lag (V4)',
                        color='#1976D2', edgecolor='white', linewidth=0.8, alpha=0.9)

    # Significance stars on top of optimal bars
    for i, (r, p, imp) in enumerate(zip(opt_r, pvals, improvement)):
        sig = ''
        if imp > 0 and p < 0.05:
            sig = '\n***' if p < 0.001 else '\n**' if p < 0.01 else '\n*'
            
        imp_str = f'+{imp:.2f}' if imp > 0 else f'{imp:.2f}'
        text_str = f"{imp_str}{sig}"
        color = '#1565C0' if imp > 0 else '#c62828'
        
        ax.text(i + width/2, r + 0.01, text_str,
                ha='center', va='bottom', fontsize=9, color=color,
                fontweight='bold', linespacing=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=35, ha='right', fontsize=11)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Pearson r (Testing Period 2023–2024)', fontsize=12)
    ax.set_title('Figure 4: City-Specific Optimal Lags vs Fixed Uniform Lag (7w, 7w)\n'
                 'V4 Model Performance on Held-Out Test Data (2023–2024)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.axhline(0.5, color='#E53935', linewidth=1.2, linestyle=':', alpha=0.7)
    ax.text(len(cities) - 0.5, 0.51, 'r = 0.5', color='#E53935', fontsize=9.5)

    fig.text(0.5, -0.06,
             '*** p<0.001 | ** p<0.01 | * p<0.05 (one-sided z-test on arctanh-transformed r)\n'
             'Numbers above bars show improvement (Δr = optimal − fixed). '
             f'8/9 cities improved. Wilcoxon p=0.006.',
             ha='center', fontsize=9.5, style='italic', color='#555555')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig04_Fixed_vs_Optimal.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("S3: Generating Figures 3 and 4")
    print("=" * 60)
    fig03_spearman()
    fig04_fixed_vs_optimal()
    print("\n✓ S3 complete. Next: run S4_Plot_Heatmaps.py")
