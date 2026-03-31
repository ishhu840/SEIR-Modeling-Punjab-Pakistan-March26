"""
S5_Plot_Model_Results.py
========================
Generates Figures 7, 8, 9, 10, 11 for the manuscript.

Fig 07 — Optimal Rain and Temp Lags by City (bar chart)
Fig 08 — Training vs Testing Performance per City (bar chart)
Fig 09 — Observed vs Predicted Time Series (9-panel, most important figure)
Fig 10 — Fitted Weather Coefficients (bR, bT, bT2) per city
Fig 11 — Pearson r Forest Plot with 95% CI per city

Inputs:
  03_Results/V4_Optimal_Lags_ByCity.xlsx
  03_Results/V4_StatTests.xlsx
  03_Results/{City}_Predictions.xlsx  (9 files)

Outputs:
  04_Figures/Fig07_Optimal_Lags.png
  04_Figures/Fig08_Train_Test_Performance.png
  04_Figures/Fig09_TimeSeries.png
  04_Figures/Fig10_WeatherCoeffs.png
  04_Figures/Fig11_PearsonCI_Forest.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', '03_Results')
FIG_DIR    = os.path.join(BASE_DIR, '..', '04_Figures')
os.makedirs(FIG_DIR, exist_ok=True)

OPTIMAL_FILE = os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx')
STATS_FILE   = os.path.join(RESULT_DIR, 'V4_StatTests.xlsx')

CITIES_ORDER = [
    'Islamabad', 'Rawalpindi', 'Gujranwala', 'Faisalabad',
    'Lahore', 'Multan', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ============================================================================
# FIG 07 — Optimal Lags Bar Chart
# ============================================================================
def fig07_optimal_lags():
    print("Generating Fig07_Optimal_Lags.png...")
    df = pd.read_excel(OPTIMAL_FILE)
    df['City_Order'] = df['City'].map({c: i for i,c in enumerate(CITIES_ORDER)})
    df = df.sort_values('City_Order')

    cities = df['City'].tolist()
    x      = np.arange(len(cities))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w/2, df['Rain_Lag'], w, label='Optimal Rainfall Lag',
           color='#1565C0', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, df['Temp_Lag'], w, label='Optimal Temperature Lag',
           color='#E53935', alpha=0.85, edgecolor='white')

    # Reference lines for biological expectations
    ax.axhline(6, color='#1565C0', linewidth=1.0, linestyle='--', alpha=0.5)
    ax.axhline(9, color='#E53935', linewidth=1.0, linestyle='--', alpha=0.5)
    ax.text(len(cities)-0.3, 6.1, 'Expected rain range', fontsize=8.5,
            color='#1565C0', alpha=0.7)
    ax.text(len(cities)-0.3, 9.1, 'Expected temp range', fontsize=8.5,
            color='#E53935', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('Optimal Lag (weeks)', fontsize=12)
    ax.set_ylim(0, 15)
    ax.set_title('Figure 7: City-Specific Optimal Weather Lags — Rainfall (4–8w) and Temperature (6–12w)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    # Add value labels
    for xi, row in zip(x, df.itertuples()):
        ax.text(xi - w/2, row.Rain_Lag + 0.2, str(int(row.Rain_Lag)),
                ha='center', fontsize=10, fontweight='bold', color='#1565C0')
        ax.text(xi + w/2, row.Temp_Lag + 0.2, str(int(row.Temp_Lag)),
                ha='center', fontsize=10, fontweight='bold', color='#E53935')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig07_Optimal_Lags.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
# FIG 08 — Training vs Testing Performance
# ============================================================================
def fig08_train_test():
    print("Generating Fig08_Train_Test_Performance.png...")
    df = pd.read_excel(STATS_FILE)
    df['City_Order'] = df['City'].map({c: i for i,c in enumerate(CITIES_ORDER)})
    df = df.sort_values('City_Order')

    cities = df['City'].tolist()
    x      = np.arange(len(cities))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w/2, df['Train_r'], w, label='Training r (2013–2022)',
           color='#546E7A', alpha=0.8, edgecolor='white')
    ax.bar(x + w/2, df['Test_r'],  w, label='Testing r (2023–2024)',
           color='#1976D2', alpha=0.9, edgecolor='white')

    ax.axhline(0.5, color='#E53935', linewidth=1.2, linestyle=':', alpha=0.7)
    ax.text(len(cities)-0.4, 0.51, 'r = 0.5', color='#E53935', fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title('Figure 8: Model Performance — Training (2013–2022) vs Testing (2023–2024)\n'
                 'All p < 0.001 | Mean Test r = 0.602', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    for xi, row in zip(x, df.itertuples()):
        ax.text(xi - w/2, row.Train_r + 0.015, f'{row.Train_r:.2f}',
                ha='center', fontsize=9, color='#546E7A')
        ax.text(xi + w/2, row.Test_r  + 0.015, f'{row.Test_r:.2f}***',
                ha='center', fontsize=9, color='#1565C0', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig08_Train_Test_Performance.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
# FIG 09 — Observed vs Predicted Time Series (most important)
# ============================================================================
def fig09_time_series():
    print("Generating Fig09_TimeSeries regional subplots...")
    
    regions = {
        'North': ['Islamabad', 'Rawalpindi', 'Gujranwala'],
        'Central': ['Lahore', 'Sheikhupura', 'Faisalabad'],
        'South': ['Sargodha', 'Multan', 'Dera Ghazi Khan']
    }
    
    for region_name, region_cities in regions.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
        axes_flat = axes.flatten()

        for idx, city in enumerate(region_cities):
            ax = axes_flat[idx]
            pred_path = os.path.join(RESULT_DIR, f'{city}_Predictions.xlsx')
            if not os.path.exists(pred_path):
                ax.text(0.5, 0.5, f'No data\n{city}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11)
                continue

            df = pd.read_excel(pred_path)
            df['Date'] = pd.to_datetime(
                df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2) + '-1',
                format='%Y-W%W-%w', errors='coerce'
            )
            df = df.sort_values('Date')

            train = df[df['Year'] <= 2022]
            test  = df[df['Year'] >  2022]

            # Observed (solid black)
            ax.plot(df['Date'], df['Number of Dengue Cases'],
                    color='#212121', linewidth=1.2, label='Observed', zorder=3)

            # Training predicted (blue)
            ax.plot(train['Date'], train['Predicted'],
                    color='#1976D2', linewidth=1.5, linestyle='-',
                    label='Predicted (Training)', alpha=0.85, zorder=2)

            # Testing predicted (red)
            ax.plot(test['Date'], test['Predicted'],
                    color='#E53935', linewidth=2.0, linestyle='-',
                    label='Predicted (Testing)', alpha=0.9, zorder=2)

            # Train/test divider
            split_date = pd.Timestamp('2023-01-01')
            max_date = df['Date'].max()
            if not pd.isna(max_date):
                ax.axvspan(split_date, max_date, color='#FF9800', alpha=0.1, zorder=1)
            ax.axvline(split_date, color='#FF9800', linewidth=1.5,
                       linestyle='--', alpha=0.8, zorder=2)
            
            ax.set_ylim(bottom=0)
            
            ax.text(split_date, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 10,
                    '← Train | Test →', fontsize=7.5, color='#F57C00', ha='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

            # Pearson r
            stats_df = pd.read_excel(STATS_FILE)
            city_stat = stats_df[stats_df['City'] == city]
            if not city_stat.empty:
                r_test = city_stat.iloc[0]['Test_r']
                ax.text(0.02, 0.94, f'Test r = {r_test:.3f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=10, fontweight='bold', color='#E53935',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#E53935', alpha=0.85), zorder=4)

            ax.set_title(city, fontsize=12, fontweight='bold', pad=4)
            ax.set_ylabel('Cases/week', fontsize=10)
            ax.tick_params(axis='x', labelsize=9, rotation=0)
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Unified legend
        handles = [
            plt.Line2D([0],[0], color='#212121', lw=1.5, label='Observed'),
            plt.Line2D([0],[0], color='#1976D2', lw=1.5, label='Predicted (Training 2013–2022)'),
            plt.Line2D([0],[0], color='#E53935', lw=2.0, label='Predicted (Testing 2023–2024)'),
            mpatches.Patch(color='#FF9800', alpha=0.2, label='Testing Period'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=11,
                   bbox_to_anchor=(0.5, 0.96), framealpha=0.95)

        fig.suptitle(f'Figure 9: Observed vs Predicted {region_name} Punjab Time Series',
                     fontsize=15, fontweight='bold', y=1.0)

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        out = os.path.join(FIG_DIR, f'Fig09_{region_name}_TimeSeries.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out}")


# ============================================================================
# FIG 10 — Weather Coefficients
# ============================================================================
def fig10_weather_coeffs():
    print("Generating Fig10_WeatherCoeffs.png...")
    df = pd.read_excel(OPTIMAL_FILE)
    
    # Exclude Dera Ghazi Khan for a cleaner figure
    df = df[df['City'] != 'Dera Ghazi Khan']
    
    df['City_Order'] = df['City'].map({c: i for i,c in enumerate(CITIES_ORDER)})
    df = df.sort_values('City_Order')

    cities = df['City'].tolist()
    x      = np.arange(len(cities))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    
    bR = df['bR'].values
    bT = df['bT'].values
    bT2 = df['bT2'].values

    limit = 8.0
    bR_cap = np.clip(bR, -limit, limit)
    bT_cap = np.clip(bT, -limit, limit)
    bT2_cap = np.clip(bT2, -limit, limit)

    ax.bar(x - w,   bR_cap,  w, label='bR (Rainfall)',      color='#1565C0', alpha=0.85, edgecolor='white')
    ax.bar(x,       bT_cap,  w, label='bT (Temperature)',   color='#E53935', alpha=0.85, edgecolor='white')
    ax.bar(x + w,   bT2_cap, w, label='bT² (Temp²)',        color='#FB8C00', alpha=0.85, edgecolor='white')

    for i, (r, t, t2) in enumerate(zip(bR, bT, bT2)):
        if abs(r) > limit:
            ax.text(x[i] - w, np.sign(r)*(limit - 1.2), f'{r:.1f}', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8, rotation=90)
        if abs(t) > limit:
            ax.text(x[i], np.sign(t)*(limit - 1.2), f'{t:.1f}', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8, rotation=90)
        if abs(t2) > limit:
            ax.text(x[i] + w, np.sign(t2)*(limit - 1.2), f'{t2:.1f}', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8, rotation=90)

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Figure 10: Fitted Weather Coefficients in β(t) — V4 SEIR Model\n'
                 'β(t) = κ·exp(b₀ + bR·Rain + bT·Temp + bT²·Temp²)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig10_WeatherCoeffs.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
# FIG 11 — Pearson r Forest Plot with 95% CI
# ============================================================================
def fig11_pearson_forest():
    print("Generating Fig11_PearsonCI_Forest.png...")
    df = pd.read_excel(STATS_FILE)
    df['City_Order'] = df['City'].map({c: i for i,c in enumerate(CITIES_ORDER)})
    df = df.sort_values('Test_r', ascending=True)   # sort by r for forest plot

    cities   = df['City'].tolist()
    r_vals   = df['Test_r'].values
    ci_lo    = df['Test_CI_lower'].values
    ci_hi    = df['Test_CI_upper'].values
    pvals    = df['Test_p'].values

    y_pos = np.arange(len(cities))

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1B5E20' if p < 0.001 else '#388E3C' if p < 0.01 else '#66BB6A'
              for p in pvals]

    # Error bars
    xerr_lo = r_vals - ci_lo
    xerr_hi = ci_hi - r_vals
    ax.errorbar(r_vals, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt='none', ecolor='#aaaaaa',
                elinewidth=1.5, capsize=5, capthick=1.5, zorder=1)

    # Dots
    scatter = ax.scatter(r_vals, y_pos, c=colors, s=100, zorder=3,
                         edgecolors='white', linewidths=1.0)

    # Reference lines
    ax.axvline(0.0, color='gray', linewidth=1.0, linestyle='-', alpha=0.4)
    ax.axvline(0.5, color='#E53935', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.text(0.51, -0.7, 'r = 0.5', color='#E53935', fontsize=9.5, va='bottom')

    # Annotations
    for i, (r, lo, hi, p) in enumerate(zip(r_vals, ci_lo, ci_hi, pvals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(hi + 0.01, i, f'  {r:.3f}{sig}  [{lo:.3f}, {hi:.3f}]',
                va='center', fontsize=9.5, color='#333333')

    # Mean line
    mean_r = r_vals.mean()
    ax.axvline(mean_r, color='#1565C0', linewidth=2.0, linestyle='-', alpha=0.8)
    ax.text(mean_r, len(cities) - 0.3, f' Mean = {mean_r:.3f}',
            color='#1565C0', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities, fontsize=11)
    ax.set_ylabel('City', fontsize=12)
    ax.set_xlabel('Pearson r (Testing Period 2023–2024)', fontsize=12)
    ax.set_xlim(-0.1, 1.05)
    
    # Increase the y-axis limits to give more breathing room between the plot and the title
    ax.set_ylim(-0.8, len(cities) + 0.5)

    ax.set_title('Figure 11: Testing Pearson Correlation with 95% Confidence Intervals\n'
                 '(Fisher Z-transformation | V4 SEIR Model | All cities p < 0.001)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig11_PearsonCI_Forest.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")

# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("S5: Generating Figures 7, 8, 9, 10, 11")
    print("=" * 60)
    fig07_optimal_lags()
    fig08_train_test()
    fig09_time_series()
    fig10_weather_coeffs()
    fig11_pearson_forest()
    print("\n✓ S5 complete. Next: run S6_Plot_Thermal_S1.py")
