import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = '../03_Results/'
OUTPUT_DIR = '../04_Figures/'
files = [f for f in os.listdir(RESULTS_DIR) if '_Predictions.xlsx' in f]

cities = sorted([f.replace('_Predictions.xlsx', '') for f in files])
n_cities = len(cities)

# Plotting setup
fig, axes = plt.subplots(n_cities, 2, figsize=(12, 3 * n_cities))
fig.suptitle('Residual Analysis (Testing Period 2023-2024)', fontsize=16, y=0.98)

# Ljung-Box Test Results Table
lb_results = []

for i, city in enumerate(cities):
    file_path = os.path.join(RESULTS_DIR, f"{city}_Predictions.xlsx")
    df = pd.read_excel(file_path)
    
    # Isolate Testing Period
    test_df = df[df['Period'] == 'Testing'].copy()
    
    # Calculate Residuals (Observed - Predicted)
    test_df['Residuals'] = test_df['Number of Dengue Cases'] - test_df['Predicted']
    
    residuals = test_df['Residuals'].values
    
    # Ljung-Box Test (checking up to 4 weeks lag)
    # A p-value > 0.05 means we CANNOT reject the null hypothesis of white noise
    # (i.e. p > 0.05 is GOOD, it means residuals are random)
    lb_stat = acorr_ljungbox(residuals, lags=[4], return_df=True)
    p_val = lb_stat['lb_pvalue'].values[0]
    is_random = "Yes" if p_val > 0.05 else "No"
    
    lb_results.append({
        'City': city,
        'LB Stat (lag=4)': lb_stat['lb_stat'].values[0],
        'p-value': p_val,
        'Random Noise?': is_random
    })

    # Plot ACF
    ax_acf = axes[i, 0]
    plot_acf(residuals, ax=ax_acf, title=f'{city} - Autocorrelation (ACF)', lags=20, zero=False)
    ax_acf.set_xlabel('Lag (Weeks)')
    ax_acf.set_ylabel('ACF')
    
    # Plot PACF
    ax_pacf = axes[i, 1]
    plot_pacf(residuals, ax=ax_pacf, title=f'{city} - Partial Autocorrelation (PACF)', lags=20, zero=False, method='ywm')
    ax_pacf.set_xlabel('Lag (Weeks)')
    ax_pacf.set_ylabel('PACF')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'FigS3_Residual_ACF_PACF.png'), dpi=300, bbox_inches='tight')
print(f"Residual plots saved to {os.path.join(OUTPUT_DIR, 'FigS3_Residual_ACF_PACF.png')}")

# Print Table
print("\nLjung-Box Test for White Noise in Residuals (Testing Period):")
print("-" * 75)
print(f"{'City':<15s} {'LB Stat':>10s} {'p-value':>10s} {'Is Random Noise?':>20s}")
print("-" * 75)
for res in lb_results:
    print(f"{res['City']:<15s} {res['LB Stat (lag=4)']:>10.2f} {res['p-value']:>10.4f}     {res['Random Noise?']:>10s}")
print("-" * 75)
print("Note: p > 0.05 indicates the residuals are indistinguishable from random noise,")
print("meaning the SEIR model successfully captured the true biological signal.")

df_lb = pd.DataFrame(lb_results)
df_lb.to_csv(os.path.join(RESULTS_DIR, 'S8_LjungBox_TestResults.csv'), index=False)
print("Results saved to 03_Results/S8_LjungBox_TestResults.csv")
