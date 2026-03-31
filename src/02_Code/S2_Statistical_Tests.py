"""
S2_Statistical_Tests.py
========================
Statistical significance testing for V4 SEIR model results.

For each city, calculates:
  - Pearson r (training + testing) with 95% CI via Fisher Z-transformation
  - p-values for training and testing correlations
  - Fixed-lag (7,7) baseline comparison from grid search results
  - Improvement significance: one-sided z-test on arctanh-transformed r values
  - Wilcoxon signed-rank test across cities (optimal vs fixed)
  - Sign test across cities

OUTPUT: V4_StatTests.xlsx saved to 03_Results/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, wilcoxon

# ============================================================================
# PATHS (relative to this script's location, i.e. 02_Code/)
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', '03_Results')

OPTIMAL_FILE = os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx')
GRID_FILE    = os.path.join(RESULT_DIR, 'V4_GridSearch_AllResults.xlsx')

CITIES = [
    'Islamabad', 'Rawalpindi', 'Gujranwala', 'Faisalabad',
    'Lahore', 'Multan', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]

TRAIN_END_YEAR = 2022

# ============================================================================
def pearson_ci(r, n, alpha=0.05):
    """
    Fisher Z-transformation to get 95% CI for Pearson r.
    Returns (r, ci_lower, ci_upper, p_value)
    """
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    # Two-sided p-value
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_val  = 2 * stats.t.sf(abs(t_stat), df=n - 2)
    return ci_lo, ci_hi, p_val

def improvement_significance(r_opt, r_fixed, n):
    """
    One-sided z-test on arctanh-transformed correlations.
    H0: r_opt <= r_fixed  |  H1: r_opt > r_fixed
    """
    if np.isnan(r_opt) or np.isnan(r_fixed):
        return np.nan, np.nan
    z_diff = (np.arctanh(r_opt) - np.arctanh(r_fixed)) / np.sqrt(2.0 / (n - 3))
    p_one_sided = stats.norm.sf(z_diff)   # one-sided (right tail)
    return float(z_diff), float(p_one_sided)

# ============================================================================
def main():
    print("=" * 70)
    print("S2: STATISTICAL SIGNIFICANCE TESTS (V4)")
    print("=" * 70)

    optimal_df = pd.read_excel(OPTIMAL_FILE)
    grid_df    = pd.read_excel(GRID_FILE)

    results = []

    for city in CITIES:
        pred_path = os.path.join(RESULT_DIR, f'{city}_Predictions.xlsx')
        if not os.path.exists(pred_path):
            print(f"  WARNING: Missing predictions for {city} — skipping")
            continue

        pred_df = pd.read_excel(pred_path)

        # Get optimal lag info for this city
        opt_row = optimal_df[optimal_df['City'] == city]
        if opt_row.empty:
            print(f"  WARNING: No optimal lag found for {city} — skipping")
            continue
        opt_row = opt_row.iloc[0]

        # Split into training and testing
        train = pred_df[pred_df['Year'] <= TRAIN_END_YEAR].copy()
        test  = pred_df[pred_df['Year'] >  TRAIN_END_YEAR].copy()

        y_train     = train['Number of Dengue Cases'].values
        yhat_train  = train['Predicted'].fillna(0).values
        y_test      = test['Number of Dengue Cases'].values
        yhat_test   = test['Predicted'].fillna(0).values

        n_train = len(y_train)
        n_test  = len(y_test)

        # --- Training correlation + CI ---
        if n_train > 3 and np.std(yhat_train) > 0:
            r_train, _ = pearsonr(y_train, yhat_train)
            ci_lo_train, ci_hi_train, p_train = pearson_ci(r_train, n_train)
        else:
            r_train = ci_lo_train = ci_hi_train = p_train = np.nan

        # --- Testing correlation + CI + Error ---
        if n_test > 3 and np.std(yhat_test) > 0:
            r_test, _ = pearsonr(y_test, yhat_test)
            ci_lo_test, ci_hi_test, p_test = pearson_ci(r_test, n_test)
            mae_test = np.mean(np.abs(y_test - yhat_test))
            rmse_test = np.sqrt(np.mean((y_test - yhat_test)**2))
        else:
            r_test = ci_lo_test = ci_hi_test = p_test = mae_test = rmse_test = np.nan

        # --- Fixed lag (7, 7) baseline from grid search ---
        fixed_row = grid_df[
            (grid_df['City'] == city) &
            (grid_df['Rain_Lag'] == 7) &
            (grid_df['Temp_Lag'] == 7)
        ]
        r_fixed = float(fixed_row['Test_r'].values[0]) if not fixed_row.empty else np.nan

        # --- Improvement significance ---
        z_diff, p_improvement = improvement_significance(r_test, r_fixed, n_test)
        improvement = r_test - r_fixed if not np.isnan(r_fixed) else np.nan

        results.append({
            'City':               city,
            'Optimal_Rain_Lag':   int(opt_row['Rain_Lag']),
            'Optimal_Temp_Lag':   int(opt_row['Temp_Lag']),
            'rho_pct':            round(float(opt_row['rho']) * 100, 1),
            'n_train':            n_train,
            'Train_r':            round(r_train,      3) if not np.isnan(r_train)      else np.nan,
            'Train_CI_lower':     round(ci_lo_train,  3) if not np.isnan(ci_lo_train)  else np.nan,
            'Train_CI_upper':     round(ci_hi_train,  3) if not np.isnan(ci_hi_train)  else np.nan,
            'Train_p':            round(p_train,       4) if not np.isnan(p_train)      else np.nan,
            'n_test':             n_test,
            'Test_r':             round(r_test,       3) if not np.isnan(r_test)       else np.nan,
            'Test_CI_lower':      round(ci_lo_test,   3) if not np.isnan(ci_lo_test)   else np.nan,
            'Test_CI_upper':      round(ci_hi_test,   3) if not np.isnan(ci_hi_test)   else np.nan,
            'Test_p':             round(p_test,        4) if not np.isnan(p_test)       else np.nan,
            'Test_MAE':           round(mae_test,      2) if not np.isnan(mae_test)     else np.nan,
            'Test_RMSE':          round(rmse_test,     2) if not np.isnan(rmse_test)    else np.nan,
            'Fixed_Lag_7_7_r':    round(r_fixed,       3) if not np.isnan(r_fixed)      else np.nan,
            'Improvement':        round(improvement,   3) if not np.isnan(improvement)  else np.nan,
            'Improvement_z':      round(z_diff,        3) if not np.isnan(z_diff)       else np.nan,
            'Improvement_p':      round(p_improvement, 4) if not np.isnan(p_improvement) else np.nan,
        })

    df_results = pd.DataFrame(results)

    # ----------------------------------------------------------------
    # ACROSS-CITY TESTS (Wilcoxon + Sign test)
    # ----------------------------------------------------------------
    valid = df_results.dropna(subset=['Test_r', 'Fixed_Lag_7_7_r'])
    opt_r   = valid['Test_r'].values
    fixed_r = valid['Fixed_Lag_7_7_r'].values
    diffs   = opt_r - fixed_r

    n_pos   = int((diffs > 0).sum())
    n_total = int(len(diffs))

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = wilcoxon(diffs, alternative='greater')
    except Exception:
        w_stat, w_p = np.nan, np.nan

    # Sign test (using binomtest for scipy >= 1.9)
    try:
        from scipy.stats import binomtest
        sign_result = binomtest(n_pos, n_total, 0.5, alternative='greater')
        sign_p = sign_result.pvalue
    except ImportError:
        sign_p = stats.binom_test(n_pos, n_total, 0.5, alternative='greater') if n_total > 0 else np.nan

    # ----------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------
    out_path = os.path.join(RESULT_DIR, 'V4_StatTests.xlsx')
    df_results.to_excel(out_path, index=False)
    print(f"\n✓ Saved: V4_StatTests.xlsx")

    # ----------------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-CITY RESULTS")
    print("=" * 70)
    print(f"\n{'City':<20} {'Rain':>5} {'Temp':>5} {'ρ%':>6} {'Test r':>8} {'95% CI':>18} {'vs Fixed(7,7)':>14}")
    print("-" * 78)

    for _, row in df_results.sort_values('Test_r', ascending=False).iterrows():
        sig = "***" if row['Test_p'] < 0.001 else "**" if row['Test_p'] < 0.01 else "*" if row['Test_p'] < 0.05 else "  "
        imp = f"+{row['Improvement']:.3f}" if not np.isnan(row['Improvement']) else "  N/A"
        print(f"{row['City']:<20} {int(row['Optimal_Rain_Lag']):>5} {int(row['Optimal_Temp_Lag']):>5} "
              f"{row['rho_pct']:>5.1f}% {row['Test_r']:>7.3f}{sig} "
              f"[{row['Test_CI_lower']:.3f}, {row['Test_CI_upper']:.3f}]  {imp}")

    print("-" * 78)
    mean_r     = df_results['Test_r'].mean()
    mean_fixed = df_results['Fixed_Lag_7_7_r'].mean()
    print(f"{'MEAN':<20} {'':>5} {'':>5} {'':>6} {mean_r:>7.3f}    "
          f"{'':18}  Fixed mean: {mean_fixed:.3f}")

    print(f"\n*** p<0.001, ** p<0.01, * p<0.05")

    print(f"\n{'='*70}")
    print(f"ACROSS-CITY TESTS")
    print(f"{'='*70}")
    print(f"  Cities where optimal > fixed (7,7): {n_pos} / {n_total}")
    print(f"  Mean improvement in Test r: {diffs.mean():+.3f}")
    print(f"  Wilcoxon signed-rank:  W={w_stat}, p={w_p:.4f}")
    print(f"  Sign test (one-sided): {n_pos}/{n_total} positive, p={sign_p:.4f}")

    print(f"\n✓ Next step: run S3_Plot_Spearman_NB.py")
    print("=" * 70)

if __name__ == '__main__':
    main()
