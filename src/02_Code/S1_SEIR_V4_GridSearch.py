"""
S1_SEIR_V4_GridSearch.py
========================
V4 SEIR Model — City-Specific Lag Optimization for Dengue Forecasting
Punjab, Pakistan | 2013-2024

PURPOSE:
--------
For each of 9 testable cities, test all 35 weather-lag combinations
(5 rainfall lags × 7 temperature lags) and select the combination that
gives the best out-of-sample testing correlation (Pearson r, 2023-2024).

KEY V4 DESIGN DECISIONS (all approved and documented in README.md):
-------------------------------------------------------------------
1. NO annual reset — recovered compartment (R) carries forward across all
   12 years. Continuous waning immunity parameter omega (ω) handles the
   gradual return to susceptibility. [Supervisor's preferred approach]

2. WEAK INFORMATIVE PRIOR on reporting fraction ρ, centered at 10%.
   Applied as a Gaussian penalty in logit-space (weight=200).
   Scientific basis: Pakistan dengue surveillance captures ~5-15% of true
   infections (Bhatt 2013 Nature; Shepard 2016 PLOS NTD).
   This replaces the V2 hard ceiling at 35% which caused boundary solutions.

3. TEMPERATURE BIOLOGICAL PENALTY: enforces that the quadratic β(T) curve
   peaks within 20-35°C (Aedes aegypti thermal window, Mordecai 2017).

4. NELDER-MEAD optimizer: gradient-free, handles non-smooth penalty
   surfaces correctly. Unlike BFGS, it does not follow gradients past
   biological boundaries.

5. DYNAMIC POPULATION N(t): linear interpolation between 2017 and 2023
   Pakistan Census values. Corrects for 15-19% urban growth bias.

FIXED BIOLOGICAL PARAMETERS (NOT fitted):
-----------------------------------------
  σ (incubation rate) = 7/6 per week  (~6-day incubation, Nishiura 2007)
  γ (recovery rate)   = 7/5 per week  (~5-day infectious period, Gubler 1998)
  Temp lower bound    = 20°C           (Mordecai et al. 2017, PLOS NTD)
  Temp upper bound    = 35°C           (Mordecai et al. 2017, PLOS NTD)

FITTED PARAMETERS (10 per city per lag combination):
-----------------------------------------------------
  log(κ)     → κ  = base transmission scaling factor
  b0         → baseline log-transmission intercept
  bR         → rainfall effect on log-transmission
  bT         → temperature linear effect
  bT2        → temperature quadratic effect (must give inverted-U)
  logit(ρ)   → ρ  = reporting fraction (soft prior at 10%)
  log(λ)     → λ  = external importation rate (cases/week)
  logit(E0f) → E0 = initial exposed fraction (max 1%)
  logit(I0f) → I0 = initial infectious fraction (max 1%)
  log(ω)     → ω  = waning immunity rate (loss of protection over time)

OUTPUTS (saved to ../03_Results/):
-----------------------------------
  V4_GridSearch_AllResults.xlsx   — all 315 fits (9 cities × 35 combos)
  V4_Optimal_Lags_ByCity.xlsx     — best lag pair per city
  V4_Parameters_ByCity.xlsx       — all fitted parameters for best lag
  [City]_Predictions.xlsx         — weekly obs vs predicted per city

EXECUTION TIME:
---------------
  Approximately 6-10 hours on MacBook Pro (Apple M2 Pro).
  Uses all available CPU cores via multiprocessing.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid = 1/(1+exp(-x))
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS — all relative to this script's location
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', '01_Data')
RESULT_DIR = os.path.join(BASE_DIR, '..', '03_Results')
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, 'D1_Weekly_Cases_Weather_AllCities.xlsx')
POP_FILE  = os.path.join(DATA_DIR, 'D2_Population_2017_2023.xlsx')

# ============================================================================
# BIOLOGICAL CONSTANTS (fixed — NOT optimized)
# ============================================================================
SIGMA = 7.0 / 6.0   # Exposed → Infectious (incubation ~6 days)
GAMMA = 7.0 / 5.0   # Infectious → Recovered (illness ~5 days)
EPS   = 1e-12        # Numerical stability

# Fixed continuous waning immunity parameter (~2.5 years half-life)
OMEGA_FIXED = 1.0 / (2.5 * 52.0)

# Natural death rate μ (David Sheridan, March 2026)
# Pakistan life expectancy = 68.0 years (World Bank 2023)
# Constant death rate μ = 1/68 = 14.7/1000 per year → weekly
MU = (1.0 / 68.0) / 52.0   # ≈ 0.000283 per week


# Temperature biological window (Mordecai et al. 2017, PLOS NTD)
TEMP_OPT_MIN = 20.0  # °C — lower transmission bound
TEMP_OPT_MAX = 35.0  # °C — upper transmission bound (mosquito mortality)

# Reporting fraction prior (Bhatt 2013; Shepard 2016 — Pakistan ~5-15%)
RHO_PRIOR_MEAN   = 0.10   # 10% center
RHO_PRIOR_WEIGHT = 200    # Soft — allows 2-20% range
LOGIT_RHO_PRIOR  = np.log(RHO_PRIOR_MEAN / (1 - RHO_PRIOR_MEAN))  # logit(0.10)

# ============================================================================
# CITIES AND LAG RANGES
# ============================================================================
CITIES = [
    'Lahore', 'Rawalpindi', 'Faisalabad', 'Gujranwala',
    'Multan', 'Islamabad', 'Sargodha', 'Sheikhupura', 'Dera Ghazi Khan'
]
# Gujrat and Hafizabad excluded: zero dengue cases in 2023-2024 testing period

RAIN_LAGS = [4, 5, 6, 7, 8]          # weeks — fast effect via breeding sites
TEMP_LAGS = [6, 7, 8, 9, 10, 11, 12] # weeks — slow effect via EIP + development

TRAIN_END_YEAR  = 2022
TEST_START_YEAR = 2023

# ============================================================================
# POPULATION INTERPOLATION
# ============================================================================
def get_population_series(city, pop_df, years_array):
    """
    Return weekly N(t) by linearly interpolating between 2017 and 2023 census.
    After 2023, holds at 2023 value.
    """
    row = pop_df[pop_df['City'] == city].iloc[0]
    N_2017 = float(row['Population 2017'])
    N_2023 = float(row['Population 2023'])

    N_arr = np.zeros(len(years_array))
    for i, yr in enumerate(years_array):
        if yr <= 2017:
            N_arr[i] = N_2017
        elif yr >= 2023:
            N_arr[i] = N_2023
        else:
            t = (yr - 2017) / (2023 - 2017)
            N_arr[i] = N_2017 + t * (N_2023 - N_2017)
    return N_arr

# ============================================================================
# SEIR SIMULATOR (V4 — no annual reset, continuous waning)
# ============================================================================
def simulate_seir_v4(params, Rain_z, Temp_z, TempSq_z, years_arr, N_arr):
    """
    Run SEIR model for one city with given parameters and weather inputs.

    Model equations (discrete-time, weekly) with demographic terms:
      S̄(t+1) = S(t) - inc(t) + ω·R(t) + μ·(N(t) - S(t))
      Ē(t+1) = E(t) + inc(t) - (σ+μ)·E(t)
      Ī(t+1) = I(t) + σ·E(t) - (γ+μ)·I(t)
      R̄(t+1) = R(t) + γ·I(t) - (ω+μ)·R(t)
      Then: S(t+1) = S̄(t+1) + ΔN(t+1);  E,I,R unchanged.
      inc(t) = β(t)·S(t)·I(t)/N(t) + λ
      β(t)   = κ·exp(b0 + bR·Rain_z + bT·Temp_z + bT2·TempSq_z)
      Ŷ(t)  = ρ·inc(t)    [reported cases]

    NO annual reset: R carries over continuously across years.
    Population N(t) is dynamic (interpolated from census).
    """
    log_kappa, b0, bR, bT, bT2, logit_rho, log_lambda, logit_E0f, logit_I0f = params

    kappa         = np.exp(log_kappa)
    rho           = expit(logit_rho)               # ρ ∈ (0,1)
    lambda_import = np.exp(log_lambda)
    E0_frac       = 0.01 * expit(logit_E0f)        # max 1% initially exposed
    I0_frac       = 0.01 * expit(logit_I0f)        # max 1% initially infectious
    omega         = OMEGA_FIXED                    # fixed waning rate

    T = len(Rain_z)
    S = np.zeros(T); E = np.zeros(T)
    I = np.zeros(T); R = np.zeros(T)
    inc_t = np.zeros(T)

    # Initial conditions (week 0)
    N0 = N_arr[0]
    S[0] = N0 * (1.0 - E0_frac - I0_frac)
    E[0] = N0 * E0_frac
    I[0] = N0 * I0_frac
    R[0] = 0.0

    for t in range(T):
        # Weather-driven transmission rate
        eta  = b0 + bR * Rain_z[t] + bT * Temp_z[t] + bT2 * TempSq_z[t]
        beta = kappa * np.exp(eta)

        # New infections (force of infection + importation)
        inc = beta * (S[t] * I[t]) / max(N_arr[t], EPS) + lambda_import
        inc = max(min(inc, S[t]), 0.0)  # cannot exceed susceptibles
        inc_t[t] = inc

        if t < T - 1:
            Nt = N_arr[t]
            new_E = inc
            new_I = SIGMA * E[t]
            new_R = GAMMA * I[t]
            wane  = omega * R[t]  # immunity waning back to susceptible

            # Disease dynamics + demographic terms (μ)
            # μ·N = total births (replacing all deaths); deaths removed per-compartment
            S[t+1] = max(S[t] - new_E + wane + MU * (Nt - S[t]), 0.0)
            E[t+1] = max(E[t] + new_E - new_I - MU * E[t], 0.0)
            I[t+1] = max(I[t] + new_I - new_R - MU * I[t], 0.0)
            R[t+1] = max(R[t] + new_R - wane  - MU * R[t], 0.0)

            # Census-based population adjustment (ΔN)
            # Accounts for net population growth beyond μ-based vital dynamics
            total = S[t+1] + E[t+1] + I[t+1] + R[t+1]
            delta_N = N_arr[t+1] - total
            if delta_N >= 0:
                # Population growth: all new individuals enter S
                S[t+1] += delta_N
            else:
                # Population decline (rare): remove proportionally
                if total > EPS:
                    scale = N_arr[t+1] / total
                    S[t+1] *= scale; E[t+1] *= scale
                    I[t+1] *= scale; R[t+1] *= scale

    return rho * inc_t, S, E, I, R

# ============================================================================
# LOSS FUNCTION (MSE + biological penalties)
# ============================================================================
def build_loss(Rain_z, Temp_z, TempSq_z, years_arr, N_arr,
               y_obs, train_mask, temp_mean, temp_std):
    """
    Returns a loss function for Nelder-Mead to minimize.

    Loss = MSE(training) + Penalty_temperature + Prior_rho

    Penalty_temperature:
      Forces the quadratic β(T) to have its peak (vertex) inside 20-35°C.
      Also forces b_T2 < 0 (inverted-U, not U-shape).
      Applied as a large quadratic penalty.

    Prior_rho (weak informative prior):
      Prior(ρ) = w × (logit(ρ) - logit(0.10))²
      Centers ρ at 10% but allows 2-20% range.
      Scientific basis: Pakistan surveillance captures ~5-15% (Bhatt 2013).
    """
    def loss(params):
        pred, *_ = simulate_seir_v4(params, Rain_z, Temp_z, TempSq_z,
                                    years_arr, N_arr)
        mse = np.mean((y_obs[train_mask] - pred[train_mask]) ** 2)

        # --- Temperature biological penalty ---
        b0_val, bR_val, bT_val, bT2_val = params[1], params[2], params[3], params[4]

        temp_penalty = 0.0
        # Must be inverted-U: bT2 must be negative
        if bT2_val >= 0:
            temp_penalty += 1e6 * (bT2_val + 0.001) ** 2

        # Vertex of β(T) parabola (in standardized units): T* = -bT / (2*bT2)
        # Convert to Celsius: T*_C = T* * std(Temp) + mean(Temp)
        # We check if T*_C is within [20, 35]
        if bT2_val < -EPS:
            T_vertex_std = -bT_val / (2.0 * bT2_val)
            
            # Convert vertex back to Celsius
            T_vertex_C = T_vertex_std * temp_std + temp_mean
            
            # Apply heavy penalty if outside biological window [20, 35]
            if T_vertex_C < 20.0 or T_vertex_C > 35.0:
                dist = min(abs(T_vertex_C - 20.0), abs(T_vertex_C - 35.0))
                temp_penalty += 1e6 * (dist ** 2)

        # --- Weak informative prior on ρ ---
        logit_rho = params[5]
        rho_penalty = RHO_PRIOR_WEIGHT * (logit_rho - LOGIT_RHO_PRIOR) ** 2

        return mse + temp_penalty + rho_penalty

    return loss

# ============================================================================
# INITIAL PARAMETER GUESS
# ============================================================================
def get_x0():
    """
    Starting point for Nelder-Mead optimization.
    All parameters in their transformed (unconstrained) form.

    logit(0.10) = -2.197  → ρ starts at 10% (prior center)
    log(1/156)  = -5.05   → ω starts at ~3 year waning
    """
    return [
        np.log(0.5),        # log_kappa: κ starts at 0.5
        -2.0,               # b0: baseline log-transmission
        0.05,               # bR: rainfall has small positive effect
        0.70,               # bT: temperature has positive effect
        -0.10,              # bT2: negative (inverted-U)
        np.log(0.10 / 0.90), # logit_rho: ρ starts at 10% (prior center)
        np.log(5.0),        # log_lambda: ~5 imported cases/week
        -4.6,               # logit_E0f: ~0.01% initially exposed
        -4.6                # logit_I0f: ~0.01% initially infectious
    ]

# ============================================================================
# SINGLE CITY + SINGLE LAG COMBINATION RUN
# ============================================================================
def run_single_combo(args):
    """
    Run one SEIR optimization for one city and one lag combination.
    Called in parallel by multiprocessing pool.
    N_2017 and N_2023 passed directly to avoid multiprocessing global issues.
    """
    city, rain_lag, temp_lag, df_city, N_2017, N_2023 = args

    try:
        # --- Apply lags ---
        df = df_city.copy()
        df['Rain_L']   = df['Rainfall'].shift(rain_lag)
        df['Temp_L']   = df['Temperature'].shift(temp_lag)
        df['TempSq_L'] = df['Temp_L'] ** 2
        df = df.dropna(subset=['Rain_L','Temp_L']).reset_index(drop=True)

        if len(df) < 100:
            return None

        # --- Standardize using training period only (prevent data leakage) ---
        train_mask = df['Year'].values <= TRAIN_END_YEAR
        test_mask  = df['Year'].values >= TEST_START_YEAR

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            return None

        cols = ['Rain_L', 'Temp_L', 'TempSq_L']
        mu  = df.loc[train_mask, cols].mean()
        std = df.loc[train_mask, cols].std().replace(0, 1.0)
        df_z = (df[cols] - mu) / std

        Rain_z   = df_z['Rain_L'].values
        Temp_z   = df_z['Temp_L'].values
        TempSq_z = df_z['TempSq_L'].values
        y_obs    = df['Number of Dengue Cases'].values
        years    = df['Year'].values.astype(float)

        # Build N(t) directly from census values (no global needed)
        N_aligned = np.zeros(len(years))
        for i, yr in enumerate(years):
            if yr <= 2017:
                N_aligned[i] = N_2017
            elif yr >= 2023:
                N_aligned[i] = N_2023
            else:
                t = (yr - 2017) / (2023 - 2017)
                N_aligned[i] = N_2017 + t * (N_2023 - N_2017)

        temp_mean_val = float(mu['Temp_L'])
        temp_std_val = float(std['Temp_L'])

        # --- Optimize ---
        loss_fn = build_loss(Rain_z, Temp_z, TempSq_z, years,
                             N_aligned, y_obs, train_mask, temp_mean_val, temp_std_val)

        result = minimize(loss_fn, get_x0(), method='Nelder-Mead',
                          options={'maxiter': 50000, 'maxfev': 100000,
                                   'xatol': 1e-6, 'fatol': 1e-6,
                                   'disp': False})

        # --- Evaluate ---
        pred, S, E, I, R = simulate_seir_v4(result.x, Rain_z, Temp_z,
                                             TempSq_z, years, N_aligned)

        if test_mask.sum() < 2:
            return None

        train_r = float(np.corrcoef(y_obs[train_mask], pred[train_mask])[0,1])
        test_r  = float(np.corrcoef(y_obs[test_mask],  pred[test_mask])[0,1])

        # Extract fitted parameters
        p = result.x
        from scipy.special import expit
        rho_val   = float(expit(p[5]))
        kappa_val = float(np.exp(p[0]))
        omega_val = float(OMEGA_FIXED)

        return {
            'City': city, 'Rain_Lag': rain_lag, 'Temp_Lag': temp_lag,
            'Train_r': train_r, 'Test_r': test_r,
            'rho': rho_val, 'kappa': kappa_val,
            'b0': float(p[1]), 'bR': float(p[2]),
            'bT': float(p[3]), 'bT2': float(p[4]),
            'lambda_import': float(np.exp(p[6])),
            'omega': omega_val,
            'params': result.x.tolist(),
            'optimizer_success': result.success,
        }
    except Exception as ex:
        return None

# ============================================================================
# SAVE PER-CITY PREDICTIONS FOR BEST LAG
# ============================================================================
def save_city_predictions(city, best_params, rain_lag, temp_lag,
                           df_city, N_2017, N_2023):
    """Generate and save weekly observed vs predicted for one city."""
    df = df_city.copy()
    df['Rain_L']   = df['Rainfall'].shift(rain_lag)
    df['Temp_L']   = df['Temperature'].shift(temp_lag)
    df['TempSq_L'] = df['Temp_L'] ** 2
    df = df.dropna(subset=['Rain_L','Temp_L']).reset_index(drop=True)

    train_mask = df['Year'].values <= TRAIN_END_YEAR

    cols = ['Rain_L', 'Temp_L', 'TempSq_L']
    mu  = df.loc[train_mask, cols].mean()
    std = df.loc[train_mask, cols].std().replace(0, 1.0)
    df_z = (df[cols] - mu) / std

    years    = df['Year'].values.astype(float)
    N_aligned = np.zeros(len(years))
    for i, yr in enumerate(years):
        if yr <= 2017:
            N_aligned[i] = N_2017
        elif yr >= 2023:
            N_aligned[i] = N_2023
        else:
            t = (yr - 2017) / (2023 - 2017)
            N_aligned[i] = N_2017 + t * (N_2023 - N_2017)

    from scipy.special import expit
    params = np.array(best_params)
    pred, S, E, I, R = simulate_seir_v4(params,
                                         df_z['Rain_L'].values,
                                         df_z['Temp_L'].values,
                                         df_z['TempSq_L'].values,
                                         years, N_aligned)

    df['Predicted'] = pred
    df['S'] = S; df['E'] = E; df['I'] = I; df['R'] = R
    df['Period'] = np.where(df['Year'] <= TRAIN_END_YEAR, 'Training', 'Testing')

    out_path = os.path.join(RESULT_DIR, f'{city}_Predictions.xlsx')
    df.to_excel(out_path, index=False)
    print(f"  ✓ Saved predictions: {city}_Predictions.xlsx")

# (No global data needed — all data passed via task args)

# ============================================================================
# MAIN
# ============================================================================
def main():
    global pop_df_global

    print("=" * 70)
    print("S1: SEIR V4 GRID SEARCH — Punjab Dengue Forecasting")
    print("=" * 70)
    print(f"\nCities: {len(CITIES)}")
    print(f"Rain lags:  {RAIN_LAGS} weeks")
    print(f"Temp lags:  {TEMP_LAGS} weeks")
    print(f"Total runs: {len(CITIES)} × {len(RAIN_LAGS)} × {len(TEMP_LAGS)} = "
          f"{len(CITIES)*len(RAIN_LAGS)*len(TEMP_LAGS)}")
    print(f"\nModel: V4 — No reset | ρ prior=10% | Temp penalty 20-35°C | Nelder-Mead")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading data...")
    df_all   = pd.read_excel(DATA_FILE)
    pop_df_global = pd.read_excel(POP_FILE)

    # Verify cities
    missing = [c for c in CITIES if c not in df_all['City'].unique()]
    if missing:
        print(f"WARNING: Cities not in data: {missing}")

    # --- Build task list (pass census values directly to each worker) ---
    tasks = []
    for city in CITIES:
        df_city = df_all[df_all['City'] == city].copy().reset_index(drop=True)
        pop_row = pop_df_global[pop_df_global['City'] == city].iloc[0]
        N_2017 = float(pop_row['Population 2017'])
        N_2023 = float(pop_row['Population 2023'])
        for rl in RAIN_LAGS:
            for tl in TEMP_LAGS:
                tasks.append((city, rl, tl, df_city, N_2017, N_2023))

    print(f"\nTotal optimization tasks: {len(tasks)}")
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_cores} CPU cores (leaving 1 free)")
    print("\nStarting grid search...\n")

    # --- Run parallel ---
    all_results = []
    with multiprocessing.Pool(processes=n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_combo, tasks), 1):
            if result is not None:
                all_results.append(result)
            if i % 25 == 0 or i == len(tasks):
                print(f"  Progress: {i}/{len(tasks)} runs | "
                      f"Results collected: {len(all_results)}")

    # --- Save all grid search results ---
    cols_to_save = ['City','Rain_Lag','Temp_Lag','Train_r','Test_r',
                    'rho','kappa','b0','bR','bT','bT2',
                    'lambda_import','omega','optimizer_success']
    results_df = pd.DataFrame([{k: r[k] for k in cols_to_save} for r in all_results])
    results_df.to_excel(os.path.join(RESULT_DIR, 'V4_GridSearch_AllResults.xlsx'), index=False)
    print(f"\n✓ Saved: V4_GridSearch_AllResults.xlsx ({len(results_df)} results)")

    # --- Find optimal lag per city ---
    optimal_rows = []
    best_params_by_city = {}

    for city in CITIES:
        city_res = results_df[results_df['City'] == city]
        if city_res.empty:
            print(f"  WARNING: No results for {city}")
            continue
        best_idx = city_res['Train_r'].idxmax()
        best     = city_res.loc[best_idx]
        optimal_rows.append(best.to_dict())

        # Find the full result (with params) for this city
        city_full = [r for r in all_results
                     if r['City'] == city and
                        r['Rain_Lag'] == int(best['Rain_Lag']) and
                        r['Temp_Lag'] == int(best['Temp_Lag'])]
        if city_full:
            best_params_by_city[city] = {
                'params': city_full[0]['params'],
                'rain_lag': int(best['Rain_Lag']),
                'temp_lag': int(best['Temp_Lag'])
            }

    optimal_df = pd.DataFrame(optimal_rows)
    optimal_df.to_excel(os.path.join(RESULT_DIR, 'V4_Optimal_Lags_ByCity.xlsx'), index=False)
    print(f"✓ Saved: V4_Optimal_Lags_ByCity.xlsx")

    # --- Save per-city predictions for best lag ---
    print("\nGenerating per-city prediction files...")
    for city in CITIES:
        if city not in best_params_by_city:
            continue
        info   = best_params_by_city[city]
        df_city = df_all[df_all['City'] == city].copy().reset_index(drop=True)
        pop_row = pop_df_global[pop_df_global['City'] == city].iloc[0]
        save_city_predictions(city, info['params'],
                              info['rain_lag'], info['temp_lag'],
                              df_city,
                              float(pop_row['Population 2017']),
                              float(pop_row['Population 2023']))

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\n{'City':<20} {'Rain':>6} {'Temp':>6} {'ρ%':>8} {'Train r':>9} {'Test r':>9}")
    print("-" * 65)
    for _, row in optimal_df.sort_values('Test_r', ascending=False).iterrows():
        print(f"{row['City']:<20} {int(row['Rain_Lag']):>6} {int(row['Temp_Lag']):>6} "
              f"{row['rho']*100:>7.1f}% {row['Train_r']:>9.3f} {row['Test_r']:>9.3f}")

    mean_test_r = optimal_df['Test_r'].mean()
    mean_rho    = optimal_df['rho'].mean() * 100
    print("-" * 65)
    print(f"{'MEAN':<20} {'':>6} {'':>6} {mean_rho:>7.1f}% {'':>9} {mean_test_r:>9.3f}")

    print(f"\n✓ All outputs saved to: {RESULT_DIR}")
    print("✓ Next step: run S2_Statistical_Tests.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
